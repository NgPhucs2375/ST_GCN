import json
import time
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stgcn import STGCN, build_hand_edge_index

# Small variant used earlier
from tools.evaluate_on_4ch import load_state

MODEL_PATH = Path("outputs_resume/stgcn_best.pt")
LABELS_PATH = Path("outputs_resume/labels.json")
DATA_PATH = Path("data/processed/train_data_4ch.npz")
OUT_CKPT = Path("outputs_resume/stgcn_finetuned_4ch.pt")

BATCH = 32
DEVICE = torch.device("cpu")
EPOCHS = 3
LR = 1e-4
VAL_RATIO = 0.1
SEED = 42
LOG_PATH = Path("outputs_resume/finetune_4ch.log")


def infer_in_channels_from_state(state):
    key = "data_bn.weight"
    if key not in state:
        raise RuntimeError("data_bn.weight not found in checkpoint")
    bn_size = int(state[key].numel())
    if bn_size % 21 != 0:
        raise RuntimeError(f"Invalid data_bn size {bn_size}")
    return bn_size // 21


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    labels = np.asarray(labels)
    train_idx = []
    val_idx = []
    classes = np.unique(labels)
    for c in classes:
        idx = np.where(labels == c)[0].tolist()
        rng.shuffle(idx)
        nval = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:nval])
        train_idx.extend(idx[nval:])
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def build_model_from_state(state, in_channels: int, num_classes: int):
    # decide variant by checking fc.weight shape
    fc_w = state.get('fc.weight') if isinstance(state, dict) else None
    target_feat = None
    if fc_w is not None:
        try:
            target_feat = int(fc_w.shape[1])
        except Exception:
            target_feat = None
    if target_feat == 256:
        # import small variant here to avoid circular deps
        from tools.evaluate_on_4ch import STGCNSmall
        model = STGCNSmall(in_channels=in_channels, num_classes=num_classes, edge_index=build_hand_edge_index())
    else:
        model = STGCN(in_channels=in_channels, num_classes=num_classes, edge_index=build_hand_edge_index())
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = crit(out, y)
            loss_sum += float(loss.item()) * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    assert MODEL_PATH.exists(), MODEL_PATH
    assert LABELS_PATH.exists(), LABELS_PATH
    assert DATA_PATH.exists(), DATA_PATH

    labels_map = json.loads(LABELS_PATH.read_text(encoding='utf-8'))
    label_to_idx = labels_map
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    data = np.load(DATA_PATH, allow_pickle=True)
    seq = data['sequences']  # (N,T,V,C)
    labels = data['labels']
    N, T, V, C = seq.shape
    print('Loaded', DATA_PATH, 'shape', seq.shape)

    # convert labels to indices
    y = np.array([label_to_idx[s] for s in labels.tolist()], dtype=np.int64)

    # stratified split
    train_idx, val_idx = stratified_split(y, VAL_RATIO, seed=SEED)
    print(f"Train/Val split: {len(train_idx)}/{len(val_idx)}")

    # prepare tensors
    # model expects (N,C,T,V)
    seq = seq.transpose(0,3,1,2)
    X = torch.from_numpy(seq).float()
    Y = torch.from_numpy(y).long()

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH, shuffle=False)

    state = load_state(MODEL_PATH)
    in_channels = infer_in_channels_from_state(state)
    assert in_channels == C, f"in_channels mismatch: state={in_channels} data={C}"

    model = build_model_from_state(state, in_channels, num_classes)
    try:
        model.load_state_dict(state, strict=True)
        print('Loaded pretrained checkpoint into model')
    except Exception as e:
        print('Warning: strict load failed, loading with strict=False')
        model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        running_corr = 0
        running_total = 0
        t0 = time.time()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            opt.zero_grad()
            out = model(x_batch)
            loss = crit(out, y_batch)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * x_batch.size(0)
            preds = out.argmax(dim=1)
            running_corr += (preds == y_batch).sum().item()
            running_total += x_batch.size(0)
        train_loss = running_loss / running_total
        train_acc = running_corr / running_total
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        t1 = time.time()
        line = f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | time={(t1-t0):.1f}s"
        print(line)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    # save checkpoint
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict()}, OUT_CKPT)
    print('Saved finetuned checkpoint to', OUT_CKPT)


if __name__ == '__main__':
    main()
