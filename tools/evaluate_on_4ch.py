import time
import json
from pathlib import Path
import numpy as np
import torch
import sys
from pathlib import Path
# allow running from tools/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.stgcn import STGCN, build_hand_edge_index
from torch import nn


# Small-version of STGCN matching older checkpoints (layers: 64,64,128,256)
class STGCNSmall(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, edge_index: torch.Tensor, num_nodes: int = 21, dropout: float = 0.0):
        super().__init__()
        from models.stgcn import STGCNBlock
        self.num_nodes = num_nodes
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.layer1 = STGCNBlock(in_channels, 64, edge_index, num_nodes, dropout=dropout)
        self.layer2 = STGCNBlock(64, 64, edge_index, num_nodes, dropout=dropout)
        self.layer3 = STGCNBlock(64, 128, edge_index, num_nodes, stride=2, dropout=dropout)
        self.layer4 = STGCNBlock(128, 256, edge_index, num_nodes, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

MODEL_PATH = Path("outputs_resume/stgcn_best.pt")
LABELS_PATH = Path("outputs_resume/labels.json")
DATA_PATH = Path("data/processed/train_data_4ch.npz")
BATCH = 32
DEVICE = torch.device("cpu")


def load_state(path):
    s = torch.load(path, map_location=DEVICE)
    if isinstance(s, dict) and "state_dict" in s:
        s = s["state_dict"]
    return s


def infer_in_channels_from_state(state):
    key = "data_bn.weight"
    if key not in state:
        raise RuntimeError("data_bn.weight not found in checkpoint")
    bn_size = int(state[key].numel())
    if bn_size % 21 != 0:
        raise RuntimeError(f"Invalid data_bn size {bn_size}")
    return bn_size // 21


def main():
    import traceback
    assert MODEL_PATH.exists(), MODEL_PATH
    assert LABELS_PATH.exists(), LABELS_PATH
    assert DATA_PATH.exists(), DATA_PATH

    labels_map = json.loads(LABELS_PATH.read_text(encoding='utf-8'))
    idx_to_label = {v:k for k,v in labels_map.items()}

    data = np.load(DATA_PATH, allow_pickle=True)
    seq = data['sequences']  # (N,T,V,C)
    labels = data['labels']
    print('data', DATA_PATH, 'seq shape', seq.shape, 'labels', labels.shape)

    state = load_state(MODEL_PATH)
    in_channels = infer_in_channels_from_state(state)
    num_classes = len(idx_to_label)
    edge_index = build_hand_edge_index()

    # choose model variant matching checkpoint feature size
    fc_weight = state.get('fc.weight') if isinstance(state, dict) else None
    target_feat = None
    if fc_weight is not None:
        try:
            target_feat = int(fc_weight.shape[1])
        except Exception:
            target_feat = None

    if target_feat == 256:
        model = STGCNSmall(in_channels=in_channels, num_classes=num_classes, edge_index=edge_index)
    else:
        model = STGCN(in_channels=in_channels, num_classes=num_classes, edge_index=edge_index)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    # prepare labels -> indices
    label_to_idx = labels_map
    y = np.array([label_to_idx[s] for s in labels.tolist()], dtype=np.int64)

    # sequences currently shape (N,T,V,C) but model expects (N,C,T,V)
    N, T, V, C = seq.shape
    assert C == in_channels, f"data channels {C} != model in_channels {in_channels}"

    seq = seq.transpose(0,3,1,2)  # (N,C,T,V)

    correct = 0
    total = 0
    timings = []

    try:
        with torch.no_grad():
            for i in range(0, N, BATCH):
                batch = seq[i:i+BATCH]
            bsize = batch.shape[0]
            x = torch.from_numpy(batch).float().to(DEVICE)
            t0 = time.perf_counter()
            out = model(x)
            t1 = time.perf_counter()
            logits = out.cpu().numpy()
            preds = logits.argmax(axis=1)
            timings.append((t1-t0)/bsize)
            correct += (preds == y[i:i+BATCH]).sum()
            total += bsize

    except Exception as e:
        print('Evaluation error:')
        traceback.print_exc()
        return

    acc = float(correct)/total
    mean_latency = float(np.mean(timings))
    p50 = float(np.percentile(timings,50))
    p95 = float(np.percentile(timings,95))
    print(f"Accuracy on {total} samples: {acc*100:.2f}%")
    print(f"Per-sample inference time (CPU): mean={mean_latency*1000:.2f} ms p50={p50*1000:.2f} ms p95={p95*1000:.2f} ms")


if __name__ == '__main__':
    main()
