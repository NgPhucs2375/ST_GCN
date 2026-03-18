import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow running as "python tools/compare_checkpoints.py" from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stgcn import STGCN, build_hand_edge_index


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_labels_file(model_path: Path) -> Optional[Path]:
    candidates = [
        model_path.with_name("labels.json"),
        model_path.parent / "labels.json",
        model_path.parent.parent / "labels.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a state_dict")
    return state


def evaluate_model(
    model_path: Path,
    labels_path: Path,
    sequences: np.ndarray,
    labels_str: List[str],
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, int]:
    with labels_path.open("r", encoding="utf-8") as f:
        label_to_index = json.load(f)

    missing = sorted(set(labels_str) - set(label_to_index.keys()))
    if missing:
        raise ValueError(f"Missing labels in labels.json: {missing}")

    y_true = np.array([label_to_index[x] for x in labels_str], dtype=np.int64)
    num_classes = len(label_to_index)
    in_channels = int(sequences.shape[-1])

    model = STGCN(
        in_channels=in_channels,
        num_classes=num_classes,
        edge_index=build_hand_edge_index(),
    ).to(device)

    state = load_state_dict(model_path, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    y_pred_parts = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            x = torch.from_numpy(sequences[i : i + batch_size]).float().permute(0, 3, 1, 2).to(device)
            p = model(x).argmax(dim=1).cpu().numpy()
            y_pred_parts.append(p)

    y_pred = np.concatenate(y_pred_parts)
    acc = float((y_pred == y_true).mean())

    # Macro average of per-class recalls for classes present in y_true.
    recalls = []
    for c in range(num_classes):
        idx = y_true == c
        total = int(idx.sum())
        if total > 0:
            recalls.append(float((y_pred[idx] == c).mean()))
    macro_recall = float(np.mean(recalls)) if recalls else 0.0

    return acc, macro_recall, num_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.npz", help="NPZ path for comparison set")
    parser.add_argument("--root", default=".", help="Root folder to search checkpoints")
    parser.add_argument("--pattern", default="stgcn_*.pt", help="Checkpoint filename pattern")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    data = np.load(data_path, allow_pickle=True)
    sequences = data["sequences"]
    labels_str = data["labels"].tolist()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpts = sorted(Path(args.root).rglob(args.pattern))
    if not ckpts:
        raise RuntimeError("No checkpoints found")

    seen_hash = set()
    rows = []
    for ckpt in ckpts:
        labels_path = find_labels_file(ckpt)
        if labels_path is None:
            rows.append((str(ckpt), "-", "-", "-", "skip(no labels.json)", "-"))
            continue

        ckpt_hash = sha256(ckpt)
        if ckpt_hash in seen_hash:
            rows.append((str(ckpt), "-", "-", "-", "skip(duplicate weights)", ckpt_hash[:12]))
            continue
        seen_hash.add(ckpt_hash)

        try:
            acc, macro_recall, num_classes = evaluate_model(
                ckpt,
                labels_path,
                sequences,
                labels_str,
                args.batch_size,
                device,
            )
            rows.append((
                str(ckpt),
                str(labels_path),
                f"{acc:.4f}",
                f"{macro_recall:.4f}",
                f"ok(classes={num_classes})",
                ckpt_hash[:12],
            ))
        except Exception as e:
            rows.append((str(ckpt), str(labels_path), "-", "-", f"error({e})", ckpt_hash[:12]))

    # Sort by accuracy descending, keep errors at bottom.
    def sort_key(row):
        try:
            return (0, -float(row[2]))
        except Exception:
            return (1, 0.0)

    rows = sorted(rows, key=sort_key)

    print(f"Dataset: {data_path}")
    print(f"Samples: {len(labels_str)} | Classes in data: {len(set(labels_str))}")
    print("")
    print("checkpoint | labels | acc | macro_recall | status | hash12")
    print("-" * 140)
    for r in rows:
        print(" | ".join(r))


if __name__ == "__main__":
    main()
