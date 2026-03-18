import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def load_labels(path: Path, num_classes: int) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        label_to_index: Dict[str, int] = json.load(f)

    index_to_label = ["" for _ in range(num_classes)]
    for label, idx in label_to_index.items():
        if 0 <= idx < num_classes:
            index_to_label[idx] = label

    for i, label in enumerate(index_to_label):
        if not label:
            index_to_label[i] = f"class_{i}"

    return index_to_label


def summarize(cm: torch.Tensor, labels: List[str], topk: int) -> None:
    total = int(cm.sum().item())
    correct = int(cm.diag().sum().item())
    overall = correct / total if total > 0 else 0.0

    print(f"Total samples: {total}")
    print(f"Overall accuracy: {overall:.3f}")
    print("")

    row_sums = cm.sum(dim=1)
    for i, label in enumerate(labels):
        total_i = int(row_sums[i].item())
        correct_i = int(cm[i, i].item())
        acc_i = correct_i / total_i if total_i > 0 else 0.0

        row = cm[i].clone()
        row[i] = 0
        values, indices = torch.topk(row, k=min(topk, row.numel()))
        confusions = []
        for v, idx in zip(values.tolist(), indices.tolist()):
            if v <= 0:
                continue
            confusions.append(f"{labels[idx]}:{int(v)}")

        conf_str = ", ".join(confusions) if confusions else "-"
        print(f"{label:>8} | acc {acc_i:.3f} ({correct_i}/{total_i}) | top confusions: {conf_str}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", required=True, help="Path to confusion_matrix.pt")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    cm = torch.load(args.confusion, map_location="cpu")
    if not torch.is_tensor(cm):
        cm = torch.tensor(cm)

    labels = load_labels(Path(args.labels), cm.size(0))
    summarize(cm, labels, args.topk)


if __name__ == "__main__":
    main()
