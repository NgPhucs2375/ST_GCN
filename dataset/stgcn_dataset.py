# dataset/stgcn_dataset.py : Định nghĩa lớp STGCNDataset để tải và xử lý dữ liệu chuỗi thời gian cho mô hình ST-GCN
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class STGCNDataset(Dataset):
    def __init__(self, npz_path: str):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = np.load(path, allow_pickle=True)
        self.sequences = data["sequences"]
        self.labels = data["labels"].tolist()
        self.label_to_index = self._build_label_map(self.labels)

    @staticmethod
    def _build_label_map(labels) -> Dict[str, int]:
        unique = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique)}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq = self.sequences[idx]
        label = self.labels[idx]
        x = torch.from_numpy(seq).float()  # (T, V, C)
        y = self.label_to_index[label]
        return x, y
