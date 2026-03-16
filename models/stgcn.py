# models/stgcn.py

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv


HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def build_hand_edge_index(num_nodes: int = 21) -> torch.Tensor:
    # Build undirected edges for the 21 hand landmarks.
    edges = []
    for a, b in HAND_CONNECTIONS:
        edges.append((a, b))
        edges.append((b, a))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if num_nodes != 21:
        raise ValueError("Only 21-node hand graph is supported")
    return edge_index


def build_batched_edge_index(edge_index: torch.Tensor, num_graphs: int, num_nodes: int) -> torch.Tensor:
    # Repeat edge_index with node offsets so each (T, N) graph stays disconnected.
    device = edge_index.device
    edges_per_graph = edge_index.size(1)
    offsets = torch.arange(num_graphs, device=device) * num_nodes
    offsets = offsets.repeat_interleave(edges_per_graph)
    edge_index = edge_index.repeat(1, num_graphs)
    return edge_index + offsets.unsqueeze(0)


class SpatialGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_index: torch.Tensor, num_nodes: int):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.register_buffer("edge_index", edge_index)
        self.num_nodes = num_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x = x.view(n * t * v, c)

        batched_edge_index = build_batched_edge_index(self.edge_index, n * t, v)
        x = self.gcn(x, batched_edge_index)

        x = x.view(n, t, v, -1).permute(0, 3, 1, 2).contiguous()
        return x


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_index: torch.Tensor,
        num_nodes: int,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (temporal_kernel - 1) // 2
        # Spatial GCN is applied per-frame using torch-geometric.
        self.gcn = SpatialGCN(in_channels, out_channels, edge_index, num_nodes)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                padding=(padding, 0),
                stride=(stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        edge_index: torch.Tensor,
        num_nodes: int = 21,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        # Normalize across node dimension to stabilize input features.
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.layer1 = STGCNBlock(in_channels, 64, edge_index, num_nodes, dropout=dropout)
        self.layer2 = STGCNBlock(64, 64, edge_index, num_nodes, dropout=dropout)
        self.layer3 = STGCNBlock(64, 128, edge_index, num_nodes, stride=2, dropout=dropout)
        self.layer4 = STGCNBlock(128, 256, edge_index, num_nodes, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
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
