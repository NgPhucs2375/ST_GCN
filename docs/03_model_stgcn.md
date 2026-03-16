# 03 — ST-GCN (PyTorch + torch-geometric) trong repo này

File chính: [models/stgcn.py](../models/stgcn.py)

## Input/Output

- Input vào model: `x` shape `(N, C, T, V)`
  - `N`: batch size
  - `C`: số kênh (2/3/4/6)
  - `T`: số frame cố định
  - `V=21`: số landmark
- Output: logits shape `(N, num_classes)`

## Spatial graph convolution (torch-geometric)

- Mỗi frame là một đồ thị 21 node.
- `build_hand_edge_index()` tạo `edge_index` cho 21 node.
- `SpatialGCN` sẽ:
  1) reshape dữ liệu để chạy `GCNConv` cho từng frame
  2) build edge_index theo batch với offset node (để các graph không nối lẫn)

Điểm quan trọng:
- torch-geometric expects node features shape `(num_nodes_total, in_channels)`
- vì ta có N batch và T frame, số graph = `N*T`

## Temporal convolution

Sau spatial GCN, mỗi block có Temporal CNN:

- `Conv2d(kernel=(temporal_kernel, 1))` để học chuyển động theo thời gian

## Residual

Giống ResNet: nếu đổi kênh hoặc stride, residual đi qua `1x1 conv`.

## Lưu ý thiết kế

Bản này là phiên bản "dễ hiểu để học" (skeleton):
- chưa implement partitioning strategy (A, B, C matrices như paper gốc)
- chưa có attention

Sau khi nắm vững, bạn có thể nâng cấp dần.
