# 03 — ST-GCN trong repo này

File chính: [models/stgcn.py](../models/stgcn.py)

## Input / Output

- Input: `x` shape `(N, C, T, V)`.
  - `N`: batch size.
  - `C`: số kênh feature.
  - `T`: số frame cố định.
  - `V=21`: số landmark.
- Output: logits shape `(N, num_classes)`.

## Graph tay

`build_hand_edge_index()` tạo đồ thị xương tay 21 node bằng các cạnh chuẩn MediaPipe hand.

Mỗi frame được xem như một graph riêng, nên `SpatialGCN` sẽ:

1. reshape batch thành `N*T` graph.
2. tạo `edge_index` batched với offset node.
3. chạy `GCNConv` cho từng graph độc lập.

## Temporal path

Sau spatial GCN, mỗi block đi qua temporal convolution:

- `Conv2d(kernel=(temporal_kernel, 1))`.
- kernel mặc định là 9.
- stride có thể giảm chiều thời gian ở các block sau.

## Block hiện tại

`STGCN` trong repo dùng 4 block chính:

- 64 channels.
- 64 channels.
- 128 channels, stride 2.
- 256 channels, stride 2.

Sau đó là adaptive average pooling và một fully-connected layer ra `num_classes`.

## Residual và normalization

- Nếu không đổi shape, residual là identity.
- Nếu đổi kênh hoặc stride, residual đi qua `1x1 conv + BatchNorm`.
- `data_bn` chuẩn hóa theo node dimension để ổn định input feature.

## Ý nghĩa thực tế

Mô hình này là biến thể dễ đọc của ST-GCN, đủ cho bài toán gesture classification trong repo:

- dễ train.
- dễ debug shape.
- phù hợp để cải tiến tiếp bằng việc thêm augmentation, dropout, scheduler, hoặc kiến trúc sâu hơn.
