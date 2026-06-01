# 01 — Tổng quan pipeline

## Mục tiêu

Biến webcam stream thành sequence landmarks ổn định, sau đó train ST-GCN để phân loại cử chỉ tay.

## Luồng thực tế trong repo

1. Browser chạy MediaPipe Hands trong `web/` và xuất 21 landmark cho mỗi frame.
2. Người dùng ghi lại nhiều frame thành một sequence và lưu ra JSON.
3. `tools/data_quality.py` quét sequence lỗi, trùng, ngắn, hoặc tracking bất thường.
4. `tools/convert_sequences.py` chuẩn hóa, pad/trim, rồi ghi ra NPZ.
5. `train.py` đọc NPZ, chia train/val, train ST-GCN và lưu checkpoint.
6. `tools/infer.py` suy luận offline trên JSON hoặc NPZ.
7. `tools/demo_webcam.py` chạy realtime, hiển thị nhãn và có thể gửi phím/lệnh theo `gesture_config.json`.

## Vì sao tách web và Python

- Web phù hợp cho capture realtime và lưu dữ liệu thô ngay trên trình duyệt.
- Python phù hợp cho batch processing, training, và kiểm soát checkpoint/metrics.

## Dữ liệu đi qua hệ thống

- JSON thô: `label` + `frames`.
- NPZ: `sequences` shape `(N, T, V, C)` và `labels` shape `(N,)`.
- Checkpoint: `stgcn_best.pt`, `stgcn_last.pt`, `labels.json`, `confusion_matrix.pt`.

## Ghi chú về feature

Repo hiện có các nhánh feature sau:

- `C=2`: x, y.
- `C=3`: x, y, z.
- `C=4`: x, y + velocity.
- `C=6`: x, y + velocity + acceleration.
- `C=9`: x, y, z + velocity + acceleration.

## Hướng triển khai sau khi train

- Demo webcam trên máy local.
- Export sang runtime khác nếu muốn chạy in-browser hoặc server-side inference.
