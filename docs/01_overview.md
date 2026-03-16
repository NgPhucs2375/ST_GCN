# 01 — Tổng quan pipeline

## Mục tiêu

Biến webcam stream → landmarks bàn tay → tensor chuỗi thời gian → phân loại cử chỉ.

## End-to-end flow

1) Browser chạy MediaPipe Hands
   - Mỗi frame trả về 21 landmark (x, y, z)
2) Web UI buffer liên tiếp `T` frame để tạo một sequence
3) Lưu sequence ra JSON (thô)
4) Python convert:
   - normalize (center + scale)
   - optional velocity
   - pad/trim để cố định `T`
   - lưu `npz` cho training
5) PyTorch train ST-GCN
6) Evaluate:
   - accuracy (tỉ lệ đúng)
   - confusion matrix (nhầm lẫn giữa các lớp)

## Tại sao tách web và python?

- Web: tối ưu cho capture realtime + chạy trên trình duyệt.
- Python: tối ưu cho xử lý batch + train nhanh + debug thuận tiện.

Sau khi model ổn định, bạn có thể triển khai inference:
- Server inference (PyTorch backend)
- In-browser inference (ONNX Runtime Web / TFJS)
