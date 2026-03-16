# 05 — Checklist triển khai web game (gesture → game loop)

## Mục tiêu realtime

- Render loop: 60 FPS (requestAnimationFrame)
- Gesture inference: 15–30 FPS (chạy mỗi 2–3 frame)
- Output ổn định: dùng smoothing + cooldown

## Checklist kỹ thuật

- Chốt `T` (vd 30) và `C` (vd x,y + velocity)
- Collect dữ liệu đủ lớn và cân bằng giữa các lớp
- Train + xem confusion matrix
- Nếu nhầm nhiều:
  - tăng data ở các class hay nhầm
  - thử tăng T hoặc bật velocity

## Tối ưu chất lượng dữ liệu (Data quality)

- Thêm dữ liệu: tăng số sample mỗi lớp, ưu tiên các lớp hay nhầm trong confusion matrix.
- Cân bằng lớp: tránh lớp A có quá nhiều sample so với lớp B.
- Loại bỏ sample nhiễu: frame bị miss landmark, bàn tay ra khỏi khung, bị che quá nhiều.
- Đa dạng điều kiện: ánh sáng, góc quay, khoảng cách camera, tốc độ thực hiện cử chỉ.

## Augmentation (tăng tính đa dạng dữ liệu)

- Jitter tọa độ nhỏ: cộng nhiễu Gaussian nhẹ lên (x,y,z).
- Time-warping: co giãn/giãn nén theo thời gian (sequence nhanh/chậm khác nhau).
- Flip trái/phải: nếu bài toán không phân biệt tay trái/tay phải (hoặc muốn tổng quát hơn).
- Random drop frame: bỏ ngẫu nhiên 1–2 frame để mô phỏng tracking bị mất ngắn hạn.

## Chuẩn hoá nâng cao (Normalize kỹ)

- Center + scale + velocity: hiện repo đã có trong `convert_sequences.py`.
- Rotation align (tuỳ chọn): xoay bàn tay về một hướng chuẩn (giảm phụ thuộc góc camera).

## Tối ưu kiến trúc model (Model capacity)

- Điều chỉnh channels: 64/128/256 có thể tăng/giảm theo độ phức tạp và dữ liệu.
- Điều chỉnh số block: thêm/bớt ST-GCN blocks.
- Điều chỉnh temporal kernel: kernel lớn hơn học chuyển động dài hơn nhưng chậm hơn.
- Dropout: thêm dropout trong block/FC để giảm overfit.

## Training tricks

- Scheduler: cosine decay / step LR để hội tụ ổn định hơn.
- Early stopping: dừng khi validation không cải thiện sau N epoch.
- Label smoothing: giảm over-confident, thường giúp generalize.

## Regularization

- Weight decay (L2): thường dùng với Adam/SGD.
- Mixup (trên sequence): trộn 2 sequence và trộn nhãn để tăng generalize.

## Tối ưu tốc độ inference (đặc biệt quan trọng cho game loop)

- Giảm `T` (ít frame hơn) và/hoặc giảm `C` (ít feature hơn).
- Chạy inference mỗi 2–3 frame thay vì mỗi frame.
- Caching: giữ prediction gần nhất và chỉ cập nhật khi đủ frame mới.
- Post-process ổn định: majority vote + cooldown để tránh nhấp nháy hành động.

## Nén/giảm kích thước model (Compression)

- Quantization (int8): giảm kích thước và tăng tốc (tuỳ runtime hỗ trợ).
- Pruning: bỏ bớt trọng số ít quan trọng.
- Knowledge distillation: train model nhỏ học theo model lớn.

## Deployment

- Export ONNX và chạy bằng runtime tối ưu (vd ONNX Runtime).
- Nếu chạy in-browser: cân nhắc ONNX Runtime Web/TFJS và kiểm soát latency.

## Integration pattern

- Buffer landmarks trong game runtime
- Mỗi lần inference ra label + confidence
- Post-process:
  - threshold confidence (vd 0.7)
  - majority vote trong cửa sổ N lần dự đoán
  - cooldown để tránh spam

## Triển khai inference

- Option A: Server inference
  - Ưu: dễ, không cần export
  - Nhược: trễ mạng

- Option B: In-browser inference
  - Ưu: nhanh, offline
  - Nhược: cần export ONNX/TFJS
