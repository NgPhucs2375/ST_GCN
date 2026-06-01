# 05 — Checklist triển khai web game

## Mục tiêu realtime

- Render loop mượt trong trình duyệt.
- Inference không cần chạy mỗi frame.
- Kết quả ổn định, tránh spam hành động.

## Checklist theo code hiện tại

- Chốt `T` trước khi train và deploy, hiện pipeline mặc định dùng `T=30`.
- Chốt feature set: `C=2`, `C=3`, hoặc thêm velocity/acceleration nếu cần.
- Train xong phải kiểm tra confusion matrix trước khi ghép vào game loop.
- Nếu nhầm nhiều, tăng dữ liệu cho lớp đó trước khi tăng độ phức tạp model.

## Khi ghép vào game loop

- Buffer landmarks đủ `T` frame mới chạy inference.
- Dùng confidence threshold để lọc dự đoán yếu.
- Dùng smoothing/EMA để giảm nhấp nháy nhãn.
- Dùng cooldown để tránh gửi cùng một action quá nhiều lần.

## Với `tools/demo_webcam.py`

- Script có thể tự tìm model và `labels.json` trong các thư mục output phổ biến.
- Nó đọc `gesture_config.json` để map cử chỉ sang phím hoặc command.
- `D0X` được dùng như nhãn không cử chỉ.
- Nếu checkpoint có số kênh khác nhau, script sẽ suy ra cấu hình feature từ checkpoint.

## Tối ưu dữ liệu cho demo

- Cần thêm mẫu cho lớp đang bị nhầm nhiều.
- Giữ dữ liệu cân bằng giữa các lớp.
- Loại sample tracking lỗi, bàn tay bị che, hoặc mất landmark.
- Thu dữ liệu ở nhiều góc quay, nhiều tốc độ thực hiện, và nhiều điều kiện ánh sáng.

## Tối ưu inference

- Giảm `T` nếu cần latency thấp hơn.
- Giảm số lần chạy inference nếu game loop đang nặng.
- Giữ post-process đơn giản: threshold + EMA + cooldown.

## Nếu muốn lên production

- Dùng server inference nếu cần triển khai nhanh.
- Dùng runtime tối ưu nếu muốn chạy local/offline.
- Kiểm tra kỹ độ trễ tổng: capture + landmark + model + action dispatch.
