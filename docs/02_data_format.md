# 02 — Dữ liệu, format tensor, và chuẩn hoá

## JSON thô từ web

Mỗi sample được lưu thành JSON với cấu trúc:

- `label`: tên cử chỉ.
- `frames`: danh sách frame.
  - mỗi frame có 21 landmark.
  - mỗi landmark là `{x, y, z}`.

## Từ JSON sang NPZ

`tools/convert_sequences.py` đọc từng file JSON, rồi chuyển thành mảng `numpy`:

- `array` có shape `(T, V, C)`.
  - `T`: số frame trong sequence.
  - `V=21`: số landmark bàn tay.
  - `C=2` nếu bỏ z.
  - `C=3` nếu giữ x, y, z.

Sau đó script pad/trim về chiều dài cố định `--length` và lưu ra:

- `sequences`: shape `(N, T, V, C)`.
- `labels`: shape `(N,)`.

## Chuẩn hoá đang dùng

Trong `normalize_frames()`:

1. Center theo cổ tay.
   - lấy landmark 0.
   - `frames = frames - wrist`.
2. Scale theo lòng bàn tay.
   - dùng landmark 9 sau khi đã center.
   - `frames = frames / ||palm||`.

Mục tiêu là giảm phụ thuộc vào vị trí và kích thước tay trong khung hình.

## Feature mở rộng

`tools/convert_sequences.py --use-velocity` hiện nối thêm cả velocity và acceleration:

- input `C=2` → output `C=6`.
- input `C=3` → output `C=9`.

Đây là điểm khác với `tools/infer.py --json`, vì file infer hiện chỉ tạo position + velocity cho đường JSON đơn lẻ.

## Pad / trim

Vì model cần `T` cố định:

- sequence dài hơn `T` thì cắt bớt.
- sequence ngắn hơn `T` thì lặp frame cuối để đệm.

## Gợi ý thực tế

- Nếu bạn ưu tiên ổn định và tốc độ, bắt đầu với `C=2` hoặc `C=3`.
- Nếu dữ liệu đủ tốt và bạn muốn tăng độ giàu thông tin, thử `--use-velocity` để có thêm động học.
