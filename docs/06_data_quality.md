# 06 — Data quality

Mục tiêu của bước này là tránh đưa vào train những sample bị lỗi tracking, sequence quá ngắn, bị duplicate, hoặc lệch phân phối quá mạnh giữa các lớp.

## Script trong repo

File chính: [tools/data_quality.py](../tools/data_quality.py)

Script quét thư mục JSON và tạo:

- `outputs/data_quality_report.csv`.
- `outputs/data_quality_summary.json`.
- `--copy-ok-to` để copy các file đạt sang folder clean.

## Cách chạy

```bat
python tools/data_quality.py --input data/raw
python tools/data_quality.py --input data/raw --copy-ok-to data/raw_clean
```

## Rule mặc định

- `--min-frames 10`: loại sequence quá ngắn.
- `--max-frames 0`: không giới hạn trên nếu để 0.
- `--expected-landmarks 21`: mỗi frame phải có đủ 21 điểm.
- `--tol-xy 0.05`: chấp nhận x/y hơi ngoài [0, 1].
- `--max-wrist-jump 0.25`: loại tracking bị nhảy mạnh ở cổ tay.
- `--max-mean-jump 0.15`: loại tracking bị nhảy mạnh trung bình toàn tay.
- `--dedup-decimals 3`: phát hiện duplicate bằng hash trên tọa độ đã làm tròn.

## Ý nghĩa report

`data_quality_report.csv` cho từng file:

- tên file.
- label.
- số frame.
- trạng thái OK/bad.
- danh sách lý do fail.

`data_quality_summary.json` cho tổng quan:

- số file OK.
- số file bad.
- đếm theo từng lý do fail.
- `per_class_total` và `per_class_ok`.

## Workflow khuyến nghị

1. Capture JSON vào `data/raw/` hoặc một thư mục raw tương đương.
2. Chạy data quality.
3. Copy file OK sang `data/raw_clean/`.
4. Convert từ folder clean sang NPZ.
5. Train model.

Ví dụ:

```bat
python tools/data_quality.py --input data/raw --copy-ok-to data/raw_clean
python tools/convert_sequences.py --input data/raw_clean --output data/processed/train.npz --length 30 --use-velocity
python train.py --data data/processed/train.npz --out outputs
```

## Cách đọc để cải thiện dữ liệu

- Nếu `bad_files` nhiều, thường là lỗi capture hoặc landmark không ổn định.
- Nếu một class ít mẫu hơn hẳn, cần thu thêm dữ liệu cho class đó.
- Nếu duplicate nhiều, cần ghi lại sequence đa dạng hơn thay vì chụp lặp.
