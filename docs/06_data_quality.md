# 06 — Data quality (lọc dữ liệu trước khi train)

Mục tiêu của data quality là:
- tránh train trên sample bị lỗi tracking
- cân bằng số lượng sample giữa các lớp
- giảm nhầm lẫn (nhìn qua confusion matrix)

## Tool trong repo

Script: [tools/data_quality.py](../tools/data_quality.py)

Tool này quét `data/raw/*.json` và tạo:
- `outputs/data_quality_report.csv`: thống kê từng file
- `outputs/data_quality_summary.json`: tổng hợp theo lý do lỗi + theo lớp
- (tuỳ chọn) copy file OK sang folder sạch

## Chạy cơ bản

```bat
python tools/data_quality.py --input data/raw
```

## Copy các file đạt sang thư mục clean

```bat
python tools/data_quality.py --input data/raw --copy-ok-to data/raw_clean
```

## Các rule (heuristic) mặc định

- `--min-frames 10`: loại sequence quá ngắn
- `--expected-landmarks 21`: mỗi frame phải đủ 21 điểm
- `--tol-xy 0.05`: cho phép x/y hơi ngoài [0,1]
- `--max-wrist-jump 0.25`: loại tracking bị nhảy mạnh ở cổ tay
- `--max-mean-jump 0.15`: loại tracking bị nhảy mạnh trung bình
- `--dedup-decimals 3`: phát hiện duplicate bằng hash trên tọa độ đã làm tròn

Các ngưỡng này là gợi ý ban đầu. Bạn nên xem report rồi chỉnh lại cho phù hợp.

## Cách dùng kết hợp với train

Workflow đề xuất:
1) Capture JSON → `data/raw/`
2) Chạy data quality → copy OK sang `data/raw_clean/`
3) Convert `data/raw_clean/` → `data/processed/train.npz`
4) Train ST-GCN

Ví dụ:

```bat
python tools/data_quality.py --input data/raw --copy-ok-to data/raw_clean
python tools/convert_sequences.py --input data/raw_clean --output data/processed/train.npz --length 30 --use-velocity
python train.py --data data/processed/train.npz --out outputs
```

## Cân bằng lớp

Trong `outputs/data_quality_summary.json` có 2 thống kê:
- `per_class_total`: số file mỗi lớp trong raw
- `per_class_ok`: số file đạt sau lọc

Nếu có lớp ít hơn hẳn, bạn nên thu thập thêm lớp đó (hoặc giảm lớp quá nhiều).
