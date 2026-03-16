# DL_DEMO — Hand Gesture Recognition (MediaPipe + ST-GCN)

Repo này là skeleton để bạn:

1) Thu thập landmarks bàn tay bằng MediaPipe trên trình duyệt (web)
2) Chuẩn hoá + chuyển dữ liệu sang tensor cố định chiều dài (Python)
3) Train mô hình ST-GCN (PyTorch + torch-geometric)
4) Lấy metric accuracy và confusion matrix để đánh giá

## Cấu trúc thư mục

Xem chi tiết trong [DATA_STRUCTURE.md](DATA_STRUCTURE.md) và docs trong thư mục [docs/](docs/).

- `web/`: UI chạy trong browser để capture landmarks và xuất JSON
- `data/raw/`: nơi bạn lưu JSON thô từ web
- `tools/convert_sequences.py`: chuyển JSON → `npz` (sequences + labels)
- `dataset/`: PyTorch Dataset đọc `npz`
- `models/stgcn.py`: ST-GCN (spatial GCNConv + temporal conv)
- `train.py`: train + lưu model + metrics

## Quickstart (Windows)

### 1) Chạy web capture

```bat
python -m http.server 8000
```

Mở: `http://localhost:8000/web/`

- Nhập `Gesture label`
- Bấm `Start Camera`
- Bấm `Record` → làm cử chỉ → bấm `Stop`
- Bấm `Save JSON`

Sau khi tải về JSON, bạn hãy copy file đó vào `data/raw/`.

### 2) Convert JSON → NPZ

Ví dụ: cố định độ dài `T=30`, dùng velocity:

```bat
python tools/convert_sequences.py --input data/raw --output data/processed/train.npz --length 30 --use-velocity
```

Output `train.npz` chứa:
- `sequences`: shape `(N, T, V, C)` với `V=21`
- `labels`: shape `(N,)` (string label)

### 3) Train ST-GCN

```bat
python train.py --data data/processed/train.npz --epochs 30 --batch-size 16 --lr 0.001 --out outputs
```

Sản phẩm trong `outputs/`:
- `stgcn_best.pt`: model tốt nhất theo validation accuracy
- `stgcn_last.pt`: model cuối
- `labels.json`: map `label -> class_id`
- `confusion_matrix.pt`: confusion matrix của best model

## Đọc nhanh docs

- Tổng quan pipeline: [docs/01_overview.md](docs/01_overview.md)
- Dữ liệu & chuẩn hoá: [docs/02_data_format.md](docs/02_data_format.md)
- ST-GCN & torch-geometric: [docs/03_model_stgcn.md](docs/03_model_stgcn.md)
- Train & metrics: [docs/04_training_metrics.md](docs/04_training_metrics.md)
- Checklist triển khai web game: [docs/05_web_game_checklist.md](docs/05_web_game_checklist.md)
- Data quality (lọc dữ liệu): [docs/06_data_quality.md](docs/06_data_quality.md)
