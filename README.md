# DL_DEMO

Project này nhận dạng cử chỉ tay từ MediaPipe landmarks bằng ST-GCN, rồi dùng kết quả đó cho demo webcam hoặc gắn phím/lệnh trên Windows.

## Luồng chính

1. Thu thập landmarks từ web capture trong `web/`.
2. Lọc dữ liệu bằng `tools/data_quality.py`.
3. Chuyển JSON sang NPZ bằng `tools/convert_sequences.py`.
4. Train ST-GCN bằng `train.py`.
5. Suy luận offline bằng `tools/infer.py` hoặc chạy realtime bằng `tools/demo_webcam.py`.

## Cài đặt nhanh trên Windows

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Thu thập dữ liệu

Serve thư mục project:

```bat
python -m http.server 8000
```

Mở `http://localhost:8000/web/` rồi:

1. Nhập nhãn cử chỉ.
2. Bật camera.
3. Ghi sequence.
4. Lưu JSON.

File JSON lưu `label` và `frames`, mỗi frame có 21 landmark `{x, y, z}`.

## Lọc và chuyển đổi dữ liệu

```bat
python tools/data_quality.py --input data/raw_ipn
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity
```

Ghi chú:
- `convert_sequences.py --use-velocity` hiện nối thêm cả velocity và acceleration, nên số kênh có thể tăng từ 2/3 lên 6/9.
- `train.py` nhận trực tiếp NPZ với shape `(N, T, V, C)`.

## Train

```bat
python train.py --data data/processed/train.npz --epochs 30 --batch-size 16 --lr 0.001 --out outputs
```

Các artifact chính trong `outputs/`:
- `stgcn_best.pt`
- `stgcn_last.pt`
- `labels.json`
- `confusion_matrix.pt`

## Suy luận và demo

```bat
python tools/infer.py --model outputs/stgcn_best.pt --labels outputs/labels.json --npz data/processed/train.npz --index 0 --topk 3 --device auto
python tools/demo_webcam.py --model outputs/stgcn_best.pt --labels outputs/labels.json --device auto
```

`tools/demo_webcam.py` có thể tự tìm checkpoint và `labels.json` trong các thư mục output phổ biến nếu bạn không truyền `--model` hoặc `--labels`.

## Tài liệu chi tiết

- [Tổng quan pipeline](docs/01_overview.md)
- [Format dữ liệu](docs/02_data_format.md)
- [Mô hình ST-GCN](docs/03_model_stgcn.md)
- [Train và metrics](docs/04_training_metrics.md)
- [Checklist web game](docs/05_web_game_checklist.md)
- [Data quality](docs/06_data_quality.md)
- [Gesture actions](docs/07_gesture_actions.md)

## Cấu trúc chính

- `web/`: giao diện capture landmarks.
- `tools/`: convert, quality check, infer, demo webcam.
- `dataset/`: loader NPZ cho PyTorch.
- `models/`: kiến trúc ST-GCN.
- `data/`: raw, clean, processed, annotations, labels.
- `outputs/`: checkpoint và report huấn luyện.
