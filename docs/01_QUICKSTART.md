# 01 - Quickstart & Pipeline

## Muc tieu

Bien webcam stream thanh landmarks ban tay, sau do chuyen sang chuoi tensor de train ST-GCN va suy luan cu chi.

## Quy trinh chuan

1. Trinh duyet chay MediaPipe Hands va lay 21 landmark cho moi frame.
2. Web UI gom lien tiep `T` frame thanh 1 sample.
3. Sample duoc luu ra JSON tho.
4. Python tien xu ly JSON:
   - center theo co tay
   - scale theo ban tay
   - them velocity neu can
   - pad/trim ve do dai co dinh
   - luu sang NPZ
5. `train.py` train mo hinh ST-GCN.
6. Danh gia bang accuracy va confusion matrix.

## Cau hinh nhanh

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, torch_geometric, mediapipe; print('ok')"
```

## Nhom file chinh can biet

- `web/`: giao dien capture JSON tu webcam.
- `tools/convert_sequences.py`: JSON -> NPZ.
- `tools/data_quality.py`: loc mau loi va sinh report.
- `train.py`: train/evaluate ST-GCN.
- `tools/demo_webcam.py`: demo real-time.

## Mau du lieu dau vao / dau ra

### JSON tho tu web

Moi sample co dang:

- `label`: ten cu chi.
- `frames`: danh sach frame.
- moi frame chua 21 diem co `{x, y, z}`.

### NPZ sau khi convert

- `sequences`: shape `(N, T, V, C)`.
- `labels`: shape `(N,)`.

Trong do:

- `T`: so frame co dinh.
- `V=21`: so landmark.
- `C=2` hoac `3`, co the tang len `4` hoac `6` neu them velocity.

## Lenh mau

```bat
python -m http.server 8000
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity
```

## Luong tong quan

```mermaid
flowchart LR
    A[Webcam + MediaPipe] --> B[JSON raw]
    B --> C[Data quality]
    C --> D[NPZ processed]
    D --> E[ST-GCN train]
    E --> F[Evaluate + demo]
```
