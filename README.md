# DL_DEMO - Huong dan day du (Thu cong + Kaggle)

Project nay dung de nhan dang cu chi tay bang chuoi landmarks (MediaPipe) va mo hinh ST-GCN.

Pipeline tong quat:
1. Thu thap landmarks tu webcam (web).
2. Chuan hoa du lieu va chuyen sang file NPZ.
3. Train mo hinh ST-GCN.
4. Danh gia ket qua va demo webcam real-time.

---

## 1) Cach 1 - Chay thu cong tren may (Windows)

### Buoc 1: Tao moi truong va cai thu vien

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, torch_geometric, mediapipe; print('ok')"
```

### Buoc 2: Thu thap du lieu bang web capture

```bat
python -m http.server 8000
```

Mo trinh duyet: `http://localhost:8000/web/`

Thao tac:
1. Nhap ten nhan cu chi (vi du: `G01`, `G10`, `D0X`...).
2. Bam `Start Camera`.
3. Bam `Record`, thuc hien cu chi, bam `Stop`.
4. Bam `Save JSON`.
5. Copy cac file JSON vao thu muc du lieu dau vao (nen dung `data/raw_ipn/` hoac `data/raw_ipn_clean/`).

### Buoc 3: Kiem tra/chat luong du lieu (khuyen nghi)

```bat
python tools/data_quality.py --input data/raw_ipn
```

Neu muon copy cac mau dat chat luong sang folder moi:

```bat
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
```

### Buoc 4: Chuyen JSON -> NPZ

```bat
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity
```

File output `data/processed/train.npz` gom:
- `sequences`: kich thuoc `(N, T, V, C)`
- `labels`: nhan tuong ung `(N,)`

### Buoc 5: Train mo hinh

```bat
python train.py --data data/processed/train.npz --epochs 30 --batch-size 16 --lr 0.001 --out outputs
```

Sau khi train xong, folder `outputs/` thuong co:
- `stgcn_best.pt`: model tot nhat theo val accuracy
- `stgcn_last.pt`: model epoch cuoi
- `labels.json`: mapping nhan -> class id
- `confusion_matrix.pt`: ma tran nham lan

### Buoc 6: Suy luan nhanh voi file du lieu

```bat
python tools/infer.py --model outputs/stgcn_best.pt --labels outputs/labels.json --npz data/processed/train.npz --index 0 --topk 3 --device auto
```

### Buoc 7: Demo webcam real-time

```bat
python tools/demo_webcam.py --model outputs/stgcn_best.pt --labels outputs/labels.json --device auto
```

---

## 2) Cach 2 - Chay tren Kaggle bang link Notebook

Ban co the train/infer tren Kaggle de dung GPU mien phi.

### 2.1. Link Notebook

Them link notebook cua ban tai day:
- `Kaggle Notebook: <dan_link_kaggle_vao_day>`

Neu ban da train xong tren Kaggle, artifact thuong nam o dang:
- `kaggle/working/.../stgcn_best.pt`
- `kaggle/working/.../labels.json`

### 2.2. Cac buoc trong Kaggle (goi y)

1. Upload source code project (hoac add dataset zip chua source).
2. Cai dependencies tu `requirements.txt` (neu can).
3. Chuan bi du lieu JSON trong working directory.
4. Chay convert:

```bash
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity
```

5. Chay train:

```bash
python train.py --data data/processed/train.npz --epochs 30 --batch-size 16 --lr 0.001 --out outputs
```

6. Luu/copy artifact (`stgcn_best.pt`, `labels.json`) tu `kaggle/working`.

### 2.3. Mang model tu Kaggle ve may de demo

Dat file model va labels vao project, vi du:
- `final_model/final_model_v2/kaggle/working/final_model_v2/stgcn_best.pt`
- `final_model/final_model_v2/kaggle/working/final_model_v2/labels.json`

Chay demo:

```bat
python tools/demo_webcam.py --model "final_model/final_model_v2/kaggle/working/final_model_v2/stgcn_best.pt" --labels "final_model/final_model_v2/kaggle/working/final_model_v2/labels.json" --device auto
```

---

## Tac dung/chuc nang tung file, tung thu muc chinh

### Thu muc goc

- `README.md`: tai lieu tong hop huong dan su dung project.
- `SETUP.md`: huong dan cai dat moi truong nhanh.
- `DATA_STRUCTURE.md`: mo ta cau truc thu muc de de quan ly du lieu.
- `requirements.txt`: danh sach thu vien Python can cai.
- `train.py`: script train ST-GCN, chia train/val, luu checkpoints va metrics.

### Thu muc `web/`

- `web/index.html`: giao dien web capture landmarks.
- `web/app.js`: logic webcam + MediaPipe + luu JSON.
- `web/style.css`: style cho giao dien web.
- `web/README.md`: huong dan rieng cho web capture.

### Thu muc `tools/`

- `tools/convert_sequences.py`: chuyen JSON landmarks thanh NPZ de train.
- `tools/data_quality.py`: kiem tra chat luong va loc sequence xau.
- `tools/infer.py`: suy luan offline voi 1 JSON hoac 1 sample trong NPZ.
- `tools/demo_webcam.py`: demo nhan dang cu chi real-time tu webcam.
- `tools/augment_balance.py`: tang cuong/can bang du lieu.
- `tools/compare_checkpoints.py`: so sanh checkpoint theo metric.
- `tools/inspect_confusion.py`: doc/hien thi confusion matrix.
- `tools/ipn_to_json.py`: ho tro chuyen doi du lieu IPN sang JSON theo pipeline.

### Thu muc model va dataset

- `models/stgcn.py`: dinh nghia kien truc ST-GCN.
- `dataset/stgcn_dataset.py`: Dataset loader doc NPZ cho PyTorch.

### Thu muc du lieu `data/`

- `data/annotations/`: metadata/phan chia train-test tu bo du lieu.
- `data/raw_ipn/`: JSON thuc te chua loc.
- `data/raw_ipn_clean/`: JSON da loc chat luong.
- `data/raw_ipn_full*`, `data/raw_ipn_merged*`: cac tap du lieu tong hop/merge theo cac phien ban.
- `data/processed/`: du lieu NPZ san sang de train.
- `data/videos/`: video tham chieu (neu co).

### Thu muc ket qua

- `outputs/`, `outputs_resume/`: ket qua train, checkpoint, labels, confusion matrix.
- `final_model/`: model chot de demo/bao cao.

### Tai lieu

- `docs/01_overview.md`: tong quan pipeline.
- `docs/02_data_format.md`: mo ta format va chuan hoa du lieu.
- `docs/03_model_stgcn.md`: giai thich mo hinh ST-GCN.
- `docs/04_training_metrics.md`: y nghia metric trong qua trinh train.
- `docs/05_web_game_checklist.md`: checklist tich hop web game.
- `docs/06_data_quality.md`: huong dan kiem tra chat luong du lieu.

---

## Lenh mau nhanh (copy-paste)

```bat
python -m http.server 8000
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity
python train.py --data data/processed/train.npz --epochs 30 --batch-size 16 --lr 0.001 --out outputs
python tools/demo_webcam.py --model outputs/stgcn_best.pt --labels outputs/labels.json --device auto
```
