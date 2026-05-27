# 03 - Progress, Missing Items & Deployment

## Tinh trang hien tai cua repo

Da co san nhieu artifact de tiep tuc lam viec:

- dataset processed trong `data/processed/`;
- checkpoint va history trong `outputs_resume/`;
- model, labels, confusion matrix va training log da co san;
- web capture UI nam trong `web/`;
- script phan chia buoc ro rang trong `tools/`.

## Phan da co

- Capture JSON tu webcam.
- Convert JSON sang NPZ.
- Train ST-GCN.
- Luu checkpoint va metrics.
- Co demo webcam va cac script quality check.

## Phan con thieu / can hoan thien

### Tai lieu

- Da gom ve 3 file chinh trong `docs/`.
- Chua co phan ghi nhan tien do theo ngay/lan chay.

### Du lieu

- Can xac nhan tap raw chuan cuoi cung la `data/raw_ipn_clean` hay mot tap merge khac.
- Can them quy tac ro rang cho so luong mau moi lop.
- Can chot class nao la class chinh, class nao dang bi nham nhieu nhat.

### Model / train

- Can chot cau hinh `T`, `C`, batch size va so epoch uu tien.
- Can co 1 bang so sanh giua cac checkpoint de biet ban nao tot nhat.
- Neu muon len san pham, can benchmark latency thuc te.

### Deployment

- Chua chot huong server inference hay in-browser inference.
- Neu co web game, can them smoothing, majority vote va cooldown.
- Chua co file export ONNX/TFJS chuan hoa cho runtime trinh duyet.

## Checklist uu tien

1. Dong bo lai tai lieu tong hop trong 3 file `docs/`.
2. Xac nhan tap du lieu cuoi cung de train.
3. Chot checkpoint best va labels.json final.
4. Chay lai evaluation de co confusion matrix final.
5. Benchmark realtime neu muc tieu la demo webcam hoac web game.

## Mau tinh trang can cap nhat

Neu muon bao cao nhanh, co the ghi theo mau sau:

- Da xong: capture, convert, train, evaluate.
- Dang co: checkpoint, history, confusion matrix.
- Con thieu: dong bo tai lieu, chot dataset final, chot deployment path, benchmark latency.

## Goi y workflow gan nhat

```bash
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity
python train.py --data data/processed/train.npz --epochs 30 --batch-size 16 --lr 0.001 --out outputs
python tools/demo_webcam.py --model outputs_resume/stgcn_best.pt --labels outputs_resume/labels.json --device auto
```

## Ket luan ngan

Repo da o trang thai co the tiep tuc train/demo, nhung chua dong bo tai lieu va chua chot day du phan deployment. 3 file trong `docs/` bay gio gom du y chinh de xem nhanh va cap nhat tien do.
