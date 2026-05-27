# Hướng dẫn: Số lượng mẫu & Workflow

## Bao nhiêu mẫu cần lấy?

### Minimum viable (Bước 1)
- **Số mẫu/class**: 30-50 mẫu
- **Mục đích**: Test pipeline, không mong chất lượng cao
- **Thời gian**: 1-2 ngày (2-3 người lấy)
- **Kỳ vọng**: Accuracy ~60-70%

```bash
# Sau khi có 30-50 mẫu/class
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity
python train.py --data data/processed/train.npz --epochs 20 --batch-size 16 --lr 0.001 --out outputs_test
```

**Kiểm tra**: Xem `outputs_test/confusion_matrix.pt` có class nào nhầm nhiều không.

---

### Production quality (Bước 2 - Khuyến khích)
- **Số mẫu/class**: 100-150 mẫu
- **Mục đích**: Model chất lượng tốt, có thể deploy
- **Thời gian**: 3-5 ngày (2-4 người lấy)
- **Kỳ vọng**: Accuracy 80-85%

**Yêu cầu chất lượng:**
- Mỗi mẫu có 20-40 frame (không quá ngắn/dài)
- Tracking không bị miss (đủ 21 landmark mỗi frame)
- Bàn tay nằm trong frame (không cắt)
- Góc quay, ánh sáng đa dạng

```bash
# Chạy pipeline đầy đủ
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train_final.npz --length 30 --use-velocity
python train.py --data data/processed/train_final.npz --epochs 40 --batch-size 16 --lr 0.001 --out outputs_final
```

**Kiểm tra**: 
- Xem training log (accuracy tăng ổn định không)
- Xem confusion matrix (class nào vẫn nhầm)
- Xem class imbalance report

---

### Optimal (Bước 3 - Nếu muốn model siêu tốt)
- **Số mẫu/class**: 200-300 mẫu
- **Mục đích**: Model very high quality
- **Thời gian**: 1-2 tuần (3-4 người lấy)
- **Kỳ vọng**: Accuracy > 90%

**Thêm**:
- Augmentation (Người B): rotate, noise, time-warp
- Fine-tuning: train lại model đã tối ưu

```bash
# Với augmentation
python tools/augment_balance.py --input data/raw_ipn_clean --output data/raw_ipn_augmented --factor 2
python tools/convert_sequences.py --input data/raw_ipn_augmented --output data/processed/train_augmented.npz --length 30 --use-velocity
python train.py --data data/processed/train_augmented.npz --epochs 60 --batch-size 16 --lr 0.0005 --out outputs_best
```

---

## Workflow: Sau khi có mẫu → Model tốt hơn

### Phase 1: Lấy mẫu ban đầu (30-50/class)

**Bước 1a**: Mở web capture
```bash
python -m http.server 8000
# Mở http://localhost:8000/web/
```

**Bước 1b**: Ghi mẫu
- Nhập label: `G01`, `G02`, ... (đặt tên cử chỉ)
- Bấm "Start Camera" → "Record" → thực hiện cử chỉ → "Stop" → "Save JSON"
- Lưu vào `data/raw_ipn/` hoặc `data/raw_ipn_clean/` tùy chất lượng

**Bước 1c**: Kiểm tra nhanh
```bash
ls data/raw_ipn/ | wc -l  # Đếm số file JSON
# Mỗi class nên có ≥ 30 file
```

---

### Phase 2: Data quality (lọc mẫu xấu)

**Bước 2**: Chạy quality check
```bash
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean
```

**Output**:
- `outputs/data_quality_report.csv`: từng file (pass/fail + lý do)
- `outputs/data_quality_summary.json`: tổng kết theo class

**Kiểm tra**:
```python
import json
summary = json.load(open('outputs/data_quality_summary.json'))
print(f"Class stats: {summary['per_class_ok']}")  # Xem mỗi class còn bao nhiêu sau lọc
```

**Nếu class nào < 20 mẫu sau lọc**: Quay lại Bước 1, lấy thêm

---

### Phase 3: Convert sang tensor (NPZ)

**Bước 3**: Chuyển JSON → NPZ
```bash
python tools/convert_sequences.py \
  --input data/raw_ipn_clean \
  --output data/processed/train.npz \
  --length 30 \
  --use-velocity
```

**Output**: `data/processed/train.npz` (chứa sequences + labels)

**Kiểm tra**:
```python
import numpy as np
data = np.load('data/processed/train.npz')
print(f"Shape: {data['sequences'].shape}")  # Nên là (N, 30, 21, C)
print(f"Classes: {np.unique(data['labels'])}")  # Xem bao nhiêu class
```

---

### Phase 4: Train & Evaluate

**Bước 4a**: Train baseline
```bash
python train.py \
  --data data/processed/train.npz \
  --epochs 30 \
  --batch-size 16 \
  --lr 0.001 \
  --out outputs_test
```

**Bước 4b**: Kiểm tra kết quả
```python
import torch
import json

# Xem accuracy
history = json.load(open('outputs_test/training_history.json'))
print(f"Final train acc: {history['train_acc'][-1]:.2%}")
print(f"Final val acc: {history['val_acc'][-1]:.2%}")

# Xem confusion matrix
cm = torch.load('outputs_test/confusion_matrix.pt')
labels = json.load(open('outputs_test/labels.json'))
print(f"Confusion matrix:\n{cm}")
print(f"Class mapping: {labels}")
```

---

## Khi nào thì chất lượng đủ?

### Checklist lấy mẫu OK

- [ ] Mỗi class ≥ 30 mẫu (nếu phase 1) hoặc ≥ 100 mẫu (phase 2)
- [ ] Sau lọc quality, vẫn giữ ≥ 80% số mẫu gốc (tức không lọc quá nhiều)
- [ ] Không có class nào thiếu hơn 20% so với class đông nhất
- [ ] Mỗi mẫu có 20-40 frame (xem trong convert log)

### Checklist training OK

- [ ] Train accuracy ≥ 85% (hoặc > 75% cho phase 1)
- [ ] Val accuracy không quá cách xa train (< 10 điểm chênh)
- [ ] Training loss đi xuống smooth (không nhảy loạn)
- [ ] Confusion matrix không có class nào bị nhầm > 50%

---

## Cải thiện nếu model yếu

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-----------|----------|
| Accuracy thấp | Data không đủ | Quay lại Bước 1, lấy thêm 50 mẫu/class |
| Accuracy thấp | Data xấu (tracking bị miss) | Kiểm tra lại quality rule trong data_quality.py |
| Train ↑ nhưng Val → | Overfit | Thêm dropout, weight decay, hoặc augmentation |
| Một class bị nhầm nhiều | Class không rõ | Quay lại xem mẫu của class đó, lấy thêm hoặc xoá mẫu xấu |
| Latency cao (> 200ms) | Model quá nặng | Giảm channels, giảm T, hoặc quantize |

---

## Tóm tắt workflow nhanh

```
1. Lấy 30-50 mẫu/class từ web
   ↓
2. Chạy data_quality.py (lọc mẫu xấu)
   ↓
3. Chạy convert_sequences.py (tạo NPZ)
   ↓
4. Chạy train.py (train 20-30 epoch)
   ↓
5. Kiểm tra accuracy & confusion matrix
   ↓
6. Nếu accuracy < 75%:
   - Quay lại bước 1 (lấy thêm mẫu)
   - Hoặc thêm augmentation (bước 3)
   ↓
7. Nếu OK: Lưu model + deploy
```

---

## Lệnh tất cả trong 1 dòng (Quick start)

```bash
# Sau khi có raw JSON trong data/raw_ipn/
python tools/data_quality.py --input data/raw_ipn --copy-ok-to data/raw_ipn_clean && \
python tools/convert_sequences.py --input data/raw_ipn_clean --output data/processed/train.npz --length 30 --use-velocity && \
python train.py --data data/processed/train.npz --epochs 30 --batch-size 16 --out outputs_v1
```

Xong: check `outputs_v1/training_history.json` và `outputs_v1/confusion_matrix.pt`
