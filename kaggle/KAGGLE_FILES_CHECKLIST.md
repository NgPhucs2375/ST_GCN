# 📦 DANH SÁCH UPLOAD KAGGLE - Chính Xác 100%

## 🎯 MỤC ĐÍCH
Danh sách chính xác từng file + folder cần upload lên Kaggle để training.
Copy tên này cho đúng, không thiếu, không dư!

---

## 📊 **PHẦN 1: DATASET (1 dataset)**

### Dataset Name
```
Dataset Name: gesture-recognition-final
(Hoặc tên khác, nhưng nhớ tên này!)
```

### File trong Dataset:
```
📁 gesture-recognition-final/
└─ train_merged_t60_accel.npz    (17 MB)
   └─ 898 samples, 6 channels, T=60
   └─ Ready to train
```

**Cách upload:**
1. Kaggle → Create Dataset
2. Upload file: `data/processed/train_merged_t60_accel.npz`
3. Title: `gesture-recognition-final`
4. Visibility: Private (hoặc Public)
5. Publish

**Path trong Kaggle sau upload:**
```
/kaggle/input/gesture-recognition-final/train_merged_t60_accel.npz
```

---

## 💻 **PHẦN 2: CODE FILES (5 files + 1 folder)**

Upload cách này (3 option):

### **Option A: Upload qua Files Panel (DỄ NHẤT)**

Trong Kaggle Notebook, click `+ Add input` → `Upload`

Upload file này:
```
1. train.py                    (151 KB)
2. requirements.txt            (1 KB)
3. models/ (folder)
   ├─ __init__.py
   ├─ stgcn.py
4. dataset/ (folder)
   ├─ __init__.py
   ├─ stgcn_dataset.py
5. tools/
   └─ analyze_confusion_matrix.py
```

---

### **Option B: Copy-Paste Code (Nếu ko tải được)**

Tạo cells này trong Kaggle notebook:

#### **Cell: Create train.py**
```python
# Copy toàn bộ nội dung train.py từ local
# Paste vào Kaggle cell này
# Chạy: %writefile train.py
%writefile train.py
# [PASTE NỘI DUNG train.py ĐÂY]
```

#### **Cell: Create models/**
```python
%writefile models/__init__.py
# Leave empty

%writefile models/stgcn.py
# [PASTE NỘI DUNG stgcn.py]
```

Tương tự cho dataset/ và tools/

---

### **Option C: Download từ GitHub (Nếu có repo)**

```python
!git clone https://github.com/your-username/gesture-recognition.git
!cp -r gesture-recognition/* .
```

---

## 📝 **PHẦN 3: STRUCTURE CHÍNH XÁC**

Sau khi upload, cấu trúc phải như thế này:

```
/kaggle/working/
├── train.py                    ✓
├── requirements.txt            ✓
├── models/
│   ├── __init__.py            ✓
│   └── stgcn.py               ✓
├── dataset/
│   ├── __init__.py            ✓
│   └── stgcn_dataset.py       ✓
└── tools/
    └── analyze_confusion_matrix.py  ✓

/kaggle/input/gesture-recognition-final/
└── train_merged_t60_accel.npz  ✓
```

**Kiểm tra bằng code:**
```python
import os

files = [
    'train.py',
    'requirements.txt',
    'models/__init__.py',
    'models/stgcn.py',
    'dataset/__init__.py',
    'dataset/stgcn_dataset.py',
    'tools/analyze_confusion_matrix.py'
]

for f in files:
    status = "✅" if os.path.exists(f) else "❌"
    print(f"{status} {f}")

# Check dataset
if os.path.exists('/kaggle/input/gesture-recognition-final/train_merged_t60_accel.npz'):
    print("✅ Dataset: train_merged_t60_accel.npz (17 MB)")
else:
    print("❌ Dataset not found!")
```

---

## 🔍 **PHẦN 4: NỘI DUNG TỪNG FILE**

### **1. train.py**
```
Dòng: ~500 lines
Nội dung chính:
├─ Main training loop
├─ Data loading με weighted sampler
├─ Model training with 3 optimizations
├─ Validation + early stopping
└─ Save best model + confusion matrix

Requirements:
├─ torch
├─ torch_geometric
├─ numpy
└─ pathlib (built-in)

Key hyperparameters:
├─ epochs: 100 (default)
├─ batch_size: 32 (default)
├─ lr: 0.001 (default)
├─ All optimizations built-in ✓
```

### **2. requirements.txt**
```
Nội dung chính xác:
┌──────────────────────────┐
│ torch                    │
│ torch-geometric          │
│ numpy                    │
│ numpy==1.24.0           │  (specific for compat)
└──────────────────────────┘
```

### **3. models/stgcn.py**
```
Dòng: ~200 lines
Nội dung:
├─ STGCNBlock class (spatial-temporal convolution)
├─ STGCN model class (4 blocks stacked)
├─ Forward pass
├─ Batch norm, dropout, residual connections
└─ Capacity: 96-96-192-384 (upgraded)

Inputs:
├─ (batch_size, channels=6, time=60, vertices=21)

Outputs:
├─ (batch_size, 14 classes)

Parameters:
├─ 510K (vs 340K original, +50% capacity)
```

### **4. dataset/stgcn_dataset.py**
```
Dòng: ~150 lines
Nội dung:
├─ Dataset class (loads NPZ)
├─ Augmentation pipeline
├─ Data normalization
├─ Auto-detect channels (4, 6, or any)
└─ Returns (X, y) tuples

Features:
├─ Jitter augmentation
├─ Random flip
├─ Time-warp
├─ Drop-frames
└─ All 6-channel compatible ✓
```

### **5. tools/analyze_confusion_matrix.py**
```
Dòng: ~400 lines
Nội dung:
├─ Load confusion matrix
├─ Calculate per-class metrics
├─ Identify weak classes
├─ Print recommendations
├─ Save visualizations
└─ Export CSV metrics

Usage after training:
python tools/analyze_confusion_matrix.py \
  --cm-file outputs_kaggle/confusion_matrix.pt \
  --labels-file outputs_kaggle/labels.json
```

---

## ✅ **CHECKLIST UPLOAD**

Trước khi start training, check:

```
DATASET:
[ ] File: train_merged_t60_accel.npz (17 MB)
[ ] Location: /kaggle/input/gesture-recognition-final/
[ ] Dataset published: YES
[ ] Kernel attached: YES

CODE:
[ ] train.py uploaded/created
[ ] requirements.txt uploaded/created
[ ] models/stgcn.py uploaded/created
[ ] models/__init__.py uploaded/created
[ ] dataset/stgcn_dataset.py uploaded/created
[ ] dataset/__init__.py uploaded/created
[ ] tools/analyze_confusion_matrix.py uploaded/created

VERIFICATION:
[ ] All files present: Run check code
[ ] No ❌ marks: YES
[ ] Ready to train: YES
```

---

## 🚀 **TRAINING COMMAND (COPY-PASTE READY)**

```python
import subprocess
import sys

cmd = [
    sys.executable, 'train.py',
    '--data', '/kaggle/input/gesture-recognition-final/train_merged_t60_accel.npz',
    '--epochs', '100',
    '--batch-size', '32',
    '--lr', '0.001',
    '--weight-decay', '1e-4',
    '--dropout', '0.05',
    '--weighted-sampler',
    '--class-weighted-loss',
    '--max-class-weight', '4.0',
    '--label-smoothing', '0.15',
    '--scheduler', 'cosine',
    '--patience', '20',
    '--aug-jitter-std', '0.02',
    '--aug-flip-prob', '0.5',
    '--aug-time-warp', '0.05',
    '--aug-drop-frames', '1',
    '--val-ratio', '0.2',
    '--seed', '42',
    '--out', '/kaggle/working/outputs_kaggle'
]

# ⚠️ IMPORTANT: Replace 'gesture-recognition-final' with YOUR dataset name!

result = subprocess.run(cmd, capture_output=False)
```

**Thay dataset name:**
- Nếu dataset tên: `my-gesture-data` 
- Thay: `gesture-recognition-final` → `my-gesture-data`

---

## 📊 **FILE SIZES VALIDATION**

```
train_merged_t60_accel.npz     17 MB    ✓
train.py                        ~151 KB  ✓
models/stgcn.py                 ~25 KB   ✓
dataset/stgcn_dataset.py        ~18 KB   ✓
tools/analyze_confusion_matrix.py ~30 KB ✓
requirements.txt                ~1 KB    ✓

TOTAL SIZE: ~17.2 MB
(Kaggle allows 25 GB, so no problem)
```

---

## 🔄 **BƯỚC TỪ A-Z**

```
1. Local Machine:
   ├─ Check: train_merged_t60_accel.npz exists
   ├─ Check: train.py, models/, dataset/ exist
   └─ Check: requirements.txt exists

2. Kaggle Create Dataset:
   ├─ Upload: train_merged_t60_accel.npz
   ├─ Name: gesture-recognition-final (or your preferred name)
   ├─ Publish: DONE
   └─ Note: Dataset name/path

3. Kaggle Notebook:
   ├─ Create: New Python notebook
   ├─ Attach: Dataset just created
   ├─ Upload/Create: train.py + code files
   ├─ Cell 1: !pip install torch-geometric -q
   ├─ Cell 2: Verify structure (check code)
   └─ Cell 3: Run training command

4. Wait:
   ├─ Training: 25-35 minutes
   ├─ Watch: Epoch progress
   └─ Check: val_acc increasing

5. Analyze:
   ├─ Cell 4: Confusion matrix analysis
   ├─ View: Per-class accuracy
   ├─ Download: outputs_kaggle/ folder
   └─ Next: Plan improvements

```

---

## 💾 **OUTPUT FILES (After Training)**

Kaggle sẽ auto-save tại: `/kaggle/working/outputs_kaggle/`

```
outputs_kaggle/
├─ stgcn_best.pt               (best model) ⭐
├─ stgcn_last.pt               (last checkpoint)
├─ confusion_matrix.pt         (for analysis)
├─ labels.json                 (class names)
├─ training_log.json           (epoch logs)
└─ config.json                 (config used)

Download:
1. Right panel → Output → Download as Zip
2. Or download file by file
```

---

## ⚠️ **COMMON MISTAKES TO AVOID**

```
❌ MISTAKE 1: Wrong dataset path
   └─ /kaggle/input/gesture- recognition-final/  ← space/typo
   └─ Correct: /kaggle/input/gesture-recognition-final/

❌ MISTAKE 2: Missing __init__.py
   └─ models/ missing __init__.py
   └─ dataset/ missing __init__.py
   └─ Fix: Create both with `!touch models/__init__.py`

❌ MISTAKE 3: Wrong file name
   └─ train_merged_t60_accelere.npz  ← typo
   └─ Correct: train_merged_t60_accel.npz

❌ MISTAKE 4: Forgot to publish dataset
   └─ Dataset created but not published
   └─ Fix: Publish dataset first

❌ MISTAKE 5: Folder upload instead of file
   └─ Kaggle won't preserve folder structure
   └─ Better: Create files individually with %writefile

✅ AVOID THESE → Training will work!
```

---

## 📞 **FINAL CHECKLIST**

Before clicking "Run All":

```
1. Files uploaded: ALL ✓
2. Dataset attached: YES ✓
3. Dataset path correct: YES ✓
4. Code structure verified: YES ✓
5. requirements.txt present: YES ✓
6. No ❌ in verification: YES ✓
7. Ready to train: YES ✓

→→→ LET'S GO! 🚀
```

---

**Now follow KAGGLE_COMPLETE_GUIDE.md for step-by-step training!**
