# 🚀 HƯỚNG DẪN KAGGLE TRAINING - BẢN HOÀN CHỈNH


---

## 📋 **BƯỚC 0: CHUẨN BỊ - Các File Cần Upload**

### **A. Tệp Dữ Liệu Chính (1 file)**

```
❗ QUAN TRỌNG: File này là dữ liệu huấn luyện
📁 data/processed/train_merged_t60_accel.npz  (17 MB)
   └─ Chứa 898 mẫu gesture recognition
   └─ 6 channels (position + velocity + acceleration)
   └─ Sẵn sàng để train
```

**Kiểm tra file có phải là:**
- ✅ Dung lượng: khoảng 17 MB
- ✅ Tên đầy đủ: `train_merged_t60_accel.npz` (ko thiếu chữ)
- ✅ Vị trí: `data/processed/` folder

---

### **B. Code Files (5 file)**

```
📁 train.py                    (Main training script)
📁 models/
   └─ __init__.py
   └─ stgcn.py               (Model architecture)
📁 dataset/
   └─ __init__.py
   └─ stgcn_dataset.py       (Data loader)
📁 tools/
   └─ analyze_confusion_matrix.py  (Analysis after training)
```

**Cách check:**
- Mở mỗi file trong VS Code
- Kiểm tra có nội dung code không
- Ko phải file rỗng

---

### **C. Configuration**

```
❗ QUAN TRỌNG
requirements.txt            (Libraries cần cài)
```

**Nội dung phải có:**
```
torch
torch-geometric
numpy
numpy==1.24.0   (specific version for compatibility)
```

---

## 📊 **BƯỚC 1: TẠO NOTEBOOK KAGGLE**

### **Cách 1: Trực Tiếp Upload Dataset**

**Step 1.1:** Vào https://www.kaggle.com

**Step 1.2:** Click "Notebooks" → "Create Notebook"
```
┌─────────────────────────────────┐
│  Notebooks  Create Notebook  ▼  │
└─────────────────────────────────┘
         👆 Click here
```

**Step 1.3:** Chọn Python kernel
```
┌────────────────────────────────┐
│ Language: Python               │
│ Data: None (chọn sau)          │
│ Title: GestureNet_Training     │
└────────────────────────────────┘
   👆 Copy tên này làm tiêu đề
```

**Step 1.4:** Tạo dataset trên Kaggle
- Vào "Create Dataset" 
- Chọn "NEW DATASET"
- Upload file: `train_merged_t60_accel.npz`
- Chọn "PRIVATE" hoặc "PUBLIC"
- Publish

**Step 1.5:** Attach dataset vào notebook
```
Phía trên code editor:
┌──────────────────────┐
│ + Add Input          │  👈 Click đây
│ Select a File        │
└──────────────────────┘

Chọn dataset vừa tạo
```

---

## 💾 **BƯỚC 2: UPLOAD CODE FILES**

### **Cách Upload Files**

**Option A: Upload từ Files Panel (DỄ NHẤT)**

```
Bên trái notebook, click tab "Files" (🗂️)
└─ Click "Upload"
   ├─ Chọn train.py
   ├─ Chọn models/ folder (drag thả)
   ├─ Chọn dataset/ folder (drag thả)
   ├─ Chọn requirements.txt
   └─ Upload xong
```

**Option B: Dùng Terminal (Nếu Option A ko work)**

Copy và paste từng cụm lệnh sau vào notebook cell:

```python
# Cụm 1: Download code from GitHub/local
!mkdir -p models dataset tools
!wget -q https://your-repo/train.py
!wget -q https://your-repo/models/__init__.py
# ... (nếu có repo)
```

---

## 🎯 **BƯỚC 3: SETUP - Cài Dependencies**

### **Cell 1: Cài Thư Viện**

**Copy code này vào cell đầu tiên:**

```python
# Cài thư viện cầu thiết
!pip install torch-geometric -q
!pip install -q torch torchvision torchaudio
```

**Nhấn Shift+Enter để chạy**

```
⏳ Chờ khoảng 2-3 phút
✅ Xong: "Successfully installed..."
```

---

## 📁 **BƯỚC 4: KIỂM TRA CẤU TRÚC**

### **Cell 2: Xác Nhận Files Có Sẵn**

```python
import os
from pathlib import Path

# Kiểm tra files tồn tại
print("=" * 60)
print("KIỂM TRA CẤU TRÚC DỰ ÁN")
print("=" * 60)

# Check dataset
if os.path.exists('/kaggle/input/your-dataset/train_merged_t60_accel.npz'):
    print("✅ Dataset found: train_merged_t60_accel.npz")
else:
    print("❌ Dataset NOT found - kiểm tra upload lại")

# Check code files
files_to_check = [
    'train.py',
    'models/stgcn.py',
    'models/__init__.py',
    'dataset/stgcn_dataset.py',
    'dataset/__init__.py',
    'tools/analyze_confusion_matrix.py'
]

for f in files_to_check:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        print(f"❌ {f} - CẦN UPLOAD")

print("=" * 60)
```

**Nhấn Shift+Enter**

```
Expected output:
✅ Dataset found: train_merged_t60_accel.npz
✅ train.py
✅ models/stgcn.py
...
```

Nếu thấy ❌ → Upload lại file đó

---

## 🚀 **BƯỚC 5: CHẠY TRAINING**

### **Cell 3: Training Command (Copy-Paste Ready)**

```python
import subprocess
import sys

# Gọi Python để chạy train.py
cmd = [
    sys.executable, 'train.py',
    '--data', '/kaggle/input/your-dataset/train_merged_t60_accel.npz',
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

print("=" * 80)
print("🚀 BẮT ĐẦU TRAINING")
print("=" * 80)
print(f"Command: {' '.join(cmd)}")
print("=" * 80)

result = subprocess.run(cmd, capture_output=False)
sys.exit(result.returncode)
```

**⚠️ QUAN TRỌNG: Thay `your-dataset` bằng tên dataset thực tế của bạn**

Ví dụ:
- Nếu dataset tên: `gesture-recognition-data`
- Thay: `/kaggle/input/your-dataset/` 
- Thành: `/kaggle/input/gesture-recognition-data/`

---

## ⏱️ **BƯỚC 6: QUAN SÁT TRAINING**

### **Khi training chạy:**

```
Epoch 001 | train loss 2.8439 acc 0.065 | val loss 2.8235 acc 0.088 | lr 1.00e-03
    ↓
Epoch 010 | train loss 0.9234 acc 0.745 | val loss 1.0121 acc 0.721 | lr 9.99e-04
    ↓
Epoch 030 | train loss 0.1823 acc 0.942 | val loss 0.3842 acc 0.879 | lr 9.70e-04 ← BEST
    ↓
Epoch 050 | train loss 0.0512 acc 0.981 | val loss 0.3821 acc 0.877 | lr 8.91e-04
    ↓
Early stopping at epoch 50 (best val acc 0.879)
✅ Training complete!
```

**Ý nghĩa:**
- `train loss` = lỗi trên dữ liệu training (thấp → tốt)
- `train acc` = độ chính xác training (cao → tốt)
- `val loss` = lỗi trên validation data
- `val acc` = độ chính xác validation (cao → tốt) ⭐
- `lr` = learning rate (tự giảm dần)

**Expected:**
```
Kỳ vọng:
├─ val acc tăng từ ~8% (epoch 1) → ~87-89% (epoch 35-50) ✓
├─ Training time: 25-35 phút
├─ Best epoch: 35-50 (không phải full 100)
└─ Auto stop khi val_acc ko improve thêm 20 epochs
```

---

## 📊 **BƯỚC 7: PHÂN TÍCH KẾT QUẢ**

### **Cell 4: Xem Confusion Matrix**

```python
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load confusion matrix
cm_path = '/kaggle/working/outputs_kaggle/confusion_matrix.pt'
if os.path.exists(cm_path):
    cm = torch.load(cm_path).numpy()
    
    # Load class names
    with open('/kaggle/working/outputs_kaggle/labels.json') as f:
        labels = json.load(f)['class_names']
    
    print("=" * 80)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    
    # Tính metrics mỗi class
    recalls = np.diag(cm) / cm.sum(axis=1)
    precisions = np.diag(cm) / cm.sum(axis=0)
    f1s = 2 * recalls * precisions / (recalls + precisions + 1e-10)
    
    # Sort by recall (weak first)
    idx = np.argsort(recalls)
    
    print("\nOP (Recall) per class (yếu → mạnh):")
    print("-" * 80)
    for i in idx:
        print(f"{labels[i]:8} | Recall: {recalls[i]:>8.1%} | Precision: {precisions[i]:>10.1%} | n={int(cm[i,:].sum())}")
    
    # Vẽ heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, 
                yticklabels=labels, cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Gesture Recognition (14 classes)', fontsize=16)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
    plt.show()
    
    print("\n✅ Confusion matrix saved to: confusion_matrix.png")
else:
    print("❌ Confusion matrix file not found - training may have failed")

print("=" * 80)
```

**Nhấn Shift+Enter để chạy analysis**

---

## 💾 **BƯỚC 8: DOWNLOAD KẾT QUẢ**

### **Output Files (sau training xong)**

```
/kaggle/working/outputs_kaggle/
├─ stgcn_best.pt                ❌ USE THIS! (best model)
├─ stgcn_last.pt                (checkpoint cuối)
├─ confusion_matrix.pt          (cho phân tích)
├─ labels.json                  (class names)
├─ training_log.json            (epoch-by-epoch metrics)
├─ config.json                  (settings used)
└─ confusion_matrix.png         (visualization)
```

**Download files:**
```
Click phía bên phải notebook:
"Output" tab → "Download as Zip"

Hoặc tải từng file riêng
```

---

## 🎓 **BƯỚC 9: HIỂU KẾT QUẢ**

### **Ví dụ Output Analysis:**

```
🔴 WEAKEST CLASS:
G04: Recall 72.7% | Precision 88.9% | n=22
    → Model chỉ tìm được 72.7% trong 22 mẫu G04
    → Nhưng khi predict G04 thì thường đúng (88.9%)
    → Lý do: quá ít mẫu (22) → khó học
    → Cải thiện tiếp theo: tăng weight, augmentation

🟡 WEAK CLASS:
G05: Recall 78.3% | Precision 82.1% | n=35
    → Trung bình, chưa tốt
    → Nên cải thiện

🟢 STRONG CLASSES:
G10: Recall 94.1% | Precision 95.2% | n=83
G11: Recall 92.8% | Precision 93.5% | n=82
    → Rất tốt, ko cần cải thiện
```

### **Overall Accuracy: 87.9%**
```
Nghĩa: Trong 898 mẫu, model predict đúng 790 mẫu
Success rate: 87.9% ✓ (Tốt!)
```

---

## ⚠️ **BƯỚC 10: TROUBLESHOOTING**

### **❌ Lỗi: "ModuleNotFoundError: No module named 'torch_geometric'"**

**Giải pháp:**
```python
# Cell mới: Cài lại
!pip install torch-geometric -q

# Chạy lại training command
```

---

### **❌ Lỗi: "FileNotFoundError: train_merged_t60_accel.npz"**

**Kiểm tra:**
1. Dataset đã upload chưa?
2. Tên đúng chưa? (chính xác: `train_merged_t60_accel.npz`)
3. Path đúng chưa? (phải là `/kaggle/input/dataset-name/...`)

**Giải pháp:**
```python
# Kiểm tra path đúng
import os
for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        if 'train_merged' in file:
            print(os.path.join(root, file))
            
# Copy path chính xác vào cmd training
```

---

### **❌ Lỗi: "CUDA out of memory"**

**Giải pháp - Giảm batch size:**
```python
# Trong Cell training, thay:
'--batch-size', '32',

# Thành:
'--batch-size', '16',

# Accuracy sẽ giảm ~0.5%, nhưng ko crash
```

---

### **❌ Lỗi: "No module named 'models'" hoặc "No module named 'dataset'"**

**Kiểm tra:**
- Có file `__init__.py` trong folder `models/` không?
- Có file `__init__.py` trong folder `dataset/` không?

**Giải pháp:**
```python
# Upload files:
models/__init__.py    (có thể trống)
dataset/__init__.py   (có thể trống)
```

---

### **❌ Training bị interrupt (timeout)**

**Lý do:** Kaggle free tier timeout sau 9 giờ

**Giải pháp:**
```python
# Giảm epochs:
'--epochs', '100',  →  '--epochs', '50',
'--patience', '20',  →  '--patience', '10',
```

---

## 📋 **CHECKLIST TRƯỚC KHI START**

Kiểm tra trước khi chạy:

- [ ] **Dataset uploaded?**
  - [ ] File: `train_merged_t60_accel.npz`
  - [ ] Dung lượng: ~17 MB
  - [ ] Dataset name: ___________

- [ ] **Code files uploaded?**
  - [ ] train.py
  - [ ] models/stgcn.py
  - [ ] models/__init__.py
  - [ ] dataset/stgcn_dataset.py
  - [ ] dataset/__init__.py
  - [ ] tools/analyze_confusion_matrix.py

- [ ] **requirements.txt có?**
  - [ ] torch
  - [ ] torch-geometric
  - [ ] numpy

- [ ] **Path trong training command đúng?**
  - [ ] `/kaggle/input/YOUR-DATASET-NAME/train_merged_t60_accel.npz`
  - [ ] Replace `YOUR-DATASET-NAME` ← Tên thực tế

- [ ] **Kernel version?**
  - [ ] Python 3.10+
  - [ ] GPU available (P100 hoặc T4)

---

## 📈 **EXPECTED RESULTS**

```
Nếu mọi thứ OK:

Training: 25-35 phút
├─ Epoch 1: val_acc ~8% (random guessing)
├─ Epoch 10: val_acc ~60%
├─ Epoch 20: val_acc ~80%
├─ Epoch 30: val_acc ~87% ← BEST
├─ Epoch 40: val_acc ~87% (stable)
└─ Epoch 50: STOP (early stopping)

Final Results:
├─ Overall Accuracy: 87-89% ✓
├─ Best Epoch: 30-35 (trước khi plateau)
├─ Weak classes (G04): 72-78% (+5% vs normal training)
├─ Strong classes (G10): 94-96%
└─ Best model: /kaggle/working/outputs_kaggle/stgcn_best.pt
```

---

## 🎯 **NEXT STEPS AFTER TRAINING**

### **Lần 1 (Iteration 1):**
1. ✅ Train xong
2. ✅ Xem confusion matrix
3. ✅ Ghi lại accuracy từng class
4. ✅ Lưu model + results

### **Lần 2 (Iteration 2) - Cải Thiện Weak Classes:**

Nếu G04 chỉ 72.7%, có thể:

**Option A: Tăng Sampling Weight**
```python
'--max-class-weight', '4.0',   →   '--max-class-weight', '8.0',
```

**Option B: Tăng Augmentation**
```python
'--aug-jitter-std', '0.02',    →   '--aug-jitter-std', '0.05',
'--aug-time-warp', '0.05',     →   '--aug-time-warp', '0.10',
```

**Option C: Longer Training**
```python
'--epochs', '100',             →   '--epochs', '150',
'--patience', '20',            →   '--patience', '30',
```

---

## 📞 **QUICK REFERENCE - Thứ Tự Commands**

```
1. Cell 1: !pip install torch-geometric -q
2. Cell 2: Kiểm tra files
3. Cell 3: python train.py ... (MAIN TRAINING)
4. Cell 4: Phân tích confusion matrix
5. Cell 5: Download outputs
```

---

## ✨ **SUMMARY**

**Nếu làm đúng các bước:**
- ✅ 25-35 phút training xong
- ✅ 87-89% accuracy
- ✅ Có confusion matrix analysis
- ✅ Có best model ready to use

**Nếu có lỗi:**
- 👉 Check troubleshooting section
- 👉 Verify paths
- 👉 Download logs để debug

---

**🚀 Ready? Bắt đầu từ BƯỚC 1!**

Nếu có câu hỏi, xem lại bước nào đó hoặc check CURRENT_STATUS.md
