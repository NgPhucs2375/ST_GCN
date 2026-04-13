# 🎯 KAGGLE TRAINING - QUICK VISUAL GUIDE

**Đọc guide này trước! Chỉ có 3 bước chính!**

---

## 📋 **QUY TRÌNH 3 BƯỚC**

```
BƯỚC 1: CHUẨN BỊ
   └─ Upload dataset + code
       ⏱️ 10-15 phút

BƯỚC 2: CHẠY TRAINING
   └─ Copy-paste lệnh & chạy
       ⏱️ 25-35 phút (Kaggle tự làm việc)
       
BƯỚC 3: NHÌN KẾT QUẢ
   └─ Phân tích accuracy
       ⏱️ 2-3 phút
```

---

## 🎬 **BƯỚC 1: CHUẨN BỊ (10-15 phút)**

### **1a. Chuẩn Bị File Local**

Run command này trên máy để kiểm tra file ready:

**Option A: PowerShell (Chạy từng cái)**
```powershell
cd d:\Univer\Nam_3\HKII\DL_DEMO
dir data\processed\train_merged_t60_accel.npz
dir train.py
dir models\stgcn.py
dir dataset\stgcn_dataset.py
dir tools\analyze_confusion_matrix.py
```

**Option B: Cmd (Hoặc kiểm tra nhanh)**
```cmd
cd d:\Univer\Nam_3\HKII\DL_DEMO
dir data\processed\train_merged_t60_accel.npz
dir train.py
dir models
dir dataset
dir tools
```

**Check:**
- ✅ All files exist?
- ✅ train_merged_t60_accel.npz is ~17 MB?

---

### **1b. Tạo Dataset trên Kaggle (CHỈ LẦN ĐẦUI)**

⚠️ **CHỈ CẦN LÀM 1 LẦN!** Publish xong dùng lại

**Step 1:** Vào https://www.kaggle.com/settings/datasets

**Step 2:** Click **"Create New Dataset"** → Tab **"Upload"**

**Step 3:** Upload file `train_merged_t60_accel.npz` (17 MB)

**Step 4:** Điền thông tin
```
Title: gesture-recognition-final
Description: Training data for ST-GCN hand gesture recognition
Subtitle: 898 samples, 6 channels, T=60 frames, 21 vertices
```

**Step 5:** Chọn Framework & License

⚠️ **FRAMEWORK:**
```
Dropdown: Chọn "Deep Learning"
hoặc "PyTorch"
(hoặc skip - ko quan trọng)
```

⚠️ **LICENSE:**
```
Dropdown: Chọn "CC0: Public Domain"
hoặc "CC-BY 4.0"
(Public domain tốt nhất)
```

**Step 6:** Click **"Create"** → Chờ 2-3 phút publish

✅ **Giờ bạn có dataset, dùng lại được lần sau!**

💡 **Lưu ý:** Framework & License chỉ là metadata, ko ảnh hưởng việc train!


---

### **1c. Tạo Notebook + Upload Code Files**

**Step 1:** Vào https://www.kaggle.com → "Code" → "New Notebook"

**Step 2:** Ở trên code editor → Click **"+ Add Input"**
```
Chọn dataset vừa tạo: gesture-recognition-final
Click [Attach]
```

**Step 3:** Upload Code Files vào Notebook

**Option A: Files Panel (DỄ - Khuyến Khích)**
```
Bên trái notebook → Click tab "Files" 🗂️
   ↓
Click [Upload]
   ↓
Chọn các file này từ máy:
  ├─ train.py
  ├─ requirements.txt
  ├─ models/ (folder)
  ├─ dataset/ (folder)
  └─ tools/
      └─ analyze_confusion_matrix.py  ⭐ BẮTBUỘC
      (các file khác tools/ tùy chọn - ko cần cho training)
   ↓
Click [Upload]
```

💡 **TIP:** Kéo thả (drag-drop) folder cũng được!

⚠️ **CHỈ CẦN:**
- ✅ train.py
- ✅ requirements.txt
- ✅ models/ (folder)
- ✅ dataset/ (folder)
- ✅ tools/analyze_confusion_matrix.py

❌ **KO CẦN (tùy chọn cho sau):**
- ❌ tools/augment_balance.py
- ❌ tools/compare_checkpoints.py
- ❌ tools/convert_sequences.py
- ❌ tools/data_quality.py
- ❌ tools/demo_webcam.py
- ❌ tools/infer.py
- ❌ tools/inspect_confusion.py
- ❌ tools/ipn_to_json.py


---

**Option B: %writefile (Nếu files panel ko work)**

Tạo cell đầu tiên:
```python
%writefile train.py
# [Copy toàn bộ nội dung file train.py từ máy, dán vào đây]

%writefile requirements.txt
torch
torch-geometric
numpy
numpy==1.24.0

%mkdir -p models dataset tools

%writefile models/__init__.py
# Leave empty

%writefile models/stgcn.py
# [Copy nội dung stgcn.py]

# Tương tự cho dataset/stgcn_dataset.py, tools/analyze_confusion_matrix.py
```

✅ **Dùng Option A nếu có thể, dễ hơn!**

---

## 🚀 **BƯỚC 2: CHẠY TRAINING (25-35 phút)**

### **Cell 1: Cài Thư Viện**

```python
!pip install torch-geometric -q
```

**Chạy:** Shift+Enter  
**Chờ:** 2-3 phút loading packages

---

### **Cell 2: Kiểm Tra Cấu Trúc**

```python
import os

# Check files
print("Checking files...")

# --- NEW: Check if working directory is empty ---
if not os.listdir('/kaggle/working/'):
    print("‼️ ERROR: Your code files are missing!")
    print("➡️ Please go to the 'Files' tab on the left, click 'Upload', and add:")
    print("   - train.py")
    print("   - requirements.txt")
    print("   - models/ (folder)")
    print("   - dataset/ (folder)")
    print("   - tools/ (folder)")
    print("-" * 50)
# ---------------------------------------------

files = [
    'train.py',
    'models/stgcn.py',
    'dataset/stgcn_dataset.py',
    'tools/analyze_confusion_matrix.py'
]

for f in files:
    status = "✅" if os.path.exists(f) else "❌"
    print(f"{status} {f}")

# Check dataset
import os
dataset_path = '/kaggle/input/gesture-recognition-final/train_merged_t60_accel.npz'
if os.path.exists(dataset_path):
    print(f"✅ Dataset: {dataset_path}")
else:
    print(f"❌ Dataset not found - check dataset name!")

print("\nAll OK? Continue to Cell 3!")
```

**Expected Output:**
```
✅ train.py
✅ models/stgcn.py
✅ dataset/stgcn_dataset.py
✅ tools/analyze_confusion_matrix.py
✅ Dataset: /kaggle/input/gesture-recognition-final/train_merged_t60_accel.npz

All OK? Continue to Cell 3!
```

---

### **Cell 3: TRAINING (Main Command)**

```python
import subprocess
import sys

# ⚠️ IMPORTANT: Replace 'gesture-recognition-final' with YOUR dataset name!

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

print("=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80)

result = subprocess.run(cmd, capture_output=False)
print("\n" + "=" * 80)
if result.returncode == 0:
    print("✅ TRAINING COMPLETE!")
else:
    print("❌ Training failed - check error above")
print("=" * 80)
```

**Nhấn:** Shift+Enter

**Xem Output:**
```
Epoch 001 | train loss 2.8439 acc 0.065 | val loss 2.8235 acc 0.088 | lr 1.00e-03
Epoch 002 | train loss 2.6123 acc 0.124 | val loss 2.5821 acc 0.156 | lr 1.00e-03
...
Epoch 030 | train loss 0.1823 acc 0.942 | val loss 0.3842 acc 0.879 | lr 9.70e-04 ← BEST
...
Epoch 050 | train loss 0.0512 acc 0.981 | val loss 0.3821 acc 0.877 | lr 8.91e-04
Early stopping at epoch 50 (best val acc 0.879)
✅ TRAINING COMPLETE!
```

---

## 📊 **BƯỚC 3: NHÌN KẾT QUẢ (2-3 phút)**

### **Cell 4: Confusion Matrix Analysis**

```python
import torch
import json
import numpy as np

print("=" * 80)
print("CONFUSION MATRIX ANALYSIS")
print("=" * 80)

# Load results
cm = torch.load('/kaggle/working/outputs_kaggle/confusion_matrix.pt').numpy()

with open('/kaggle/working/outputs_kaggle/labels.json') as f:
    labels = json.load(f)['class_names']

# Calculate metrics
recalls = np.diag(cm) / cm.sum(axis=1)
precisions = np.diag(cm) / cm.sum(axis=0)

# Sort by recall (weakest first)
idx = np.argsort(recalls)

print("\n🔴 WEAKEST CLASSES (Cần cải thiện):")
print("-" * 80)
for i in idx[:5]:
    print(f"{labels[i]:8} | Recall: {recalls[i]:>8.1%} | Precision: {precisions[i]:>10.1%}")

print("\n🟢 STRONGEST CLASSES (Tốt rồi):")
print("-" * 80)
for i in idx[-3:][::-1]:
    print(f"{labels[i]:8} | Recall: {recalls[i]:>8.1%} | Precision: {precisions[i]:>10.1%}")

overall_accuracy = np.trace(cm) / cm.sum()
print(f"\n📈 OVERALL ACCURACY: {overall_accuracy:.1%}")
print("=" * 80)
```

**Expected Output:**
```
🔴 WEAKEST CLASSES (Cần cải thiện):
────────────────────────────────────────────────────────────────────────────────
G04      | Recall:    72.7% | Precision:     88.9%
G05      | Recall:    78.3% | Precision:     82.1%
B0B      | Recall:    90.2% | Precision:     89.5%
...

🟢 STRONGEST CLASSES (Tốt rồi):
────────────────────────────────────────────────────────────────────────────────
G11      | Recall:    92.8% | Precision:     93.5%
G10      | Recall:    94.1% | Precision:     95.2%

📈 OVERALL ACCURACY: 87.9%
════════════════════════════════════════════════════════════════════════════════
```

---

### **Cell 5: Download Results**

```python
import shutil
import os

# Save outputs
output_dir = '/kaggle/working/outputs_kaggle'
if os.path.exists(output_dir):
    shutil.make_archive('/kaggle/working/results', 'zip', output_dir)
    print("✅ Results saved to: results.zip")
    print("Download from Output panel →")
else:
    print("❌ No outputs found")
```

**Download:**
```
Right panel → Output tab → results.zip
└─ Click download icon
```

---

## ✅ **TROUBLESHOOTING QUICK FIX**

| Problem | Fix |
|---------|-----|
| ❌ "ModuleNotFoundError: torch_geometric" | Run Cell 1 again: `!pip install torch-geometric -q` |
| ❌ "FileNotFoundError: train_merged_t60_accel.npz" | Check: /kaggle/input/YOUR-DATASET-NAME/ (correct name?) |
| ❌ "CUDA out of memory" | Change Cell 3: `'--batch-size', '16',` (was 32) |
| ❌ "No module named 'models'" | Upload files to /kaggle/working/ (check Cell 2) |
| ⏱️ Timeout (>9 hours) | Reduce epochs: `'--epochs', '50',` |

---

## 🎯 **WHAT YOU'LL GET**

```
After training (35 phút):

Files tự động save tại: /kaggle/working/outputs_kaggle/
├─ stgcn_best.pt         ← Best model (87.9% accuracy)
├─ confusion_matrix.pt   ← For analysis
├─ training_log.json     ← Epoch-by-epoch metrics
└─ config.json           ← Settings used

Metrics:
├─ Overall Accuracy: 87-89% ✓
├─ G04 (weak): 72.7%
├─ G10 (strong): 94.1%
└─ Training time: 25-35 minutes ✓
```

---

## 📱 **EXPECTED PROGRESS**

```
Timeline:
├─ Cell 1: 2-3 phút (package install)
├─ Cell 2: 1 phút (verify)
├─ Cell 3: 25-35 phút (training) ← Main work
│   ├─ Epoch 1: val_acc 8% (random)
│   ├─ Epoch 10: val_acc 60% (learning)
│   ├─ Epoch 20: val_acc 80% (improving)
│   ├─ Epoch 30: val_acc 87% ← BEST
│   └─ Epoch 50: STOP (auto)
└─ Cell 4-5: 2 phút (analysis + download)

Total: ~45 phút (includes wait time)
```

---

## 💡 **KEY POINTS**

```
✅ DO:
  ├─ Check dataset path exactly
  ├─ Verify all files uploaded
  ├─ Wait for training complete
  ├─ Check accuracy result
  └─ Download outputs

❌ DON'T:
  ├─ Click "Interrupt" during training
  ├─ Close notebook tab
  ├─ Modify training command (unless you know)
  ├─ Change batch-size > 32 (might OOM)
  └─ Give up if first epoch is slow
```

---

## 📞 **EVERYTHING YOU NEED**

**Before Starting:**
- ✅ Dataset ready: train_merged_t60_accel.npz
- ✅ Code ready: train.py + models/ + dataset/
- ✅ Config ready: requirements.txt

**During Training:**
- ⏳ Watch epochs progress (should see val_acc increase)
- ✅ If val_acc goes 8% → 60% → 87%, you're good!
- ❌ If stuck at 10%, check files/paths

**After Training:**
- ✅ Download results
- ✅ Check confusion matrix
- ✅ Celebrate! 🎉

---

## 🚀 **LET'S START!**

```
1. Follow BƯỚC 1 (chuẩn bị)       ← 10-15 phút
2. Follow BƯỚC 2 (training)        ← 25-35 phút
3. Follow BƯỚC 3 (kết quả)        ← 2-3 phút

Total: ~50 phút

Your model ready! 🎯
```

---

**Ready? Go to KAGGLE_COMPLETE_GUIDE.md for full details!**
