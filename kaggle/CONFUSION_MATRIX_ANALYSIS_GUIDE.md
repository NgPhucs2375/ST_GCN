# Confusion Matrix Analysis Guide - Cải Thiện Weak Classes

## 🎯 Mục Đích

Sau khi train xong, confusion matrix sẽ show:
- **Accuracy từng lớp** (recall): Bao nhiêu % sample đúng được predict đúng
- **Precision từng lớp**: Khi model predict lớp X, bao nhiêu % đúng
- **Nhầm lẫn pattern**: Lớp X thường bị nhầm với lớp Y?

---

## 📊 Confusion Matrix là gì?

### Ví dụ 4 lớp (B0A, B0B, D0X, G01):

```
                PREDICTED
                B0A  B0B  D0X  G01
ACTUAL    B0A   93    2    3    2    ← Row = actual samples
          B0B    1   90    5    4
          D0X    2    4   92    2
          G01    3    2    1   94
          
Diagonal = đúng ✓
Off-diagonal = nhầm ✗
```

Từ ma trận này:
- **B0A đúng 93%** (93 out of 100 predicted correctly)
- **B0B đúng 90%**
- **B0B thường nhầm với D0X** (5 samples)
- **D0X thường nhầm với B0B** (4 samples)

---

## 📈 Metrics từ Confusion Matrix

### 1. Recall (Sensitivity) - "Tìm được bao nhiêu?"
```python
Recall[class i] = TP[i] / (TP[i] + FN[i])
                = Số dự đoán đúng / Tổng actual samples

Ví dụ: 
Recall[G04] = 16 / (16 + 6) = 16/22 = 72.7%
             ↑ dự đoán đúng  ↑ bị bỏ qua
```

**Meaning:** "Trong 22 samples G04 thực tế, model tìm được 72.7%"

### 2. Precision - "Dự đoán đó chính xác bao nhiêu?"
```python
Precision[class i] = TP[i] / (TP[i] + FP[i])
                   = Dự đoán đúng / Tất cả dự đoán là lớp i

Ví dụ:
Precision[G04] = 16 / (16 + 2) = 16/18 = 88.9%
                ↑ đúng  ↑ false positive
```

**Meaning:** "Khi model nói 'G04', nó đúng 88.9% lần"

### 3. F1-Score - "Balance giữa Recall vs Precision"
```python
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Ví dụ:
F1[G04] = 2 * (0.889 * 0.727) / (0.889 + 0.727)
        = 2 * 0.646 / 1.616
        = 0.799 (79.9%)
```

**Meaning:** "Tổng thể, model xử lý G04 được 79.9% (giữa tìm được (72.7%) và chính xác (88.9%))"

---

## 🔴 Xác Định Weak Classes

### A. Overall Accuracy từng lớp (Recall)

```
Sort theo recall:
G04:  72.7% ← WEAKEST  (22 samples)
G05:  78.3% ← weak     (35 samples)
B0B:  90.2% ← medium   (67 samples)
G10:  94.1% ← strong   (83 samples)
```

**Why G04 yếu?**
1. Có ít nhất sample (22) → overfitting risk cao
2. Gesture tương tự các lớp khác → khó phân biệt
3. Augmentation có thể không enough

### B. Nhầm Lẫn Pattern (Where does G04 go?)

```
G04 actual samples (22):
- 16 predict as G04 ✓
- 3 predict as G05 ← Thường nhầm vào lớp này
- 2 predict as B0B
- 1 predict as D0X
```

**Insight:** G04 thường bị nhầm với **G05** → có thể cải thiện bằng:
- Thêm augmentation khác nhau cho G04 vs G05
- Tăng model capacity học fine-grained difference
- Xem data quality G04 vs G05

---

## 🔍 Phân Tích Chi Tiết Per-Class

### Công thức tính từ confusion matrix:

```python
# For class i
TP[i] = diagonal[i,i]          # đúng
FP[i] = sum(col i) - TP[i]     # dự đoán i nhưng sai
FN[i] = sum(row i) - TP[i]     # actual i nhưng dự đoán sai
TN[i] = sum_all - TP[i] - FP[i] - FN[i]

Recall[i] = TP[i] / (TP[i] + FN[i])     # nhạy cảm
Precision[i] = TP[i] / (TP[i] + FP[i])  # chính xác
F1[i] = 2*(Precision*Recall)/(Precision+Recall)
Support[i] = TP[i] + FN[i]             # số sample actual
```

---

## 📋 Confusion Matrix Analysis Workflow

### Step 1: Load Matrix từ Training Output

```python
import json
import numpy as np

# File tự động save sau training
confusion_matrix = np.load("outputs_kaggle/confusion_matrix.pt")
with open("outputs_kaggle/labels.json") as f:
    class_names = json.load(f)["class_names"]
```

### Step 2: Tính Metrics Per-Class

```python
def compute_metrics(cm):
    """Compute recall, precision, f1 for each class"""
    metrics = {}
    
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        
        recall = TP / (TP + FN + 1e-10)
        precision = TP / (TP + FP + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        support = TP + FN
        
        metrics[class_names[i]] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "support": support
        }
    
    return metrics
```

### Step 3: Rank Weak Classes

```python
# Sort by recall (accuracy per class)
sorted_metrics = sorted(metrics.items(), 
                       key=lambda x: x[1]['recall'])

for cls, m in sorted_metrics:
    print(f"{cls:5} | Recall: {m['recall']:.1%} | "
          f"Precision: {m['precision']:.1%} | "
          f"F1: {m['f1']:.1%} | n={m['support']}")
```

**Output:**
```
G04   | Recall: 72.7% | Precision: 88.9% | F1: 79.9% | n=22
G05   | Recall: 78.3% | Precision: 82.1% | F1: 80.1% | n=35
B0B   | Recall: 90.2% | Precision: 89.5% | F1: 89.8% | n=67
G10   | Recall: 94.1% | Precision: 95.2% | F1: 94.6% | n=83
```

### Step 4: Analyze Confusion Pattern

```python
def show_confusions(cm, class_names, top_k=3):
    """Show what each weak class gets confused with"""
    
    for i in range(len(cm)):
        # Exclude diagonal (correct predictions)
        confusions = cm[i, :].copy()
        confusions[i] = 0
        
        # Top K confusion classes
        top_confused = np.argsort(confusions)[-top_k:][::-1]
        
        if confusions[top_confused].sum() > 0:
            cls = class_names[i]
            print(f"\n{cls} (recall {cm[i,i]/cm[i,:].sum():.1%}):")
            for j in top_confused:
                if confusions[j] > 0:
                    pct = confusions[j] / cm[i, :].sum() * 100
                    print(f"  → {class_names[j]}: {confusions[j]} samples ({pct:.1f}%)")
```

**Output:**
```
G04 (recall 72.7%):
  → B0B: 3 samples (13.6%)
  → G05: 2 samples ( 9.1%)
  → D0X: 1 sample  ( 4.5%)

G05 (recall 78.3%):
  → G04: 4 samples (11.4%)
  → G06: 3 samples ( 8.6%)
```

---

## 💡 Insights từ Confusion Matrix

### A. "Accuracy thấp vs Precision cao" (e.g., Recall=72%, Precision=89%)
```
Meaning: Model predict lớp này ít, nhưng khi predict thì thường đúng
Cause: Too conservative (thiếu confidence)

Solution:
├─ Lower classification threshold (nếu possible)
├─ Tăng weight cho lớp này trong loss
├─ Thêm data/augmentation cho lớp này
└─ Tăng model capacity học fine detail
```

### B. "Recall và Precision đều thấp" (e.g., cả 72%, 88%)
```
Meaning: Model chưa tìm được cách phân biệt tốt
Cause: Khó phân biệt gesture, data quality thấp

Solution:
├─ Xem lại data quality samples thấp
├─ Kiểm tra có duplicate/mislabel không
├─ Tăng augmentation để đa dạng
├─ Check correlation với lớp nhầm lẫn
└─ Nếu gesture quá giống → có thể combine classes
```

### C. "Confusion pattern A→B" (e.g., G04 → G05)
```
Meaning: Gesture G04 thường bị predict thành G05
Cause: Gesture quá giống nhau

Solution:
├─ Analyze video: G04 vs G05 khác gì?
├─ Tăng augmentation chỉ cho pair này
├─ Add explicit discrimination for this pair
├─ Check landmark extraction cho pair này
└─ Nếu quá giống → xem lại annotation
```

---

## 🛠️ Cải Thiện Per Weak Class

### Pattern 1: Weak Class with High Precision → Low Recall
```
Condition: Precision > 0.85, Recall < 0.80
Meaning: Model predict ít nhưng đúng

Actions (thứ tự ưu tiên):
1. ✓ WeightedSampler - oversample class này (ALREADY DONE: 4x)
2. ✓ Class-weighted loss - penalty lớn khi sai (ALREADY DONE)
3. ✓ Label smoothing - soften targets (ALREADY DONE: 0.15)
4. ✗ Lower threshold - adjust decision boundary
   └─ Requires post-training tuning
5. ✗ Augmentation specifically for this class
   → Can do in next iteration
```

### Pattern 2: Weak Class with Low Precision → High Recall
```
Condition: Precision < 0.85, Recall > 0.80
Meaning: Model predict nhiều nhưng thường sai

Actions:
1. ✗ Check data quality (mislabeled?)
2. ✗ Increase dropout/regularization
3. ✗ Reduce class weight (model too confident)
4. ✗ Reduce augmentation (too noisy?)
```

### Pattern 3: Both Low (Low Precision AND Recall)
```
Condition: Both < 0.80
Meaning: Model really struggles

Actions (priority order):
1. ✓ Check data: quality, duplicates, mislabels
   → Already done (quality_filtering)
2. ✗ Increase data: collect more, aggressive augmentation
3. ✗ Feature engineering: better landmarks, more features
4. ✗ Model capacity: larger model
5. ✗ Training time: longer training, different scheduler
```

---

## 📊 Reading Confusion Matrix

### Visual Example (14 classes):

```
         B0A B0B D0X G01 G02 G03 G04 G05 G06 G07 G08 G09 G10 G11
B0A [ 93   2   1   0   0   1   0   0   0   1   0   2   0   0]
B0B [  1  90   2   0   1   0   0   3   1   0   1   1   0   1]
D0X [  1   4  92   0   1   0   0   1   1   0   0   0   0   0]
G01 [  0   0   0  96   1   1   0   0   1   0   1   0   0   0]
G02 [  0   1   0   2  91   3   1   1   0   0   0   1   0   0]
G03 [  1   0   1   1   2  94   0   0   1   0   0   1   0   0]
G04 [  0   3   1   0   0   2  16   2   0   0   0   0   0   0]  ← WEAK!
G05 [  0   2   0   1   0   1   4  78   1   0   0   0   0   0]  ← weak
G06 [  0   1   1   1   1   0   0   1  89   0   0   1   1   0]
G07 [  0   0   0   0   0   1   1   0   0  92   1   0   0   0]
G08 [  1   1   0   0   0   1   0   0   1   1  89   0   0   1]
G09 [  2   1   0   1   0   0   0   0   1   0   2  92   0   1]
G10 [  0   0   1   0   0   0   0   0   2   0   0   0  94   0]
G11 [  0   0   0   0   1   0   0   0   1   0   1   0   2  92]
```

Key observations:
- **G04 (row 7):** Recall 16/22=72.7% ← Lowest
- **G04 confusion:** Mostly to B0B (3), G05 (2), D0X (1)
- **G05 (row 8):** Recall 78/100=78% ← Second lowest
- **G05 confusion:** To G04 (4), B0B (2)

---

## 🎯 Analysis Template (After Training)

Run script (provided) to get:

```
CONFUSION MATRIX ANALYSIS REPORT
════════════════════════════════════════════

OVERALL: 87.9% accuracy (best epoch 30)
Sample distribution: 14 classes, n=898

WEAK CLASSES (sorted by recall):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. G04     | Recall: 72.7% | Precision: 88.9% | F1: 79.9% | n=22
2. G05     | Recall: 78.3% | Precision: 82.1% | F1: 80.1% | n=35
3. B0B     | Recall: 90.2% | Precision: 89.5% | F1: 89.8% | n=67
...

CONFUSION PATTERNS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
G04 (n=22, recall 72.7%):
  ✗ → B0B: 3 samples (13.6%) [most common confusion]
  ✗ → G05: 2 samples (9.1%)
  ✗ → D0X: 1 sample  (4.5%)

G05 (n=35, recall 78.3%):
  ✗ → G04: 4 samples (11.4%)
  ✗ → B0B: 2 samples (5.7%)

RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For G04 (Recall=72.7%, Precision=88.9%):
├─ Pattern: High precision, low recall
├─ Cause: Model conservative, few samples (n=22)
├─ Next steps:
│  ├─ ✓ Already applied: WeightedSampler (4x)
│  ├─ ✓ Already applied: Class weights in loss
│  ├─ ✓ Already applied: Label smoothing 0.15
│  └─ Next iteration: Specific G04↔G05 augmentation
│     (since most confusion is with G05)
│
├─ Check data quality:
│  ├─ G04 vs G05: Gesture quá giống?
│  ├─ Landmark extraction robust?
│  └─ Annotation lại G04 sample?
│
└─ Potential fixes:
   ├─ Aggressive augmentation for G04
   ├─ Larger model (more capacity)
   ├─ Longer training (more epochs)
   └─ Better landmarks (check MediaPipe extraction)
```

---

## Next Steps After Analysis

### If Recall is Low but Precision is High:
→ Use more of that class (WeightedSampler with higher weight)
→ Current: 4x, try 6x-8x next

### If Precision is Low:
→ Model needs to learn better decision boundary
→ Collect more data for that class
→ Or: Reduce weight if mislabeled data

### If Both Low:
→ Data quality issue or gesture too hard
→ Check annotation, check augmentation, check landmarks

### If Specific Confusion Pattern (A→B):
→ Add targeted augmentation for that pair
→ Or: Feature engineering (what makes them different?)

---

## 📝 Summary Checklist

After running analysis script, fill this:

- [ ] Overall accuracy: _____%
- [ ] Best epoch: _____
- [ ] Weakest class: _______ (recall _____%)
- [ ] Second weakest: _______ (recall _____%)
- [ ] Most common confusion pattern: _____ → _____
- [ ] Recommendation #1: _________________________________
- [ ] Recommendation #2: _________________________________

**Then:** Based on findings, decide next iteration improvements!

---

## Useful Python Code Snippets

### Load confusion matrix from Kaggle output
```python
import torch
import json

cm = torch.load('outputs_kaggle/confusion_matrix.pt').numpy()
with open('outputs_kaggle/labels.json') as f:
    labels = json.load(f)['class_names']
```

### Calculate all metrics
```python
def get_full_metrics(cm):
    recalls = np.diag(cm) / cm.sum(axis=1)
    precisions = np.diag(cm) / cm.sum(axis=0)
    f1s = 2 * recalls * precisions / (recalls + precisions)
    supports = cm.sum(axis=1)
    return recalls, precisions, f1s, supports

recalls, precisions, f1s, supports = get_full_metrics(cm)
```

### Visualize confusion matrix heatmap
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, 
            yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix - 14 Gesture Classes')
plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png', dpi=300)
```

---

## Kết Luận

Confusion matrix là đồ thị cơ bản nhất để:
1. **Identify weak classes** (recall thấp)
2. **Understand confusion patterns** (nhầm lẫn với lớp nào)
3. **Plan improvements** (targeted fixes)

Sau khi train xong, run script để get report tự động!
