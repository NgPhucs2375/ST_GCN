# Quick Start: Confusion Matrix Analysis on Kaggle

## 🚀 Step-by-Step Workflow

### Step 1: Train Model (25-35 minutes)

```bash
# In Kaggle notebook, run:
!pip install torch-geometric -q

!python train.py \
    --data data/processed/train_merged_t60_accel.npz \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --dropout 0.05 \
    --weighted-sampler \
    --class-weighted-loss \
    --max-class-weight 4.0 \
    --label-smoothing 0.15 \
    --scheduler cosine \
    --patience 20 \
    --aug-jitter-std 0.02 \
    --aug-flip-prob 0.5 \
    --aug-time-warp 0.05 \
    --aug-drop-frames 1 \
    --val-ratio 0.2 \
    --seed 42 \
    --out outputs_kaggle
```

**Expected output:**
```
Epoch 001 | train loss 2.8439 acc 0.065 | val loss 2.8235 acc 0.088 | lr 1.00e-03
...
Epoch 030 | train loss 0.1823 acc 0.942 | val loss 0.3842 acc 0.879 | lr 9.70e-04 ← BEST
...
Epoch 050 | Early stopping at epoch 50 (best acc 0.879)
Training complete!
```

**Output files created:**
```
outputs_kaggle/
├── stgcn_best.pt              ← Best model
├── stgcn_last.pt              ← Last model
├── confusion_matrix.pt        ← For analysis
├── labels.json                ← Class names
├── training_log.json          ← Epoch metrics
└── config.json                ← Your config
```

---

### Step 2: Run Confusion Matrix Analysis

#### Option A: Python Script (Recommended)

```bash
# In Kaggle notebook, run:
!python tools/analyze_confusion_matrix.py \
    --cm-file outputs_kaggle/confusion_matrix.pt \
    --labels-file outputs_kaggle/labels.json \
    --output-dir results
```

**Output:**
```
════════════════════════════════════════════════════════════════════════════════
CONFUSION MATRIX ANALYSIS REPORT
════════════════════════════════════════════════════════════════════════════════
Overall Accuracy: 87.90%
Total Samples: 898
Total Classes: 14
════════════════════════════════════════════════════════════════════════════════

📉 WEAK CLASSES (Top 5 by Recall):
────────────────────────────────────────────────────────────────────────────────
Rank  Class    Recall      Precision    F1         n     
────────────────────────────────────────────────────────────────────────────────
1     G04      72.7%       88.9%        79.9%      22    🔴 CRITICAL
2     G05      78.3%       82.1%        80.1%      35    🟡 WEAK
3     B0B      90.2%       89.5%        89.8%      67    🟢 OK
...
────────────────────────────────────────────────────────────────────────────────

✅ STRONG CLASSES (Top 3 by Recall):
────────────────────────────────────────────────────────────────────────────────
...
────────────────────────────────────────────────────────────────────────────────

🔀 CONFUSION PATTERNS - Where weak classes go wrong:
────────────────────────────────────────────────────────────────────────────────

G04 (Recall: 72.7%, n=22):
  ✓ Correct: 16
  🔴 → B0B: 3 samples (13.6%) [most common confusion]
  🟡 → G05: 2 samples (9.1%)
  🟢 → D0X: 1 sample  (4.5%)

G05 (Recall: 78.3%, n=35):
  ✓ Correct: 27
  🔴 → G04: 4 samples (11.4%)
  🟡 → B0B: 2 samples (5.7%)

💡 PATTERN ANALYSIS - What to improve:
────────────────────────────────────────────────────────────────────────────────

G04 (n=22):
  Recall:    72.7% ⬇️ LOW
  Precision: 88.9% ✓
  Pattern: HIGH PRECISION, LOW RECALL
  Cause: Model is conservative (predicts class rarely)
  Solutions:
    ✓ Already applied: WeightedSampler (4x)
    ✓ Already applied: Class weights in loss
    ✓ Already applied: Label smoothing (0.15)
    → Next: Try threshold adjustment or more augmentation

📊 MISCLASSIFICATION SUMMARY:
────────────────────────────────────────────────────────────────────────────────
Total samples: 898
Correct predictions: 790 (87.9%)
Incorrect predictions: 108 (12.1%)

Errors per class (FN + FP):
Class    Errors   Error Rate   Type
────────────────────────────────────────────────────────────────────────────────
G04      6        27.3%        FN:6, FP:2
G05      8        22.9%        FN:8, FP:4
...
════════════════════════════════════════════════════════════════════════════════
✅ Analysis complete! Check outputs:
  - confusion_matrix_heatmap.png
  - confusion_matrix_metrics.csv
════════════════════════════════════════════════════════════════════════════════
```

**Generated files:**
```
results/
├── confusion_matrix_heatmap.png    ← Visual matrix
└── confusion_matrix_metrics.csv    ← Per-class metrics
```

---

#### Option B: Manual Analysis (If script fails)

```python
import torch
import json
import numpy as np

# Load data
cm = torch.load('outputs_kaggle/confusion_matrix.pt').numpy()
with open('outputs_kaggle/labels.json') as f:
    labels = json.load(f)['class_names']

# Calculate per-class metrics
recalls = np.diag(cm) / cm.sum(axis=1)
precisions = np.diag(cm) / cm.sum(axis=0)
f1s = 2 * recalls * precisions / (recalls + precisions + 1e-10)

# Sort by recall (weakest first)
idx = np.argsort(recalls)

print("WEAK CLASSES (sorted by recall):")
for i in idx:
    print(f"{labels[i]:8} | Recall: {recalls[i]:.1%} | Precision: {precisions[i]:.1%} | F1: {f1s[i]:.1%}")

# Show confusions for G04
print("\nG04 confusion pattern:")
g04_idx = labels.index('G04')
for j, count in enumerate(cm[g04_idx, :]):
    if count > 0 and j != g04_idx:
        print(f"  → {labels[j]}: {int(count)} samples")
```

---

### Step 3: Analyze Results

#### Read the Report

**Key metrics to understand:**

```
Metric                  | Meaning
────────────────────────────────────────────────────────────────
Overall Accuracy: 87.9% | Out of 898 samples, 790 predicted correctly

G04 Recall: 72.7%       | Out of 22 G04 samples, 16 predicted correctly
                        | → Missing 6 G04 samples (false negatives)

G04 Precision: 88.9%    | When model predicts G04, it's right 88.9% times
                        | → Sometimes predicts G04 when it's not (2 FP)

G04 F1: 79.9%           | Balance between recall (72.7%) and precision (88.9%)
                        | → Overall performance score for this class

Pattern:                | HIGH precision, LOW recall
Interpretation:         | Model doesn't predict G04 often, but when it does,
                        | it's usually right → conservative model

Most confused with:     | B0B (3 samples, 13.6% of G04 errors)
                        | → G04 often mistaken as B0B
```

#### Identify Pattern

```
Check your weakest class (e.g., G04):

IF recall > 0.80 and precision > 0.85:
   Pattern: BALANCED, good model
   
IF recall < 0.75 and precision > 0.85:
   Pattern: HIGH PRECISION, LOW RECALL
   Cause: Model conservative (predicts rarely)
   Solution: Increase sampling weight, more augmentation
   
IF recall < 0.75 and precision < 0.85:
   Pattern: BOTH LOW
   Cause: Hard to learn, possible data quality issue
   Solutions:
      1. Check data quality (videos, annotations)
      2. Increase model capacity
      3. More aggressive augmentation
      4. Check what class it confuses with most
      
IF recall > 0.85 and precision < 0.80:
   Pattern: HIGH RECALL, LOW PRECISION
   Cause: Model overconfident
   Solution: Reduce class weight, increase regularization
```

---

### Step 4: Plan Next Iteration

Use the **WEAK_CLASS_IMPROVEMENT_STRATEGY.md** guide to decide:

```
From confusion matrix analysis:

1. **Weakest class:** G04 (recall 72.7%)
2. **Pattern:** High precision, low precision
3. **Confusion:** Mostly with B0B
4. **Decision:**
   → This is Pattern 1 (low recall, high precision)
   → Model is conservative
   → Solution: Increase sampling weight
   
NEXT ITERATION CONFIG:
   --max-class-weight 8.0  # (was 4.0)
   --aug-jitter-std 0.05   # (was 0.02)
   --patience 30           # (was 20)
```

---

## 📊 Interpretation Examples

### Example 1: Weak Class with Issue

```
G04: Recall 72.7%, Precision 88.9%
     ↑ Low recall        ↑ High precision
     → Model is CONSERVATIVE
     → Predicts G04 rarely, but usually right
     
Most confused: B0B (13.6%)
     → When model is wrong on G04, it guesses B0B
     
Action:
     → Class is undersampled (only 22 samples)
     → Increase WeightedSampler: 4x → 8x
     → More augmentation for G04
     → Maybe check: Is G04 really different from B0B?
```

### Example 2: Strong Class

```
G10: Recall 94.1%, Precision 95.2%
     ↑ High both ways
     → Model does great
     
No action needed - this class is learned well!
```

### Example 3: Problematic Pair

```
From confusion patterns:
G04 → B0B: 13.6% of G04 errors
B0B → G04: 4.5% of B0B errors

This is a BIDIRECTIONAL confusion:
├─ G04 samples predicted as B0B
└─ B0B samples predicted as G04

Root cause options:
├─ Gesture G04 is too similar to B0B
├─ Landmarks not distinctive enough
├─ Data quality issue (mislabeled?)
└─ Model confusion (needs better features)

Actions:
├─ Check videos: Is annotation correct?
├─ Add specific augmentation for this pair
├─ Feature engineering: What makes them different?
└─ Or: Maybe they should be one class?
```

---

## 🛠️ Implementing Improvements

### Change 1: Increase Class Weight (Easiest)

```bash
# Old command:
!python train.py --max-class-weight 4.0 ...

# New command:
!python train.py --max-class-weight 8.0 ...
```

---

### Change 2: More Augmentation (Easy)

```bash
# Old command:
!python train.py \
    --aug-jitter-std 0.02 \
    --aug-flip-prob 0.5 \
    --aug-time-warp 0.05 \
    --aug-drop-frames 1

# New command:
!python train.py \
    --aug-jitter-std 0.05 \     # INCREASE
    --aug-flip-prob 0.5 \
    --aug-time-warp 0.10 \      # INCREASE
    --aug-drop-frames 2         # INCREASE
```

---

### Change 3: Longer Training (Very Easy)

```bash
# Old command:
!python train.py --epochs 100 --patience 20 ...

# New command:
!python train.py --epochs 150 --patience 30 ...
```

---

### Change 4: Better Model (Requires Code Edit)

In `models/stgcn.py`:

```python
# BEFORE:
in_channels = 6
out_channels = [96, 96, 192, 384]  # ← Current

# AFTER:
in_channels = 6
out_channels = [128, 128, 256, 512]  # ← Larger
```

Then retrain.

---

## ✅ Verification Checklist

After 2nd iteration training:

- [ ] Analysis complete: `python tools/analyze_confusion_matrix.py`
- [ ] Check: Did weakest class accuracy improve?
   - Old: G04 recall 72.7%
   - New: G04 recall ___% (should be > 72.7%)
- [ ] Check: Did strong classes stay strong?
   - G10 recall still ~94%?
- [ ] Check: Overall accuracy better?
   - Old: 87.9%
   - New: ___% (should be ≥ 87.9%)
- [ ] Document: What worked/didn't work

**Success:**
```
If G04 recall improved from 72.7% → 78%:
  ✅ Check! The change helped
  
If G04 recall improved by only 0.5%:
  🤔 Maybe try bigger change next time
  
If overall accuracy dropped:
  ❌ The change made things worse
  → Revert and try different approach
```

---

## 📞 Troubleshooting

### Error: "File not found: confusion_matrix.pt"
**Solution:** Make sure training completed successfully. Check for:
```bash
ls outputs_kaggle/
# Should show: stgcn_best.pt, confusion_matrix.pt, labels.json, etc.
```

### Error: "ModuleNotFoundError" in analysis script
**Solution:**
```bash
!pip install matplotlib seaborn torch torchvision scipy -q
```

### Analysis script runs but output confusing
**Solution:** Use the manual Python code below and debug step-by-step:
```python
import torch, json, numpy as np
cm = torch.load('outputs_kaggle/confusion_matrix.pt').numpy()
print(cm.shape)  # Should be (14, 14)
print(cm.sum())  # Should be 898
```

---

## 🎯 Summary

**3-Step Confusion Matrix Workflow:**

```
1️⃣ TRAIN
   └─ python train.py ... → outputs_kaggle/

2️⃣ ANALYZE
   └─ python tools/analyze_confusion_matrix.py
   └─ Get: weak classes, patterns, recommendations

3️⃣ IMPROVE
   └─ Pick ONE change based on pattern
   └─ Retrain
   └─ Compare results
   └─ Repeat
```

**Time Estimate:**
- Training: 25-35 minutes
- Analysis: 2 minutes
- Planning improvements: 5 minutes
- **Total per iteration: ~40 minutes**

**Expected improvement per iteration:**
- Pattern 1 (conservative): +4-6% recall on weak class
- Pattern 2 (moderate): +2-4% recall
- Pattern 3 (both low): +1-3% (needs deeper changes)

---

Good luck! 🚀 See you after the first training! 🎉
