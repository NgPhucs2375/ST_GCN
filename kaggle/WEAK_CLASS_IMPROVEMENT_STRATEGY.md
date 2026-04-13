# Weak Class Improvement Strategy - Cải Thiện Chi Tiết

## 🎯 Overview: From Analysis to Action

After running confusion matrix analysis, bạn sẽ biết:
1. **Lớp nào yếu nhất?** (lowest recall)
2. **Nhầm với những lớp gì?** (confusion pattern)
3. **Pattern gì?** (low recall + high precision? vs low recall + low precision?)

Dựa vào pattern, bạn sẽ chọn **strategy cải thiện** phù hợp.

---

## 📊 5 Patterns & Solutions

### Pattern 1: Recall 70-80%, Precision 85%+
**"Model too conservative"**

```
Example: G04 (recall 72%, precision 89%)
Meaning: Model chỉ predict G04 khi rất confident
         → Bỏ lỡ 28% G04 samples
         → Nhưng khi predict G04 thường đúng
```

**Root causes:**
- Ít samples trong class (22 files)
- Model học pattern yếu
- Loss function không encourage high recall

**Solutions (Try in Order):**

✅ **Already Implemented (Current Model):**
```python
# 1. WeightedRandomSampler - oversample G04 4x
#    → G04 samples appear 4 times per epoch
batch = [... B0A, B0B, D0X, G04, G04, G04, G04 ...]

# 2. Class-weighted loss - penalty when wrong on G04
#    weight[G04] = 4.0 (highest)
#    if model predicts wrong on G04, loss is 4x higher

# 3. Label smoothing 0.15 - soft targets
#    y_true = [0, 0, 1, 0] → [0.0107, 0.0107, 0.9071, 0.0107]
#    → Model less confident, learns more patterns
```

🔄 **Next Iteration (If Still Weak):**

```python
# Solution 1: Increase sampling weight
# Current: weighted_sampler with class size weight
# Try: Even higher weight for G04
weight[G04] = 8.0 or 10.0  # vs 4.0 now

# Solution 2: Threshold adjustment
# Current: pred probability > 0.5 → classify as class
# Try: pred[G04] > 0.3 for G04 specifically
#      (require less confidence to predict G04)

# Solution 3: Specific augmentation for G04
# Add more aggressive augmentation for undersampled classes
aug_strength[G04] = 2x aug of normal classes

# Solution 4: Focal loss (advanced)
# Focus extra penalty on hard samples of weak classes
from torch.nn import functional as F
focal_loss = -alpha_t * (1 - p_t)**gamma * log(p_t)
#           ↑ focuses on hard negatives of weak classes

# Solution 5: Longer training + different scheduler
# Current: 100 epochs, cosine annealing
# Try: 150-200 epochs, step scheduler with smaller gamma
```

**Decision Tree:**
```
If recall[G04] < 75% and precision[G04] > 85%:
├─ Is n[G04] << n[other classes]?
│  └─ YES → Increase sampling weight (try 6-8x)
├─ Is G04 confused mostly with G05?
│  └─ YES → Add pair-specific augmentation
├─ Training stopped early?
│  └─ YES (early stopping at epoch 30)
│     → Try --patience 30 (wait longer before stopping)
└─ Still low?
   └─ Try Solution 5: longer training + focal loss
```

---

### Pattern 2: Recall 75-85%, Precision 82-88%
**"Model struggles, but improving"**

```
Example: G05 (recall 78%, precision 82%)
Meaning: Model has moderate difficulty, not huge problems
```

**Solutions:**

```python
# 1. Check confusion pattern
# If G05 → G04 frequently:
#    → Add augmentation to better separate G04-G05
#    → Check if gesture truly different in videos

# 2. More data/augmentation
batch_aug_multiplier[G05] = 1.5x

# 3. Feature engineering
#    → Add better landmarks (MediaPipe subsets)
#    → Velocity + Acceleration (already done ✓)
#    → Try other features (angles, distances, velocities)

# 4. Model capacity
#    → Increase hidden dims (already 96→96→192→384)
#    → Try 128→128→256→512
#    → Or add more STGCN blocks (currently 4, try 5-6)

# 5. Different scheduler
#    → Current: cosine with T_max=100
#    → Try: Step scheduler with step_size=20, gamma=0.5
#           (big drops every 20 epochs, exploration then fine-tune)
```

**Priority:**
1. Check confusion pattern (specific lớp?)
2. Add augmentation for that pair
3. If no pattern: increase model capacity

---

### Pattern 3: Both Recall AND Precision < 80%
**"Model really struggles - hard class"**

```
Example: If hypothetically both 70%
Meaning: Model not finding good decision boundary
         → Hard gesture to recognize
         → Or data quality issue
```

**Emergency Solutions:**

```python
# 1. CHECK DATA FIRST!
# → Visual inspection: is annotation correct?
# → Landmark quality: good MediaPipe extraction?
# → Duplicates: any duplicate samples?
# → Outliers: any very short/corrupted videos?

from tools.data_quality import check_sample_quality
quality = check_sample_quality("raw_data/sample.json")
print(f"Quality score: {quality['score']}")
print(f"Issues: {quality['issues']}")

# 2. Data cleaning in next iteration
# → Re-annotate if mislabeled
# → Remove low-quality landmarks
# → Add more diverse augmentation

# 3. Feature engineering
# → Different landmark subsets
# → Try different time windows (T=30, T=90)
# → Normalize differently (whole body vs hand only)

# 4. Model architecture changes
# → Deeper model (5-6 blocks instead of 4)
# → Different GCN variant (GCN vs GraphSAGE)
# → Attention mechanism for gesture regions

# 5. Training changes
# → Higher learning rate (0.002 instead of 0.001)
# → Different optimizer (Adam vs SGD)
# → Longer training (200+ epochs)
# → Different scheduler (warmer start with CosineAnnealingWarmRestarts)
```

**Diagnostic Workflow:**
```python
# Save predictions + confidence
def save_predictions():
    for batch in dataloader:
        x, y = batch
        logits = model(x)
        probs = softmax(logits)
        confidence = max(probs)
        
        # Save: [sample_id, true_label, pred_label, confidence, correct?]
        # Analyze:
        # - Which samples have low confidence?
        # - Are they all from certain class?
        # - Check those specific samples for quality

# Confusion pattern analysis
# If G04 confused with G05 70% of time:
# → They're too similar → data quality check
# → Or model not learning difference → need augmentation/features
```

---

### Pattern 4: High Recall, Low Precision
**"Model overconfident about this class"**

```
Example: Recall 92%, Precision 70%
Meaning: Find most G04 samples ✓
         But predict G04 for non-G04 samples too ✗
```

**Solutions:**

```python
# 1. Reduce class weight (current weight too high)
class_weight[weak_class] = 2.0  # instead of 4.0
# → Less penalty for being wrong on this class
# → Model doesn't overly focus on it

# 2. Increase regularization
dropout = 0.1  # instead of 0.05
weight_decay = 5e-4  # instead of 1e-4

# 3. Check for data issues
# → Mislabeled data (some G04 labeled as G05 accidentally)
# → Duplicates across classes
# → Too similar samples in different classes

# 4. Reduce augmentation for this class
# → If augmented too much, model learns too flexible representation
# → Makes it harder to differentiate from similar classes
```

---

### Pattern 5: Balanced but Low Overall
**"All classes struggling equally"**

```
Example: All recall 75-82%, precision 80-87%
Meaning: Model has general learning problem, not specific to one class
```

**Systematic Solutions:**

```python
# 1. Feature/Data quality
# → Better landmarks (try different body parts subset)
# → Better preprocessing (different normalization)
# → More augmentation variety

# 2. Model architecture
# → More capacity (larger dims)
# → Deeper model (more blocks)
# → Different architecture altogether (CNN + temporal model)

# 3. Training
# → Higher learning rate (0.002-0.005)
# → Longer training (200+ epochs)
# → Different optimizer (Adam vs SGD with momentum)
# → Warmup scheduler (gradual increase then decay)

# 4. Dataset
# → Check data is balanced well
# → Verify augmentation isn't corrupting data
# → Ensure validation split is representative
```

---

## 📋 Practical Improvement Roadmap

### Week 1: Quick Wins (Based on Current Threeoptimizations)

```
✓ Already have: 
  - Label smoothing 0.15 → soft targets
  - Cosine scheduler → smooth LR decay
  - Early stopping patience=20 → prevent overfitting
  - WeightedRandomSampler → class balance
  - Class-weighted loss → per-class penalty

→ Just train!
→ Analyze confusion matrix
→ See what pattern emerges
```

### Week 2: Targeted Improvements

**If weak class has Pattern 1 (high precision, low recall):**
```python
# Iteration 2 config:
--weighted-sampler \
--class-weighted-loss \
--max-class-weight 8.0 \      # INCREASE from 4.0
--label-smoothing 0.2 \        # INCREASE from 0.15
--patience 30 \                # INCREASE from 20
--scheduler cosine \
```

**If weak class has Pattern 3 (both low):**
```python
# Iteration 2 config:
# Expand model capacity
MODEL_DIMS = {
    'channel_0': 128, 128, 256, 512  # vs 96, 96, 192, 384
}
# + more augmentation
--aug-jitter-std 0.05 \        # INCREASE from 0.02
--aug-time-warp 0.10 \         # INCREASE from 0.05
```

### Week 3: Advanced Techniques

- **Focal loss** for hard samples
- **Mixup/CutMix** augmentation
- **Different architecture** (deeper, wider)
- **Different scheduler** (warmup + cosine)
- **Ensemble methods** (train multiple models)

---

## 🎯 Decision Making Framework

After confusion matrix analysis, ask these 5 questions:

### Q1: Which class is weakest?
→ Check recall ranking

### Q2: Is it low samples problem?
```python
if support[weak_class] < 30:
    # Likely undersampling
    → Increase weighted_sampler weight
    → Check if data quality acceptable
else:
    # Enough data but still weak
    → Check confusion pattern
    → Feature/model issue
```

### Q3: Is precision high?
```python
if precision[weak_class] > 0.85 and recall < 0.80:
    # Model conservative
    → Increase sampling weight
    → More augmentation
else:
    # Model confused
    → Check what it confuses with
    → Feature engineering
```

### Q4: What's the main confusion?
```python
top_confusion = argmax(cm[weak_class, :])

if top_confusion in [related_classes]:
    # Reasonable confusion
    → Add pair-specific augmentation
else:
    # Strange confusion
    → Data quality issue?
    → Model learning wrong features?
```

### Q5: Is it fixable with these knobs?
```python
knobs = [
    "sampling_weight",      # ✓ Easy
    "class_weight",         # ✓ Easy
    "label_smoothing",      # ✓ Easy
    "augmentation",         # ✓ Medium
    "model_capacity",       # ✓ Medium
    "training_schedule",    # ✓ Medium
    "scheduler_type",       # ✓ Advanced
    "feature_engineering",  # ✗ Hard
]

if recall > 0.70 and precision > 0.80:
    # Quick wins available
    → Tighten sampler/augmentation knobs
else:
    # Need more work
    → Feature engineering or data quality check
```

---

## 📊 Summary: By Pattern

| Pattern | Root Cause | Primary Fix | Secondary Fix |
|---------|-----------|-------------|---------------|
| Recall 70%, Prec 89% | Too conservative | ↑ sampling weight | ↑ augmentation |
| Recall 75%, Prec 82% | Moderate struggle | Check confusion | ↑ model capacity |
| Recall 60%, Prec 70% | Hard to learn | ✓ Check data quality | ↑ model/features |
| Recall 90%, Prec 70% | Overconfident | ↓ class weight | ↑ regularization |
| All 75-80% | System issue | ↑ model capacity | ↑ learning rate |

---

## 💾 Template: After Analysis

```
CONFUSION MATRIX ANALYSIS RESULTS:
═════════════════════════════════════

Overall Accuracy: ____%

WEAKEST 3 CLASSES:
1. _________ (recall ___%, precision ___%)
2. _________ (recall ___%, precision ___%)
3. _________ (recall ___%, precision ___%)

MAIN CONFUSION PATTERN:
__________ → __________ (__% of confusion)

PATTERN IDENTIFIED:
[ ] Pattern 1: High precision, low recall
[ ] Pattern 2: Moderate both
[ ] Pattern 3: Very low both
[ ] Pattern 4: High recall, low precision
[ ] Pattern 5: Balanced but low

RECOMMENDED NEXT ITERATION:
Action 1: ________________________
Action 2: ________________________
Action 3: ________________________

IMPLEMENTATION CHECKLIST:
[ ] Modify train.py with new hyperparameter
[ ] Update convert_sequences.py if needed
[ ] Run new training
[ ] Compare results
```

---

## 🚀 How to Implement Improvements

### Example 1: Increase Sampling Weight

**File: train.py** (around line 250-270)
```python
# BEFORE:
class_weights = 1.0 / class_counts
class_weights = np.clip(class_weights, 1.0, 4.0)
sampler_weights = class_weights[labels]

# AFTER (if G04 still weak):
class_weights = 1.0 / class_counts
class_weights = np.clip(class_weights, 1.0, 8.0)  # ← CHANGED: 4.0 → 8.0
sampler_weights = class_weights[labels]
```

**Run:**
```bash
python train.py --weighted-sampler --class-weighted-loss \
    --max-class-weight 8.0 ...  # ← Changed from 4.0
```

### Example 2: More Augmentation

**File: train.py** (around line 180-200)
```python
# BEFORE:
augmented = augment_sample(
    x,
    jitter_std=0.02,
    flip_prob=0.5,
    time_warp=0.05,
    drop_frames=1
)

# AFTER:
augmented = augment_sample(
    x,
    jitter_std=0.05,      # ← 0.02 → 0.05
    flip_prob=0.5,
    time_warp=0.10,       # ← 0.05 → 0.10
    drop_frames=2         # ← 1 → 2
)
```

**Run:**
```bash
python train.py --aug-jitter-std 0.05 \
    --aug-time-warp 0.10 --aug-drop-frames 2 ...
```

### Example 3: Longer Training

**Run:**
```bash
python train.py --epochs 150 \        # ← 100 → 150
    --patience 30 \                    # ← 20 → 30
    ...
```

---

## ✅ Validation Checklist

After each improvement iteration:

- [ ] Retrain with new config
- [ ] Re-run analysis: `python tools/analyze_confusion_matrix.py`
- [ ] Compare: New recall vs old recall for weak classes
- [ ] Check: Did improvement help without hurting others?
- [ ] Log: Document what worked/didn't

---

## 🎓 Remember

> "Perfect is the enemy of good"
> - Don't over-optimize one class at cost of others
> - Track metrics for ALL classes
>
> "Do one thing per iteration"
> - Change ONE hyperparameter or feature at a time
> - Know what caused the change (good or bad)
>
> "Trust the data"
> - If pattern is weird, check data quality first
> - Annotation, MediaPipe extraction, duplicates

---

Let's get to training! 🚀
