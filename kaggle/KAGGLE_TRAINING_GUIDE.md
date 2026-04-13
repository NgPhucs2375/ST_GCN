# ST-GCN Hand Gesture Recognition - Kaggle Training Guide

## Setup Environment
```bash
!pip install torch-geometric -q
!pip install torchvision torchaudio torch-geometric torch -q
```

## Load Data & Convert to NPZ (if needed)
```bash
# If you already have NPZ files, skip this step and jump to Training

# If you have raw JSON files in data/raw_ipn_final_merged/:
!python tools/convert_sequences.py \
    --input data/raw_ipn_final_merged \
    --output data/processed/train_merged_t60_accel.npz \
    --length 60 \
    --use-velocity
```

## Training with Optimal Settings
```bash
!python train.py \
    --data data/processed/train_merged_t60_accel.npz \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --weighted-sampler \
    --class-weighted-loss \
    --max-class-weight 4.0 \
    --label-smoothing 0.1 \
    --scheduler cosine \
    --patience 20 \
    --aug-jitter-std 0.02 \
    --aug-flip-prob 0.5 \
    --aug-time-warp 0.05 \
    --aug-drop-frames 1 \
    --val-ratio 0.2 \
    --out outputs_kaggle
```

## Key Features:
✅ **Data:** 898 files, T=60 frames, 6 channels (pos+vel+accel)
✅ **WeightedRandomSampler:** Balance weak classes (G04=22 files)
✅ **Class-weighted loss:** Extra penalty for light classes
✅ **Label smoothing:** Prevent overfitting
✅ **Cosine scheduler:** Smooth LR decay
✅ **Early stopping:** patience=20 epochs
✅ **Data augmentation:** jitter + flip + time-warp + drop-frames

## Monitoring:
- Training logs: stdout
- Best model: outputs_kaggle/stgcn_best.pt
- Confusion matrix: outputs_kaggle/confusion_matrix.pt
- Labels mapping: outputs_kaggle/labels.json

## Expected Results:
- Training: 50-100 epochs
- Best accuracy: ~85-90% (with acceleration features)
- Weak class (G04): ~70-75% accuracy
- Strong classes (G10/G11): ~95%+ accuracy
