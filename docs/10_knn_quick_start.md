# 🎯 KNN Mode - Quick Start Guide

## Overview

KNN (K-Nearest Neighbors) gesture recognition is a **lightweight alternative to ST-GCN** that:
- ✅ Runs at **60+ FPS on CPU** (no GPU needed)
- ✅ Uses personalized gesture templates from calibration
- ✅ Is **faster and more responsive** than neural networks
- ❌ Requires gesture calibration first

---

## Step 1: Calibrate Gestures

Before using KNN mode, you must record gesture templates:

```bash
python tools/calibrate_gestures.py
```

**Interactive workflow:**
1. Program displays each gesture (14 total: G01-G14)
2. **SPACE** = Start/stop recording gesture landmarks
3. **N** = Move to next gesture
4. **Q** = Quit and save

**Output:** `data/gesture_templates.json` (contains mean + std for each gesture)

**Example recording:**
```
📺 Recording gesture: G01 (Click 1 ngón)
   Press SPACE to start recording...
   Press SPACE again to stop...
   Recorded 20 samples ✅
   Press N to move to next gesture...
```

---

## Step 2: Run Demo with KNN Mode

Once calibration is complete:

```bash
python tools/demo_webcam.py --use-knn
```

**Available KNN-specific flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use-knn` | bool | False | Enable KNN matching mode |
| `--knn-threshold` | float | 5.0 | Distance threshold (lower = stricter matching) |
| `--knn-k` | int | 3 | Number of neighbors for voting |
| `--template-file` | path | `data/gesture_templates.json` | Custom template path |

**Example with tuning:**

```bash
# Stricter matching (threshold=3.0, k=5 neighbors)
python tools/demo_webcam.py --use-knn --knn-threshold 3.0 --knn-k 5

# Looser matching (threshold=8.0, k=1 = closest only)
python tools/demo_webcam.py --use-knn --knn-threshold 8.0 --knn-k 1
```

---

## Step 3: Tune for Your Environment

### If gestures are **too sensitive** (false positives):
- **Increase threshold:** `--knn-threshold 7.0` or `--knn-threshold 10.0`
- Or use `--knn-k 5` (more consensus needed)

### If gestures are **not detected** (missed:
- **Decrease threshold:** `--knn-threshold 3.0` or `--knn-threshold 2.0`
- Or use `--knn-k 1` (closest match only)
- Recalibrate: ensure you record clear, consistent gestures in `calibrate_gestures.py`

### Best starting point:
```bash
python tools/demo_webcam.py --use-knn --knn-threshold 5.0 --knn-k 3
```

---

## Output & Monitoring

When KNN mode initializes, you'll see:

```
✅ KNN matcher initialized with threshold=5.0, k=3
```

During execution:
- **Green text:** High confidence match → action triggered
- **Orange/yellow text:** Low confidence match → requires stability
- **Gray text:** "Khong chac" = No confident match yet

Real-time distance metrics are displayed in overlay.

---

## Troubleshooting

### Error: "Template file not found"
```
⚠️  Template file not found: data/gesture_templates.json
   Please run: python tools/calibrate_gestures.py
```

**Solution:** Run calibration first:
```bash
python tools/calibrate_gestures.py
```

### Error: "Failed to load KNN matcher"
- Check that `data/gesture_templates.json` is valid JSON
- Ensure all 14 gesture templates (G01-G14) are present
- Try re-calibrating

### Gestures not working in KNN mode
1. **Check gesture config:** Ensure gestures are mapped in `Gan_nut/gesture_config.json`
2. **Re-calibrate:** Run `python tools/calibrate_gestures.py` again with clearer motions
3. **Adjust threshold:** Increase or decrease `--knn-threshold` to match your hand size/lighting
4. **Fallback to ST-GCN:** Run without `--use-knn` to test if ST-GCN works

### Performance / FPS drops
- KNN is CPU-only; ensure no other GPU-intensive tasks running
- Reduce camera resolution: `--camera-width 640 --camera-height 480`
- Reduce confidence threshold: `--min-confidence 0.4`

---

## Combining KNN + Gesture Options

You can still use all gesture options with KNN mode:

```json
{
  "G01": "hotkey:ctrl+c|stable_count=3|follow_hold",
  "G02": "mouse:follow_on",
  "G03": "run:notepad.exe|instant"
}
```

**Example command:**
```bash
python tools/demo_webcam.py --use-knn --knn-threshold 4.0 --config Gan_nut/gesture_config.json
```

---

## Comparing KNN vs ST-GCN

| Aspect | KNN | ST-GCN |
|--------|-----|--------|
| **Speed** | 60+ FPS (CPU) | 30-45 FPS (GPU) |
| **Accuracy** | Very good (personalized) | Excellent (trained on dataset) |
| **Setup** | Needs calibration | Pre-trained model |
| **GPU Required** | ❌ No | ✅ Yes (recommended) |
| **Best for** | Personal use | Production / general dataset |
| **Latency** | ~16ms | ~23ms |

**Recommendation:** Use KNN for **real-time personal use** (webcam apps, mouse control). Use ST-GCN for **dataset-wide evaluation**.

---

## Demo Workflow

```bash
# Terminal 1: Calibrate (one-time setup, ~2 minutes)
python tools/calibrate_gestures.py

# Terminal 2: Run demo with KNN
python tools/demo_webcam.py --use-knn --knn-threshold 5.0

# Adjust settings if needed, re-run with different thresholds
python tools/demo_webcam.py --use-knn --knn-threshold 4.0  # Stricter
python tools/demo_webcam.py --use-knn --knn-threshold 6.0  # Looser
```

---

## See Also

- **[Calibration Guide](./09_calibration_knn_guide.md)** — Detailed calibration workflow with screenshots
- **[Gesture Actions](./07_gesture_actions.md)** — Mapping syntax (run:, hotkey:, mouse:, etc.)
- **[Mouse Follow Troubleshooting](./08_mouse_follow_troubleshooting.md)** — Debug mouse follow issues
