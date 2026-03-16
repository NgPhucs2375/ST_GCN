# Suggested Folder Structure

```
DL_DEMO/
  web/                      # MediaPipe capture UI
    index.html
    app.js
    style.css
  docs/                     # Tài liệu giải thích code/pipeline
    01_overview.md
    02_data_format.md
    03_model_stgcn.md
    04_training_metrics.md
    05_web_game_checklist.md
  data/
    raw/                    # JSON sequences saved from web
    processed/              # npz files for training
  tools/
    convert_sequences.py    # JSON -> npz
    data_quality.py         # Data quality report + filter raw JSON
  dataset/
    stgcn_dataset.py        # PyTorch Dataset
  models/
    stgcn.py                # ST-GCN model (torch-geometric + temporal conv)
  train.py                  # Training entry point
  requirements.txt
  SETUP.md
  README.md
```

# Web Deployment Checklist (Gesture Game)

- Collect 3-5 gestures with at least 50 sequences per class.
- Normalize and pad sequences to fixed length (e.g. 30 frames).
- Train ST-GCN and export model (pt, onnx).
- Choose inference path:
  - Server: Flask/FastAPI + WebSocket
  - Browser: ONNX Runtime Web or TFJS
- Inference settings:
  - Run every 2-3 frames
  - Confidence threshold (e.g. 0.7)
  - Majority vote across last N predictions
- Game loop integration:
  - Emit gesture events only when stable
  - Add cooldown to avoid spam actions
- UX:
  - Show live overlay of detected gesture
  - Visualize confidence bar for debug
