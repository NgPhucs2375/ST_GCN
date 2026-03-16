# Setup

## Create venv (Windows)

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

## Install dependencies

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick import check

```bat
python -c "import torch, torch_geometric, mediapipe; print('ok')"
```

## Architecture flow (Mermaid)

```mermaid
flowchart TB
	subgraph Web[Browser (Client)]
		W1[Webcam Stream] --> W2[MediaPipe Hands]
		W2 -->|21 landmarks per frame| W3[Keypoint Buffer]
		W3 --> W4[Preprocess]
		W4 -->|Center at wrist| W4a[Normalize Translation]
		W4 -->|Scale by palm size| W4b[Normalize Scale]
		W4 -->|Optional: add velocity| W4c[Delta Features]
		W4a --> W5[Sequence Builder]
		W4b --> W5
		W4c --> W5
		W5 -->|Tensor: T x V x C| W6[Inference Input]
	end

	subgraph Inference[Inference Options]
		W6 --> S1[Option A: Backend API]
		W6 --> B1[Option B: In-Browser Runtime]

		S1 --> S2[ST-GCN (PyTorch)]
		S2 --> S3[Gesture Label + Confidence]

		B1 --> B2[ONNX/TFJS Runtime]
		B2 --> B3[ST-GCN (Web)]
		B3 --> B4[Gesture Label + Confidence]
	end

	S3 --> W7[UI Overlay]
	B4 --> W7

	subgraph Train[Training Pipeline]
		T1[Data Collection UI] --> T2[Saved Sequences]
		T2 --> T3[Train ST-GCN]
		T3 --> T4[Export Model]
		T4 --> S2
		T4 --> B2
	end
```
