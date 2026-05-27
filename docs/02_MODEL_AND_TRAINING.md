# 02 - Model & Training

## Kien truc chinh

File trung tam la [`models/stgcn.py`](../models/stgcn.py). Mo hinh nhan dau vao shape `(N, C, T, V)` va tra logits shape `(N, num_classes)`.

## Y nghia input

- `N`: batch size.
- `C`: so kenh feature, thuong la `2`, `3`, `4` hoac `6`.
- `T`: so frame co dinh sau khi pad/trim.
- `V=21`: so landmark ban tay.

## Cach mo hinh hoat dong

- Moi frame la mot do thi 21 node.
- `build_hand_edge_index()` tao connectivity giua cac landmark.
- Spatial GCN xu ly moi frame theo graph.
- Temporal CNN hoc chuoi chuyen dong theo thoi gian.
- Residual connection giup on dinh khi tang/giam channel.

## Dat diem hien tai

Ban hien tai la ban skeleton de hoc va de debug:

- chua co partitioning strategy phuc tap nhu paper goc;
- chua co attention module;
- uu tien ro rang, de mo rong ve sau.

## Training flow

File train la [`train.py`](../train.py).

- Dataset tra ve `x` shape `(T, V, C)` va label `y`.
- Truoc khi dua vao model, tensor duoc chuyen thanh `(N, C, T, V)`.
- Sau moi epoch, repository theo doi accuracy va confusion matrix.

## Metrics can doc ngay

- Accuracy: ti le du doan dung tren tong mau.
- Confusion matrix: xem class nao hay bi nham sang class nao.

Artifact hien co trong repo:

- `outputs_resume/stgcn_best.pt`
- `outputs_resume/stgcn_finetuned_4ch.pt`
- `outputs_resume/stgcn_trained_4ch.pt`
- `outputs_resume/stgcn_trained_6ch.pt`
- `outputs_resume/stgcn_trained_9ch.pt`
- `outputs_resume/confusion_matrix.pt`
- `outputs_resume/labels.json`
- `outputs_resume/training_history.json`

## Cach doc confusion matrix

```python
import json
import torch

cm = torch.load("outputs_resume/confusion_matrix.pt")
labels = json.load(open("outputs_resume/labels.json", "r", encoding="utf-8"))
```

## Goi y tinh chinh

- Dung scheduler nhu cosine decay hoac step LR.
- Them early stopping neu val accuracy dung lai.
- Dung label smoothing, weight decay va dropout neu overfit.
- Neu du lieu it, uu tien tang data hon la tang model qua manh.

## Toi uu inference

- Giam `T` hoac `C` neu can latency thap hon.
- Chi chay suy luan sau 2-3 frame.
- Dung majority vote + cooldown de tranh nhay class.
- Neu can deployment nhanh, co the export ONNX.

## Dung quan trong

- Model hien co phu hop cho webcam demo va thuc nghiem.
- Muon len san pham thi can them chuan hoa input, smooth output va benchmark latency tren may dich.
