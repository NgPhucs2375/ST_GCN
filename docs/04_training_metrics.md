# 04 — Train và Metrics (accuracy + confusion matrix)

File chính: [train.py](../train.py)

## DataLoader

- Dataset trả về `x` shape `(T, V, C)` và label index `y`.
- Trước khi feed vào model: `x.permute(0, 3, 1, 2)` → `(N, C, T, V)`.

## Accuracy là gì?

Accuracy = (số mẫu dự đoán đúng) / (tổng số mẫu).

Trong code:
- `preds = logits.argmax(dim=1)`
- so sánh `preds == y`

## Confusion matrix là gì?

Confusion matrix là bảng `(num_classes x num_classes)`:
- hàng (row) = nhãn thật (ground truth)
- cột (col) = nhãn dự đoán

Ví dụ:
- `cm[2,5]` lớn nghĩa là class 2 hay bị nhầm thành class 5.

Repo lưu:
- `outputs/confusion_matrix.pt` (PyTorch tensor)
- `outputs/labels.json` để mapping label → id

## Cách đọc nhanh confusion matrix

Bạn có thể load trong python:

```python
import json
import torch

cm = torch.load('outputs/confusion_matrix.pt')
labels = json.load(open('outputs/labels.json', 'r', encoding='utf-8'))
# labels: {"swipe_left": 0, ...}
```

Gợi ý: normalize theo row để xem tỉ lệ nhầm.
