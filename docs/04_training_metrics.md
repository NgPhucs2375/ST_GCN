# 04 — Train và metrics

File chính: [train.py](../train.py)

## Dòng dữ liệu

- `STGCNDataset` trả về `(x, y)` với `x` shape `(T, V, C)`.
- Trước khi vào model, `train.py` đổi sang `(N, C, T, V)` bằng `permute(0, 3, 1, 2)`.

## Metrics chính

### Accuracy

Accuracy = số dự đoán đúng / tổng số mẫu.

Trong code:

- `preds = logits.argmax(dim=1)`.
- so sánh `preds == y`.

### Confusion matrix

Confusion matrix có shape `(num_classes, num_classes)`:

- row = nhãn thật.
- col = nhãn dự đoán.

Nếu `cm[2, 5]` lớn, class 2 đang hay bị nhầm sang class 5.

Repo lưu:

- `outputs/confusion_matrix.pt`.
- `outputs/labels.json`.

## Các tùy chọn train hiện có

`train.py` hỗ trợ:

- split `random` hoặc `stratified`.
- `--resume` để load checkpoint.
- `--weighted-sampler`.
- `--class-weighted-loss`.
- `--label-smoothing`.
- `--scheduler none|cosine|step`.
- `--patience` để early stopping.
- augmentation: jitter, time-warp, flip, drop-frames.
- `--mixup-alpha` để train với mixup trên sequence.

## Cách đọc kết quả

Sau mỗi epoch, log sẽ có:

- train loss / train acc.
- val loss / val acc.
- learning rate hiện tại.

`stgcn_best.pt` được lưu khi validation accuracy cải thiện.

## Ví dụ kiểm tra confusion matrix

```python
import json
import torch

cm = torch.load('outputs/confusion_matrix.pt')
labels = json.load(open('outputs/labels.json', 'r', encoding='utf-8'))
```

Gợi ý thực tế:

- xem row-wise normalization để thấy tỉ lệ nhầm.
- ưu tiên tăng dữ liệu cho các lớp bị nhầm nhiều nhất.
