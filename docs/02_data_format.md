py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 02 — Dữ liệu, format tensor, và chuẩn hoá

## JSON thô từ web

Web UI lưu mỗi sample dưới dạng:

- `label`: tên cử chỉ (string)
- `frames`: list các frame
  - mỗi frame là list 21 điểm
  - mỗi điểm có `{x, y, z}` (x/y thường trong [0,1])

## Từ JSON → tensor

Script [tools/convert_sequences.py](../tools/convert_sequences.py) convert thành mảng numpy:

- `array`: shape `(T, V, C)`
  - `T`: số frame
  - `V=21`: số landmark
  - `C=2` nếu bỏ z, hoặc `C=3` nếu dùng z

Sau đó lưu vào `npz`:

- `sequences`: shape `(N, T, V, C)`
- `labels`: shape `(N,)`

## Chuẩn hoá đang dùng

Trong `normalize_frames()`:

1) Center (translation)
   - lấy landmark 0 (wrist)
   - `frames = frames - wrist`

2) Scale
   - dùng landmark 9 (palm) sau khi đã center
   - `scale = ||palm||`
   - `frames = frames / scale`

## Velocity (tuỳ chọn)

Bật `--use-velocity`:

- tính `velocity[t] = frames[t] - frames[t-1]`
- concat theo kênh → C tăng gấp đôi

Ví dụ:
- nếu dùng x,y: C=2 → sau velocity thành 4
- nếu dùng x,y,z: C=3 → sau velocity thành 6

## Pad/trim

Vì model cần `T` cố định:
- nếu sequence dài hơn `T`: cắt
- nếu ngắn hơn: lặp frame cuối để đệm
