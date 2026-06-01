# 07 — Gesture Actions

Tài liệu này mô tả cách `tools/demo_webcam.py` gắn cử chỉ tay với phím hoặc lệnh ngoài trên Windows.

## File cấu hình

- Mặc định: `Gan_nut/gesture_config.json`.
- Có thể override bằng `--config <path>` khi chạy demo.

## Cách hoạt động

`gesture_config.json` là một JSON object:

- key là mã cử chỉ, ví dụ `G03`.
- value là hành động cần thực hiện khi model nhận đúng cử chỉ đó.

## Hành động được hỗ trợ hiện tại

- Gửi một phím: `"G03": "w"`.
- Chạy lệnh hoặc app: `"G04": "run:notepad.exe"`.

Lưu ý:

- Script hiện dùng `pyautogui.press()` cho action dạng phím.
- Script dùng `subprocess.Popen(..., shell=True)` cho action dạng `run:`.
- Hiện chưa có support `type:` trong code thực tế.

## Ví dụ cấu hình

```json
{
  "D0X": "",
  "B0A": "",
  "B0B": "",
  "G01": "",
  "G02": "",
  "G03": "w",
  "G04": "run:notepad.exe",
  "G05": "",
  "G06": "",
  "G07": "",
  "G08": "",
  "G09": "",
  "G10": "",
  "G11": ""
}
```

### Prefixes hữu ích để giảm trễ/nhảy

- `instant:` — thực hiện action ngay lập tức, bỏ qua debounce/stability (dùng cho click đơn, double click khi bạn muốn action nhanh).
  - Ví dụ: `"G01": "instant:mouse:left"`
- `repeat:N:<action>` — chạy action N lần nhanh (nên dùng để mô phỏng double-click). Ví dụ: `"G08": "repeat:2:mouse:left"` sẽ click 2 lần.

Ví dụ mapping cho click/double-click:

```
"G01": "instant:mouse:left",
"G08": "repeat:2:mouse:left",
```

## Chạy và test

```bat
python tools/demo_webcam.py
python tools/demo_webcam.py --config Gan_nut/gesture_config.json
```

### Các tuỳ chọn mới (tối ưu cho độ ổn định)

- `--action-delay`: số giây tối thiểu giữa 2 action thực sự (debounce). Mặc định 0.4s.
- `--stability-threshold`: ngưỡng dịch chuyển cổ tay (normalized) trong sequence; nếu vượt quá thì action bị bỏ qua. Mặc định 0.03.
- `--mouse-follow-smooth`: alpha EMA (0..1) để làm mượt chuyển động chuột khi bật mouse-follow. Mặc định 0.6 (cao = mượt hơn).

- `--stable-count`: số lượng khung (frames/sequence window) mà cùng một label phải xuất hiện liên tiếp trước khi action được thực hiện. Mục đích giảm false positives (ví dụ giơ 2 ngón nhưng lặp thành zoom). Mặc định 3.

- `--min-action-frames`: số khung nhỏ nhất phải có trong buffer trước khi hệ thống cố gắng infer và thực hiện hành động. Mặc định 8. Giảm giá trị này để nhận nhanh hơn (nhưng có thể giảm độ chính xác). Giá trị tối ưu thường nằm trong 8..30.

Ví dụ chạy với tuning:

```bash
python tools/demo_webcam.py --config Gan_nut/gesture_config.json --action-delay 0.5 --stability-threshold 0.02 --mouse-follow-smooth 0.7
```

3. Quan sát console log:
- Khi cử chỉ được nhận, script sẽ in thông tin (ví dụ `→ send_key: pressed 'w'` hoặc `→ send_action: launching command: notepad.exe`).
- Nếu pyautogui không được cài, sẽ có thông báo cảnh báo khi khởi động.

4. Nếu mapping là `run:...`, sau khi console in `launching command`, click chuột vào cửa sổ ứng dụng (nếu cần) để đảm bảo app có focus và nhận phím.

## Vấn đề thường gặp và cách khắc phục
- "Nhận đúng cử chỉ nhưng không thấy gõ chữ vào app":
  - Nguyên nhân phổ biến nhất: ứng dụng không có focus. Sau khi script mở app, bạn cần đưa focus vào cửa sổ đó để phím giả lập gửi đến đúng nơi.
  - Ứng dụng mới mở có thể cần vài trăm ms để sẵn sàng — nếu gửi phím quá sớm thì không có tác dụng. Có thể cấu hình delay (tôi có thể bổ sung tuỳ chọn `run:<delay_seconds>:<cmd>`).
  - Nếu app chạy dưới quyền admin, còn script chạy ở quyền thường, pyautogui có thể không tương tác được với app đó. Chạy script với quyền admin để kiểm tra.

- "Không thấy log send_key hoặc send_action":
  - Kiểm tra console xem có in `→ send_key: ...` hay `→ send_action: ...` không. Nếu không thấy, có thể cử chỉ không đạt ngưỡng confidence hoặc đang bị cooldown (`--send-cooldown`).

## Tuỳ chọn nâng cao (có thể thêm)
- `run:<delay_seconds>:<command>` — đợi `delay_seconds` trước khi gửi phím/tiến hành thao tác (hữu ích khi mở app mới).
- `type:<text>` — gõ chuỗi text thay vì một phím đơn.
- Tự động đặt focus vào cửa sổ mới (Windows): cần thêm `pywin32` hoặc gọi PowerShell + user32 API.

Nếu bạn muốn, tôi có thể cập nhật ngay để hỗ trợ `run:<delay>:...` và `type:`. Bạn muốn tôi thêm cái nào trước?