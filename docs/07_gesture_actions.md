# 07 — Gesture Actions (Gắn phím / chạy ứng dụng từ cử chỉ)

Tài liệu này mô tả cách cấu hình hệ thống demo webcam để khi nhận dạng cử chỉ tay sẽ gửi phím hoặc chạy một ứng dụng/lệnh ngoài trên Windows.

## Vị trí cấu hình
- File cấu hình mặc định: `Gan_nut/gesture_config.json` (có thể override bằng `--config <path>` khi chạy `tools/demo_webcam.py`).

## Cú pháp mapping
File `gesture_config.json` có dạng JSON với key là mã cử chỉ (ví dụ `G03`) và value là chuỗi mô tả hành động.

Supported action formats:
- Key press: `"G03": "w"` → Gửi một lần phím `w` bằng PyAutoGUI.
- Run command/app: `"G04": "run:notepad.exe"` → Chạy `notepad.exe` bằng `subprocess.Popen(..., shell=True)` (không chặn vòng lặp camera).
- Type text (chuỗi): `"G05": "type:hello world"` → Gõ chuỗi "hello world" bằng `pyautogui.typewrite` (nếu cần). (Nếu project cần, có thể bật tuỳ chọn này sau.)

Lưu ý: Hiện tại script mặc định hỗ trợ `key` (pyautogui.press) và `run:` (subprocess.Popen). Nếu bạn muốn hỗ trợ `type:` hoặc `run:<delay>:cmd`, có thể yêu cầu để tôi thêm.

## Ví dụ `gesture_config.json`
```
{
  "D0X": "",
  "B0A": "",
  "B0B": "",
  "G01": "",
  "G02": "",
  "G03": "w",
  "G04": "run:notepad.exe",
  "G05": "type:hello world",
  "G06": "run:C:\\Program Files\\MyApp\\myapp.exe",
  "G07": "",
  "G08": "",
  "G09": "",
  "G10": "",
  "G11": ""
}
```

## Chạy và test
1. Cài PyAutoGUI (nếu muốn gửi phím):

```bash
pip install pyautogui
```

2. Chạy demo webcam (từ project root):

```bash
python tools/demo_webcam.py
```

Hoặc chỉ định config khác:

```bash
python tools/demo_webcam.py --config Gan_nut/gesture_config.json
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