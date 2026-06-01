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

## Chạy và test

```bat
python tools/demo_webcam.py
python tools/demo_webcam.py --config Gan_nut/gesture_config.json
```

Nếu muốn gửi phím qua automation, cài thêm `pyautogui`:

```bat
python -m pip install pyautogui
```

## Điều cần nhớ khi dùng trong app thực tế

- Action chỉ được kích khi confidence vượt ngưỡng `--min-confidence`.
- Cùng một nhãn sẽ bị chặn bởi `--send-cooldown` để tránh spam.
- Với `run:...`, cửa sổ đích cần có focus nếu muốn nhận phím tiếp theo.

## Dấu hiệu lỗi thường gặp

- Nếu nhận đúng cử chỉ nhưng không gửi action, kiểm tra confidence và cooldown.
- Nếu mở app mới nhưng không nhận phím, thường là do app chưa focus hoặc chưa kịp sẵn sàng.