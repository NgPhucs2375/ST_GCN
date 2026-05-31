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

## Mouse-follow (theo dõi bằng đầu ngón)

Bạn có thể bật chế độ để con trỏ chuột theo vị trí đầu ngón trỏ (landmark index 8). Có 2 cách:

### Cách 1: Bật/tắt bằng gesture (persistent toggle)
Dùng 2 gesture riêng để bật/tắt mouse-follow, nó sẽ giữ trạng thái cho đến khi bạn tắt:

```json
"G07": "mouse:follow_on",
"G08": "mouse:follow_off"
```

### Cách 2: Follow chỉ khi giơ 1 gesture (follow_hold) — **RECOMMENDED**
Thêm `|follow_hold` vào mapping. Mouse-follow sẽ **tự động tắt** khi bạn chuyển sang gesture khác hoặc tay rơi:

```json
"B0A": "mouse:follow_on|follow_hold"
```

**Hành vi:**
- Khi B0A (giơ 2 ngón) được nhận diện stable, mouse-follow bật.
- Con trỏ theo vị trí đầu ngón trỏ (landmark index 8).
- Ngay khi bạn chuyển sang gesture khác (ví dụ click, throw) hoặc tay rơi, mouse-follow tự tắt.
- Không cần 2 mapping on/off riêng biệt.

**Ưu điểm:**
- Thuận tiện: chỉ cần 1 gesture.
- An toàn: tự động tắt khi không dùng, không bị stuck trong chế độ follow.
- Dễ dùng: gesture nào có `follow_hold` sẽ kiểm soát follow mode.

### Tuning mouse-follow smoothness
Dùng flag `--mouse-follow-smooth` để điều chỉnh độ mượt của chuyển động (giá trị alpha EMA: 0..1):
- Cao (0.8–0.95): mượt mà nhưng chậm, ít nhạy (phù hợp khi vẽ/thiết kế).
- Thấp (0.1–0.3): nhanh nhạy nhưng rung lắc (phù hợp khi muốn follow chính xác từng pixel).
- Mặc định: 0.6 (cân bằng).

Ví dụ:
```bash
python tools/demo_webcam.py --mouse-follow-smooth 0.8
```

### Lưu ý
- Mouse-follow có thể bị ảnh hưởng bởi crop/scale camera và môi trường ánh sáng.
- Dùng `--stability-threshold` để tăng/giảm yêu cầu tay yên tĩnh trước khi action trigger.
- Dùng `--action-delay` để giảm hành vi rác khi tay rơi.
- Nếu mouse jump/không đúng vị trí, có thể do camera bị mirror/crop; liên hệ để điều chỉnh.

  ## Mẹo giảm nhận diện rác giữa các gesture tương tự (ví dụ zoom in/out)

  - Nếu hai gesture dễ bị nhầm lẫn (ví dụ `G10` = zoom in và `G11` = zoom out), hãy tăng `--stable-count` (global) hoặc dùng per-mapping `stable_count` option:

  ```
  "G10": "run:zoom_in_script|stable_count=5|still",
  "G11": "run:zoom_out_script|stable_count=5|still",
  ```

  Alternatively you can use `mutex` to mark them as mutually exclusive (give them the same mutex group name). The runtime will block the other action for a short time after one fires:

  ```
  "G10": "run:zoom_in_script|mutex=zoom",
  "G11": "run:zoom_out_script|mutex=zoom",
  ```

  - `still` yêu cầu cổ tay thực sự đứng yên trong sequence (chặt hơn threshold), giảm tình trạng nhảy nhót do tay chạm bàn.
  - Nếu bạn muốn hành động click/double-click nhạy, dùng `instant:` hoặc `repeat:N:`:

  ```
  "G01": "instant:mouse:left",
  "G08": "repeat:2:mouse:left",
  ```

  - Nếu gesture click vẫn nhạy quá (gõ nhiều lần), giảm tần suất bằng `--action-delay` hoặc thêm `stable_count` vào mapping.

## Giải quyết xung đột gesture (zoom in/out, v.v.)

### Vấn đề
Nếu 2 gesture tương tự (ví dụ G10 zoom in và G11 zoom out) dễ nhầm lẫn, bạn có thể thấy cả 2 hành động xảy ra gần như cùng lúc, hoặc 1 gesture không ăn được.

### Giải pháp: Mutex group
Thêm `|mutex=<group_name>` vào mapping. Khi 1 gesture trong group thực hiện, gesture khác trong cùng group sẽ bị block trong 0.6 giây.

**Ví dụ:**
```json
"G10": "hotkey:ctrl+=|mutex=zoom",
"G11": "hotkey:ctrl+-|mutex=zoom"
```

**Hành vi:**
- Bạn giơ gesture G10 (phóng to) → zoom in được thực hiện + mutex "zoom" được lock.
- Trong 0.6 giây tiếp theo, G11 (thu nhỏ) sẽ bị **block** (không thực hiện).
- Sau 0.6 giây, unlock → có thể thực hiện G11.

**Lợi ích:**
- Tránh 2 gesture chạy liên tiếp trong vòng 0.6s.
- Giảm gesture rác do nhận diện sai lẫn.

### Các mutex group khác
Bạn có thể dùng mutex cho bất kỳ gesture nào dễ nhầm lẫn:
- Zoom: `|mutex=zoom`
- Scroll: `|mutex=scroll`
- Navigation: `|mutex=nav`

Tất cả các gesture có cùng mutex name sẽ **không thể chạy cùng lúc** (trong 0.6s sau khi 1 cái chạy).

- "Không thấy log send_key hoặc send_action":
  - Kiểm tra console xem có in `→ send_key: ...` hay `→ send_action: ...` không. Nếu không thấy, có thể cử chỉ không đạt ngưỡng confidence hoặc đang bị cooldown (`--send-cooldown`).

## Tuỳ chọn nâng cao (có thể thêm)
- `run:<delay_seconds>:<command>` — đợi `delay_seconds` trước khi gửi phím/tiến hành thao tác (hữu ích khi mở app mới).
- `type:<text>` — gõ chuỗi text thay vì một phím đơn.
- Tự động đặt focus vào cửa sổ mới (Windows): cần thêm `pywin32` hoặc gọi PowerShell + user32 API.

Nếu bạn muốn, tôi có thể cập nhật ngay để hỗ trợ `run:<delay>:...` và `type:`. Bạn muốn tôi thêm cái nào trước?

python tools/demo_webcam.py --model data/stgcn_best.pt --labels data/labels.json --device cpu

---

## 🆕 Nâng cấp v2 — Motion Prediction + Blur Detection + Toggle Debounce

### 1. Motion Prediction (Dự đoán hướng di chuyển)
**Cái gì:** Con trỏ không chỉ theo vị trí hiện tại mà còn **dự đoán vị trí tiếp theo** dựa vào vận tốc của ngón tay.

**Cách hoạt động:**
- Lưu trữ 5 vị trí gần nhất của ngón trỏ (landmark 8) trong `MOTION_HISTORY`.
- Tính vận tốc từ 2 vị trí cuối: `velocity = pos_n - pos_n-1`.
- Dự đoán vị trí tiếp theo: `predicted = current + velocity * 0.3`.
- Áp dụng EMA smoothing trên predicted position.

**Kết quả:** Chuyển động mượt hơn, ít lag, cảm giác "trước thấy" (anticipatory).

**Tuning:**
- Factor = 0.3 (mặc định): cân bằng giữa dự đoán và vị trí hiện tại.
- Muốn tấn công hơn → tăng factor (0.5–0.8), nhưng có thể quá xa.
- Muốn an toàn hơn → giảm factor (0.1–0.2).

### 2. Blur Detection (Khử mờ ảnh)
**Cái gì:** Script tự động **bỏ qua frame bị mờ** (do rung tay, ánh sáng kém, chuyển động nhanh, v.v.).

**Cách hoạt động:**
- Dùng Laplacian variance để phát hiện blur.
- Ngưỡng mặc định: `BLUR_THRESHOLD = 100.0`.
- Nếu variance < 100 → frame bị mờ → skip inference.
- Overlay sẽ hiển thị "⚠️ Frame blurry (skip inference)" khi skip.

**Kết quả:**
- Giảm false positives từ frame mờ.
- Cải thiện độ chính xác nhận diện.
- Vòng lặp vẫn chạy bình thường (không lag).

**Tuning:**
- Tăng ngưỡng (120–150) nếu bạn skip quá nhiều frame.
- Giảm ngưỡng (50–80) nếu vẫn thấy frame mờ mà không skip.
- Nếu camera luôn mờ → kiểm tra camera (lau lens, tăng ánh sáng, giảm ISO nếu có).

### 3. Toggle Debounce — Follow ON/OFF ổn định ⭐⭐⭐
**Vấn đề cũ:**
- Khi gửi `mouse:follow_off`, đôi khi nó "để im" (không tắt follow).
- Hoặc spam on/off liên tục → follow bị nhảy/rung.

**Giải pháp:**
- Thêm **debounce khoảng 10 frame** (~0.3s @ 30fps) giữa các toggle.
- Nếu bạn spam gesture follow_on/off, script chỉ xử lý lần đầu rồi **bỏ qua các lần tiếp theo** cho đến khi cooldown hết.
- Khi toggle được chấp nhận, `MOTION_HISTORY` bị xóa (reset dự đoán).

**Kết quả:**
- Follow ON/OFF **ổn định**, không còn "để im".
- Giảm rung lắc khi toggle liên tục.
- Log in console: `[follow_toggle] ON @ frame 150 (debounced)` hoặc `OFF @ frame 165 (debounced)`.

**Tuning:**
- Cooldown = 10 frame (mặc định) → ~0.33s @ 30fps.
- Muốn thoáng hơn → giảm cooldown (nhưng cần cẩn thận, có thể toggle lại).
- Muốn cứng hơn → tăng cooldown (20–30 frame).

### 4. Responsive Mouse-Follow Start
**Cái gì:** Follow bắt đầu **ngay khi buffer có đủ `--min-action-frames`** thay vì đợi full buffer.

**Trước:** `if len(frame_buffer) == args.length` → phải đợi 30 frame.  
**Sau:** `if len(frame_buffer) >= args.min_action_frames` → chỉ đợi 8 frame (mặc định).

**Kết quả:** Mouse-follow **phản ứng nhanh hơn** ~3–4 lần, ít lag.

---

## Recommended Tuning

Để có trải nghiệm tốt nhất, dùng lệnh này:

```bash
python tools/demo_webcam.py \
  --config Gan_nut/gesture_config.json \
  --min-action-frames 8 \
  --early-conf 0.85 \
  --early-frames 2 \
  --stable-count 3 \
  --action-delay 0.35 \
  --mouse-follow-smooth 0.7 \
  --stability-threshold 0.03
```

**Giải thích:**
- `--min-action-frames 8`: bắt đầu infer khi có 8 frame (đủ responsive).
- `--early-conf 0.85`: high-confidence early trigger (skip 32-frame wait).
- `--early-frames 2`: cần 2 frame liên tiếp high-confidence.
- `--stable-count 3`: cần 3 lần cùng label (giảm false positive).
- `--action-delay 0.35`: debounce 0.35s giữa các action (tránh spam).
- `--mouse-follow-smooth 0.7`: EMA alpha = 0.7 (mượt, nhưng vẫn nhạy).
- `--stability-threshold 0.03`: wrist stability tightness (0..1).

---

## Troubleshooting Nâng cấp v2

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-----------|---------|
| Follow vẫn rung lắc | EMA alpha quá thấp | Tăng `--mouse-follow-smooth` (0.8–0.95) |
| Follow mở/đóng vẫn không chắc | Debounce cooldown quá ngắn | Dùng `follow_hold` thay vì toggle (an toàn hơn) |
| Frame liên tục bị mờ | Camera/ánh sáng kém | Lau lens, tăng ánh sáng, hoặc tăng blur threshold |
| Motion prediction quá tấn công | Factor = 0.3 quá cao | Giảm factor trong code (hiện hardcode 0.3) |
| Follow bắt đầu chậm | Buffer = 30, min-action-frames = 30 | Giảm `--min-action-frames` (8–15) |
| Inference vẫn lâu | Hardware yếu | Dùng `--device cpu` hoặc `--device cuda` (nếu có GPU) |

---

## Gesture Mapping Ví dụ (với nâng cấp v2)

```json
{
  "D0X": "",
  "B0A": "mouse:follow_on|still|follow_hold",
  "B0B": "",
  "G01": "",
  "G02": "mouse:right",
  "G03": "up",
  "G04": "down",
  "G05": "left",
  "G06": "right",
  "G07": "f5",
  "G08": "instant:mouse:left",
  "G09": "mouse:right",
  "G10": "hotkey:ctrl+=|mutex=zoom",
  "G11": "hotkey:ctrl+-|mutex=zoom"
}
```

**Highlights:**
- `B0A` có `still` + `follow_hold` → follow bắt đầu khi gesture stable và tay yên tĩnh.
- `G08` có `instant:` → click không chờ debounce (nhanh).
- `G10/G11` có `mutex=zoom` → zoom in/out không xung đột.

---

## Tóm tắt Nâng cấp

| Feature | Tác dụng | Tuning |
|---------|---------|--------|
| Motion Prediction | Mượt hơn, ít lag | Factor 0.1–0.8 |
| Blur Detection | Ít false positive | Threshold 50–150 |
| Toggle Debounce | Follow ON/OFF ổn định | Cooldown 5–30 frame |
| Responsive Follow | Follow bắt đầu nhanh | min-action-frames 5–15 |

Nếu bạn thấy vấn đề nào, hãy báo cáo log từ console (ví dụ `[follow_toggle] ON @ frame 150`) để tôi debug.