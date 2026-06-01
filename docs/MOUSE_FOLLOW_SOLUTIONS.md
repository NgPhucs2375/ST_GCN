# Giải Pháp Thực Hành — Cách Sửa Follow Không Tắt

## 🚀 Quick Fix — 3 Tuỳ Chọn

Chọn **1 trong 3** cách dưới đây:

---

## ✅ TUỲ CHỌN 1: Explicit ON/OFF (RECOMMENDED)

### Cách Hoạt Động
Dùng 2 gesture riêng biệt để bật/tắt follow.

### Config
```json
{
  "B0A": "mouse:follow_on",
  "B0B": "mouse:follow_off",
  "G01": "mouse:left",
  ...
}
```

### Cách Dùng
```
1. Giơ 2 ngón lần 1 (B0A) → "Follow ON"
   Console: "[follow_toggle] ON @ frame 150 (debounced)"

2. Di chuyển chuột tự do → chuột theo ngón tay

3. Giơ 2 ngón lần 2 (B0B) → "Follow OFF"
   Console: "[follow_toggle] OFF @ frame 200 (debounced)"
```

### Code (Already Implemented ✓)
Debounce toggle đã có sẵn trong `toggle_mouse_follow()`:
```python
def toggle_mouse_follow(enable: bool, frame_idx: int, force: bool = False) -> bool:
    global MOUSE_FOLLOW_ENABLED, FOLLOW_TOGGLE_FRAME
    
    if not force and (frame_idx - FOLLOW_TOGGLE_FRAME) < FOLLOW_TOGGLE_COOLDOWN:
        return False  # Ignore (in cooldown)
    
    MOUSE_FOLLOW_ENABLED = enable
    FOLLOW_TOGGLE_FRAME = frame_idx
    print(f"[follow_toggle] {'ON' if enable else 'OFF'} @ frame {frame_idx}")
    return True
```

### Ưu/Nhược
- ✓ Rõ ràng, an toàn, user-controlled.
- ✓ Debounce 10-frame (~0.33s) → reliable.
- ✗ Tốn 1 gesture (B0B).

### Khuyến Cáo
**👍 DÙNG CÁI NÀY** — đơn giản, tốn ít công.

---

## ⏰ TUỲ CHỌN 2: Auto-Off Timeout

### Cách Hoạt Động
Follow tự động tắt nếu để im > 3 giây.

### Config (Vẫn Dùng follow_hold)
```json
{
  "B0A": "mouse:follow_on|follow_hold",
  ...
}
```

### Code Cần Thêm
Thêm global:
```python
FOLLOW_IDLE_START = 0.0
FOLLOW_IDLE_TIMEOUT = 3.0  # 3 seconds
```

Thêm vào inference loop (trước mouse follow section):
```python
# Auto-off follow if idle > timeout
if MOUSE_FOLLOW_ENABLED:
    now = time.perf_counter()
    if MOTION_HISTORY:
        # Check if fingertip moved significantly
        if len(MOTION_HISTORY) >= 2:
            pos_now = np.array(MOTION_HISTORY[-1])
            pos_prev = np.array(MOTION_HISTORY[-2])
            movement = np.linalg.norm(pos_now - pos_prev)
            
            if movement < 1.0:  # < 1 pixel movement
                if FOLLOW_IDLE_START == 0.0:
                    FOLLOW_IDLE_START = now
                elif now - FOLLOW_IDLE_START > FOLLOW_IDLE_TIMEOUT:
                    MOUSE_FOLLOW_ENABLED = False
                    MOTION_HISTORY.clear()
                    print(f"[follow] auto-off after {FOLLOW_IDLE_TIMEOUT}s idle")
                    FOLLOW_IDLE_START = 0.0
            else:
                FOLLOW_IDLE_START = 0.0  # Reset on movement
```

### Hành Vi
```
Frame 1-3: Bạn giơ B0A
  → follow ON

Frame 4-100: Bạn để im
  → movement < 1px, idle counter tăng
  → sau 3s (90 frames @ 30fps): follow OFF tự động

Frame 101: Bạn di chuyển lại
  → movement > 1px, idle counter reset
  → follow ON lại
```

### Ưu/Nhược
- ✓ Tự động, không tốn gesture.
- ✓ Thông minh, theo dõi chuyển động.
- ✗ Timeout cứng (phải tune timeout value).
- ✗ Cần thêm ~20 dòng code.

---

## 🧠 TUỲ CHỌN 3: Movement Detection (Advanced)

### Cách Hoạt Động
Follow chỉ ON khi gesture stable **AND** tay đang di chuyển.

### Code Cần Thêm
Global:
```python
FOLLOW_MOVEMENT_THRESHOLD = 3.0  # pixels/frame
FOLLOW_IDLE_FRAMES = 0
FOLLOW_IDLE_LIMIT = 30  # 1 second @ 30fps
```

Thêm vào inference loop:
```python
# Movement detection for follow_hold
if MOUSE_FOLLOW_ENABLED and len(MOTION_HISTORY) >= 2:
    pos_now = np.array(MOTION_HISTORY[-1])
    pos_prev = np.array(MOTION_HISTORY[-2])
    movement = np.linalg.norm(pos_now - pos_prev)
    
    if movement > FOLLOW_MOVEMENT_THRESHOLD:
        # Tay đang di chuyển → follow ON, reset idle
        FOLLOW_IDLE_FRAMES = 0
    else:
        # Tay đứng yên → tăng idle counter
        FOLLOW_IDLE_FRAMES += 1
        if FOLLOW_IDLE_FRAMES > FOLLOW_IDLE_LIMIT:
            MOUSE_FOLLOW_ENABLED = False
            MOTION_HISTORY.clear()
            print(f"[follow] auto-off after {FOLLOW_IDLE_LIMIT/30:.1f}s no movement")
            FOLLOW_IDLE_FRAMES = 0
```

### Hành Vi
```
Bạn: Giơ B0A
  → follow ON

Bạn: Di chuyển (movement > 3px/frame)
  → follow ON (vẫn)

Bạn: Để im (movement < 3px/frame)
  → FOLLOW_IDLE_FRAMES tăng
  → sau 1s: follow OFF tự động

Bạn: Di chuyển lại
  → FOLLOW_IDLE_FRAMES reset, follow ON lại
```

### Ưu/Nhược
- ✓ Rất thông minh, tự nhiên.
- ✓ Cảm giác "thực tế" nhất.
- ✗ Phức tạp hơn, cần tuning threshold.
- ✗ Có thể bị noise (rung nhỏ coi là di chuyển).

---

## 🎯 So Sánh Nhanh

| | Explicit ON/OFF | Auto Timeout | Movement Detect |
|---|---|---|---|
| **Dễ hiểu** | ✓✓ | ✓ | ✓ |
| **Tự động** | ✗ | ✓✓ | ✓✓ |
| **Độ phức tạp** | ✓✓ (đơn) | ✓ (trung bình) | ~ (phức) |
| **Tuning cần** | ✗ | ✓ (timeout) | ✓ (threshold) |
| **Code cần thêm** | 0 dòng (sẵn) | ~20 dòng | ~30 dòng |
| **Dùng gesture riêng** | ✓ (B0B) | ✗ | ✗ |

---

## 💡 My Recommendation

### Ngay bây giờ → TUỲ CHỌN 1
```json
{
  "B0A": "mouse:follow_on",
  "B0B": "mouse:follow_off"
}
```
**Lý do:** Code đã có sẵn, không cần thêm, reliable.

### Nếu muốn thông minh hơn → TUỲ CHỌN 2
Thêm auto-timeout (20 dòng code).

### Nếu muốn "tự nhiên" nhất → TUỲ CHỌN 1 + 2
Combine: Explicit OFF + auto-timeout.

---

## ❓ Bạn Chọn Cái Nào?

1. **Tuỳ chọn 1** → Cách nhanh nhất (0 code)
2. **Tuỳ chọn 2** → Cách tiện nhất (20 dòng)
3. **Tuỳ chọn 3** → Cách thông minh nhất (30 dòng)
4. **1 + 2 combine** → Cách hoàn hảo nhất

---

## 📝 Cách Áp Dụng

### Nếu Chọn Tuỳ Chọn 1
**1. Cập nhật config:**
```bash
# Sửa Gan_nut/gesture_config.json
{
  "B0A": "mouse:follow_on",
  "B0B": "mouse:follow_off",
  ...
}
```

**2. Chạy lại:**
```bash
python tools/demo_webcam.py --config Gan_nut/gesture_config.json
```

**3. Test:**
- Giơ 2 ngón lần 1 → Console: `[follow_toggle] ON @ frame X`
- Di chuyển
- Giơ 2 ngón lần 2 → Console: `[follow_toggle] OFF @ frame Y`

### Nếu Chọn Tuỳ Chọn 2 hoặc 3
Hãy báo cáo, tôi sẽ patch code vào `tools/demo_webcam.py` ngay.

---

## 🚀 Next Steps

1. **Bạn chọn tuỳ chọn nào?**
2. **Nếu chọn 1:** Update config, test, báo cáo.
3. **Nếu chọn 2 hoặc 3:** Báo cáo, tôi patch code.
