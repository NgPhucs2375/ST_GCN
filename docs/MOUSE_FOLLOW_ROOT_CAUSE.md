# Giải Thích Chi Tiết — Tại Sao Follow Không Tắt

## 📌 Tóm Tắt Ngay

**Vấn đề:** Khi bạn giữ gesture B0A (2 ngón) stable, follow vẫn ON dù bạn để im.

**Lý do:** `follow_hold` = "follow khi gesture stable", không phải "follow khi di chuyển".

**Giải pháp:** Dùng explicit ON/OFF gesture hoặc thêm auto-off timeout.

---

## 🔬 Phân Tích Code

### Hiện Tại (Line 676-680 tools/demo_webcam.py)

```python
# If this mapping is follow_hold, enable/disable follow based on stability
if follow_hold_requested:  # ← check if mapping has "|follow_hold"
    if stable_label or early_trigger or is_instant:
        MOUSE_FOLLOW_ENABLED = True   # ← ON nếu stable
    else:
        MOUSE_FOLLOW_ENABLED = False  # ← OFF nếu không stable
```

git reset --soft HEAD~1
### stable_label là gì? (Line 668-669)

```python
lh = list(label_history)
tail = lh[-args.stable_count:] if args.stable_count > 0 else lh
stable_label = (len(tail) == args.stable_count and all(l == top_label for l in tail))
#             └─ TRUE khi last N frames đều có cùng gesture
```

**Default:** `--stable-count 3` → cần 3 frame liên tiếp cùng gesture B0A.

---

## 📽️ Ví Dụ Cụ Thể

### Scenario 1: Bạn Để Im

```
Frame 1: Giơ 2 ngón (B0A)
  top_label = "B0A"
  label_history = [B0A]
  stable_label = FALSE (1 < 3)
  FOLLOW = OFF

Frame 2: Vẫn giơ (để im)
  top_label = "B0A"
  label_history = [B0A, B0A]
  stable_label = FALSE (2 < 3)
  FOLLOW = OFF

Frame 3: Vẫn để im
  top_label = "B0A"
  label_history = [B0A, B0A, B0A]
  stable_label = TRUE ✓
  FOLLOW = ON ← Bắt đầu follow!

Frame 4-99: Bạn vẫn để im (không di chuyển)
  top_label = "B0A"
  label_history = [B0A, B0A, B0A]  ← quay vòng (maxlen=3)
  stable_label = TRUE (vẫn)
  FOLLOW = ON (vẫn) ← KHÔNG TẮT!

Frame 100+: Bạn rơi gesture / khóa tay
  top_label = "D0X" (rest)
  label_history = []  ← cleared (line 722)
  FOLLOW = OFF ✓ ← Tắt được
```

**Kết luận:** Khi B0A stable, follow luôn ON dù không di chuyển. Chỉ tắt khi rơi gesture.

---

### Scenario 2: Bạn Di Chuyển

```
Frame 3: B0A stable
  FOLLOW = ON

Frame 4-50: Di chuyển tay
  top_label = "B0A" (vẫn)
  label_history = [B0A, B0A, B0A] (vẫn)
  stable_label = TRUE (vẫn) ← Key point!
  FOLLOW = ON (vẫn) ← TIẾP TỤC FOLLOW
```

**Tại sao?** Gesture vẫn được nhận diện là B0A, vẫn stable → không lý do để tắt.

---

## 🎯 Lỗi Trong Thiết Kế Hiện Tại

`follow_hold` được designed với logic này:

```
╔════════════════════════════════════════════╗
║ Follow ON/OFF dựa CHỈ vào gesture stability║
║                                             ║
║ Không có logic:                             ║
║ • "Nếu để im > N giây → tắt"               ║
║ • "Nếu không di chuyển → tắt"              ║
║ • "Nếu gesture stability giảm → tắt"       ║
╚════════════════════════════════════════════╝
```

---

## ✅ 3 Giải Pháp

### 1. Explicit ON/OFF (Khuyến Cáo)

**Config:**
```json
{
  "B0A": "mouse:follow_on",
  "B0B": "mouse:follow_off",
  ...
}
```

**Cách dùng:**
```
Bạn: Giơ 2 ngón lần 1 (B0A)
→ follow ON

Bạn: Di chuyển chuột tự do

Bạn: Giơ 2 ngón lần 2 (B0B)
→ follow OFF
```

**Code thay đổi:**
- Xóa `|follow_hold` từ B0A.
- Dùng `send_action()` với `toggle_mouse_follow()`.
- Có debounce 10-frame → reliable.

**Ưu:** Rõ ràng, an toàn, có debounce.  
**Nhược:** Tốn 1 gesture.

---

### 2. Auto-Off Timeout (Thông Minh)

**Ý tưởng:**
```python
FOLLOW_IDLE_TIME = 0.0

# Mỗi frame
if MOUSE_FOLLOW_ENABLED:
    if (now - FOLLOW_IDLE_TIME) > 3.0:  # 3 giây
        MOUSE_FOLLOW_ENABLED = False
        print("[follow] auto-off after 3s idle")
```

**Hành vi:**
```
Bạn: Giơ B0A (follow_hold)
→ follow ON, FOLLOW_IDLE_TIME = now

Bạn: Để im 3+ giây
→ follow OFF tự động

Bạn: Di chuyển lại
→ FOLLOW_IDLE_TIME reset, follow ON lại
```

**Ưu:** Tự động, không tốn gesture.  
**Nhược:** Timeout cứng (phải tune).

---

### 3. Movement Detection (Nâng Cao)

**Ý tưởng:**
```python
# Follow chỉ ON khi:
# 1. Gesture stable
# 2. Tay đang di chuyển (movement > 5px)
# 3. Nếu để im > 1s → auto-off

if MOUSE_FOLLOW_ENABLED:
    movement = distance(now_pos, prev_pos)
    if movement < 5:
        idle_frames += 1
        if idle_frames > 30:  # 1s
            FOLLOW = OFF
    else:
        idle_frames = 0  # reset
```

**Hành vi:**
```
Bạn: Giơ B0A
→ follow ON

Bạn: Di chuyển
→ follow ON (movement > 5px)

Bạn: Để im 1+ giây
→ follow OFF tự động

Bạn: Di chuyển lại
→ follow ON lại
```

**Ưu:** Thông minh, tự nhiên.  
**Nhược:** Phức tạp, có thể bị noise.

---

## 📊 So Sánh

| Aspect | 1: Explicit | 2: Timeout | 3: Movement |
|--------|------------|-----------|------------|
| Rõ ràng | ✓✓ | ✓ | ✓ |
| Tiện lợi | ✗ | ✓✓ | ✓✓ |
| Phức tạp | ✗ | ✓ | ✓✓ |
| Tuning | ✗ | ✗ (timeout) | ✓ (movement threshold) |
| Debounce | ✓ (có) | ✓ (có) | ~ |

---

## 🎯 Khuyến Cáo

**Ngay bây giờ:** Dùng **tuỳ chọn 1 (Explicit ON/OFF)**
- Đơn giản nhất.
- Có debounce sẵn.
- Bạn kiểm soát trực tiếp.

**Nếu muốn thông minh:** Dùng **tuỳ chọn 2 (Timeout)**
- Thêm 5-10 dòng code.
- Tự động tắt sau 3s idle.
- Không tốn gesture.

**Nếu muốn best:** Dùng **cả 2**
- Explicit OFF vẫn có.
- Timeout auto-off nếu quên.

---

## ❓ Câu Hỏi?

**Q: Tại sao `follow_hold` lại designed như vậy?**  
A: Vì `follow_hold` = "follow while gesture held", không phải "follow while moving". Nó dùng gesture stability làm trigger, không motion detection.

**Q: Có cách nào không cần thêm gesture?**  
A: Có, dùng auto-off timeout (tuỳ chọn 2).

**Q: Nếu tôi muốn follow tự động tắt khi không di chuyển?**  
A: Dùng tuỳ chọn 3 (movement detection).

