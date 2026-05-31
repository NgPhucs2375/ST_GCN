# 08 — Mouse-Follow Toggle Issue & Root Cause Analysis

## Vấn đề Bạn Gặp Phải

> "Tôi để im cái thế đó trong vài giây thì tắt được, nhưng nếu di chuyển thì nó tiếp tục mơ (không tắt được)."

**Root cause:** Gesture `B0A` (**chỉ 2 ngón**) vẫn được nhận diện **liên tục** khi bạn giữ ngón tay, vì vậy:
- **Khi để im** → gesture stability giảm dần → không còn stable → follow tắt ✓
- **Khi di chuyển** → gesture vẫn stable → follow vẫn ON ✓ (đúng hành vi!)

---

## Sơ đồ Luồng `follow_hold` Hiện Tại

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Inference Loop (30 FPS)                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
                   Nhận diện gesture từ camera
                              ↓
            ┌───────────────────────────────────────┐
            │ if B0A detected & confident enough    │
            │   → B0A vào label_history             │
            └───────────────────────────────────────┘
                              ↓
            ┌───────────────────────────────────────┐
            │ Check stability:                      │
            │ stable_label = (last 3 labels ==      │
            │                "B0A")                 │
            │                                       │
            │ if stable_label = TRUE:               │
            │   → MOUSE_FOLLOW_ENABLED = True ✓    │
            │   → di chuyển chuột theo ngón tay      │
            │                                       │
            │ if stable_label = FALSE:              │
            │   → MOUSE_FOLLOW_ENABLED = False ✗   │
            └───────────────────────────────────────┘
                              ↓
            ┌───────────────────────────────────────┐
            │ if top_label != "B0A" (rơi/tay                │
            │    hoặc khác gesture):                │
            │   → label_history.clear()             │
            │   → MOUSE_FOLLOW_ENABLED = False      │
            │   → follow OFF ✓                      │
            └───────────────────────────────────────┘
```

---

## Tại Sao Nó Không Tắt Khi Di Chuyển?

### Lý Do 1: Gesture Vẫn Stable (Đúng Hành Vi)
**Code:**
```python
# Line 668
lh = list(label_history)
tail = lh[-args.stable_count:] if args.stable_count > 0 else lh
stable_label = (len(tail) == args.stable_count and all(l == top_label for l in tail))

# Line 676
if follow_hold_requested:
    if stable_label or early_trigger or is_instant:
        MOUSE_FOLLOW_ENABLED = True  # ← Follow ON khi stable
    else:
        MOUSE_FOLLOW_ENABLED = False  # ← Follow OFF khi NOT stable
```

**Kịch bản:**
```
Khung 1: Bạn giơ 2 ngón
  → top_label = "B0A", confidence = 0.92
  → label_history = ["B0A"]
  → stable_label = False (chỉ có 1 entry, cần 3)
  → MOUSE_FOLLOW_ENABLED = False

Khung 2: Bạn vẫn giơ 2 ngón (di chuyển)
  → top_label = "B0A", confidence = 0.91
  → label_history = ["B0A", "B0A"]
  → stable_label = False (chỉ có 2 entry, cần 3)
  → MOUSE_FOLLOW_ENABLED = False

Khung 3: Bạn vẫn giơ 2 ngón (vẫn di chuyển)
  → top_label = "B0A", confidence = 0.90
  → label_history = ["B0A", "B0A", "B0A"]
  → stable_label = TRUE ✓ (có 3 entry liên tiếp)
  → MOUSE_FOLLOW_ENABLED = True  ← Follow ON!
  → Mouse bắt đầu di chuyển
  → Khung 4, 5, 6, ... vẫn nhận B0A
  → stable_label = TRUE (vẫn)
  → MOUSE_FOLLOW_ENABLED = True (vẫn)  ← Follow tiếp tục!

Khi bạn để im:
  → Mưu ngón không di chuyển
  → Landmark sử đổi ít
  → Nhưng gesture vẫn B0A
  → stable_label vẫn TRUE
  → MOUSE_FOLLOW_ENABLED vẫn TRUE  ← Follow vẫn ON!

Khi bạn bỏ gesture (khóa tay/rơi):
  → top_label ≠ "B0A"
  → label_history.clear()
  → MOUSE_FOLLOW_ENABLED = False  ← Follow tắt ✓
```

**Tóm lại:** `follow_hold` có nghĩa là "**follow khi gesture stable**", không phải "**follow khi tay di chuyển**".

---

## Lý Do 2: Không Có "Rơi Gesture" Detection

Hiện tại, code chỉ tắt follow khi:
1. **Gesture khác được nhận** → clear label_history → follow OFF
2. **Tay rơi hoàn toàn** (no landmarks) → missing_count tăng → frame_buffer clear → follow OFF
3. **Gesture không confident** (< min_confidence) → không enter "gửi phím" block

**Nhưng nếu:**
- Bạn giơ 2 ngón (`B0A`) ổn định → follow ON
- Bạn để im mà vẫn giữ gesture → gesture vẫn stable → follow vẫn ON ✓ (không lỗi)

---

## Giải Pháp: 3 Tuỳ Chọn

### Tuỳ Chọn 1: Dùng "Explicit On/Off" Gesture (Recommended)
**Ý tưởng:** Dùng 2 gesture riêng để bật/tắt follow (không dùng `follow_hold`).

**Config:**
```json
{
  "B0A": "mouse:follow_on",
  "B0B": "mouse:follow_off"
}
```

**Hành vi:**
```
Bạn giơ 2 ngón (B0A) → follow ON
Bạn giơ 2 ngón (B0B) → follow OFF
```

**Ưu điểm:**
- ✓ Rõ ràng: ON/OFF tường minh.
- ✓ Debounce: toggle_mouse_follow() có 10-frame cooldown → không spam.
- ✓ An toàn: bạn chủ động tắt.

**Nhược điểm:**
- ✗ Tốn 1 gesture (B0B).
- ✗ Phải nhớ để tắt (dễ quên).

---

### Tuỳ Chọn 2: Dùng "Rơi Gesture" Timeout
**Ý tưởng:** Nếu gesture stable > N giây mà không có action → tự động tắt follow.

**Cách thêm vào code:**
```python
# Global
FOLLOW_LAST_ACTION_TIME = time.time()
FOLLOW_AUTO_OFF_TIMEOUT = 3.0  # 3 giây

# In inference loop
if MOUSE_FOLLOW_ENABLED:
    now = time.time()
    # Tắt follow nếu không có action/movement trong 3s
    if now - FOLLOW_LAST_ACTION_TIME > FOLLOW_AUTO_OFF_TIMEOUT:
        MOUSE_FOLLOW_ENABLED = False
        print("[follow_timeout] OFF after 3 seconds of inactivity")
```

**Hành vi:**
```
Bạn giơ 2 ngón → follow ON
Bạn để im (không di chuyển) → sau 3s → follow OFF tự động
Bạn di chuyển → FOLLOW_LAST_ACTION_TIME reset → follow tiếp tục
```

**Ưu điểm:**
- ✓ Tự động: không phải tắt bằng tay.
- ✓ Thông minh: theo dõi hoạt động.
- ✓ Giữ `follow_hold` gọn gàng.

**Nhược điểm:**
- ✗ Timeout cứng: phải tune để vừa phải.
- ✗ Có thể bất ngờ (tắt giữa chừng).

---

### Tuỳ Chọn 3: Nâng Cao — "Movement Detection"
**Ý tưởng:** Follow chỉ ON khi **gesture vừa ổn định VÀ tay đang di chuyển**.

**Cách thêm vào code:**
```python
# Track fingertip position for movement detection
FOLLOW_PREV_POS = None
FOLLOW_MIN_MOVEMENT = 5  # pixels

# In mouse follow section
if MOUSE_FOLLOW_ENABLED and len(MOTION_HISTORY) >= 2:
    # Check if fingertip is actually moving
    pos_now = MOTION_HISTORY[-1]
    pos_prev = MOTION_HISTORY[-2]
    movement = np.linalg.norm(np.array(pos_now) - np.array(pos_prev))
    
    if movement < FOLLOW_MIN_MOVEMENT:
        # Tay không di chuyển → auto-off sau timeout
        if not hasattr(FOLLOW_PREV_POS, 'idle_frames'):
            FOLLOW_PREV_POS = {'idle_frames': 0}
        else:
            FOLLOW_PREV_POS['idle_frames'] += 1
        
        if FOLLOW_PREV_POS['idle_frames'] > 30:  # 1 giây @ 30fps
            MOUSE_FOLLOW_ENABLED = False
    else:
        # Tay đang di chuyển → reset idle counter
        FOLLOW_PREV_POS = {'idle_frames': 0}
```

**Hành vi:**
```
Bạn giơ 2 ngón → follow ON
Bạn di chuyển → follow tiếp tục (movement > 5px)
Bạn để im > 1 giây → follow OFF tự động
Bạn di chuyển lại → follow ON lại
```

**Ưu điểm:**
- ✓ Thông minh: chỉ follow khi thực sự di chuyển.
- ✓ Gọn gàng: không cần gesture riêng.
- ✓ Thuyết phục: hành vi "tự nhiên".

**Nhược điểm:**
- ✗ Phức tạp hơn: cần thêm tracking.
- ✗ Có thể bị noise: tay rung nhỏ coi là di chuyển.

---

## Khuyến Cáo

| Kịch bản | Tuỳ chọn | Lý do |
|---------|---------|------|
| Muốn rõ ràng, an toàn | 1 (Explicit ON/OFF) | Kiểm soát trực tiếp |
| Muốn tiện, tự động | 2 (Timeout) | Dễ thêm, không tốn gesture |
| Muốn "thông minh" nhất | 3 (Movement Detection) | Hành vi tự nhiên |
| Combo tốt nhất | 2 + 3 | Vừa timeout + movement |

---

## Tóm Tắt Nguyên Nhân

| Trạng thái | Lý do | Kết quả |
|-----------|------|--------|
| Giơ gesture B0A | Gesture stable → follow ON | ✓ Correct |
| Để im tay | Gesture vẫn stable | ✓ Follow vẫn ON (expected) |
| Di chuyển tay | Gesture stable + di chuyển | ✓ Follow vẫn ON (expected) |
| Rơi gesture | Không còn stable → follow OFF | ✓ Correct |

**Kết luận:** Hành vi hiện tại là **correct per spec**. Vấn đề là "follow_hold" không có "auto-off" logic. Bạn cần thêm một trong 3 tuỳ chọn trên.

---

## Đề Xuất Cải Thiện Ngay

Tôi khuyến nghị **tuỳ chọn 1** (Explicit ON/OFF) vì:
1. ✓ Rõ ràng: không nhập nhằng.
2. ✓ Debounce: có 10-frame cooldown → reliable.
3. ✓ Đơn giản: không thêm code phức tạp.

**Config mới:**
```json
{
  "B0A": "mouse:follow_on",
  "B0B": "mouse:follow_off",
  ...
}
```

**Cách dùng:**
- Giơ 2 ngón lần 1 (B0A) → follow ON
- Di chuyển chuột
- Giơ 2 ngón lần 2 (B0B) → follow OFF

Hoặc nếu bạn muốn **automatic + timeout**, tôi có thể thêm tuỳ chọn 2 vào code ngay.

**Bạn chọn tuỳ chọn nào?**
