# Hướng Dẫn Calibration Mode — Xây Dựng Gesture Templates Cho KNN

## 📋 Tổng Quan

**Calibration Mode** cho phép bạn:
1. Ghi lại landmarks (tọa độ tay) cho mỗi cử chỉ (~20 lần)
2. Tính toán mean (trung bình) & std (độ lệch chuẩn) của mỗi cử chỉ
3. Lưu vào `gesture_templates.json`
4. Dùng KNN để match gesture trong real-time (thay thế ST-GCN nếu muốn)

**Ưu điểm so với ST-GCN:**
- ✓ Lightweight (không cần GPU)
- ✓ Cá nhân hóa (mỗi người calibrate riêng)
- ✓ Nhanh: chỉ tính Euclidean distance
- ✗ Cần calibration trước (lại chiếm thời gian)

---

## 🚀 Bước 1: Chạy Calibration Tool

### Lệnh chạy (Windows cmd)

```bash
cd "D:\code\Nam3_HK2_25-26\Deep learning\ST_GCN"
.venv\Scripts\activate
python tools/calibrate_gestures.py
```

### Giao diện Calibration

```
============================================================
🎮 ST-GCN GESTURE CALIBRATION TOOL
============================================================
📋 Gesture: G01 — Click 1 ngón
============================================================
🎥 Instructions:
  1. Position your hand in front of camera
  2. Press SPACE to START recording
  3. Hold the gesture for ~20 frames
  4. Press SPACE to STOP recording
  5. System calculates mean/std and saves template
  6. Press 'N' for next gesture or 'Q' to quit

Giao diện camera:
┌─────────────────────────────────┐
│ 🔴 RECORDING | Frames: 15/20    │
│ Gesture: G01 - Click 1 ngon     │
│                                 │
│   [Hand landmarks overlay]      │
│                                 │
│ SPACE=Toggle  Q=Quit  N=Next    │
└─────────────────────────────────┘
```

---

## 🎬 Bước 2: Calibrate Từng Gesture

### Quy Trình Cho Mỗi Gesture

```
1️⃣  Xem tên gesture (ví dụ "G01 — Click 1 ngón")

2️⃣  Đặt tay vào góc nhìn camera (vị trí tự nhiên)

3️⃣  Nhấn SPACE để bắt đầu record
    → Camera bắt đầu lưu landmarks

4️⃣  Giữ gesture trong ~20 frame (~1 giây)
    → Overlay hiện "🔴 RECORDING | Frames: 20/20"

5️⃣  Nhấn SPACE để kết thúc
    → Script tính mean/std
    → In: "✅ Template saved: 20 samples, mean shape (21, 3)"

6️⃣  Chọn tiếp theo:
    - SPACE → tiếp tục gesture hiện tại
    - N → gesture tiếp theo
    - Q → thoát
```

### Mẹo Calibration Tốt

- **Vị trí tay:** Đặt tay ở khoảng 30-60cm từ camera, trong góc nhìn.
- **Độ sáng:** Đủ sáng để camera nhìn rõ tay.
- **Ổn định:** Đừng cử động quá nhiều (để tay yên tĩnh hoặc chuyển động từ từ).
- **Samples:** Cố gắng lấy ít nhất 15-20 frame cho mỗi gesture (càng nhiều càng tốt, nhưng tối thiểu 5).
- **Lặp lại:** Nếu không hài lòng, nhấn SPACE lại để re-record gesture đó.

---

## 📁 Bước 3: Output — gesture_templates.json

Sau calibration, file `data/gesture_templates.json` sẽ chứa:

```json
{
  "G01": {
    "gesture_id": "G01",
    "gesture_name": "Click 1 ngón",
    "num_samples": 20,
    "mean": [
      [0.0, 0.0, 0.0],        // landmark 0 (wrist) - normalized
      [0.15, 0.02, -0.05],    // landmark 1
      ...
      [-0.08, -0.12, 0.03]    // landmark 20 (pinky tip)
    ],
    "std": [
      [0.01, 0.02, 0.01],
      ...
    ]
  },
  "G02": { ... },
  ...
}
```

Mỗi gesture có:
- **mean**: Trung bình vị trí tay qua 20 samples
- **std**: Độ lệch chuẩn (thể hiện biến thiên)

---

## 🎮 Bước 4: Dùng KNN Matcher Trong Demo

### Chạy Demo Với KNN Mode

```bash
python tools/demo_webcam.py \
  --config Gan_nut/gesture_config.json \
  --use-knn \
  --knn-threshold 5.0 \
  --knn-k 3
```

### Tham số KNN

- `--use-knn`: Bật KNN mode (dùng gesture_templates.json thay ST-GCN)
- `--knn-threshold`: Khoảng cách tối đa để coi là match (mặc định 5.0)
  - Cao = permissive (dễ match, nhưng ít chính xác)
  - Thấp = strict (ít false positive, nhưng ít nhạy)
- `--knn-k`: Số neighbors cho voting (mặc định 3)

### Hành Vi KNN Runtime

```
Inference loop:
  ↓
Detect landmarks (MediaPipe)
  ↓
Normalize landmarks
  ↓
Compute distance to ALL templates
  ↓
Get closest gesture (hoặc vote k-nearest)
  ↓
Output: gesture_id, confidence
  ↓
Execute gesture mapping (ví dụ G01 → "mouse:left")
```

---

## 🔄 So Sánh: ST-GCN vs KNN

| Aspect | ST-GCN | KNN |
|--------|--------|-----|
| **Setup** | Sẵn model | Cần calibration |
| **Tốc độ** | Chậm (GPU nhanh) | Cực nhanh (CPU) |
| **Độ chính xác** | Cao (học từ dataset lớn) | Tùy thuộc calibration |
| **Cá nhân hóa** | ✗ Chung | ✓ Per-user |
| **Tài nguyên** | GPU/RAM cao | GPU/RAM thấp |
| **Demo tốc độ cao** | ~ 15-30 FPS | ~ 60+ FPS |

---

## 📊 Troubleshooting

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-----------|---------|
| Không detect tay | Tay ngoài góc nhìn, quá mờ | Đặt tay vào góc nhìn, tăng sáng |
| Landmarks không xuất hiện | MediaPipe chưa load | Chờ load model (~5s) |
| KNN match sai | Template chất lượng kém | Calibrate lại với vị trí ổn định hơn |
| Confidence thấp | Gesture hiện tại khác template | Calibrate lại cùng vị trí/góc |

---

## 💾 Quản Lý Templates

### Sao lưu template

```bash
# Copy template hiện tại
copy data\gesture_templates.json data\gesture_templates_backup.json
```

### Reset templates

```bash
# Xóa file template để calibrate lại
del data\gesture_templates.json
```

### So sánh templates (khác người dùng)

Nếu muốn lưu templates của nhiều người:
```
data/
  gesture_templates_user1.json
  gesture_templates_user2.json
  gesture_templates_user3.json
```

Rồi chạy demo với flag `--template-file`:
```bash
python tools/demo_webcam.py --template-file data/gesture_templates_user1.json --use-knn
```

---

## 🚀 Full Workflow — Từ Đầu Đến Cuối

```bash
# 1. Calibrate
python tools/calibrate_gestures.py
# → Lưu templates vào data/gesture_templates.json

# 2. Sửa config nếu cần (gesture_config.json)
notepad Gan_nut\gesture_config.json

# 3. Chạy demo với KNN
python tools/demo_webcam.py --use-knn --knn-threshold 5.0

# 4. Test gesture
# (Giơ tay làm cử chỉ, xem có nhận không)

# 5. Điều chỉnh
# - Nếu quá nhiều false positive → tăng --knn-threshold
# - Nếu quá ít nhạy → giảm --knn-threshold
# - Nếu vẫn sai → calibrate lại gesture đó
```

---

## 📌 Ghi Chú

- **Landmarks normalized:** Tất cả landmarks đã được normalize (tương đối với wrist) trước khi save template.
- **Robustness:** KNN cho phép variation (dùng std); nếu bạn giơ tay ở góc khác, vẫn có khả năng match.
- **Performance:** KNN rất nhanh (~1ms/match), phù hợp cho real-time demo.

---

## ❓ FAQ

**Q: Có thể mix ST-GCN + KNN không?**  
A: Hiện tại chỉ hỗ trợ chế độ này hoặc chế độ kia. Tôi có thể thêm "ensemble" mode sau.

**Q: Calibration mất bao lâu?**  
A: ~10 phút cho 14 gestures (nếu mỗi gesture 20 frame).

**Q: Có thể calibrate riêng từng gesture không?**  
A: Có, bạn có thể nhấn N để skip, rồi re-run tool để calibrate lại gesture cụ thể (nó sẽ overwrite template cũ).

**Q: Template có thể share được không?**  
A: Có, nhưng kém chính xác nếu chia sẻ giữa người dùng khác (vì tay/cử chỉ có khác biệt).
