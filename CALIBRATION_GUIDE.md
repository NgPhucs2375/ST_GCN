# 🎓 Gesture Calibration & Training Mode - Quick Start Guide

## What is Gesture Calibration?

**Calibration Mode** cho phép bạn **tạo gesture templates** riêng (không cần retrain model).  
Sau đó hệ thống dùng **KNN matching** để nhận diện gestures nhanh chóng mà không cần STGCN.

---

## 🚀 Cách dùng

### **Step 1: Setup venv & dependencies**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install mediapipe opencv-contrib-python
```

### **Step 2: Run Calibration Mode**
```bash
python tools/demo_webcam.py --calibration-mode
```

**Giao diện sẽ hiển thị:**
- Gesture hiện tại (ví dụ: G01 - Click 1 ngon)
- Số frame đã ghi / target (ví dụ: 15/20)
- Progress bar
- Hướng dẫn phím (SPACE, S, Q)

### **Step 3: Record gestures**
Lần lượt:
1. **SPACE** - Bắt đầu ghi
2. Thực hiện cử chỉ ~20 lần (tự do, không cần đều)
3. **SPACE** - Kết thúc & chuyển gesture tiếp theo
4. Lặp lại cho tất cả 14 gestures

**Phím tắt:**
- **SPACE**: Start/Stop recording
- **S**: Skip gesture (bỏ qua, chuyển tiếp)
- **Q**: Quit & save

### **Step 4: Xem templates đã save**
```bash
# Các templates sẽ được lưu tại:
Gan_nut/gesture_templates.json

# Format:
{
  "G01": {
    "count": 20,
    "mean_landmarks": [...],  # Shape: [21, 3]
    "std_landmarks": [...],   # Shape: [21, 3]
    "timestamp": 1234567890.5
  },
  "G02": { ... },
  ...
}
```

---

## 🔥 Use KNN Matching (instead of STGCN)

Sau khi calibration xong, bạn có thể chạy demo với **KNN matching** (nhanh hơn STGCN):

```bash
python tools/demo_webcam.py --knn-mode
```

**Ưu điểm KNN:**
- ⚡ Nhanh hơn STGCN (chỉ cần compute distance → vector comparison)
- 📱 Có thể chạy CPU-only, không cần GPU
- 🎯 Dễ debug (xem được top-3 nearest gestures)
- ✅ Personalized: templates từ riêng bạn

**Nhược điểm:**
- 📊 Cần calibration trước
- 🎨 Không robust như deep learning (nếu lighting/angle thay đổi)

---

## 📊 Kết hợp cả hai: STGCN + KNN Matching

**Idea**: Dùng STGCN để có top-3 candidates, rồi verify bằng KNN:

```python
# Pseudocode
top3_stgcn = model.predict(frames)  # STGCN top-3
knn_match = knn_matcher.predict(current_landmarks)  # KNN top-1

if knn_match and top3_stgcn includes knn_match:
    final_gesture = knn_match  # Confirmed by both
else:
    final_gesture = top3_stgcn[0]  # Fall back to STGCN
```

---

## ⚙️ Parameters

### Calibration Mode
```bash
python tools/demo_webcam.py --calibration-mode \
  --camera-id 0 \           # Camera device ID
  --camera-width 640 \      # Resolution width
  --camera-height 480 \     # Resolution height
  --det-conf 0.6 \          # MediaPipe detection confidence
  --track-conf 0.35 \       # MediaPipe tracking confidence
  --landmark-filter ema     # ema or oneeuro
```

### KNN Mode
```bash
python tools/demo_webcam.py --knn-mode \
  --config Gan_nut/gesture_config.json  # Gesture → action mapping
```

---

## 🐛 Troubleshooting

### Q: Camera không detect tay?
**A:** 
- Kiểm tra lighting (đủ sáng)
- Thử `--det-conf 0.4` (threshold thấp hơn)
- Kiểm tra quyền camera

### Q: Calibration không đủ frames?
**A:** 
- Hãy thực hiện gesture chậm hơn & rõ ràng hơn
- Phải có ít nhất 10 frames để lưu (default target 20)

### Q: KNN match lúc tốt lúc xấu?
**A:** 
- Điều chỉnh `threshold` trong KNN (`--knn-threshold 2.0`)
- Calibrate lại với nhiều angles/lighting khác nhau
- Dùng combination: STGCN + KNN để verify

---

## 📈 Next Steps

1. **Implement multi-hand support** (num_hands=2)
2. **Add gesture analytics** (confusion matrix, accuracy report)
3. **Gesture combo detection** (G01 + G02 together → special action)
4. **Web UI** để visualize templates + match confidence

---

**Made with ❤️ for hand gesture recognition**
