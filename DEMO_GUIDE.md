# 🎥 ST-GCN Demo - Hướng dẫn chạy

## Cách chạy demo webcam

### Cách 1: Chạy file batch (Windows - Đơn giản nhất) ⭐
```bash
run_demo.bat
```
**Hoặc:** Double-click file `run_demo.bat` trong thư mục dự án

---

### Cách 2: Chạy PowerShell script
```powershell
.\run_demo.ps1
```

---

### Cách 3: Chạy từ terminal (lệnh đầy đủ)

**Nếu chưa kích hoạt venv:**
```cmd
venv\Scripts\activate.bat
python tools/demo_webcam.py --model outputs/stgcn_best.pt --labels outputs/labels.json
```

**Nếu đã kích hoạt venv:**
```cmd
python tools/demo_webcam.py --model outputs/stgcn_best.pt --labels outputs/labels.json
```

---

## Tùy chọn nâng cao

Bạn có thể thêm các tham số để tùy chỉnh:

```cmd
python tools/demo_webcam.py ^
  --model outputs/stgcn_best.pt ^
  --labels outputs/labels.json ^
  --camera-width 1280 ^
  --camera-height 960 ^
  --camera-fps 30 ^
  --length 30 ^
  --min-confidence 0.35 ^
  --topk 3
```

### Các tham số quan trọng:

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `--model` | Đường dẫn model (.pt) | outputs/stgcn_best.pt |
| `--labels` | Đường dẫn labels.json | outputs/labels.json |
| `--camera-width` | Chiều rộng camera | 1280 |
| `--camera-height` | Chiều cao camera | 960 |
| `--camera-fps` | FPS của camera | 30 |
| `--length` | Số frame dự đoán | 30 |
| `--min-confidence` | Ngưỡng tin cậy tối thiểu | 0.35 |
| `--topk` | Hiển thị top-k kết quả | 3 |
| `--det-conf` | Confidence detection tay | 0.5 |
| `--track-conf` | Confidence tracking tay | 0.5 |

---

## Phím tắt khi chạy demo

| Phím | Chức năng |
|------|----------|
| **Q** | Thoát chương trình |
| **C** | Reset sequence (xóa buffer hiện tại) |

---

## Ghi chú

- **Camera được khuyên:** 1280x960 hoặc 1920x1080 để hiệu suất tốt
- **Giảm FPS** nếu máy chậm: `--camera-fps 15`
- **Tăng confidence threshold** để chỉ nhận diện khi chắc chắn: `--min-confidence 0.5`

---

## Khắc phục sự cố

### Nếu không thấy camera:
```cmd
--camera-id 0  # Thử camera 0
--camera-id 1  # Hoặc camera 1 nếu có nhiều camera
```

### Nếu chạy chậm:
- Giảm `--camera-width` và `--camera-height`
- Giảm `--camera-fps`
- Đảm bảo CUDA được cài (GPU sẽ nhanh hơn CPU)

---

✅ Hoàn tất! Giờ bạn có thể test model với camera.
