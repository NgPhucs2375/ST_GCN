# 📋 Bảng Class Cử Chỉ Tay

## Danh sách tất cả các cử chỉ

| ID | Code | Tên Tiếng Anh | Mô Tả Tiếng Việt | 
|----|----|--------------|------------------|
| 0 | B0A | Pointing with one finger | ☝️ Chỉ tay với 1 ngón |
| 1 | B0B | Pointing with two fingers | ✌️ Chỉ tay với 2 ngón |
| 2 | D0X | Non-gesture | ❌ Không có cử chỉ (tay thường) |
| 3 | G01 | Click with one finger | 👆 Nhấp với 1 ngón |
| 4 | G02 | Click with two fingers | 👆 Nhấp với 2 ngón |
| 5 | G03 | Throw up | 👋 Ném lên |
| 6 | G04 | Throw down | 👋 Ném xuống |
| 7 | G05 | Throw left | 👋 Ném sang trái |
| 8 | G06 | Throw right | 👋 Ném sang phải |
| 9 | G07 | Open twice | 👌 Mở 2 lần |
| 10 | G08 | Double click with one finger | 👆 Nhấp đôi 1 ngón |
| 11 | G09 | Double click with two fingers | 👆 Nhấp đôi 2 ngón |
| 12 | G10 | Zoom in | 🔍 Phóng to |
| 13 | G11 | Zoom out | 🔍 Thu nhỏ |

---

## Cách đọc kết quả khi chạy demo

Khi chạy demo, bạn sẽ thấy output như:

```
B0A: 0.92
D0X: 0.05
B0B: 0.03
```

**Nghĩa là:**
- **B0A (0.92)** ← Model 92% chắc đó là "Chỉ tay với 1 ngón"
- **D0X (0.05)** ← 5% khả năng là "Không có cử chỉ"
- **B0B (0.03)** ← 3% khả năng là "Chỉ tay với 2 ngón"

---

## Hướng dẫn test từng cử chỉ

### 1️⃣ **D0X - Non-gesture (Tay thường)**
- Đặt tay thường vào camera
- Model sẽ nhận diện tay nhưng **không phải là cử chỉ**

### 2️⃣ **B0A - Pointing with one finger (Chỉ tay 1 ngón)**
- Giơ 1 ngón (thường là ngón trỏ) lên
- Model sẽ nhận diện là "Chỉ tay 1 ngón"

### 3️⃣ **B0B - Pointing with two fingers (Chỉ tay 2 ngón)**
- Giơ 2 ngón lên (ngón trỏ + ngón giữa)
- Model sẽ nhận diện là "Chỉ tay 2 ngón"

### 4️⃣ **G01/G02 - Click (Nhấp)**
- G01: Nhấp nhanh 1 lần với 1 ngón
- G02: Nhấp nhanh 1 lần với 2 ngón

### 5️⃣ **G03-G06 - Throw (Ném)**
- G03: Cử chỉ ném **lên**
- G04: Cử chỉ ném **xuống**
- G05: Cử chỉ ném **sang trái**
- G06: Cử chỉ ném **sang phải**

### 6️⃣ **G07 - Open twice (Mở 2 lần)**
- Mở tay nhanh 2 lần liên tiếp

### 7️⃣ **G08/G09 - Double click (Nhấp đôi)**
- G08: Nhấp đôi với 1 ngón
- G09: Nhấp đôi với 2 ngón

### 8️⃣ **G10/G11 - Zoom (Phóng thu)**
- G10: Cử chỉ **phóng to** (2 tay hoặc 2 ngón mở ra)
- G11: Cử chỉ **thu nhỏ** (2 tay hoặc 2 ngón đóng lại)

---

## 📁 Vị trí các file

| File | Vị trí | Mô Tả |
|------|--------|-------|
| **labels.json** | `outputs/labels.json` | Map code → index |
| **class_details.txt** | `data/annotations/class_details.txt` | Chi tiết các class |
| **classIdx.txt** | `data/annotations/classIdx.txt` | Index của các class |

---

## 💡 Tips khi test

1. **Sáng tốt**: Đảm bảo ánh sáng phù hợp để camera nhận diện tay rõ
2. **Tay vào frame**: Đặt tay trong khung hình camera
3. **Chờ 30 frame**: Model chờ 30 frame (khoảng 1 giây) để dự đoán
4. **Cử chỉ rõ ràng**: Làm cử chỉ rõ ràng, không quá nhanh hoặc quá chậm
5. **Xem confidence**: Nếu confidence < 0.35, model sẽ hiển thị "Không chắc"

---

## 🔍 Để xem code mapping

File: `data/annotations/classIdx.txt`
```
B0A 0
B0B 1
D0X 2
G01 3
G02 4
...
```

Hoặc mở file `outputs/labels.json` để xem mapping đầy đủ.

---

✅ Bây giờ bạn có thể test model với camera!
