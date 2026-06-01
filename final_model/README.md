# Final Model V2 - Hand Gesture ST-GCN

## Muc dich
Folder nay luu model cuoi cung de demo va suy luan nhanh.

## File quan trong
- final_model_v2/kaggle/working/final_model_v2/stgcn_best.pt
- final_model_v2/kaggle/working/final_model_v2/labels.json
- final_model_v2/kaggle/working/final_model_v2/confusion_matrix.pt
- final_model_v2.zip

## Danh sach cu chi (14 lop)
| ID | Code | Ten hien thi | Mo ta ngan |
|---|---|---|---|
| 0 | B0A | Chi 1 ngon | Pointing with one finger |
| 1 | B0B | Chi 2 ngon | Pointing with two fingers |
| 2 | D0X | Khong cu chi | Non-gesture |
| 3 | G01 | Click 1 ngon | Click with one finger |
| 4 | G02 | Click 2 ngon | Click with two fingers |
| 5 | G03 | Hat len | Throw up |
| 6 | G04 | Hat xuong | Throw down |
| 7 | G05 | Hat trai | Throw left |
| 8 | G06 | Hat phai | Throw right |
| 9 | G07 | Mo 2 lan | Open twice |
| 10 | G08 | Double click 1 ngon | Double click with one finger |
| 11 | G09 | Double click 2 ngon | Double click with two fingers |
| 12 | G10 | Phong to | Zoom in |
| 13 | G11 | Thu nho | Zoom out |

## Chay demo webcam
Chay tu thu muc goc project:

```bash
python tools/demo_webcam.py --model "final_model/final_model_v2/kaggle/working/final_model_v2/stgcn_best.pt" --labels "final_model/final_model_v2/kaggle/working/final_model_v2/labels.json" --device auto
```

### Cau hinh khuyen nghi cho camera laptop yeu (nhieu/rung/anh sang kem)

```bash
python tools/demo_webcam.py --model "final_model/final_model_v2/kaggle/working/final_model_v2/stgcn_best.pt" --labels "final_model/final_model_v2/kaggle/working/final_model_v2/labels.json" --device auto --length 36 --det-conf 0.60 --track-conf 0.60 --ema-alpha 0.80 --landmark-ema-alpha 0.75 --max-hand-jump 0.10 --missing-reset-frames 8 --min-confidence 0.40 --camera-width 640 --camera-height 480 --camera-fps 24
```

Neu camera chinh khong mo duoc, thu camera khac:

```bash
python tools/demo_webcam.py --model "final_model/final_model_v2/kaggle/working/final_model_v2/stgcn_best.pt" --labels "final_model/final_model_v2/kaggle/working/final_model_v2/labels.json" --camera-id 1 --device auto
```

## Luu y khi test de nhan dung hon
- Bat dau bang trang thai D0X (khong cu chi) trong khoang 0.5s.
- Dat tay o giua khung hinh, tranh ra khoi frame.
- Cac lop dong (G03-G09) can lam ro huong va du bien do.
- Giu dong tac o diem cuoi them mot nhip ngan de model on dinh.
- Anh sang deu, tranh nhoe tay do di chuyen qua nhanh.
- Neu doi cu chi lien tuc ma nhan sai, bam phim c de clear buffer roi lam lai.

## Nhan xet thuc te
Model nay thuong nhan ra tot cac lop sau:
- D0X (Khong cu chi)
- G11 (Thu nho)
- G10 (Phong to)
- G06 (Hat phai)
- G05 (Hat trai)

Cac lop click/double-click thuong kho hon, can thao tac ro nhip va dung bien do.
