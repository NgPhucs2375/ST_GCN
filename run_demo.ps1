# Script chạy demo webcam ST-GCN (PowerShell)
# Sử dụng: .\run_demo.ps1

Write-Host "===================================" -ForegroundColor Green
Write-Host "ST-GCN Hand Gesture Demo" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""

# Kích hoạt virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Launching demo..." -ForegroundColor Yellow
Write-Host ""

# Chạy demo
python tools/demo_webcam.py --model outputs/stgcn_best.pt --labels outputs/labels.json --camera-width 1280 --camera-height 960 --flip-camera

Write-Host ""
Write-Host "Demo closed." -ForegroundColor Yellow
