@echo off
REM Script chạy demo webcam ST-GCN
REM Sử dụng: chỉ cần double-click hoặc gõ: run_demo.bat

cd /d "%~dp0"
echo.
echo ===================================
echo ST-GCN Hand Gesture Demo
echo ===================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Launching demo...
echo.
python tools/demo_webcam.py --model outputs/stgcn_best.pt --labels outputs/labels.json --camera-width 1280 --camera-height 960 --flip-camera

pause
