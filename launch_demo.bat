@echo off
echo ================================================
echo Traffic Sign Recognition Demo - Quick Launch
echo ELEC5308 Project 2 - Group 26
echo ================================================
echo.

echo Checking dependencies...
python -c "import gradio, torch, ultralytics" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements_demo.txt
    echo.
)

echo Checking model files...
if not exist "yolov8\runs\detect\train3\weights\best.pt" (
    echo ERROR: YOLOv8 model not found at yolov8\runs\detect\train3\weights\best.pt
    pause
    exit /b 1
)
if not exist "resnet\parameters_Resnet50.cpt" (
    echo ERROR: ResNet50 model not found at resnet\parameters_Resnet50.cpt
    pause
    exit /b 1
)

echo.
echo Starting web demo...
echo The demo will open in your browser at http://127.0.0.1:7861
echo.
echo Press Ctrl+C to stop the server
echo.

python traffic_sign_demo_final.py

pause
