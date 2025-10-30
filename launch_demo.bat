@echo off
REM Traffic Sign Detection Demo Launcher
REM This script launches the Gradio web interface

echo ========================================
echo Traffic Sign Detection Demo
echo ELEC5308 Project 1 - Group 26
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Starting demo application...
echo.
echo Once started, open your browser and go to:
echo http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the demo
echo ========================================
echo.

python traffic_sign_demo.py

pause
