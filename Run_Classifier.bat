@echo off
title Image Classifier
color 0A

cd /d "%~dp0"

echo ============================================================
echo                   IMAGE CLASSIFIER
echo ============================================================
echo.
echo Loading classifier...
echo.

python classifier.py --interactive

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Something went wrong!
    echo.
    echo Possible fixes:
    echo   1. Make sure Python is installed
    echo   2. Run: pip install Pillow imagehash numpy opencv-python
    echo   3. Run: python build_database.py
    echo ============================================================
)

echo.
pause

