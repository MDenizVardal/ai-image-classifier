@echo off
title AI Image Classifier
color 0A

cd /d "%~dp0"

echo ============================================================
echo            AI IMAGE CLASSIFIER (Deep Learning)
echo ============================================================
echo.
echo Loading AI model... (this may take a moment)
echo.

python classifier_ai.py --interactive

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Something went wrong!
    echo.
    echo Run Install_AI_Dependencies.bat first!
    echo ============================================================
)

echo.
pause

