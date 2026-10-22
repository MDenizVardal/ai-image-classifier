@echo off
title Build AI Database
color 0E

cd /d "%~dp0"

echo ============================================================
echo              BUILD AI DATABASE (Deep Learning)
echo ============================================================
echo.
echo This will extract AI features from all reference images.
echo This may take a few minutes...
echo.

python classifier_ai.py --build

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Run Install_AI_Dependencies.bat first!
    echo ============================================================
)

echo.
pause

