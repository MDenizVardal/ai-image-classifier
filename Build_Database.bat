@echo off
title Build Reference Database
color 0E

cd /d "%~dp0"

echo ============================================================
echo              BUILD REFERENCE DATABASE
echo ============================================================
echo.
echo This will scan your reference images and build the database.
echo.

python build_database.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Something went wrong!
    echo.
    echo Possible fixes:
    echo   1. Make sure Python is installed
    echo   2. Run: pip install Pillow imagehash numpy opencv-python
    echo   3. Check that reference folder path is correct in build_database.py
    echo ============================================================
)

echo.
echo ============================================================
echo Done! You can now run Run_Classifier.bat
echo ============================================================
pause

