@echo off
title Install Dependencies
color 0B

echo ============================================================
echo              INSTALL DEPENDENCIES
echo ============================================================
echo.
echo Installing required Python packages...
echo.

pip install Pillow imagehash numpy opencv-python

if errorlevel 1 (
    echo.
    echo Trying with py -m pip...
    py -m pip install Pillow imagehash numpy opencv-python
)

echo.
echo ============================================================
echo Installation complete!
echo.
echo Next steps:
echo   1. Run Build_Database.bat (if you haven't already)
echo   2. Run Run_Classifier.bat to classify images
echo ============================================================
pause

