@echo off
title Install AI Dependencies
color 0B

echo ============================================================
echo           INSTALL AI DEPENDENCIES (PyTorch)
echo ============================================================
echo.
echo This will install PyTorch and other required packages.
echo This may take a few minutes and download ~2GB...
echo.

pip install torch torchvision pillow numpy

if errorlevel 1 (
    echo.
    echo Trying with py -m pip...
    py -m pip install torch torchvision pillow numpy
)

echo.
echo ============================================================
echo Installation complete!
echo.
echo Next steps:
echo   1. Run Build_AI_Database.bat
echo   2. Run Run_AI_Classifier.bat
echo ============================================================
pause

