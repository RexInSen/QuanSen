@echo off
REM ============================================================
REM  QuanSen — Full Build Script
REM  Run from the folder containing all source files.
REM ============================================================
setlocal EnableDelayedExpansion

echo.
echo  =========================================================
echo    QuanSen Build Script
echo  =========================================================
echo.

pyinstaller --version >nul 2>&1
if errorlevel 1 ( echo [ERROR] PyInstaller missing. pip install pyinstaller & pause & exit /b 1 )
python -c "from PIL import Image" >nul 2>&1
if errorlevel 1 ( echo [ERROR] Pillow missing. pip install pillow & pause & exit /b 1 )
python -c "import streamlit" >nul 2>&1
if errorlevel 1 ( echo [ERROR] Streamlit missing. pip install streamlit & pause & exit /b 1 )

echo  All dependencies found.
echo.

echo  [1/2] Building quansen_app (onedir)...
pyinstaller quansen_app.spec --clean --noconfirm > build_app.log 2>&1
if errorlevel 1 (
    echo  [ERROR] App build FAILED. Last 30 lines of log:
    powershell -Command "Get-Content build_app.log | Select-Object -Last 30"
    pause & exit /b 1
)
echo  [OK] dist\quansen_app\ folder built.
echo.

echo  [2/2] Building QuanSen.exe (splash, onefile)...
pyinstaller quansen_splash.spec --clean --noconfirm > build_splash.log 2>&1
if errorlevel 1 (
    echo  [ERROR] Splash build FAILED.
    powershell -Command "Get-Content build_splash.log | Select-Object -Last 20"
    pause & exit /b 1
)
echo  [OK] dist\QuanSen.exe built.
echo.

echo  =========================================================
echo   BUILD COMPLETE — Final structure in dist\:
echo.
echo   dist\
echo     QuanSen.exe            ^<-- users double-click this
echo     quansen_app\
echo       quansen_app.exe      ^<-- Streamlit app (auto-launched)
echo       ^<all supporting .dll and .pyd files^>
echo.
echo   IMPORTANT: The entire dist\ folder must be distributed
echo   together. QuanSen.exe and quansen_app\ must stay
echo   side-by-side.
echo  =========================================================
echo.
pause
