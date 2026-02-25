@echo off
echo ==========================================
echo    CYBER-SIGHT INSTALLATION SCRIPT
echo    ML & AI Based Cyber Crime Detection
echo ==========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo Please install Python 3.10 or higher from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

:: Activate virtual environment and install packages
echo Installing required packages (this may take a few minutes)...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)

echo.
echo ==========================================
echo    INSTALLATION COMPLETE!
echo ==========================================
echo.
echo To run the application:
echo   1. Double-click "RUN_APP.bat"
echo   OR
echo   2. Open terminal and run: streamlit run app.py
echo.
echo Default Login Credentials:
echo   Username: admin
echo   Password: admin123
echo.
pause
