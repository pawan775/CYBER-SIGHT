@echo off
echo Starting Cyber-Sight...
echo.

cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run INSTALL.bat first.
    pause
    exit /b 1
)

:: Activate and run
call venv\Scripts\activate.bat
streamlit run app.py --server.headless true

pause
