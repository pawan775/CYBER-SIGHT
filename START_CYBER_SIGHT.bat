@echo off
title Cyber-Sight Server
echo.
echo ========================================
echo    CYBER-SIGHT - Starting Server...
echo ========================================
echo.
echo Please wait while the server starts...
echo.
echo Once started, open in browser:
echo    - Local: http://localhost:8501
echo    - Mobile: Check Network URL below
echo.
echo ========================================
echo.

cd /d "D:\pawan project\cyber_sight"
"D:\pawan project\.venv\Scripts\streamlit.exe" run app.py --server.headless true --server.address 0.0.0.0

pause
