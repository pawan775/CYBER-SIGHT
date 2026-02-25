@echo off
echo Stopping Cyber-Sight Server...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM streamlit.exe /T 2>nul
echo Server stopped!
timeout /t 2 >nul
