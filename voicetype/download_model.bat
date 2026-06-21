@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   Скачивание модели Whisper medium (~1.53 ГБ)
echo   Совет: отключи VPN для скорости (HuggingFace доступен напрямую)
echo ============================================================
echo.
call venv\Scripts\activate.bat
python download_model.py
echo.
pause
