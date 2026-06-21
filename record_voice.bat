@echo off
cd /d "%~dp0\voicetype"
call venv\Scripts\activate
cd ..
python -m tests.e2e.record_voice
pause
