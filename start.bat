@echo off
cd /d "%~dp0server"
call ..\env\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
