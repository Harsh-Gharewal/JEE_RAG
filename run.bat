@echo off
REM JEE-RAG Startup Script for Windows
echo Starting JEE-RAG Server...

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python main.py

