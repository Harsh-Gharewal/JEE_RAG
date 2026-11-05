# JEE-RAG Startup Script for PowerShell
Write-Host "Starting JEE-RAG Server..." -ForegroundColor Cyan

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

python main.py

