@echo off
REM ClauseInsight Backend Startup Script for Windows

echo ========================================
echo ClauseInsight Backend Setup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Check for .env file
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Creating .env from .env.example...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env and add your OPENAI_API_KEY
    echo.
    pause
)

REM Start the server
echo ========================================
echo Starting FastAPI Server...
echo Backend will be available at http://localhost:8000
echo API docs at http://localhost:8000/docs
echo ========================================
echo.

cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
