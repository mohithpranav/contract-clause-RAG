#!/bin/bash
# ClauseInsight Backend Startup Script for Linux/Mac

echo "========================================"
echo "ClauseInsight Backend Setup"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Please edit .env and add your OPENAI_API_KEY"
    echo ""
    read -p "Press enter to continue..."
fi

# Start the server
echo "========================================"
echo "Starting FastAPI Server..."
echo "Backend will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo "========================================"
echo ""

cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
