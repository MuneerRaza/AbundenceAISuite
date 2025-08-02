@echo off
echo 🚀 Setting up IntelliFlow AI...
echo ==================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Copy environment file
if not exist .env (
    echo 📋 Creating environment file...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your API keys before running the application.
) else (
    echo ✅ .env file already exists
)

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Docker not found. Please install MongoDB and Qdrant manually or install Docker.
) else (
    echo 🐳 Docker detected. You can run 'docker-compose up -d' to start MongoDB and Qdrant.
)

echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Start services: docker-compose up -d (if using Docker)
echo 3. Run CLI: python main.py
echo 4. Or start API server: uvicorn api:app --reload
echo.
echo 📖 For more information, see README.md
pause
