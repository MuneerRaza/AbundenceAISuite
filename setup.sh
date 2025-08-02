#!/bin/bash

# IntelliFlow AI Setup Script

echo "🚀 Setting up IntelliFlow AI..."
echo "=================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. You have Python $python_version."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before running the application."
else
    echo "✅ .env file already exists"
fi

# Check if Docker is available for services
if command -v docker &> /dev/null; then
    echo "🐳 Docker detected. You can run 'docker-compose up -d' to start MongoDB and Qdrant."
else
    echo "⚠️  Docker not found. Please install MongoDB and Qdrant manually or install Docker."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Start services: docker-compose up -d (if using Docker)"
echo "3. Run CLI: python main.py"
echo "4. Or start API server: uvicorn api:app --reload"
echo ""
echo "📖 For more information, see README.md"
