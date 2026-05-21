@echo off
REM RAG System Start Script for Windows

echo.
echo 🚀 Starting RAG System...
echo.

REM Check if in correct directory
if not exist "main.py" (
    echo ❌ Error: Please run this script from the rag-system directory
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
)

REM Check .env file
if not exist ".env" (
    echo 📝 Creating .env from template...
    copy .env.example .env
    echo ⚠️  Please edit .env with your settings
)

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "chroma_db" mkdir chroma_db
if not exist "logs" mkdir logs

REM Check Ollama
echo.
echo 🤖 Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama not found at localhost:11434
    echo    Start Ollama with: ollama serve
    echo    Or update OLLAMA_BASE_URL in .env
) else (
    echo ✅ Ollama is running
)

REM Start the application
echo.
echo 📡 Starting RAG System API...
echo API will be available at: http://localhost:8000
echo Swagger UI: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

python main.py

pause
