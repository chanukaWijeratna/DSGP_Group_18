@echo off
setlocal enabledelayedexpansion
set ROOT=%~dp0

echo ============================================
echo   VOCAL INSIGHT - Setup
echo ============================================
echo.


:: ── Step 1: Create Python virtual environment ──
echo [1/4] Creating Python virtual environment...
if exist "%ROOT%venv" (
    echo       venv already exists, skipping creation.
) else (
    python -m venv "%ROOT%venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Make sure Python is installed and on PATH.
        pause & exit /b 1
    )
    echo       venv created.
)
echo.


:: ── Step 2: Check config.json ──
echo [2/5] Checking API key configuration...
if not exist "%ROOT%backend\feedback-system\config.json" (
    echo.
    echo   No API key found. Please enter your OpenRouter API key below.
    echo   ^(Get one at https://openrouter.ai/keys^)
    echo.
    set /p USER_API_KEY="  Paste API key: "
    echo { > "%ROOT%backend\feedback-system\config.json"
    echo     "api_key": "!USER_API_KEY!" >> "%ROOT%backend\feedback-system\config.json"
    echo } >> "%ROOT%backend\feedback-system\config.json"
    echo       API key saved to config.json.
)
echo       config.json ready.
echo.


:: ── Step 3: Install Python packages ──
echo [3/5] Installing Python packages from requirements.txt...
call "%ROOT%venv\Scripts\activate.bat"
pip install --upgrade pip --quiet
pip install -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo [ERROR] pip install failed. Check requirements.txt and your internet connection.
    pause & exit /b 1
)
echo       Python packages installed.
echo.


:: ── Step 3: Install frontend and backend npm packages ──
echo [4/5] Installing frontend packages...
cd /d "%ROOT%frontend"
call npm install
if errorlevel 1 (
    echo [ERROR] npm install failed. Make sure Node.js is installed and on PATH.
    pause & exit /b 1
)
echo       Frontend packages installed.

echo       Installing backend packages...
cd /d "%ROOT%backend"
call npm install
if errorlevel 1 (
    echo [ERROR] Backend npm install failed.
    pause & exit /b 1
)
echo       Backend packages installed.
echo.


:: ── Step 4: Generate RAG index ──
echo [5/5] Generating RAG knowledge index...
cd /d "%ROOT%backend\feedback-system"
python indexing.py
if errorlevel 1 (
    echo [ERROR] indexing.py failed. Check that the RAG_knowledge/original folder has content.
    pause & exit /b 1
)
echo       RAG index generated.
echo.


:: ── Done ──
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo   Run start.bat to launch the application.
echo.
pause
