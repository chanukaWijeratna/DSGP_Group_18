@echo off
echo ============================================
echo   VOCAL INSIGHT - Starting all services...
echo ============================================
echo.

set ROOT=%~dp0
set VENV=%ROOT%venv\Scripts\activate.ps1

:: Terminal 1 - Node.js Auth Backend (port 3001)
start "Auth Backend :3001" powershell -NoExit -Command "Write-Host '[Auth Backend] Starting...'; cd '%ROOT%backend'; node server.js"

:: Terminal 2 - Emotion Detection API (port 8000)
start "Emotion API :8000" powershell -NoExit -Command "Write-Host '[Emotion API] Activating env...'; & '%VENV%'; cd '%ROOT%backend\emotion-recognition\routes'; uvicorn emotion_detection_api:app --host 0.0.0.0 --port 8000"

:: Terminal 3 - Disorder Detection API (port 5000)
start "Disorder API :5000" powershell -NoExit -Command "Write-Host '[Disorder API] Activating env...'; & '%VENV%'; cd '%ROOT%backend\disorder-detection\routes'; uvicorn app:app --host 0.0.0.0 --port 5000"

:: Terminal 4 - Bad Habit Detection API (port 8001)
start "Bad Habit API :8001" powershell -NoExit -Command "Write-Host '[Bad Habit API] Activating env...'; & '%VENV%'; cd '%ROOT%backend\bad-habit-detection\routes'; uvicorn analyze_audio:app --host 0.0.0.0 --port 8001"

:: Terminal 5 - Feedback / RAG API (port 5001)
start "Feedback API :5001" powershell -NoExit -Command "Write-Host '[Feedback API] Activating env...'; & '%VENV%'; cd '%ROOT%backend\feedback-system'; uvicorn app:app --host 0.0.0.0 --port 5001"

:: Terminal 6 - Frontend (port 5173)
start "Frontend :5173" powershell -NoExit -Command "Write-Host '[Frontend] Starting...'; cd '%ROOT%frontend'; npm run dev"

echo.
echo All 6 services are starting in separate windows.
echo.
echo   Auth Backend   -^>  http://localhost:3001
echo   Emotion API    -^>  http://localhost:8000
echo   Disorder API   -^>  http://localhost:5000
echo   Bad Habit API  -^>  http://localhost:8001
echo   Feedback API   -^>  http://localhost:5001
echo   Frontend       -^>  http://localhost:5173
echo.
echo Open http://localhost:5173 once all windows are ready.
echo.
pause
