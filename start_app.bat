@echo off
echo Starting Model Benchmark Explorer...

echo Starting Backend...
start "Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn main:app --reload --port 8000"

echo Starting Frontend...
start "Frontend" cmd /k "npm run dev"

echo.
echo App started! 
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000/docs
echo.
pause
