@echo off
echo === Drowsiness Detection Setup ===

:: Find Python
set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" (where python3 >nul 2>&1 && set PYTHON=python3)
if "%PYTHON%"=="" (where py >nul 2>&1 && set PYTHON=py)

if "%PYTHON%"=="" (
    echo ERROR: Python 3.8+ is required but not found.
    echo Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

%PYTHON% --version

:: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    %PYTHON% -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

:: Download model if missing
if not exist "face_landmarker.task" (
    echo Downloading face landmark model...
    curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

echo.
echo === Setup complete! ===
echo Starting app...
echo Open http://127.0.0.1:5000 in your browser
echo.
python app.py
pause
