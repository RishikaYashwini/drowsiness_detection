#!/usr/bin/env bash
set -e

echo "=== Drowsiness Detection Setup ==="

# Find Python
PYTHON=""
for cmd in python3 python py; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            PYTHON="$cmd"
            echo "Found Python: $("$cmd" --version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.8+ is required but not found."
    echo "Install Python from https://www.python.org/downloads/"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv venv
fi

# Activate venv
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate  # Windows Git Bash
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate      # macOS / Linux
fi

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Download model if missing
MODEL="face_landmarker.task"
if [ ! -f "$MODEL" ]; then
    echo "Downloading face landmark model..."
    curl -L -o "$MODEL" "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
fi

echo ""
echo "=== Setup complete! ==="
echo "Starting app..."
echo "Open http://127.0.0.1:5000 in your browser"
echo ""
python app.py
