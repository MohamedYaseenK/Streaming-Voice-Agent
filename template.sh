#!/usr/bin/env bash
set -e

# Linux only: pyttsx3 needs espeak installed as a system package.
# macOS/Windows use their built-in TTS engines, so skip this there.
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y espeak
fi

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

if [ ! -f .env ]; then
    echo "GEMINI_API_KEY=your-gemini-key-here" > .env
    echo "Created .env -- put your real Gemini key in it (from https://aistudio.google.com/apikey)."
fi

mkdir -p logs

echo ""
echo "Setup done. To run:"
echo "  Terminal 1: source venv/bin/activate && python server.py"
echo "  Terminal 2: source venv/bin/activate && python client.py"
