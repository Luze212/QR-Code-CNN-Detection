#!/usr/bin/env bash
set -euo pipefail

echo "== QR-Code Project Setup (macOS) =="

# 1) Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
else
  echo ".venv already exists."
fi

# 2) Activate venv
source .venv/bin/activate

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install pinned requirements (TF + NumPy + OpenCV + GUI)
pip install -r requirements_gui_mac.txt

# 5) Install torch (CPU) - macOS wheels are on default pip index
pip install torch torchvision torchaudio

# 6) Prevent ultralytics from auto-upgrading deps (optional safety)
pip install --upgrade --no-deps ultralytics

echo ""
echo "✅ Setup complete."
echo "Run GUI with:"
echo "  python GUI.py"
