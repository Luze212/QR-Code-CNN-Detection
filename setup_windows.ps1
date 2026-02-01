$ErrorActionPreference = "Stop"
Write-Host "== QR-Code Project Setup (Windows) =="

Write-Host "Python:"
python --version

if (!(Test-Path ".\.venv")) {
  Write-Host "Creating .venv ..."
  python -m venv .venv
} else {
  Write-Host ".venv already exists."
}

Write-Host "Activating venv..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing requirements_gui.txt..."
pip install -r .\requirements_gui.txt

Write-Host "Installing Torch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Write-Host "Ensuring ultralytics without deps upgrade..."
pip install --upgrade --no-deps ultralytics

Write-Host "✅ Setup complete."
Write-Host "Run:"
Write-Host "  .\.venv\Scripts\python.exe GUI.py"
