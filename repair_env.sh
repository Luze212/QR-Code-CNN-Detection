#!/bin/bash

# ============================================================
# NOTFALL-SKRIPT FÜR M4 MAC (Apple Silicon)
# Repariert die Python-Umgebung und erzwingt CPU-Modus für TF
# ============================================================

# Abbruch bei Fehlern
set -e

echo "🧹 Räume auf..."
# 1. Deaktivieren, falls aktiv (ignoriere Fehler falls nicht)
deactivate 2>/dev/null || true

# 2. Lösche alte/kaputte Umgebungen restlos
rm -rf .venv venv
echo "   -> Alte Umgebung gelöscht."

echo "📦 Erstelle neue Umgebung..."
# 3. Neue saubere venv erstellen
python3 -m venv .venv

# 4. Aktivieren
source .venv/bin/activate

echo "⬆️  Aktualisiere Pip..."
# 5. Pip reparieren/aktualisieren
python3 -m pip install --upgrade pip

echo "📥 Installiere Pakete (STABILITÄTS-MODUS)..."
# 6. Installation der Pakete
# WICHTIG: Wir installieren KEIN 'tensorflow-metal', um Abstürze zu verhindern!
pip install PySide6 opencv-python matplotlib ultralytics pandas seaborn

# TensorFlow Basis (CPU-Optimiert für ARM64, aber ohne Metal-Plugin-Crash)
pip install tensorflow-macos

echo "========================================================"
echo "✅ REPARATUR ABGESCHLOSSEN!"
echo "========================================================"
echo "Starte das Programm jetzt mit:"
echo "source .venv/bin/activate"
echo "python3 QR-Code-Erkennung-Interface.py"
echo "========================================================"