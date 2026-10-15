#!/usr/bin/env bash
# Crée l'environnement virtuel, installe les dépendances et télécharge les données.
set -e

python -m venv .venv
# Linux/macOS : .venv/bin/activate ; Windows (Git Bash) : .venv/Scripts/activate
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt

python scripts/download_data.py --out data/raw
python -m src.data_processing --build
