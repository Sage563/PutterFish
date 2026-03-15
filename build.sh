#!/bin/sh
# Build uci binary with PyInstaller; uses icon.ico if present.

set -e
cd "$(dirname "$0")"

if [ ! -f "models/model.pth" ]; then
    echo "ERROR: models/model.pth not found. Train or place the model first."
    exit 1
fi

echo "Installing PyInstaller if needed..."
pip install pyinstaller torch chess -q

ICON=""
if [ -f "icon.ico" ]; then
    ICON="--icon icon.ico"
    echo "Using icon.ico"
else
    echo "No icon.ico found; add icon.ico to project for app icon."
fi

echo "Building uci..."
pyinstaller --onefile --name uci $ICON \
    --add-data "models/model.pth:models" \
    --hidden-import=chess \
    --hidden-import=torch \
    main.py

echo ""
echo "Done. Run: dist/uci"
