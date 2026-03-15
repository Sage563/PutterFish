@echo off
REM Build uci.exe with PyInstaller (bundles model; uses icon.ico if present).
setlocal
cd /d "%~dp0"

if not exist "models\model.pth" (
    echo ERROR: models\model.pth not found. Train or place the model first.
    exit /b 1
)

echo Installing PyInstaller if needed...
pip install pyinstaller torch chess --quiet

set ICON=
if exist "icon.ico" (
    set ICON=--icon icon.ico
    echo Using icon.ico
) else (
    echo No icon.ico found; add icon.ico to project for exe icon.
)

echo Building uci.exe...
pyinstaller --onefile --name uci %ICON% --add-data "models/model.pth;models" --hidden-import=chess --hidden-import=torch main.py

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b 1
)

echo.
echo Done. Run: dist\uci.exe
endlocal
