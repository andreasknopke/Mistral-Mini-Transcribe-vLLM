@echo off
REM ============================================================
REM Voxtral Windows Setup - Python + PyTorch CUDA + Dependencies
REM Kein WSL2 noetig! Laeuft nativ auf Windows mit TCC-GPU.
REM ============================================================

echo ======================================
echo  Voxtral Windows Setup (nativ)
echo ======================================
echo.

REM Pruefe Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FEHLER: Python nicht gefunden!
    echo Bitte Python 3.10+ installieren: https://www.python.org/downloads/
    echo WICHTIG: Bei der Installation "Add to PATH" ankreuzen!
    pause
    exit /b 1
)

echo [1/5] Python gefunden:
python --version
echo.

REM Pruefe NVIDIA GPU
echo [2/5] Pruefe GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNUNG: nvidia-smi nicht gefunden!
) else (
    echo GPU OK.
)
echo.

REM Erstelle venv
echo [3/5] Python Virtual Environment erstellen...
set VENV_DIR=%~dp0..\voxtral-env
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo   voxtral-env existiert bereits.
) else (
    python -m venv "%VENV_DIR%"
    echo   voxtral-env erstellt.
)

REM Aktiviere venv
call "%VENV_DIR%\Scripts\activate.bat"
echo   Environment aktiviert.
echo.

REM Installiere PyTorch mit CUDA + Dependencies
echo [4/5] PyTorch mit CUDA und Dependencies installieren...
echo   (Das dauert einige Minuten)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate "mistral-common[audio]" huggingface_hub
pip install fastapi uvicorn python-multipart pydantic soundfile librosa

echo.
echo [5/5] Pruefe PyTorch CUDA...
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'KEINE GPU!')"

echo.
echo ======================================
echo  Installation abgeschlossen!
echo ======================================
echo.
echo  Naechster Schritt - Server starten:
echo    start_voxtral_windows.bat
echo.
pause
