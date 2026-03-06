@echo off
REM ============================================================
REM Voxtral Server starten (Windows nativ)
REM ============================================================

echo ======================================
echo  Voxtral Server starten (Windows)
echo ======================================
echo.

set VENV_DIR=%~dp0..\voxtral-env
set SERVER=%~dp0..\voxtral_server.py

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo FEHLER: voxtral-env nicht gefunden!
    echo Bitte zuerst setup_voxtral_windows.bat ausfuehren.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Modell:  mistralai/Voxtral-Mini-3B-2507
echo Port:    8000
echo API:     http://localhost:8000/v1/audio/transcriptions
echo Health:  http://localhost:8000/health
echo.
echo Erster Start dauert einige Minuten (Modell-Download).
echo Stoppen mit Ctrl+C
echo.

python "%SERVER%"

pause
