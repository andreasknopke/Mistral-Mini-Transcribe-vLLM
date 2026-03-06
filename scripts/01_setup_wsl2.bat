@echo off
REM ============================================================
REM Voxtral WSL2 Setup - Schritt 1: WSL2 + Ubuntu einrichten
REM Rechtsklick -> "Als Administrator ausfuehren"!
REM ============================================================

REM Admin-Check
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo FEHLER: Bitte Rechtsklick auf diese Datei
    echo         und "Als Administrator ausfuehren" waehlen!
    echo.
    pause
    exit /b 1
)

echo ======================================
echo  Voxtral WSL2 Setup
echo ======================================
echo.

REM 1. WSL2 Status pruefen
echo [1/4] Pruefe WSL2 Status...
wsl --status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   WSL2 wird installiert...
    echo   Das kann einige Minuten dauern.
    echo.
    wsl --install
    echo.
    echo ======================================
    echo  WICHTIG: PC neu starten!
    echo ======================================
    echo.
    echo  Nach dem Neustart diese .bat erneut
    echo  als Administrator ausfuehren.
    echo.
    pause
    exit /b 0
)
echo   WSL2 ist bereits installiert.

REM 2. Ubuntu pruefen
echo [2/4] Pruefe Ubuntu Distribution...
wsl --list --quiet 2>nul | findstr /i "Ubuntu" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   Ubuntu wird installiert...
    wsl --install -d Ubuntu
    echo   Ubuntu installiert.
    echo   Beim ersten Start Benutzername und Passwort setzen.
) else (
    echo   Ubuntu ist bereits installiert.
)

REM 3. NVIDIA Treiber pruefen
echo [3/4] Pruefe NVIDIA Treiber...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   WARNUNG: nvidia-smi nicht gefunden!
    echo   NVIDIA Treiber installieren: https://www.nvidia.com/Download/index.aspx
) else (
    echo   GPU gefunden.
)

REM 4. GPU in WSL2 pruefen
echo [4/4] Pruefe GPU in WSL2...
wsl -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   WARNUNG: GPU in WSL2 nicht verfuegbar.
    echo   Versuche: wsl --shutdown  und dann erneut starten.
) else (
    echo   GPU in WSL2 verfuegbar.
)

echo.
echo ======================================
echo  Naechster Schritt:
echo ======================================
echo.
echo  1. Oeffne ein WSL2/Ubuntu Terminal:
echo     Tippe "wsl" in PowerShell oder CMD
echo.
echo  2. Fuehre aus:
echo     bash /mnt/d/GitHub/Mistral/HTML/scripts/02_install_voxtral.sh
echo.
echo ======================================
echo.
pause
