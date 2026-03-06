@echo off
title Voxtral Local Server
echo ============================================
echo   Voxtral Local Server - V100
echo ============================================
echo.

call C:\Users\Andre.AUDIO-WS1\miniconda3\condabin\conda.bat activate voxtral

echo Starte Server auf Port 8000...
echo.
python "%~dp0voxtral_server.py"

pause
