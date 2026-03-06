@echo off
REM ============================================================
REM Voxtral Server testen (Windows)
REM ============================================================

echo ======================================
echo  Voxtral Server Test
echo ======================================
echo.

set SERVER_URL=http://localhost:8000

echo [1/2] Health-Check...
curl -s %SERVER_URL%/health
echo.
echo.

echo [2/2] Modell-Info...
curl -s %SERVER_URL%/v1/models
echo.
echo.

echo Manueller Transkriptions-Test:
echo   curl %SERVER_URL%/v1/audio/transcriptions -F file=@deine-datei.wav -F language=de -F response_format=verbose_json
echo.
pause
