# ============================================================
# Voxtral WSL2 Setup - Schritt 1: WSL2 + Ubuntu einrichten
# Dieses Skript als Administrator in PowerShell ausführen!
# ============================================================

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Voxtral WSL2 Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Prüfe ob als Administrator ausgeführt
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "FEHLER: Bitte als Administrator ausfuehren!" -ForegroundColor Red
    Write-Host "Rechtsklick auf PowerShell -> 'Als Administrator ausfuehren'" -ForegroundColor Yellow
    exit 1
}

# 1. Prüfe ob WSL2 bereits installiert ist
Write-Host "[1/4] Pruefe WSL2 Status..." -ForegroundColor Yellow
$wslStatus = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WSL2 wird installiert..." -ForegroundColor Yellow
    wsl --install
    Write-Host ""
    Write-Host "WICHTIG: Bitte den PC neu starten und dieses Skript erneut ausfuehren!" -ForegroundColor Red
    Write-Host "Nach dem Neustart wird Ubuntu automatisch eingerichtet." -ForegroundColor Yellow
    Read-Host "Druecke Enter zum Schliessen"
    exit 0
} else {
    Write-Host "  WSL2 ist bereits installiert." -ForegroundColor Green
}

# 2. Prüfe ob Ubuntu installiert ist
Write-Host "[2/4] Pruefe Ubuntu Distribution..." -ForegroundColor Yellow
$distros = wsl --list --quiet 2>&1
if ($distros -notmatch "Ubuntu") {
    Write-Host "  Ubuntu wird installiert..." -ForegroundColor Yellow
    wsl --install -d Ubuntu
    Write-Host "  Ubuntu installiert. Bitte Benutzername und Passwort setzen." -ForegroundColor Green
} else {
    Write-Host "  Ubuntu ist bereits installiert." -ForegroundColor Green
}

# 3. Prüfe NVIDIA Treiber
Write-Host "[3/4] Pruefe NVIDIA Treiber..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  GPU gefunden: $nvidiaSmi" -ForegroundColor Green
    } else {
        Write-Host "  WARNUNG: nvidia-smi nicht gefunden!" -ForegroundColor Red
        Write-Host "  Bitte NVIDIA Treiber installieren: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  WARNUNG: nvidia-smi nicht gefunden!" -ForegroundColor Red
    Write-Host "  Bitte NVIDIA Treiber installieren: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
}

# 4. Prüfe GPU in WSL2
Write-Host "[4/4] Pruefe GPU in WSL2..." -ForegroundColor Yellow
$gpuCheck = wsl -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  GPU in WSL2 verfuegbar: $gpuCheck" -ForegroundColor Green
} else {
    Write-Host "  WARNUNG: GPU in WSL2 nicht verfuegbar." -ForegroundColor Red
    Write-Host "  Starte WSL2 neu mit: wsl --shutdown" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Naechster Schritt:" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Oeffne ein WSL2/Ubuntu Terminal (tippe 'wsl' in PowerShell)" -ForegroundColor White
Write-Host "2. Fuehre aus:" -ForegroundColor White
Write-Host "   bash /mnt/d/GitHub/Mistral/HTML/scripts/02_install_voxtral.sh" -ForegroundColor Green
Write-Host ""
