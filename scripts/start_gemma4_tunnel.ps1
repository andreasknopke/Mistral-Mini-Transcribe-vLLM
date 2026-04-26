param(
    [string]$RemoteHost = "192.168.188.173",
    [string]$RemoteUser = "ksai0001_local",
    [int]$LocalPort = 9000,
    [int]$RemotePort = 9000,
    [string]$RemoteBindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$sshTarget = "$RemoteUser@$RemoteHost"
$forwardSpec = "127.0.0.1:${LocalPort}:${RemoteBindHost}:${RemotePort}"

Write-Host "Starte SSH-Tunnel für Gemma-4 ..." -ForegroundColor Cyan
Write-Host "  Lokal:  http://127.0.0.1:$LocalPort/v1" -ForegroundColor DarkGray
Write-Host "  Remote: $sshTarget -> ${RemoteBindHost}:$RemotePort" -ForegroundColor DarkGray
Write-Host "Terminal offen lassen, solange der Tunnel gebraucht wird." -ForegroundColor Yellow

& ssh -N -L $forwardSpec $sshTarget
exit $LASTEXITCODE