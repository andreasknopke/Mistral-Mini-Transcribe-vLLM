param(
    [string]$RemoteHost = "192.168.188.173",
    [string]$RemoteUser = "ksai0001_local",
    [string]$RemoteDir = "~/voxtral-setup"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$voxtralServer = Join-Path $repoRoot "voxtral_server.py"
$installScript = Join-Path $PSScriptRoot "03_install_voxtral_dgx_spark.sh"
$containerScript = Join-Path $PSScriptRoot "04_install_voxtral_dgx_spark_container.sh"

$sshTarget = $RemoteUser + "@" + $RemoteHost
$targetDisplay = $sshTarget + ":" + $RemoteDir
Write-Host "Kopiere Voxtral-Dateien nach $targetDisplay ..." -ForegroundColor Cyan
ssh $sshTarget "mkdir -p $RemoteDir"
scp $voxtralServer ($targetDisplay + "/voxtral_server.py")
scp $installScript ($targetDisplay + "/03_install_voxtral_dgx_spark.sh")
scp $containerScript ($targetDisplay + "/04_install_voxtral_dgx_spark_container.sh")

Write-Host "" 
Write-Host "Dateien kopiert." -ForegroundColor Green
Write-Host "Danach auf dem DGX Spark ausführen:" -ForegroundColor Yellow
Write-Host "  cd ~/voxtral-setup"
Write-Host "  chmod +x 03_install_voxtral_dgx_spark.sh"
Write-Host "  ./03_install_voxtral_dgx_spark.sh"
