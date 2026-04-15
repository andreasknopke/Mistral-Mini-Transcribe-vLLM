param(
    [string]$RemoteHost = "192.168.188.173",
    [string]$RemoteUser = "ksai0001_local",
    [string]$RemoteDir = "~/voxtral-setup"
)

$ErrorActionPreference = "Stop"

$sshTarget = $RemoteUser + "@" + $RemoteHost
$targetDisplay = $sshTarget + ":" + $RemoteDir
Write-Host "Kopiere Voxtral-Dateien nach $targetDisplay ..." -ForegroundColor Cyan
ssh $sshTarget "mkdir -p $RemoteDir"
scp "./voxtral_server.py" ($targetDisplay + "/voxtral_server.py")
scp "./scripts/03_install_voxtral_dgx_spark.sh" ($targetDisplay + "/03_install_voxtral_dgx_spark.sh")
scp "./scripts/04_install_voxtral_dgx_spark_container.sh" ($targetDisplay + "/04_install_voxtral_dgx_spark_container.sh")

Write-Host "" 
Write-Host "Dateien kopiert." -ForegroundColor Green
Write-Host "Danach auf dem DGX Spark ausführen:" -ForegroundColor Yellow
Write-Host "  cd ~/voxtral-setup"
Write-Host "  chmod +x 03_install_voxtral_dgx_spark.sh"
Write-Host "  ./03_install_voxtral_dgx_spark.sh"
