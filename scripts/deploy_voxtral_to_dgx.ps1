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
$whisperxInstallScript = Join-Path $PSScriptRoot "05_install_whisperx_dgx_spark.sh"
$correctionLlmInstallScript = Join-Path $PSScriptRoot "06_install_correction_llm_dgx_spark.sh"
$stackInstallScript = Join-Path $PSScriptRoot "07_install_dgx_spark_ai_stack.sh"
$whisperxSparkDir = Join-Path $repoRoot "whisperx_spark"
$stackReadme = Join-Path $repoRoot "README_DGX_SPARK_STACK.md"

$sshTarget = $RemoteUser + "@" + $RemoteHost
$targetDisplay = $sshTarget + ":" + $RemoteDir
Write-Host "Kopiere Voxtral-Dateien nach $targetDisplay ..." -ForegroundColor Cyan
ssh $sshTarget "mkdir -p $RemoteDir"
scp $voxtralServer ($targetDisplay + "/voxtral_server.py")
scp $installScript ($targetDisplay + "/03_install_voxtral_dgx_spark.sh")
scp $containerScript ($targetDisplay + "/04_install_voxtral_dgx_spark_container.sh")
scp $whisperxInstallScript ($targetDisplay + "/05_install_whisperx_dgx_spark.sh")
scp $correctionLlmInstallScript ($targetDisplay + "/06_install_correction_llm_dgx_spark.sh")
scp $stackInstallScript ($targetDisplay + "/07_install_dgx_spark_ai_stack.sh")
scp $stackReadme ($targetDisplay + "/README_DGX_SPARK_STACK.md")
scp -r $whisperxSparkDir ($targetDisplay + "/whisperx_spark")

Write-Host "" 
Write-Host "Dateien kopiert." -ForegroundColor Green
Write-Host "Danach auf dem DGX Spark ausführen:" -ForegroundColor Yellow
Write-Host "  cd ~/voxtral-setup"
Write-Host "  chmod +x *.sh"
Write-Host "  ./07_install_dgx_spark_ai_stack.sh"
