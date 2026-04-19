param(
    [string]$RemoteHost = "192.168.188.173",
    [string]$RemoteUser = "ksai0001_local",
    [string]$RemoteDir = "~/voxtral-setup"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$voxtralServer = Join-Path $repoRoot "voxtral_server.py"
$voxtralCompatProxy = Join-Path $repoRoot "voxtral_vllm_proxy.py"
$installScript = Join-Path $PSScriptRoot "03_install_voxtral_dgx_spark.sh"
$containerScript = Join-Path $PSScriptRoot "04_install_voxtral_dgx_spark_container.sh"
$whisperxInstallScript = Join-Path $PSScriptRoot "05_install_whisperx_dgx_spark.sh"
$correctionLlmInstallScript = Join-Path $PSScriptRoot "06_install_correction_llm_dgx_spark.sh"
$stackInstallScript = Join-Path $PSScriptRoot "07_install_dgx_spark_ai_stack.sh"
$sparkAdminInstallScript = Join-Path $PSScriptRoot "08_install_spark_admin_dgx_spark.sh"
$gemma4InstallScript = Join-Path $PSScriptRoot "09_install_gemma4_dgx_spark.sh"
$vibevoiceInstallScript = Join-Path $PSScriptRoot "10_install_vibevoice_dgx_spark.sh"
$whisperxSparkDir = Join-Path $repoRoot "whisperx_spark"
$sparkAdminDir = Join-Path $repoRoot "spark_admin"
$vibevoiceSparkDir = Join-Path $repoRoot "vibevoice_spark"
$stackReadme = Join-Path $repoRoot "README_DGX_SPARK_STACK.md"
$localEnv = Join-Path $repoRoot ".env.local"

$sshTarget = $RemoteUser + "@" + $RemoteHost
$targetDisplay = $sshTarget + ":" + $RemoteDir
Write-Host "Kopiere Voxtral-Dateien nach $targetDisplay ..." -ForegroundColor Cyan
ssh $sshTarget "mkdir -p $RemoteDir"
scp $voxtralServer ($targetDisplay + "/voxtral_server.py")
scp $voxtralCompatProxy ($targetDisplay + "/voxtral_vllm_proxy.py")
scp $installScript ($targetDisplay + "/03_install_voxtral_dgx_spark.sh")
scp $containerScript ($targetDisplay + "/04_install_voxtral_dgx_spark_container.sh")
scp $whisperxInstallScript ($targetDisplay + "/05_install_whisperx_dgx_spark.sh")
scp $correctionLlmInstallScript ($targetDisplay + "/06_install_correction_llm_dgx_spark.sh")
scp $stackInstallScript ($targetDisplay + "/07_install_dgx_spark_ai_stack.sh")
scp $sparkAdminInstallScript ($targetDisplay + "/08_install_spark_admin_dgx_spark.sh")
scp $gemma4InstallScript ($targetDisplay + "/09_install_gemma4_dgx_spark.sh")
scp $vibevoiceInstallScript ($targetDisplay + "/10_install_vibevoice_dgx_spark.sh")
scp $stackReadme ($targetDisplay + "/README_DGX_SPARK_STACK.md")
if (Test-Path $localEnv) {
    scp $localEnv ($targetDisplay + "/.env.local")
}
scp -r $whisperxSparkDir ($targetDisplay + "/whisperx_spark")
scp -r $sparkAdminDir ($targetDisplay + "/spark_admin")
scp -r $vibevoiceSparkDir ($targetDisplay + "/vibevoice_spark")

Write-Host "" 
Write-Host "Dateien kopiert." -ForegroundColor Green
Write-Host "Danach auf dem DGX Spark ausführen:" -ForegroundColor Yellow
Write-Host "  cd ~/voxtral-setup"
Write-Host "  chmod +x *.sh"
Write-Host "  ./07_install_dgx_spark_ai_stack.sh"
