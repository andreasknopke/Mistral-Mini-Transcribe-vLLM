param(
    [string]$RemoteHost = "192.168.188.173",
    [string]$RemoteUser = "ksai0001_local"
)

$ErrorActionPreference = "Stop"

$remoteTarget = "$RemoteUser@$RemoteHost"
$remoteAppFile = "/home/$RemoteUser/voxtral-local/voxtral_server.py"
$localFile = Join-Path $PSScriptRoot "..\voxtral_server.py"

Write-Host "Uploading $localFile to $remoteTarget`:$remoteAppFile" -ForegroundColor Cyan
scp $localFile "${remoteTarget}:$remoteAppFile"
if ($LASTEXITCODE -ne 0) {
    throw "scp upload failed"
}

$remoteCommand = @'
bash -lc 'source ~/voxtral-env/bin/activate && pip install "mistral-common[audio]" && sudo systemctl restart voxtral && sudo systemctl status voxtral --no-pager -l'
'@
Write-Host "Running remote repair command on $remoteTarget" -ForegroundColor Cyan
ssh -t $remoteTarget $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "remote repair command failed"
}

Write-Host "Done. To watch logs, run:" -ForegroundColor Green
Write-Host "ssh $remoteTarget 'journalctl -u voxtral -f'"
