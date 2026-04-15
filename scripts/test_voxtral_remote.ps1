param(
    [string]$ServerUrl = "http://192.168.188.173:8000"
)

$ErrorActionPreference = "Stop"
Write-Host "Prüfe Health-Endpoint: $ServerUrl/health" -ForegroundColor Cyan
Invoke-RestMethod -Method Get -Uri "$ServerUrl/health" | ConvertTo-Json -Depth 5
