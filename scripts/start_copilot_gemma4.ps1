param(
    [string]$BaseUrl = "http://127.0.0.1:9000/v1",
    [string]$Model = "gemma-4",
    [string]$ProviderType = "openai",
    [string]$ApiKey = "",
    [string]$BearerToken = "",
    [string]$ModelId = "",
    [int]$MaxPromptTokens = 0,
    [int]$MaxOutputTokens = 0,
    [switch]$Offline,
    [switch]$NoOffline,
    [switch]$SkipHealthcheck,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CopilotArgs
)

$ErrorActionPreference = "Stop"

if ($env:GEMMA4_COPILOT_BASE_URL) {
    $BaseUrl = $env:GEMMA4_COPILOT_BASE_URL
}
if ($env:GEMMA4_COPILOT_MODEL) {
    $Model = $env:GEMMA4_COPILOT_MODEL
}
if ($env:GEMMA4_COPILOT_PROVIDER_TYPE) {
    $ProviderType = $env:GEMMA4_COPILOT_PROVIDER_TYPE
}
if (-not $ApiKey -and $env:GEMMA4_COPILOT_API_KEY) {
    $ApiKey = $env:GEMMA4_COPILOT_API_KEY
}
if (-not $BearerToken -and $env:GEMMA4_COPILOT_BEARER_TOKEN) {
    $BearerToken = $env:GEMMA4_COPILOT_BEARER_TOKEN
}
if (-not $ModelId -and $env:GEMMA4_COPILOT_MODEL_ID) {
    $ModelId = $env:GEMMA4_COPILOT_MODEL_ID
}

$normalizedBaseUrl = $BaseUrl.TrimEnd('/')
$modelsUrl = if ($normalizedBaseUrl.EndsWith('/v1')) {
    "$normalizedBaseUrl/models"
} else {
    "$normalizedBaseUrl/v1/models"
}

if (-not $SkipHealthcheck) {
    Write-Host "Prüfe Gemma-4 Endpoint: $modelsUrl" -ForegroundColor Cyan
    try {
        $response = Invoke-RestMethod -Uri $modelsUrl -Method Get -TimeoutSec 8
        $modelNames = @()
        if ($response.data) {
            $modelNames = @($response.data | ForEach-Object { $_.id })
        }

        if ($modelNames.Count -gt 0) {
            Write-Host ("Endpoint erreichbar. Modelle: " + ($modelNames -join ', ')) -ForegroundColor Green
        } else {
            Write-Host "Endpoint erreichbar, aber /models lieferte keine IDs zurück." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Gemma-4 Endpoint nicht erreichbar." -ForegroundColor Red
        Write-Host "Tipps:" -ForegroundColor Yellow
        Write-Host "  1) Läuft auf dem Spark 'correction-llm' auf Port 9000?" -ForegroundColor Yellow
        Write-Host "  2) Falls der Spark remote ist: zuerst den SSH-Tunnel starten." -ForegroundColor Yellow
        Write-Host "  3) Danach dieses Skript erneut ausführen oder -SkipHealthcheck nutzen." -ForegroundColor Yellow
        throw
    }
}

$env:COPILOT_PROVIDER_TYPE = $ProviderType
$env:COPILOT_PROVIDER_BASE_URL = $normalizedBaseUrl
$env:COPILOT_MODEL = $Model

if ($ApiKey) {
    $env:COPILOT_PROVIDER_API_KEY = $ApiKey
}
if ($BearerToken) {
    $env:COPILOT_PROVIDER_BEARER_TOKEN = $BearerToken
}
if ($ModelId) {
    $env:COPILOT_PROVIDER_MODEL_ID = $ModelId
}
if ($MaxPromptTokens -gt 0) {
    $env:COPILOT_PROVIDER_MAX_PROMPT_TOKENS = [string]$MaxPromptTokens
}
if ($MaxOutputTokens -gt 0) {
    $env:COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = [string]$MaxOutputTokens
}

$offlineEnabled = $true
if ($NoOffline) {
    $offlineEnabled = $false
} elseif ($Offline) {
    $offlineEnabled = $true
}

if ($offlineEnabled) {
    $env:COPILOT_OFFLINE = "true"
} else {
    Remove-Item Env:COPILOT_OFFLINE -ErrorAction SilentlyContinue
}

Write-Host "Starte Copilot CLI mit lokalem Gemma-4-Provider ..." -ForegroundColor Cyan
Write-Host "  COPILOT_PROVIDER_BASE_URL=$($env:COPILOT_PROVIDER_BASE_URL)" -ForegroundColor DarkGray
Write-Host "  COPILOT_MODEL=$($env:COPILOT_MODEL)" -ForegroundColor DarkGray
Write-Host "  COPILOT_PROVIDER_TYPE=$($env:COPILOT_PROVIDER_TYPE)" -ForegroundColor DarkGray
Write-Host "  COPILOT_OFFLINE=$($env:COPILOT_OFFLINE)" -ForegroundColor DarkGray

& copilot @CopilotArgs
exit $LASTEXITCODE