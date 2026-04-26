# Gemma 4 mit GitHub Copilot CLI in VS Code

Wichtig: Das im April 2026 veröffentlichte BYOK-/Local-Models-Feature gilt für **GitHub Copilot CLI**.
Es ersetzt **nicht automatisch** das normale Copilot-Chat-Panel in VS Code.

Für dieses Workspace ist der passende Weg daher:

1. Gemma 4 auf dem DGX Spark als OpenAI-kompatiblen `vLLM`-Server auf Port `9000` laufen lassen.
2. In VS Code die integrierte PowerShell oder Tasks nutzen.
3. `copilot` per `COPILOT_PROVIDER_BASE_URL` auf diesen Endpoint zeigen lassen.

## Was im Workspace schon vorhanden ist

- `scripts/09_install_gemma4_dgx_spark.sh` installiert Gemma 4 als Dienst `correction-llm`.
- Der Dienst stellt `/v1/models` und `/v1/chat/completions` bereit.
- Standard-Modellname für Clients ist `gemma-4`.

## Schnellstart in VS Code

### Fall A: VS Code läuft direkt auf dem Spark

Starte in einem VS-Code-Terminal:

```powershell
pwsh -File .\scripts\start_copilot_gemma4.ps1
```

### Fall B: VS Code läuft auf Windows, Gemma 4 läuft remote auf dem Spark

Default im Workspace:

- Spark-IP: `192.168.188.173`
- Spark-User: `ksai0001_local`

1. Erst SSH-Tunnel öffnen:

```powershell
pwsh -File .\scripts\start_gemma4_tunnel.ps1
```

2. In einem zweiten Terminal Copilot CLI gegen den lokalen Tunnel starten:

```powershell
pwsh -File .\scripts\start_copilot_gemma4.ps1
```

Standardmäßig wird dabei `http://127.0.0.1:9000/v1` verwendet.

## VS Code Tasks

Es gibt fünf Tasks:

- `Gemma4: Start SSH Tunnel`
- `Gemma4: Copilot CLI`
- `Gemma4: Copilot CLI direkt am Spark`
- `Gemma4: Copilot CLI ohne Offline`
- `Gemma4: Repair Tool Calling am Spark`

Öffnen über `Terminal -> Run Task`.

`Gemma4: Copilot CLI direkt am Spark` nutzt direkt `http://192.168.188.173:9000/v1`.
Das ist praktisch, wenn dein Windows-Rechner den Spark-Port direkt im LAN erreicht.
Wenn du maximale Isolation oder stabilere Erreichbarkeit willst, bleib beim Tunnel.

## Relevante Umgebungsvariablen

Das Launcher-Skript setzt diese Copilot-Variablen:

- `COPILOT_PROVIDER_TYPE=openai`
- `COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:9000/v1`
- `COPILOT_MODEL=gemma-4`
- `COPILOT_OFFLINE=true`

Optional kannst du vor dem Start eigene Werte setzen:

- `GEMMA4_COPILOT_BASE_URL`
- `GEMMA4_COPILOT_MODEL`
- `GEMMA4_COPILOT_PROVIDER_TYPE`
- `GEMMA4_COPILOT_API_KEY`
- `GEMMA4_COPILOT_BEARER_TOKEN`
- `GEMMA4_COPILOT_MODEL_ID`

Beispiel:

```powershell
$env:GEMMA4_COPILOT_BASE_URL = "http://192.168.188.173:9000/v1"
$env:GEMMA4_COPILOT_MODEL = "correction-llm"
pwsh -File .\scripts\start_copilot_gemma4.ps1
```

## Healthcheck

Vor dem Start prüft das Skript automatisch `GET /v1/models`.

Manuell:

```powershell
curl http://127.0.0.1:9000/v1/models
```

## Wenn Copilot mit `transient_bad_request` oder `400 tool_choice="auto"` scheitert

Dann läuft auf dem Spark sehr wahrscheinlich noch eine ältere `vLLM`-Startkonfiguration ohne:

- `--enable-auto-tool-choice`
- `--tool-call-parser gemma4`

Für dieses Workspace gibt es dafür jetzt einen Einmal-Fix:

1. `Terminal -> Run Task`
2. `Gemma4: Repair Tool Calling am Spark`
3. Spark-Passwort eingeben, wenn `ssh` oder `sudo` danach fragt

Der Task verbindet sich nach `192.168.188.173`, patcht remote
`/home/ksai0001_local/correction-llm-vllm/run_gemma4.sh`, erstellt ein Backup und startet `correction-llm` neu.

Für **neue** Deployments ist der Fix schon in `scripts/09_install_gemma4_dgx_spark.sh` enthalten.

## Wichtige Einschränkung

Copilot CLI erwartet **Streaming** und **Tool Calling**.
Der vorhandene Gemma-4-vLLM-Endpoint deckt die OpenAI-Chat-API ab, aber je nach Modell-/vLLM-Stand können einzelne agentische Funktionen weiterhin eingeschränkt sein.
Wenn einfache Chats funktionieren, komplexe Agent-Features aber scheitern, ist das meist ein Modell-/Tool-Calling-Thema und kein VS-Code-Fehler.