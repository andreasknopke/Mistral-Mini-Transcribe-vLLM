# DGX Spark AI Stack

Dieses Setup ergänzt den bestehenden Voxtral-Pfad um zwei weitere Dienste auf dem DGX Spark:

- Mistral-Transcribe / Voxtral über vLLM auf Port `8000`
- WhisperX mit Timestamps und Worker-Pool auf Port `7860`
- optional: OpenAI-kompatibles Korrektur-LLM über vLLM auf Port `8001`

## Zielbild

- `voxtral-vllm` liefert schnelle OpenAI-kompatible Audio-Transkription für Mistral/Voxtral.
- `whisperx` liefert segmentierte Timestamps und optional LLM-basierte Nachkorrektur.
- optional kann `correction-llm` `/v1/models` und `/v1/chat/completions` für Textkorrektur oder Review liefern.

## Deployment

Vom Windows-Rechner aus:

```powershell
pwsh -File .\scripts\deploy_voxtral_to_dgx.ps1 -RemoteUser <dein-user>
```

Dadurch landen auf dem Spark unter `~/voxtral-setup`:

- `voxtral_server.py`
- `scripts/03_install_voxtral_dgx_spark.sh`
- `scripts/04_install_voxtral_dgx_spark_container.sh`
- `scripts/05_install_whisperx_dgx_spark.sh`
- `scripts/06_install_correction_llm_dgx_spark.sh`
- `scripts/07_install_dgx_spark_ai_stack.sh`
- `whisperx_spark/`

## Installationsreihenfolge

Auf dem DGX Spark:

```bash
cd ~/voxtral-setup
chmod +x *.sh
./04_install_voxtral_dgx_spark_container.sh
./05_install_whisperx_dgx_spark.sh
./06_install_correction_llm_dgx_spark.sh   # nur falls das LLM auf dem Spark laufen soll
```

Alternativ als Sammelaufruf:

```bash
./07_install_dgx_spark_ai_stack.sh
```

## Ports

| Dienst | Port | Zweck |
|---|---:|---|
| Voxtral / vLLM | `8000` | OpenAI-kompatible Audio-Transkription |
| WhisperX | `7860` | Gradio-UI + API mit Timestamps |
| Korrektur-LLM | `8001` | optionale OpenAI-kompatible Textkorrektur |

## Starten und prüfen

```bash
sudo systemctl start voxtral-vllm whisperx
sudo systemctl status voxtral-vllm whisperx --no-pager -l
```

Voxtral:

```bash
curl http://127.0.0.1:8000/health
```

WhisperX:

```bash
curl -I http://127.0.0.1:7860
```

Optionales Korrektur-LLM:

```bash
curl http://127.0.0.1:8001/v1/models \
  -H "Authorization: Bearer local-correction-llm"
```

## WhisperX-API

Das Spark-Backend orientiert sich an der bisherigen WhisperX-Gradio-API.
Wichtiger Endpoint:

- `POST /gradio_api/call/start_process`

Im UI ist derselbe Prozess unter Port `7860` verfügbar.
Zusätzlich gibt es Admin-Funktionen für:

- `system_cleanup`
- `system_kill_zombies`
- `system_reboot`
- `system_pool_status`

## Parallelität und Sizing

Wichtiger Punkt: Alle drei Dienste teilen sich dieselbe GPU / denselben Gerätespeicher.
Darum sind konservative Defaults gesetzt. Wenn das Korrektur-LLM extern von der Schreibdienst-App verwaltet wird, bleiben auf dem Spark nur Voxtral und WhisperX aktiv.

### Empfohlene Defaults für den ersten Test

- `VOXTRAL_MAX_NUM_SEQS=4`
- `WHISPERX_POOL_SIZE=2`
- optional: `CORRECTION_LLM_MAX_NUM_SEQS=2`
- optional: `CORRECTION_LLM_MODEL=mistralai/Ministral-8B-Instruct-2410`

### Wenn du ein größeres Korrektur-LLM willst

Beispiel:

```bash
export CORRECTION_LLM_MODEL=<dein-großes-mistral-modell>
export CORRECTION_LLM_GPU_MEMORY_UTILIZATION=0.55
export WHISPERX_POOL_SIZE=1
export VOXTRAL_MAX_NUM_SEQS=2
./06_install_correction_llm_dgx_spark.sh
```

Die sichere Reihenfolge ist:

1. Erst alle drei Dienste mit kleinen Defaults stabil starten.
2. Dann WhisperX-Worker oder vLLM-Sequenzen langsam erhöhen.
3. Erst danach ein größeres Korrektur-LLM ausprobieren.

## WhisperX + Korrektur-LLM koppeln

Wenn das Korrektur-LLM auf dem Spark laufen soll, kann `05_install_whisperx_dgx_spark.sh` diese Anbindung in `.env` bekommen:

- `LLM_OPENAI_BASE_URL=http://127.0.0.1:8001/v1`
- `LLM_OPENAI_MODEL=correction-llm`
- `LLM_OPENAI_API_KEY=local-correction-llm`

Wenn diese Werte leer bleiben, läuft WhisperX ohne lokale LLM-Nachkontrolle.

## Troubleshooting

### WhisperX startet, aber keine GPU wird genutzt

```bash
source ~/whisperx-env/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### WhisperX lädt, aber Requests stauen sich

- `WHISPERX_POOL_SIZE` erhöhen
- Alignment auf CPU lassen (`WHISPERX_ALIGNMENT_DEVICE=cpu`)
- Großes Korrektur-LLM vorübergehend stoppen

### Zu wenig Speicher bei Gesamtstack

- optional: `sudo systemctl stop correction-llm`
- `WHISPERX_POOL_SIZE=1`
- `VOXTRAL_MAX_NUM_SEQS=1`
- kleineres `CORRECTION_LLM_MODEL` wählen
