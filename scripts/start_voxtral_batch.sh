#!/bin/bash
# ============================================================
# Voxtral Batch-Modell (3B) starten
# Für Datei-Transkription (POST /v1/audio/transcriptions)
# ============================================================

set -e

echo "======================================"
echo " Voxtral Batch Server (3B) starten"
echo "======================================"

# Environment aktivieren
source "$HOME/voxtral-env/bin/activate"

echo "Modell: mistralai/Voxtral-Mini-3B-2507"
echo "Port:   8000"
echo "API:    http://localhost:8000/v1/audio/transcriptions"
echo ""
echo "Erster Start dauert 5-15 Minuten (Modell-Download ~6 GB)"
echo "Stoppen mit Ctrl+C"
echo ""

vllm serve mistralai/Voxtral-Mini-3B-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --max-model-len 8192 \
  --enforce-eager \
  --dtype half \
  --gpu-memory-utilization 0.9
