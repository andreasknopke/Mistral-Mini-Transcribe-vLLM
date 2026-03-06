#!/bin/bash
# ============================================================
# Voxtral Realtime-Modell (4B) starten
# Für Live-Diktat via WebSocket (ws://localhost:8000/v1/realtime)
# ============================================================

set -e

echo "======================================"
echo " Voxtral Realtime Server (4B) starten"
echo "======================================"

# Environment aktivieren
source "$HOME/voxtral-env/bin/activate"

echo "Modell:    mistralai/Voxtral-Mini-4B-Realtime-2602"
echo "Port:      8000"
echo "WebSocket: ws://localhost:8000/v1/realtime"
echo ""
echo "Erster Start dauert 5-10 Minuten (Download ~8 GB)"
echo "Stoppen mit Ctrl+C"
echo ""

vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --max-model-len 8192 \
  --enforce-eager \
  --dtype half \
  --gpu-memory-utilization 0.9
