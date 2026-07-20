#!/usr/bin/env bash
# run_vllm.sh - Voxtral vLLM mit nativ gemergtem Mistral-Modell
# 2026-07-20: LoRA -> native Mistral merge (merge_lora_native_mistral.py)
#             voxtral_patched.py NICHT mehr noetig da Mistral-Format = original
set -euo pipefail

exec /usr/bin/docker run --rm \
  --name voxtral-vllm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  --env-file /home/owc/voxtral-vllm/vllm.env \
  -p 8000:8000 \
  -v /home/owc/.cache/huggingface:/root/.cache/huggingface \
  -v /home/owc/voxtral-consolidated:/models/merged:ro \
  -v /home/owc/voxtral-setup/voxtral_vllm_proxy.py:/app/voxtral_vllm_proxy.py:ro \
  -v /home/owc/voxtral-vllm/entrypoint_merged.sh:/app/entrypoint_merged.sh:ro \
  voxtral-vllm-dgx:latest \
  /bin/bash /app/entrypoint_merged.sh
