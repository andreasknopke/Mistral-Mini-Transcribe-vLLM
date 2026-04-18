#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/docker run --rm \
  --name correction-llm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  --env-file /home/ksai0001_local/correction-llm-vllm/llm.env \
  -p 9000:8000 \
  -v /home/ksai0001_local/.cache/huggingface:/root/.cache/huggingface \
  correction-llm-vllm:latest \
  vllm serve llmat/Mistral-Small-24B-Instruct-2501-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name correction-llm \
  --gpu-memory-utilization 0.25 \
  --max-model-len 8192 \
  --max-num-seqs 4
