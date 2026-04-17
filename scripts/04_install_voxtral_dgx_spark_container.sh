#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HOME}/voxtral-vllm"
IMAGE_NAME="voxtral-vllm-dgx:latest"
SERVICE_NAME="voxtral-vllm"
BASE_IMAGE="${VOXTRAL_VLLM_BASE_IMAGE:-nvcr.io/nvidia/vllm:26.03-py3}"
MODEL_ID="${VOXTRAL_LOCAL_MODEL:-mistralai/Voxtral-Mini-3B-2507}"
HOST_PORT="${VOXTRAL_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${VOXTRAL_GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${VOXTRAL_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VOXTRAL_MAX_NUM_SEQS:-4}"
FORCE_REBUILD="${VOXTRAL_FORCE_REBUILD:-0}"

run_sudo() {
  if [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

echo "======================================"
echo " Voxtral vLLM auf DGX Spark installieren"
echo "======================================"
echo ""
echo "Modell:                  ${MODEL_ID}"
echo "Port:                    ${HOST_PORT}"
echo "Max parallele Seqs:      ${MAX_NUM_SEQS}"
echo "GPU Memory Utilization:  ${GPU_MEMORY_UTILIZATION}"
echo "Base Image:              ${BASE_IMAGE}"
echo "Force Rebuild:           ${FORCE_REBUILD}"
echo ""

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi nicht gefunden. NVIDIA-Treiber prüfen."
  exit 1
fi

echo "[1/6] Docker + NVIDIA Container Toolkit prüfen"
if ! command -v docker >/dev/null 2>&1; then
  run_sudo apt update
  run_sudo apt install -y docker.io nvidia-container-toolkit
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  run_sudo apt update
  run_sudo apt install -y nvidia-container-toolkit
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
  run_sudo nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
fi

run_sudo systemctl enable docker >/dev/null 2>&1 || true
run_sudo systemctl restart docker

echo "[2/6] NGC Login prüfen"
if ! run_sudo docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  NGC_LOGIN_TOKEN="${NGC_API_KEY:-${NGC_TOKEN:-}}"
  if [ -z "${NGC_LOGIN_TOKEN}" ]; then
    echo "Bitte NGC API Key eingeben (https://org.ngc.nvidia.com/setup/api-key):"
    read -r -s NGC_LOGIN_TOKEN
    echo ""
  fi
  run_sudo /bin/sh -c "printf '%s' $(printf '%q' "${NGC_LOGIN_TOKEN}") | docker login nvcr.io --username '\$oauthtoken' --password-stdin"
fi

echo "[3/6] Hugging Face Token prüfen"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Bitte Hugging Face Token mit Read-Recht eingeben:"
  read -r -s HF_TOKEN
  echo ""
fi

mkdir -p "${APP_DIR}"
mkdir -p "${HOME}/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SOURCE="${SCRIPT_DIR}/voxtral_vllm_proxy.py"
if [ ! -f "${PROXY_SOURCE}" ]; then
  echo "Proxy-Datei nicht gefunden: ${PROXY_SOURCE}"
  exit 1
fi
cp "${PROXY_SOURCE}" "${APP_DIR}/voxtral_vllm_proxy.py"

cat > "${APP_DIR}/Dockerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip "setuptools<82" && \
  python -m pip install "fastapi" "uvicorn[standard]" "httpx" "python-multipart" "mistral-common[audio]" "huggingface_hub[hf_transfer]"
RUN python - <<'PY'
from pathlib import Path
import re

voxtral_path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/voxtral.py')
text = voxtral_path.read_text(encoding='utf-8')

new = "\n".join([
  "        window = torch.hann_window(",
  "            self.config.window_size, device=\"cpu\"",
  "        )",
  "        stft = torch.stft(",
  "            audio_waveforms.float().cpu(),",
  "            self.config.window_size,",
  "            self.config.hop_length,",
  "            window=window,",
  "            return_complex=True,",
  "        )",
  "        magnitudes = stft[..., :-1].abs() ** 2",
  "        magnitudes = magnitudes.to(",
  "            device=audio_waveforms.device, dtype=self.mel_filters.dtype",
  "        )",
  "        mel_spec = self.mel_filters.T @ magnitudes",
])

pattern = re.compile(
  r"""        window = torch\.hann_window\(
\s+self\.config\.window_size, device=audio_waveforms\.device
\s+\)
\s+stft = torch\.stft\(
\s+audio_waveforms,
\s+self\.config\.window_size,
\s+self\.config\.hop_length,
\s+window=window,
\s+return_complex=True,
\s+\)
\s+magnitudes = stft\[\.\.\., :-1\]\.abs\(\) \*\* 2
\s+mel_spec = self\.mel_filters\.T @ magnitudes"""
)

text, count = pattern.subn(new, text, count=1)
if count != 1:
  raise SystemExit(f'cuFFT patch pattern not found in {voxtral_path}')

voxtral_path.write_text(text, encoding='utf-8')
print(f'Patched {voxtral_path}')
PY
COPY voxtral_vllm_proxy.py /app/voxtral_vllm_proxy.py
EXPOSE 8000
EOF

echo "[4/6] Docker-Image bauen"
if [ "${FORCE_REBUILD}" = "1" ] || ! run_sudo docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  run_sudo docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${IMAGE_NAME}" "${APP_DIR}"
else
  echo "Lokales Image ${IMAGE_NAME} bereits vorhanden - überspringe Rebuild"
fi

cat > "${APP_DIR}/vllm.env" <<EOF
HF_TOKEN=${HF_TOKEN}
HF_HOME=/root/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VOXTRAL_VLLM_UPSTREAM_URL=http://127.0.0.1:8001
EOF

cat > "${APP_DIR}/run_vllm.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/docker run --rm \
  --name ${SERVICE_NAME} \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  --env-file ${APP_DIR}/vllm.env \
  -p ${HOST_PORT}:8000 \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  ${IMAGE_NAME} \
  /bin/bash -lc 'python /app/voxtral_vllm_proxy.py & exec vllm serve ${MODEL_ID} \
  --host 127.0.0.1 \
  --port 8001 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --enforce-eager \
  --dtype half \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS}'
EOF
chmod +x "${APP_DIR}/run_vllm.sh"

echo "[5/6] systemd Service einrichten"
run_sudo systemctl disable --now voxtral >/dev/null 2>&1 || true
run_sudo systemctl disable --now ${SERVICE_NAME} >/dev/null 2>&1 || true
run_sudo docker rm -f ${SERVICE_NAME} >/dev/null 2>&1 || true

SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Voxtral vLLM Server (DGX Spark)
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f ${SERVICE_NAME}
ExecStart=/bin/bash ${APP_DIR}/run_vllm.sh
ExecStop=/usr/bin/docker stop ${SERVICE_NAME}
Restart=always
RestartSec=5
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

run_sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
run_sudo systemctl daemon-reload
run_sudo systemctl enable "${SERVICE_NAME}"

echo "[6/6] Fertig"
echo ""
echo "Server starten:   sudo systemctl start ${SERVICE_NAME}"
echo "Status anzeigen:  sudo systemctl status ${SERVICE_NAME} --no-pager -l"
echo "Logs anzeigen:    journalctl -u ${SERVICE_NAME} -f"
echo "Health-Check:     curl http://127.0.0.1:${HOST_PORT}/health"
echo "Audio API:        http://127.0.0.1:${HOST_PORT}/v1/audio/transcriptions"
echo ""
echo "Hinweis: Dieser Pfad ist für parallele Benutzer mit vLLM gedacht."
echo "Falls du lieber den direkten Transformers-Server willst, nutze 03_install_voxtral_dgx_spark.sh"
