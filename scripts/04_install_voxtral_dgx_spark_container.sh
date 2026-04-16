#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HOME}/voxtral-vllm"
IMAGE_NAME="voxtral-vllm-dgx:latest"
SERVICE_NAME="voxtral-vllm"
BASE_IMAGE="${VOXTRAL_VLLM_BASE_IMAGE:-nvcr.io/nvidia/vllm:26.03-py3}"
MODEL_ID="${VOXTRAL_LOCAL_MODEL:-mistralai/Voxtral-Mini-3B-2507}"
HOST_PORT="${VOXTRAL_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${VOXTRAL_GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${VOXTRAL_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${VOXTRAL_MAX_NUM_SEQS:-4}"
FORCE_REBUILD="${VOXTRAL_FORCE_REBUILD:-0}"

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
  sudo apt update
  sudo apt install -y docker.io nvidia-container-toolkit
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y nvidia-container-toolkit
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
  sudo nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
fi

sudo systemctl enable docker >/dev/null 2>&1 || true
sudo systemctl restart docker

echo "[2/6] NGC Login prüfen"
if ! sudo docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  if [ -z "${NGC_API_KEY:-}" ]; then
    echo "Bitte NGC API Key eingeben (https://org.ngc.nvidia.com/setup/api-key):"
    read -r -s NGC_API_KEY
    echo ""
  fi
  printf '%s' "${NGC_API_KEY}" | sudo docker login nvcr.io --username '$oauthtoken' --password-stdin
fi

echo "[3/6] Hugging Face Token prüfen"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Bitte Hugging Face Token mit Read-Recht eingeben:"
  read -r -s HF_TOKEN
  echo ""
fi

mkdir -p "${APP_DIR}"
mkdir -p "${HOME}/.cache/huggingface"

cat > "${APP_DIR}/Dockerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip "setuptools<82" && \
  python -m pip install "mistral-common[audio]" "huggingface_hub[hf_transfer]"
EXPOSE 8000
EOF

echo "[4/6] Docker-Image bauen"
if [ "${FORCE_REBUILD}" = "1" ] || ! sudo docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  sudo docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${IMAGE_NAME}" "${APP_DIR}"
else
  echo "Lokales Image ${IMAGE_NAME} bereits vorhanden - überspringe Rebuild"
fi

cat > "${APP_DIR}/vllm.env" <<EOF
HF_TOKEN=${HF_TOKEN}
HF_HOME=/root/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
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
  vllm serve ${MODEL_ID} \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --enforce-eager \
  --dtype half \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS}
EOF
chmod +x "${APP_DIR}/run_vllm.sh"

echo "[5/6] systemd Service einrichten"
sudo systemctl disable --now voxtral >/dev/null 2>&1 || true
sudo systemctl disable --now ${SERVICE_NAME} >/dev/null 2>&1 || true
sudo docker rm -f ${SERVICE_NAME} >/dev/null 2>&1 || true

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

sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

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
