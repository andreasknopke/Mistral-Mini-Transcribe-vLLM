#!/usr/bin/env bash
set -euo pipefail

STACK_OWNER="${SPARK_STACK_OWNER:-${SUDO_USER:-${USER:-$(id -un)}}}"
STACK_HOME="${SPARK_STACK_HOME:-/home/${STACK_OWNER}}"
APP_DIR="${STACK_HOME}/correction-llm-vllm"
IMAGE_NAME="correction-llm-vllm:latest"
SERVICE_NAME="correction-llm"
BASE_IMAGE="${CORRECTION_LLM_VLLM_BASE_IMAGE:-nvcr.io/nvidia/vllm:26.03-py3}"
CORRECTION_LLM_PROFILE="${CORRECTION_LLM_PROFILE:-spark-shared}"
DEFAULT_MODEL="mistral-small-24b-nvfp4"
DEFAULT_GPU_MEMORY_UTILIZATION="0.50"
DEFAULT_MAX_MODEL_LEN="8192"
DEFAULT_MAX_NUM_SEQS="4"
DEFAULT_QUANTIZATION="compressed-tensors"
DEFAULT_KV_CACHE_DTYPE="fp8"
DEFAULT_DTYPE="auto"
if [ "${CORRECTION_LLM_PROFILE}" = "exclusive" ]; then
  DEFAULT_GPU_MEMORY_UTILIZATION="0.85"
  DEFAULT_MAX_MODEL_LEN="32768"
  DEFAULT_MAX_NUM_SEQS="8"
fi
REQUESTED_MODEL="${CORRECTION_LLM_MODEL:-${DEFAULT_MODEL}}"
MODEL_ID="${REQUESTED_MODEL}"
QUANTIZATION="${DEFAULT_QUANTIZATION}"
KV_CACHE_DTYPE="${DEFAULT_KV_CACHE_DTYPE}"
DTYPE="${DEFAULT_DTYPE}"
if [ "${REQUESTED_MODEL}" = "mistral-small-24b-nvfp4" ]; then
  MODEL_ID="llmat/Mistral-Small-24B-Instruct-2501-NVFP4"
fi
SERVED_MODEL_NAME="${CORRECTION_LLM_SERVED_NAME:-${REQUESTED_MODEL}}"
HOST_PORT="${CORRECTION_LLM_PORT:-9000}"
GPU_MEMORY_UTILIZATION="${CORRECTION_LLM_GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
MAX_MODEL_LEN="${CORRECTION_LLM_MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${CORRECTION_LLM_MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
FORCE_REBUILD="${CORRECTION_LLM_FORCE_REBUILD:-0}"
API_KEY="${CORRECTION_LLM_API_KEY:-}"
if [ -n "${API_KEY}" ]; then
  API_KEY_FLAG="--api-key ${API_KEY}"
else
  API_KEY_FLAG=""
fi

run_sudo() {
  if [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

echo "==============================================="
echo " Korrektur-LLM (OpenAI-kompatibel) installieren"
echo "==============================================="
echo ""
echo "Modell:                  ${MODEL_ID}"
echo "Profil:                  ${CORRECTION_LLM_PROFILE}"
echo "Served name:             ${SERVED_MODEL_NAME}"
echo "Port:                    ${HOST_PORT}"
echo "Max parallele Seqs:      ${MAX_NUM_SEQS}"
echo "GPU Memory Utilization:  ${GPU_MEMORY_UTILIZATION}"
echo "Base Image:              ${BASE_IMAGE}"
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
  if [ -f "${STACK_HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '\r\n' < "${STACK_HOME}/.cache/huggingface/token")"
  fi
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Bitte Hugging Face Token mit Read-Recht eingeben:"
  read -r -s HF_TOKEN
  echo ""
fi

mkdir -p "${APP_DIR}"
mkdir -p "${STACK_HOME}/.cache/huggingface"

cat > "${APP_DIR}/sitecustomize.py" <<'EOF'
import os


def _should_fix(pretrained_model_name_or_path):
  if os.environ.get("FIX_MISTRAL_REGEX", "1") != "1":
    return False
  model_name = str(pretrained_model_name_or_path).lower()
  return "mistral" in model_name or "qwen" in model_name


def _patch_loader(auto_cls):
  original = auto_cls.from_pretrained

  def wrapped(pretrained_model_name_or_path, *args, **kwargs):
    if (
      _should_fix(pretrained_model_name_or_path)
      and "fix_mistral_regex" not in kwargs
    ):
      kwargs["fix_mistral_regex"] = True
    return original(pretrained_model_name_or_path, *args, **kwargs)

  auto_cls.from_pretrained = wrapped


try:
  from transformers import AutoProcessor, AutoTokenizer

  _patch_loader(AutoTokenizer)
  _patch_loader(AutoProcessor)
except Exception:
  pass
EOF

cat > "${APP_DIR}/Dockerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONPATH=/opt/correction-llm-patches${PYTHONPATH:+:${PYTHONPATH}}
RUN python -m pip install --upgrade pip "setuptools<82" && \
  python -m pip install "huggingface_hub[hf_transfer]"
COPY sitecustomize.py /opt/correction-llm-patches/sitecustomize.py
EXPOSE 8000
EOF

echo "[4/6] Docker-Image bauen"
if [ "${FORCE_REBUILD}" = "1" ] || ! run_sudo docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  run_sudo docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${IMAGE_NAME}" "${APP_DIR}"
else
  echo "Lokales Image ${IMAGE_NAME} bereits vorhanden - überspringe Rebuild"
fi

cat > "${APP_DIR}/llm.env" <<EOF
HF_TOKEN=${HF_TOKEN}
HF_HOME=/root/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
FIX_MISTRAL_REGEX=1
EOF

cat > "${APP_DIR}/run_llm.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/docker run --rm \
  --name ${SERVICE_NAME} \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  --env-file ${APP_DIR}/llm.env \
  -p ${HOST_PORT}:8000 \
  -v ${STACK_HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ${APP_DIR}/sitecustomize.py:/opt/correction-llm-patches/sitecustomize.py:ro \
  ${IMAGE_NAME} \
  vllm serve ${MODEL_ID} \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name ${SERVED_MODEL_NAME} \
  ${API_KEY_FLAG} \
  --quantization ${QUANTIZATION} \
  --dtype ${DTYPE} \
  --kv-cache-dtype ${KV_CACHE_DTYPE} \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --tokenizer-mode hf \
  --chat-template-content-format string \
  --trust-remote-code
EOF
chmod +x "${APP_DIR}/run_llm.sh"

echo "[5/6] systemd Service einrichten"
run_sudo systemctl disable --now ${SERVICE_NAME} >/dev/null 2>&1 || true
run_sudo docker rm -f ${SERVICE_NAME} >/dev/null 2>&1 || true

SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Correction LLM vLLM Server (DGX Spark)
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f ${SERVICE_NAME}
ExecStart=/bin/bash ${APP_DIR}/run_llm.sh
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
if [ -n "${API_KEY}" ]; then
  echo "Models API:       curl http://127.0.0.1:${HOST_PORT}/v1/models -H \"Authorization: Bearer ${API_KEY}\""
else
  echo "Models API:       curl http://127.0.0.1:${HOST_PORT}/v1/models"
fi
echo "Hinweis: Default-Profil 'spark-shared' nutzt Mistral-Small-24B NVFP4 (~15 GiB) mit 30% GPU-Speicher für parallelen Betrieb mit Voxtral."
echo "Für exklusiven Betrieb mit maximalem Kontext: CORRECTION_LLM_PROFILE=exclusive"
