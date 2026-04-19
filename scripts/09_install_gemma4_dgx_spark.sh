#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Gemma 4 26B-A4B MoE (NVFP4) auf DGX Spark installieren
#  Modell: bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
#  - 25.2B total params, 3.8B aktiv pro Token (128 Experts, Top-8)
#  - NVFP4 W4A4 Quantisierung → ~16.5 GB auf Disk / ~15.7 GiB GPU
#  - Getestet auf DGX Spark GB10 Blackwell SM 12.1
#  - Benötigt vLLM mit transformers >= 5.4 + gemma4_patched.py
# ============================================================================

STACK_OWNER="${SPARK_STACK_OWNER:-${SUDO_USER:-${USER:-$(id -un)}}}"
STACK_HOME="${SPARK_STACK_HOME:-/home/${STACK_OWNER}}"
APP_DIR="${STACK_HOME}/correction-llm-vllm"
SERVICE_NAME="correction-llm"
MODEL_ID="${GEMMA4_MODEL:-bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4}"
# Beide Modellnamen registrieren, damit bestehende Clients
# sowohl "gemma-4" als auch "correction-llm" als model= nutzen können.
SERVED_MODEL_NAMES="${GEMMA4_SERVED_NAMES:-gemma-4 correction-llm}"
HOST_PORT="${GEMMA4_PORT:-9000}"
GEMMA4_PROFILE="${GEMMA4_PROFILE:-spark-shared}"

# ---------- Profil-Defaults ----------
DEFAULT_GPU_MEMORY_UTILIZATION="0.30"
DEFAULT_MAX_MODEL_LEN="32768"
DEFAULT_MAX_NUM_SEQS="4"

if [ "${GEMMA4_PROFILE}" = "exclusive" ]; then
  DEFAULT_GPU_MEMORY_UTILIZATION="0.85"
  DEFAULT_MAX_MODEL_LEN="131072"
  DEFAULT_MAX_NUM_SEQS="4"
fi
if [ "${GEMMA4_PROFILE}" = "max-context" ]; then
  DEFAULT_GPU_MEMORY_UTILIZATION="0.85"
  DEFAULT_MAX_MODEL_LEN="262144"
  DEFAULT_MAX_NUM_SEQS="1"
fi

GPU_MEMORY_UTILIZATION="${GEMMA4_GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
MAX_MODEL_LEN="${GEMMA4_MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${GEMMA4_MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
FORCE_REBUILD="${GEMMA4_FORCE_REBUILD:-0}"

# ---------- Base-Image ----------
# Das Modell braucht transformers >= 5.4 für die Gemma 4-Architektur.
# Option 1: spark-vllm-docker mit --tf5  (empfohlen, siehe github.com/eugr/spark-vllm-docker)
# Option 2: Standard NGC vLLM-Image + pip install im Dockerfile
BASE_IMAGE="${GEMMA4_VLLM_BASE_IMAGE:-nvcr.io/nvidia/vllm:26.03-py3}"

run_sudo() {
  if [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

echo "================================================"
echo " Gemma 4 26B-A4B MoE (NVFP4) auf DGX Spark"
echo "================================================"
echo ""
echo "Modell:                  ${MODEL_ID}"
echo "Served Names:            ${SERVED_MODEL_NAMES}"
echo "Profil:                  ${GEMMA4_PROFILE}"
echo "Port:                    ${HOST_PORT}"
echo "Max parallele Seqs:      ${MAX_NUM_SEQS}"
echo "Max Context Len:         ${MAX_MODEL_LEN}"
echo "GPU Memory Utilization:  ${GPU_MEMORY_UTILIZATION}"
echo "Base Image:              ${BASE_IMAGE}"
echo ""

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi nicht gefunden. NVIDIA-Treiber prüfen."
  exit 1
fi

# ---- [1/7] Docker + NVIDIA Container Toolkit ----
echo "[1/7] Docker + NVIDIA Container Toolkit prüfen"
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

# ---- [2/7] NGC Login ----
echo "[2/7] NGC Login prüfen"
if ! run_sudo docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  NGC_LOGIN_TOKEN="${NGC_API_KEY:-${NGC_TOKEN:-}}"
  if [ -z "${NGC_LOGIN_TOKEN}" ]; then
    echo "Bitte NGC API Key eingeben (https://org.ngc.nvidia.com/setup/api-key):"
    read -r -s NGC_LOGIN_TOKEN
    echo ""
  fi
  run_sudo /bin/sh -c "printf '%s' $(printf '%q' "${NGC_LOGIN_TOKEN}") | docker login nvcr.io --username '\$oauthtoken' --password-stdin"
fi

# ---- [3/7] Hugging Face Token ----
echo "[3/7] Hugging Face Token prüfen"
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

# ---- [4/7] Modell vorab downloaden + gemma4_patched.py extrahieren ----
echo "[4/7] Modell und Patch-Datei vorbereiten"

# Wir laden das Modell vorab, damit der Patch im Image verfügbar ist.
# Die gemma4_patched.py liegt im HF-Repo des Modells.
PATCH_FILE="${APP_DIR}/gemma4_patched.py"
if [ ! -f "${PATCH_FILE}" ]; then
  echo "  → gemma4_patched.py aus dem Modell-Repo herunterladen ..."
  run_sudo docker run --rm \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HOME=/root/.cache/huggingface \
    -v "${STACK_HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${APP_DIR}:/output" \
    "${BASE_IMAGE}" \
    python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('${MODEL_ID}', 'gemma4_patched.py')
shutil.copy(path, '/output/gemma4_patched.py')
print(f'Copied {path} → /output/gemma4_patched.py')
"
  echo "  → gemma4_patched.py bereit: ${PATCH_FILE}"
fi

# ---- [5/7] Docker-Image bauen ----
IMAGE_NAME="gemma4-vllm-dgx:latest"

cat > "${APP_DIR}/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Gemma 4 benötigt transformers >= 5.4
RUN python -m pip install --upgrade pip "setuptools<82" && \
    python -m pip install "transformers>=5.4" "huggingface_hub[hf_transfer]"

# gemma4_patched.py für NVFP4 MoE scale key loading
COPY gemma4_patched.py /opt/gemma4_patched.py

EXPOSE 8000
DOCKERFILE

echo "[5/7] Docker-Image bauen"
if [ "${FORCE_REBUILD}" = "1" ] || ! run_sudo docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  run_sudo docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${IMAGE_NAME}" "${APP_DIR}"
else
  echo "Lokales Image ${IMAGE_NAME} bereits vorhanden - überspringe Rebuild"
fi

# ---- [6/7] Run-Script und Service ----
cat > "${APP_DIR}/llm.env" <<EOF
HF_TOKEN=${HF_TOKEN}
HF_HOME=/root/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_NVFP4_GEMM_BACKEND=marlin
EOF

# Patch-Ziel im Container (vLLM's gemma4.py ersetzen)
VLLM_GEMMA4_PATH="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py"

# --served-model-name: vLLM akzeptiert kommagetrennte Liste für Aliase
SERVED_NAMES_CSV="$(echo ${SERVED_MODEL_NAMES} | tr ' ' ',')"

cat > "${APP_DIR}/run_gemma4.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/docker run --rm \\
  --name ${SERVICE_NAME} \\
  --gpus all \\
  --ipc=host \\
  --ulimit memlock=-1 \\
  --shm-size=16g \\
  --env-file ${APP_DIR}/llm.env \\
  -p ${HOST_PORT}:8000 \\
  -v ${STACK_HOME}/.cache/huggingface:/root/.cache/huggingface \\
  -v ${APP_DIR}/gemma4_patched.py:${VLLM_GEMMA4_PATH} \\
  ${IMAGE_NAME} \\
  vllm serve ${MODEL_ID} \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --served-model-name ${SERVED_NAMES_CSV} \\
  --quantization modelopt \\
  --dtype auto \\
  --kv-cache-dtype fp8 \\
  --moe-backend marlin \\
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --max-num-seqs ${MAX_NUM_SEQS} \\
  --trust-remote-code
EOF
chmod +x "${APP_DIR}/run_gemma4.sh"

echo "[6/7] systemd Service einrichten"
# Alten correction-llm UND alten gemma4-llm Service stoppen
run_sudo systemctl disable --now correction-llm >/dev/null 2>&1 || true
run_sudo systemctl disable --now gemma4-llm >/dev/null 2>&1 || true
run_sudo docker rm -f correction-llm >/dev/null 2>&1 || true
run_sudo docker rm -f gemma4-llm >/dev/null 2>&1 || true

SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Gemma 4 26B-A4B NVFP4 MoE Server (DGX Spark)
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f ${SERVICE_NAME}
ExecStart=/bin/bash ${APP_DIR}/run_gemma4.sh
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

# ---- [7/7] Fertig ----
echo "[7/7] Fertig"
echo ""
echo "========================================"
echo " Gemma 4 26B-A4B MoE NVFP4 installiert"
echo "========================================"
echo ""
echo "Server starten:   sudo systemctl start ${SERVICE_NAME}"
echo "Status anzeigen:  sudo systemctl status ${SERVICE_NAME} --no-pager -l"
echo "Logs anzeigen:    journalctl -u ${SERVICE_NAME} -f"
echo "Models API:       curl http://127.0.0.1:${HOST_PORT}/v1/models"
echo ""
echo "Served Model-Namen: ${SERVED_MODEL_NAMES}"
echo "Dein Sprachprogramm kann model=\"gemma-4\" oder model=\"correction-llm\" verwenden."
echo ""
echo "Chat testen:"
echo "  curl http://127.0.0.1:${HOST_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo '      "model": "gemma-4",'
echo '      "messages": [{"role": "user", "content": "Hallo! Erzähl mir einen Witz."}],'
echo '      "max_tokens": 200'
echo "    }'"
echo ""
echo "Profil-Info:"
echo "  spark-shared  (default) → 30% GPU, 32K ctx – parallel mit Voxtral nutzbar"
echo "  exclusive               → 85% GPU, 128K ctx – allein auf dem Spark"
echo "  max-context             → 85% GPU, 256K ctx – maximale Kontextlänge"
echo ""
echo "Memory-Footprint: ~15.7 GiB Modell + FP8 KV-Cache"
echo "Das Modell lässt massig Platz für parallelen Betrieb mit Voxtral + WhisperX!"
