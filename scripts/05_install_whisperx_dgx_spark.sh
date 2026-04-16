#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

APP_DIR="${HOME}/whisperx-spark"
VENV_DIR="${HOME}/whisperx-env"
SERVICE_NAME="whisperx"
HOST="${WHISPERX_HOST:-0.0.0.0}"
PORT="${WHISPERX_PORT:-7860}"
MODEL_NAME="${WHISPERX_MODEL:-large-v3}"
POOL_SIZE="${WHISPERX_POOL_SIZE:-2}"
DEVICE="${WHISPERX_DEVICE:-cuda}"
ALIGNMENT_DEVICE="${WHISPERX_ALIGNMENT_DEVICE:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LLM_OPENAI_BASE_URL="${LLM_OPENAI_BASE_URL:-}"
LLM_OPENAI_MODEL="${LLM_OPENAI_MODEL:-}"
LLM_OPENAI_API_KEY="${LLM_OPENAI_API_KEY:-}"

log() {
    echo -e "${CYAN}$1${NC}"
}

step() {
    echo -e "${YELLOW}$1${NC}"
}

fail() {
    echo -e "${RED}$1${NC}"
    exit 1
}

log "======================================"
log " WhisperX auf DGX Spark installieren"
log "======================================"

echo "App-Verzeichnis: ${APP_DIR}"
echo "Venv:            ${VENV_DIR}"
echo "Modell:          ${MODEL_NAME}"
echo "Pool Size:       ${POOL_SIZE}"
echo "Host/Port:       ${HOST}:${PORT}"
echo "Device:          ${DEVICE}"
echo "Alignment:       ${ALIGNMENT_DEVICE}"
echo "LLM Endpoint:    ${LLM_OPENAI_BASE_URL:-extern verwaltet / optional}"
echo ""

if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo wird benötigt."
fi

step "[1/7] System prüfen"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi nicht gefunden. NVIDIA-Treiber prüfen."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} nicht gefunden."
[ -d "${PWD}/whisperx_spark" ] || fail "whisperx_spark Verzeichnis nicht gefunden. Bitte zuerst Deploy-Skript ausführen."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" --version

step "[2/7] Ubuntu-Pakete installieren"
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev build-essential ffmpeg libsndfile1 git curl

step "[3/7] Projektdateien kopieren"
rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}"
cp -R "${PWD}/whisperx_spark/." "${APP_DIR}/"

step "[4/7] Python-Umgebung vorbereiten"
if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"

step "[5/7] Python-Abhängigkeiten installieren"
pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
pip install -r "${APP_DIR}/requirements/requirements_cuda.txt"

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA-Version: {torch.version.cuda}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA ist in PyTorch nicht verfügbar. WhisperX würde sonst nur im CPU-Modus laufen.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

step "[6/7] .env konfigurieren"
cat > "${APP_DIR}/.env" <<EOF
WHISPER_AUTH_USERNAME=${WHISPER_AUTH_USERNAME:-admin}
WHISPER_AUTH_PASSWORD=${WHISPER_AUTH_PASSWORD:-change-me}
WHISPERX_MODEL=${MODEL_NAME}
WHISPERX_DEVICE=${DEVICE}
WHISPERX_POOL_SIZE=${POOL_SIZE}
WHISPERX_ALIGNMENT_DEVICE=${ALIGNMENT_DEVICE}
WHISPERX_HOST=${HOST}
WHISPERX_PORT=${PORT}
WHISPERX_ESTIMATED_WORKER_GB=${WHISPERX_ESTIMATED_WORKER_GB:-6}
LLM_TIMEOUT=${LLM_TIMEOUT:-120}
LLM_OPENAI_BASE_URL=${LLM_OPENAI_BASE_URL}
LLM_OPENAI_MODEL=${LLM_OPENAI_MODEL}
LLM_OPENAI_API_KEY=${LLM_OPENAI_API_KEY}
OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-}
OLLAMA_MODEL=${OLLAMA_MODEL:-}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
EOF

step "[7/7] systemd Service einrichten"
SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=WhisperX Server (DGX Spark)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${APP_DIR}
Environment=HF_HOME=${HOME}/.cache/huggingface
EnvironmentFile=${APP_DIR}/.env
ExecStart=${VENV_DIR}/bin/python ${APP_DIR}/app.py
Restart=always
RestartSec=5
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

echo ""
log "Installation abgeschlossen."
echo "Server starten:   sudo systemctl start ${SERVICE_NAME}"
echo "Status anzeigen:  sudo systemctl status ${SERVICE_NAME} --no-pager -l"
echo "Logs anzeigen:    journalctl -u ${SERVICE_NAME} -f"
echo "UI/API:           http://127.0.0.1:${PORT}"
echo "Gradio API:       http://127.0.0.1:${PORT}/gradio_api/openapi.json"
echo "Hinweis: Für mehr Parallelität WHISPERX_POOL_SIZE erhöhen, aber VRAM-/RAM-Budget im Blick behalten."
echo "Hinweis: Das Korrektur-LLM ist optional und kann extern von der Schreibdienst-App verwaltet werden."
