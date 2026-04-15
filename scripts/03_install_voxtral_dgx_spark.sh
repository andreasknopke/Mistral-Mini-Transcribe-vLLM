#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

APP_DIR="${HOME}/voxtral-local"
VENV_DIR="${HOME}/voxtral-env"
SERVICE_NAME="voxtral"
MODEL_ID="${VOXTRAL_LOCAL_MODEL:-mistralai/Voxtral-Mini-3B-2507}"
HOST="${VOXTRAL_HOST:-0.0.0.0}"
PORT="${VOXTRAL_PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_BIN="${VENV_DIR}/bin/hf"

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
log " Voxtral Installation auf DGX Spark"
log "======================================"

echo "App-Verzeichnis: ${APP_DIR}"
echo "Venv:            ${VENV_DIR}"
echo "Modell:          ${MODEL_ID}"
echo "Host/Port:       ${HOST}:${PORT}"
echo ""

if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo wird benötigt."
fi

step "[1/7] System prüfen"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi nicht gefunden. NVIDIA-Treiber prüfen."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} nicht gefunden."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"${PYTHON_BIN}" --version

step "[2/7] Ubuntu-Pakete installieren"
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev build-essential ffmpeg libsndfile1 git curl

step "[3/7] Projektdateien kopieren"
mkdir -p "${APP_DIR}"
cp "${PWD}/voxtral_server.py" "${APP_DIR}/voxtral_server.py"

step "[4/7] Python-Umgebung vorbereiten"
if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"

step "[5/7] Python-Abhängigkeiten installieren"
pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
pip install \
  "numpy<3" \
  "soundfile" \
  "librosa" \
  "fastapi" \
  "uvicorn[standard]" \
  "python-multipart" \
    "mistral-common[audio]" \
  "transformers>=4.53.0" \
  "accelerate>=1.8.0" \
  "sentencepiece" \
  "safetensors" \
  "huggingface_hub[cli]"

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA-Version: {torch.version.cuda}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA ist in PyTorch nicht verfügbar. Nutze als Fallback das NVIDIA-NGC-Container-Setup.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

step "[6/7] Hugging Face Login prüfen"
if "${HF_BIN}" auth whoami >/dev/null 2>&1; then
    echo "Hugging Face Login vorhanden: $("${HF_BIN}" auth whoami | head -1)"
else
    echo "Bitte jetzt mit deinem Hugging Face Token anmelden."
    echo "Vorher Lizenz freischalten: https://huggingface.co/${MODEL_ID}"
    "${HF_BIN}" auth login
fi

step "[7/7] systemd Service einrichten"
SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Voxtral Local Speech-to-Text Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${APP_DIR}
Environment=VOXTRAL_LOCAL_MODEL=${MODEL_ID}
Environment=VOXTRAL_HOST=${HOST}
Environment=VOXTRAL_PORT=${PORT}
Environment=HF_HOME=${HOME}/.cache/huggingface
ExecStart=${VENV_DIR}/bin/python ${APP_DIR}/voxtral_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

echo ""
log "Installation abgeschlossen."
echo "Server starten:   sudo systemctl start ${SERVICE_NAME}"
echo "Status anzeigen:  sudo systemctl status ${SERVICE_NAME} --no-pager"
echo "Logs anzeigen:    journalctl -u ${SERVICE_NAME} -f"
echo "Health-Check:     curl http://127.0.0.1:${PORT}/health"
echo "Container-Fallback: siehe scripts/04_install_voxtral_dgx_spark_container.sh"
