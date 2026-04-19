#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

APP_DIR="${HOME}/vibevoice-spark"
VENV_DIR="${HOME}/vibevoice-env"
SERVICE_NAME="vibevoice"
HOST="${VIBEVOICE_HOST:-0.0.0.0}"
PORT="${VIBEVOICE_PORT:-7862}"
MODEL_NAME="${QWEN_ALIGNER_MODEL:-Qwen/Qwen3-ForcedAligner-0.6B}"
DEVICE="${VIBEVOICE_DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { echo -e "${CYAN}$1${NC}"; }
step() { echo -e "${YELLOW}$1${NC}"; }
fail() { echo -e "${RED}$1${NC}"; exit 1; }

load_env_file() {
    local env_file="$1"
    if [ -f "${env_file}" ]; then
        echo "Lade Umgebungswerte aus ${env_file}"
        local cleaned
        cleaned=$(tr -d '\r' < "${env_file}")
        set -a; eval "${cleaned}"; set +a
    fi
}

run_sudo() {
    if [ -n "${SUDO_PASSWORD:-}" ]; then
        printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
    else
        sudo "$@"
    fi
}

log "======================================"
log " Qwen3-ForcedAligner auf DGX Spark installieren"
log "======================================"
echo ""
echo "App-Verzeichnis: ${APP_DIR}"
echo "Venv:            ${VENV_DIR}"
echo "Modell:          ${MODEL_NAME}"
echo "Host/Port:       ${HOST}:${PORT}"
echo "Device:          ${DEVICE}"
echo ""

if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo wird benötigt."
fi

step "[1/7] System prüfen"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi nicht gefunden. NVIDIA-Treiber prüfen."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} nicht gefunden."
[ -d "${PWD}/vibevoice_spark" ] || fail "vibevoice_spark Verzeichnis nicht gefunden. Bitte aus dem Projektroot ausführen."
load_env_file "${PWD}/.env.local"
load_env_file "${PWD}/vibevoice_spark/.env.local"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" --version

step "[2/7] Ubuntu-Pakete installieren"
run_sudo apt update
run_sudo apt install -y python3-venv python3-pip python3-dev build-essential ffmpeg libsndfile1 git curl

step "[3/7] Projektdateien kopieren"
rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}"
cp -R "${PWD}/vibevoice_spark/." "${APP_DIR}/"

step "[4/7] Python-Umgebung vorbereiten"
if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"

step "[5/7] Python-Abhängigkeiten installieren"
pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio

# qwen-asr (ForcedAligner)
pip install -U qwen-asr

# Restliche Abhängigkeiten
pip install -r "${APP_DIR}/requirements.txt"

# Verifizierung
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA-Version: {torch.version.cuda}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA ist in PyTorch nicht verfügbar.")
print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    from qwen_asr import Qwen3ForcedAligner
    print("Qwen3-ForcedAligner Import: OK")
except ImportError as e:
    raise SystemExit(f"qwen-asr nicht korrekt installiert: {e}")
PY

step "[6/7] .env konfigurieren"
cat > "${APP_DIR}/.env" <<EOF
VIBEVOICE_AUTH_USERNAME=${VIBEVOICE_AUTH_USERNAME:-admin}
VIBEVOICE_AUTH_PASSWORD=${VIBEVOICE_AUTH_PASSWORD:-change-me}
QWEN_ALIGNER_MODEL=${MODEL_NAME}
VIBEVOICE_HOST=${HOST}
VIBEVOICE_PORT=${PORT}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
HF_HOME=${HOME}/.cache/huggingface
EOF

# Modell vorab herunterladen
step "[6b/7] Modell vorab herunterladen"
python - <<PY
from huggingface_hub import snapshot_download
print("Lade ${MODEL_NAME} herunter...")
snapshot_download("${MODEL_NAME}", cache_dir="${HOME}/.cache/huggingface/hub")
print("Modell-Download abgeschlossen.")
PY

step "[7/7] systemd Service einrichten"
SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Qwen3-ForcedAligner Server (DGX Spark)
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

run_sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
run_sudo systemctl daemon-reload
run_sudo systemctl enable "${SERVICE_NAME}"

echo ""
log "Installation abgeschlossen."
echo "Server starten:   sudo systemctl start ${SERVICE_NAME}"
echo "Status anzeigen:  sudo systemctl status ${SERVICE_NAME} --no-pager -l"
echo "Logs anzeigen:    journalctl -u ${SERVICE_NAME} -f"
echo "API:              http://127.0.0.1:${PORT}"
echo "Docs:             http://127.0.0.1:${PORT}/docs"
echo ""
echo "Endpunkt: POST /v1/align  (Audio + Text → Timestamps)"
echo "Wird vom Voxtral-Proxy genutzt um Timestamps zu erzeugen."
echo "Qwen3-ForcedAligner-0.6B benötigt nur ~1.5 GB VRAM."
