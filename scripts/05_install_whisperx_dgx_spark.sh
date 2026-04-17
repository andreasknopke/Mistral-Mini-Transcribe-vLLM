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
GERMAN_ALIGNMENT_MODEL="${WHISPERX_GERMAN_ALIGNMENT_MODEL:-jonatasgrosman/wav2vec2-large-xlsr-53-german}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CTRANSLATE2_VERSION="${CTRANSLATE2_VERSION:-4.7.1}"
CTRANSLATE2_INSTALL_DIR="${HOME}/ctranslate2-install"
CTRANSLATE2_LD_LIBRARY_PATH="${CTRANSLATE2_INSTALL_DIR}/lib:/usr/local/cuda/lib64:${VENV_DIR}/lib/python3.12/site-packages/nvidia/cu13/lib:${VENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn/lib"

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

load_env_file() {
    local env_file="$1"
    if [ -f "${env_file}" ]; then
        echo "Lade Umgebungswerte aus ${env_file}"
        set -a
        . "${env_file}"
        set +a
    fi
}

run_sudo() {
    if [ -n "${SUDO_PASSWORD:-}" ]; then
        printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
    else
        sudo "$@"
    fi
}

install_ctranslate2_cuda_arm64() {
    local src_dir="${HOME}/src/CTranslate2"
    local build_dir="${src_dir}/build"
    local cuda_pkg="${VENV_DIR}/lib/python3.12/site-packages/nvidia/cu13"
    local cudnn_pkg="${VENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn"

    step "[5b/7] CTranslate2 mit CUDA für ARM64 bauen"
    run_sudo apt install -y ninja-build libopenblas-dev

    mkdir -p "${HOME}/src"
    if [ ! -d "${src_dir}/.git" ]; then
        git clone --recursive https://github.com/OpenNMT/CTranslate2.git "${src_dir}"
    fi

    pushd "${src_dir}" >/dev/null
    git fetch --tags --force
    git checkout "v${CTRANSLATE2_VERSION}"
    git submodule update --init --recursive
    popd >/dev/null

    run_sudo mkdir -p /usr/local/cuda/include /usr/local/cuda/lib64
    run_sudo ln -sf "${cudnn_pkg}"/include/cudnn*.h /usr/local/cuda/include/
    run_sudo ln -sf "${cudnn_pkg}"/lib/libcudnn*.so.9 /usr/local/cuda/lib64/
    run_sudo ln -sf /usr/local/cuda/lib64/libcudnn.so.9 /usr/local/cuda/lib64/libcudnn.so

    rm -rf "${build_dir}" "${CTRANSLATE2_INSTALL_DIR}"
    mkdir -p "${build_dir}" "${CTRANSLATE2_INSTALL_DIR}"

    export LD_LIBRARY_PATH="${CTRANSLATE2_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="/usr/local/cuda/lib64:${cuda_pkg}/lib:${cudnn_pkg}/lib:${LIBRARY_PATH:-}"
    export C_INCLUDE_PATH="/usr/local/cuda/include:${cuda_pkg}/include:${cudnn_pkg}/include:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="/usr/local/cuda/include:${cuda_pkg}/include:${cudnn_pkg}/include:${CPLUS_INCLUDE_PATH:-}"

    pushd "${build_dir}" >/dev/null
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${CTRANSLATE2_INSTALL_DIR}" \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DCUDA_DYNAMIC_LOADING=ON \
        -DWITH_MKL=OFF \
        -DWITH_DNNL=OFF \
        -DWITH_OPENBLAS=ON \
        -DOPENMP_RUNTIME=COMP \
        -DBUILD_CLI=OFF \
        -DBUILD_TESTS=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
    ninja -j4
    ninja install
    popd >/dev/null

    pushd "${src_dir}/python" >/dev/null
    pip install -r install_requirements.txt
    CTRANSLATE2_ROOT="${CTRANSLATE2_INSTALL_DIR}" python setup.py bdist_wheel
    pip install --force-reinstall --no-deps dist/*.whl
    popd >/dev/null

    pip install --force-reinstall --no-deps numpy==1.26.4 "setuptools<82"
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
echo "DE Align Model:  ${GERMAN_ALIGNMENT_MODEL}"
echo ""

if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo wird benötigt."
fi

step "[1/7] System prüfen"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi nicht gefunden. NVIDIA-Treiber prüfen."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} nicht gefunden."
[ -d "${PWD}/whisperx_spark" ] || fail "whisperx_spark Verzeichnis nicht gefunden. Bitte zuerst Deploy-Skript ausführen."
load_env_file "${PWD}/.env.local"
load_env_file "${PWD}/whisperx_spark/.env.local"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" --version

step "[2/7] Ubuntu-Pakete installieren"
run_sudo apt update
run_sudo apt install -y python3-venv python3-pip python3-dev build-essential ffmpeg libsndfile1 git curl

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

if [ "$(uname -m)" = "aarch64" ]; then
    install_ctranslate2_cuda_arm64
fi

python - <<'PY'
import ctranslate2
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA-Version: {torch.version.cuda}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA ist in PyTorch nicht verfügbar. WhisperX würde sonst nur im CPU-Modus laufen.")
print(f"CTranslate2: {ctranslate2.__version__}")
print(f"CTranslate2 CUDA Devices: {ctranslate2.get_cuda_device_count()}")
if ctranslate2.get_cuda_device_count() < 1:
    raise SystemExit("CTranslate2 erkennt keine CUDA-Geräte. WhisperX würde sonst im CPU-Modus laufen.")
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
WHISPERX_GERMAN_ALIGNMENT_MODEL=${GERMAN_ALIGNMENT_MODEL}
WHISPERX_HOST=${HOST}
WHISPERX_PORT=${PORT}
WHISPERX_ESTIMATED_WORKER_GB=${WHISPERX_ESTIMATED_WORKER_GB:-6}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
LD_LIBRARY_PATH=${CTRANSLATE2_LD_LIBRARY_PATH}
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

run_sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
run_sudo systemctl daemon-reload
run_sudo systemctl enable "${SERVICE_NAME}"

echo ""
log "Installation abgeschlossen."
echo "Server starten:   sudo systemctl start ${SERVICE_NAME}"
echo "Status anzeigen:  sudo systemctl status ${SERVICE_NAME} --no-pager -l"
echo "Logs anzeigen:    journalctl -u ${SERVICE_NAME} -f"
echo "UI/API:           http://127.0.0.1:${PORT}"
echo "Gradio API:       http://127.0.0.1:${PORT}/gradio_api/openapi.json"
echo "Hinweis: Für mehr Parallelität WHISPERX_POOL_SIZE erhöhen, aber VRAM-/RAM-Budget im Blick behalten."
echo "Hinweis: Auf ARM64 baut das Skript CTranslate2 automatisch mit CUDA/cuDNN aus dem lokalen NVIDIA-Stack."
