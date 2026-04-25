#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Gemma 4 26B-A4B MoE (NVFP4) auf DGX Spark – Version 2 (eugr/spark-vllm-docker)
#
#  Modell:  bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
#           - 25.2B total / 3.8B aktiv (128 Experts, Top-8)
#           - NVFP4 W4A4 Quantisierung → ~16.5 GB Disk / ~15.7 GiB GPU
#
#  Warum dieser Stack:
#    - NGC vllm:26.03-py3 hatte noch keine native Gemma4ForCausalLM-Klasse,
#      der TransformersMultiModalMoEForCausalLM-Fallback hängt stundenlang.
#    - eugr/spark-vllm-docker liefert nightly auf DGX Spark getestete
#      vLLM-Wheels mit nativem Gemma 4 + transformers 5.x (--tf5).
#    - Modellkarte fordert ZUSÄTZLICH:
#        * gemma4_patched.py  (NVFP4 expert-scale-key mapping fix)
#        * --moe-backend marlin
#        * VLLM_NVFP4_GEMM_BACKEND=marlin
# ============================================================================

STACK_OWNER="${SPARK_STACK_OWNER:-${SUDO_USER:-${USER:-$(id -un)}}}"
STACK_HOME="${SPARK_STACK_HOME:-/home/${STACK_OWNER}}"
APP_DIR="${STACK_HOME}/correction-llm-vllm"
SPARK_VLLM_DIR="${STACK_HOME}/spark-vllm-docker"
SERVICE_NAME="correction-llm"
MODEL_ID="${GEMMA4_MODEL:-bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4}"
SERVED_MODEL_NAMES="${GEMMA4_SERVED_NAMES:-gemma-4 correction-llm}"
HOST_PORT="${GEMMA4_PORT:-9000}"
GEMMA4_PROFILE="${GEMMA4_PROFILE:-spark-shared}"
IMAGE_TAG="${GEMMA4_IMAGE_TAG:-vllm-node-tf5}"
FORCE_REBUILD_IMAGE="${GEMMA4_FORCE_REBUILD:-0}"

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

run_sudo() {
  if [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

echo "================================================"
echo " Gemma 4 26B-A4B MoE (NVFP4) – DGX Spark v2"
echo "================================================"
echo "Modell:                 ${MODEL_ID}"
echo "Served Names:           ${SERVED_MODEL_NAMES}"
echo "Profil:                 ${GEMMA4_PROFILE}"
echo "Port:                   ${HOST_PORT}"
echo "Max parallele Seqs:     ${MAX_NUM_SEQS}"
echo "Max Context Len:        ${MAX_MODEL_LEN}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Image Tag:              ${IMAGE_TAG}"
echo ""

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FEHLER: nvidia-smi nicht gefunden. NVIDIA-Treiber prüfen."
  exit 1
fi

# ---- [1/8] Docker + NVIDIA Container Toolkit + git ----
echo "[1/8] Docker + Werkzeuge prüfen"
if ! command -v docker >/dev/null 2>&1; then
  run_sudo apt update
  run_sudo apt install -y docker.io nvidia-container-toolkit git curl
fi
if ! command -v git >/dev/null 2>&1; then
  run_sudo apt update
  run_sudo apt install -y git
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

# Aktuellen Benutzer in docker-Gruppe (idempotent)
run_sudo usermod -aG docker "${STACK_OWNER}" >/dev/null 2>&1 || true

# ---- [2/8] Hugging Face Token ----
echo "[2/8] Hugging Face Token prüfen"
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
mkdir -p "${STACK_HOME}/.cache/huggingface"
echo -n "${HF_TOKEN}" > "${STACK_HOME}/.cache/huggingface/token"

mkdir -p "${APP_DIR}"

# ---- [3/8] eugr/spark-vllm-docker holen / aktualisieren ----
echo "[3/8] eugr/spark-vllm-docker bereitstellen"
if [ ! -d "${SPARK_VLLM_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/eugr/spark-vllm-docker.git "${SPARK_VLLM_DIR}"
else
  echo "Repo existiert – aktualisiere"
  git -C "${SPARK_VLLM_DIR}" fetch --depth 1 origin main
  git -C "${SPARK_VLLM_DIR}" reset --hard origin/main
fi
chown -R "${STACK_OWNER}":"${STACK_OWNER}" "${SPARK_VLLM_DIR}" 2>/dev/null || true

# ---- [4/8] vLLM-Image bauen (--tf5: transformers 5.x + native Gemma4) ----
echo "[4/8] vLLM-Image bauen (kann beim ersten Mal ~5–10 Min dauern)"
NEED_BUILD=0
if [ "${FORCE_REBUILD_IMAGE}" = "1" ]; then
  NEED_BUILD=1
elif ! run_sudo docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  NEED_BUILD=1
fi

if [ "${NEED_BUILD}" = "1" ]; then
  pushd "${SPARK_VLLM_DIR}" >/dev/null
  # --tf5 → transformers >= 5.x (für Gemma 4 Architektur), zieht zertifizierte
  # vLLM-Wheels (April 2026 Build mit nativem Gemma4ForCausalLM).
  # NICHT --rebuild-vllm: dann werden die getesteten Prebuilt-Wheels verwendet.
  ./build-and-copy.sh --tf5 -t "${IMAGE_TAG}"
  popd >/dev/null
else
  echo "Image ${IMAGE_TAG} bereits vorhanden – überspringe Build"
  echo "(Erzwingen mit GEMMA4_FORCE_REBUILD=1)"
fi

# ---- [5/8] gemma4_patched.py vom HF-Repo holen ----
echo "[5/8] gemma4_patched.py aus dem Modell-Repo laden"
PATCH_FILE="${APP_DIR}/gemma4_patched.py"
PATCH_URL="https://huggingface.co/${MODEL_ID}/resolve/main/gemma4_patched.py"

# huggingface_hub-CLI nicht zwingend nötig – direkter Download via curl
HTTP_CODE="$(curl -sSL -w '%{http_code}' \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -o "${PATCH_FILE}.tmp" \
  "${PATCH_URL}")" || true

if [ "${HTTP_CODE}" = "200" ] && [ -s "${PATCH_FILE}.tmp" ]; then
  mv "${PATCH_FILE}.tmp" "${PATCH_FILE}"
  # vLLM 0.19.2rc1 (April 2026) hat reduce_results aus FusedMoE entfernt
  sed -i '/reduce_results=True,/d' "${PATCH_FILE}"
  echo "OK: $(wc -l < "${PATCH_FILE}") Zeilen heruntergeladen (reduce_results entfernt)"
else
  rm -f "${PATCH_FILE}.tmp"
  echo "WARNUNG: Konnte gemma4_patched.py nicht laden (HTTP ${HTTP_CODE})."
  echo "         Versuche, ohne Patch zu starten – kann zu fehlerhaften MoE-Scales führen."
  PATCH_FILE=""
fi

# ---- [6/8] env-Datei + Run-Wrapper ----
echo "[6/8] env-Datei + Run-Wrapper schreiben"
cat > "${APP_DIR}/llm.env" <<EOF
HF_TOKEN=${HF_TOKEN}
HF_HOME=/root/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
VLLM_NVFP4_GEMM_BACKEND=marlin
VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF
chmod 600 "${APP_DIR}/llm.env"

# Optional: Mount für gemma4_patched.py – nur wenn Datei vorhanden
PATCH_MOUNT_LINE=""
if [ -n "${PATCH_FILE}" ] && [ -s "${PATCH_FILE}" ]; then
  PATCH_MOUNT_LINE="  -v ${PATCH_FILE}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py:ro \\"
fi

cat > "${APP_DIR}/run_gemma4.sh" <<EOF
#!/usr/bin/env bash
# Auto-generiert von 09_install_gemma4_dgx_spark.sh – nicht von Hand bearbeiten.
set -euo pipefail

exec /usr/bin/docker run --rm \\
  --name ${SERVICE_NAME} \\
  --gpus all \\
  --ipc=host \\
  --network host \\
  --ulimit memlock=-1 \\
  --shm-size=16g \\
  --env-file ${APP_DIR}/llm.env \\
  -v ${STACK_HOME}/.cache/huggingface:/root/.cache/huggingface \\
${PATCH_MOUNT_LINE}
  ${IMAGE_TAG} \\
  vllm serve ${MODEL_ID} \\
  --host 0.0.0.0 \\
  --port ${HOST_PORT} \\
  --served-model-name ${SERVED_MODEL_NAMES} \\
  --quantization modelopt \\
  --dtype auto \\
  --kv-cache-dtype fp8 \\
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --max-num-seqs ${MAX_NUM_SEQS} \\
  --max-num-batched-tokens 8192 \\
  --moe-backend marlin \\
  --tokenizer-mode hf \\
  --chat-template-content-format string \\
  --trust-remote-code
EOF
chmod +x "${APP_DIR}/run_gemma4.sh"

# ---- [7/8] systemd Service ----
echo "[7/8] systemd Service einrichten"
run_sudo systemctl disable --now correction-llm >/dev/null 2>&1 || true
run_sudo systemctl disable --now gemma4-llm >/dev/null 2>&1 || true
run_sudo docker rm -f correction-llm >/dev/null 2>&1 || true
run_sudo docker rm -f gemma4-llm >/dev/null 2>&1 || true

SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Gemma 4 26B-A4B NVFP4 MoE Server (DGX Spark, eugr/spark-vllm-docker)
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f ${SERVICE_NAME}
ExecStart=/bin/bash ${APP_DIR}/run_gemma4.sh
ExecStop=/usr/bin/docker stop ${SERVICE_NAME}
Restart=always
RestartSec=10
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

run_sudo mv "${SERVICE_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
run_sudo systemctl daemon-reload
run_sudo systemctl enable "${SERVICE_NAME}"

# ---- [8/8] Fertig ----
echo "[8/8] Fertig"
echo ""
echo "=========================================================="
echo " Gemma 4 26B-A4B MoE NVFP4 installiert (eugr-Stack, --tf5)"
echo "=========================================================="
echo "Image:           ${IMAGE_TAG}"
echo "Patch gemounted: $([ -n "${PATCH_FILE}" ] && echo "ja → ${PATCH_FILE}" || echo "NEIN")"
echo "Profil:          ${GEMMA4_PROFILE}"
echo "Port:            ${HOST_PORT}"
echo ""
echo "Starten:   sudo systemctl start ${SERVICE_NAME}"
echo "Status:    sudo systemctl status ${SERVICE_NAME} --no-pager -l"
echo "Logs:      journalctl -u ${SERVICE_NAME} -f"
echo "Test API:  curl http://127.0.0.1:${HOST_PORT}/v1/models"
echo ""
echo "Erwartete Lade-Zeit beim ersten Start: ~3–5 Min"
echo "(natives Gemma4ForCausalLM, NICHT mehr der hängende Transformers-Fallback)"
