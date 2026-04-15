#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HOME}/voxtral-local"
IMAGE_NAME="voxtral-local:dgx-spark"
CONTAINER_NAME="voxtral-local"
MODEL_ID="${VOXTRAL_LOCAL_MODEL:-mistralai/Voxtral-Mini-3B-2507}"
HOST_PORT="${VOXTRAL_PORT:-8000}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker ist nicht installiert. Bitte zuerst Docker + NVIDIA Container Toolkit einrichten."
    exit 1
fi

mkdir -p "${APP_DIR}"
cp "${PWD}/voxtral_server.py" "${APP_DIR}/voxtral_server.py"

cat > "${APP_DIR}/Dockerfile" <<'EOF'
FROM nvcr.io/nvidia/pytorch:26.03-py3
WORKDIR /app
COPY voxtral_server.py /app/voxtral_server.py
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
      "numpy<3" \
      soundfile \
      librosa \
      fastapi \
      "uvicorn[standard]" \
      python-multipart \
  "mistral-common[audio]" \
      "transformers>=4.53.0" \
      "accelerate>=1.8.0" \
      sentencepiece \
      safetensors \
      "huggingface_hub[cli]"
EXPOSE 8000
CMD ["python", "/app/voxtral_server.py"]
EOF

cd "${APP_DIR}"
echo "Baue Docker-Image ${IMAGE_NAME} ..."
docker build -t "${IMAGE_NAME}" .

echo "Starte Container ${CONTAINER_NAME} ..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --restart unless-stopped \
  -p "${HOST_PORT}:8000" \
  -e VOXTRAL_LOCAL_MODEL="${MODEL_ID}" \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "${IMAGE_NAME}"

echo "Container läuft. Logs: docker logs -f ${CONTAINER_NAME}"
echo "Health: curl http://127.0.0.1:${HOST_PORT}/health"
