#!/bin/bash
# Quantize Mistral-Small-24B-Instruct-2501 to NVFP4 on DGX Spark
# Based on NVIDIA NVFP4 Quantization Playbook
set -e

MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
OUTPUT_DIR="/home/ksai0001_local/nvfp4-quantization/output_models"
HF_CACHE="/home/ksai0001_local/.cache/huggingface"
HF_TOKEN=$(cat "$HF_CACHE/token")

echo "=== NVFP4 Quantization of $MODEL ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Prepare output directory
mkdir -p "$OUTPUT_DIR"
chmod 755 "$OUTPUT_DIR"

# Step 2: Flush memory cache to maximize available RAM
echo "Flushing memory caches..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Step 3: Stop correction-llm to free GPU memory for quantization
echo "Stopping correction-llm containers to free memory..."
sudo docker stop correction-llm-test correction-llm 2>/dev/null || true
echo "Containers stopped."

# Step 4: Run quantization
echo ""
echo "Starting NVFP4 quantization (this will take 30-60 minutes)..."
echo "Container: nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"
echo ""

sudo docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$OUTPUT_DIR:/workspace/output_models" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    pip install hf_transfer && \
    echo '>>> Cloning TensorRT Model Optimizer...'
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/TensorRT-Model-Optimizer && \
    cd /app/TensorRT-Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    echo '>>> Starting quantization of $MODEL to NVFP4...' && \
    /app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
      --model '$MODEL' \
      --quant nvfp4 \
      --tp 1 \
      --export_fmt hf && \
    echo '>>> Quantization complete!' && \
    ls -lh /workspace/output_models/
  "

echo ""
echo "=== Quantization finished ==="
echo "Output files:"
ls -lh "$OUTPUT_DIR/"
find "$OUTPUT_DIR" -name "config.json" -exec echo "Found: {}" \;
