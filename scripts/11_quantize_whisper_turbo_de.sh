#!/usr/bin/env bash
# ==============================================================================
# 11_quantize_whisper_turbo_de.sh
# ------------------------------------------------------------------------------
# Konvertiert primeline/whisper-large-v3-turbo-german mit ct2-transformers-converter
# nach CTranslate2 (int8_bfloat16) – optimal für Blackwell GB10 (Spark).
#
# Idempotent. Schreibt nach $TARGET_DIR. Bestehender Ordner wird NICHT überschrieben,
# außer FORCE=1.
#
# Rollback: Verzeichnis einfach löschen, ENV WHISPERX_MODEL zurücksetzen.
# ==============================================================================
set -euo pipefail

VENV="${WHISPERX_VENV:-/home/ksai0001_local/whisperx-env}"
APP_ENV_FILE="${WHISPERX_ENV_FILE:-/home/ksai0001_local/whisperx-spark/.env}"
SRC_MODEL="${WHISPERX_QUANT_SRC:-primeline/whisper-large-v3-turbo-german}"
TOKENIZER_FALLBACK_MODEL="${WHISPERX_QUANT_TOKENIZER_SRC:-openai/whisper-large-v3-turbo}"
TARGET_DIR="${WHISPERX_QUANT_DST:-/home/ksai0001_local/models/primeline-turbo-de-int8_bf16}"
QUANT="${WHISPERX_QUANT_TYPE:-int8_bfloat16}"

echo "=============================================================="
echo " WhisperX Turbo-DE Quantisierung"
echo "  Quelle : $SRC_MODEL"
echo "  Fallback Tokenizer : $TOKENIZER_FALLBACK_MODEL"
echo "  Ziel   : $TARGET_DIR"
echo "  Typ    : $QUANT"
echo "  Venv   : $VENV"
echo "=============================================================="

if [ ! -f "$VENV/bin/activate" ]; then
  echo "FEHLER: venv $VENV nicht gefunden." >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ -f "$APP_ENV_FILE" ] && [ -z "${LD_LIBRARY_PATH:-}" ]; then
  EXISTING_LD_LIBRARY_PATH="$(grep '^LD_LIBRARY_PATH=' "$APP_ENV_FILE" | head -n1 | cut -d= -f2- || true)"
  if [ -n "$EXISTING_LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$EXISTING_LD_LIBRARY_PATH"
  fi
fi

if [ -d "$TARGET_DIR" ] && [ -f "$TARGET_DIR/model.bin" ]; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "Zielordner existiert bereits – Skip (FORCE=1 zum Überschreiben)."
    exit 0
  fi
  echo "FORCE=1 gesetzt – lösche $TARGET_DIR ..."
  rm -rf "$TARGET_DIR"
fi

mkdir -p "$(dirname "$TARGET_DIR")"

pip install -q -U "transformers>=4.42"

python - <<'PY'
import ctranslate2
print(f"ct2_version={ctranslate2.__version__}")
print(f"ct2_cuda_devices={ctranslate2.get_cuda_device_count()}")
PY

LOCAL_SRC_DIR="$(mktemp -d /tmp/whisper_src_XXXXXX)"
export LOCAL_SRC_DIR SRC_MODEL TOKENIZER_FALLBACK_MODEL
cleanup() {
  rm -rf "$LOCAL_SRC_DIR"
}
trap cleanup EXIT

python - <<'PY'
import os
import shutil
from huggingface_hub import snapshot_download

target = os.environ["LOCAL_SRC_DIR"]
src_model = os.environ["SRC_MODEL"]
fallback_model = os.environ["TOKENIZER_FALLBACK_MODEL"]
required_files = [
  "tokenizer.json",
  "tokenizer_config.json",
  "preprocessor_config.json",
  "normalizer.json",
  "generation_config.json",
]

src_dir = snapshot_download(repo_id=src_model)
for name in os.listdir(src_dir):
  source_path = os.path.join(src_dir, name)
  target_path = os.path.join(target, name)
  if os.path.isdir(source_path):
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
  else:
    shutil.copy2(source_path, target_path)

missing = [name for name in required_files if not os.path.exists(os.path.join(target, name))]
if missing:
  fallback_dir = snapshot_download(repo_id=fallback_model)
  for name in missing:
    fallback_path = os.path.join(fallback_dir, name)
    if os.path.exists(fallback_path):
      shutil.copy2(fallback_path, os.path.join(target, name))

still_missing = [name for name in required_files if not os.path.exists(os.path.join(target, name))]
if still_missing:
  raise SystemExit(f"Fehlende Dateien nach Fallback: {still_missing}")

print(f"prepared_model_dir={target}")
PY

python -m ctranslate2.converters.transformers \
  --model "$LOCAL_SRC_DIR" \
  --output_dir "$TARGET_DIR" \
  --copy_files tokenizer.json preprocessor_config.json normalizer.json generation_config.json \
  --quantization "$QUANT" \
  --force

echo
echo "Größe:"
du -sh "$TARGET_DIR"
echo
echo "FERTIG. Modell-Pfad: $TARGET_DIR"
echo "Setze in ~/whisperx-spark/.env:"
echo "  WHISPERX_MODEL=$TARGET_DIR"
echo "  WHISPERX_COMPUTE_TYPE=$QUANT"
