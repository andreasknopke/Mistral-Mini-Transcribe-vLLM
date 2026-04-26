#!/usr/bin/env bash
# ==============================================================================
# 12_rebuild_ctranslate2_flashattn.sh
# ------------------------------------------------------------------------------
# Baut CTranslate2 ARM64 mit FlashAttention für SM 12.1 (Blackwell GB10).
#
# ROLLBACK-SICHER:
#   - Vor Build wird das aktuell installierte ctranslate2 als Wheel nach
#     ~/ct2_backup/ctranslate2-<oldver>-<ts>.whl gesichert.
#   - Build läuft in /tmp/ct2_build_<ts>, alter Build bleibt unangetastet
#     bis zum finalen `pip install --force-reinstall`.
#   - Rollback einfach mit:
#         pip install --force-reinstall ~/ct2_backup/ctranslate2-<oldver>-<ts>.whl
#
# Aktivierung danach: WHISPERX_FLASH_ATTENTION=1 in .env
# ==============================================================================
set -euo pipefail

VENV="${WHISPERX_VENV:-/home/ksai0001_local/whisperx-env}"
APP_ENV_FILE="${WHISPERX_ENV_FILE:-/home/ksai0001_local/whisperx-spark/.env}"
ARCH="${CT2_CUDA_ARCH:-121}"        # Blackwell GB10
JOBS="${CT2_BUILD_JOBS:-$(nproc)}"
OPENMP_RUNTIME="${CT2_OPENMP_RUNTIME:-COMP}"
TS="$(date +%Y%m%d_%H%M%S)"
BUILD_DIR="/tmp/ct2_build_${TS}"
BACKUP_DIR="${CT2_BACKUP_DIR:-/home/ksai0001_local/ct2_backup}"

echo "=============================================================="
echo " CTranslate2 Rebuild mit FlashAttention"
echo "  Venv     : $VENV"
echo "  CUDA Arch: $ARCH"
echo "  Jobs     : $JOBS"
echo "  OpenMP   : $OPENMP_RUNTIME"
echo "  Build    : $BUILD_DIR"
echo "  Backup   : $BACKUP_DIR"
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

mkdir -p "$BACKUP_DIR"

# ---------- 1) Backup des aktuellen ctranslate2 als Wheel ---------------------
OLD_VER="$(python -c 'import ctranslate2,sys;print(ctranslate2.__version__)' 2>/dev/null || echo unknown)"
echo "Aktuelle ctranslate2-Version: $OLD_VER"
if [ "$OLD_VER" != "unknown" ]; then
  echo "Backup als Wheel ..."
  pip wheel --no-deps "ctranslate2==$OLD_VER" -w "$BACKUP_DIR/" || \
    echo "WARN: Wheel-Backup fehlgeschlagen (kein PyPI-Wheel für ARM64?)."
  # Zusätzlich: bestehende installierte Dateien als Tarball sichern
  SITE="$(python -c 'import ctranslate2,os,sys;print(os.path.dirname(ctranslate2.__file__))')"
  tar czf "$BACKUP_DIR/ctranslate2-installed-${OLD_VER}-${TS}.tar.gz" -C "$(dirname "$SITE")" "$(basename "$SITE")" || true
  echo "Installations-Backup: $BACKUP_DIR/ctranslate2-installed-${OLD_VER}-${TS}.tar.gz"
fi

# ---------- 2) Build-Deps -----------------------------------------------------
echo "Prüfe Build-Tools ..."
command -v cmake  >/dev/null || { echo "cmake fehlt: sudo apt install cmake"; exit 3; }
command -v ninja  >/dev/null || echo "(optional) ninja fehlt – make wird genutzt."

# ---------- 3) Source ziehen --------------------------------------------------
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
git clone --recursive --depth 1 https://github.com/OpenNMT/CTranslate2 .

# ---------- 4) Konfigurieren + Bauen -----------------------------------------
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CLI=OFF \
  -DWITH_MKL=OFF \
  -DWITH_RUY=ON \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_FLASH_ATTN=ON \
  -DOPENMP_RUNTIME="$OPENMP_RUNTIME" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all"

make -j"$JOBS"
sudo make install
sudo ldconfig

# ---------- 5) Python-Bindings ------------------------------------------------
cd ../python
pip install -r install_requirements.txt
python setup.py bdist_wheel
NEW_WHEEL="$(ls -t dist/ctranslate2-*.whl | head -n1)"
echo "Neues Wheel: $NEW_WHEEL"
cp "$NEW_WHEEL" "$BACKUP_DIR/$(basename "$NEW_WHEEL" .whl)-flashattn-${TS}.whl"

pip install --force-reinstall --no-deps "$NEW_WHEEL"

# ---------- 6) Smoke-Test -----------------------------------------------------
python - <<'PY'
import ctranslate2
print("ctranslate2:", ctranslate2.__version__)
cuda_count = getattr(ctranslate2, "get_cuda_device_count", None)
if callable(cuda_count):
  print("CUDA devices:", cuda_count())
else:
  print("CUDA devices: API nicht exponiert; Import erfolgreich")
PY

echo
echo "=============================================================="
echo " FERTIG. FlashAttention ist im Build aktiviert."
echo " Aktivierung im Service: WHISPERX_FLASH_ATTENTION=1 in .env"
echo
echo " ROLLBACK:"
echo "   pip install --force-reinstall \\"
echo "     $BACKUP_DIR/ctranslate2-${OLD_VER}-*.whl"
echo "   (oder Tarball entpacken: $BACKUP_DIR/ctranslate2-installed-${OLD_VER}-${TS}.tar.gz)"
echo "=============================================================="
