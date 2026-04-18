#!/usr/bin/env bash
set -euo pipefail

echo "==============================================="
echo " DGX Spark AI Stack installieren"
echo "==============================================="
echo ""
echo "Dieser Helper installiert nacheinander:"
echo "  1) Voxtral / Mistral-Transcribe via vLLM"
echo "  2) WhisperX Server"
echo "  3) optional: Korrektur-LLM via vLLM"
echo "  4) Spark Admin Control Center"
echo ""

echo "[1/4] Voxtral installieren"
./04_install_voxtral_dgx_spark_container.sh

echo "[2/4] WhisperX installieren"
./05_install_whisperx_dgx_spark.sh

if [ "${INSTALL_CORRECTION_LLM:-0}" = "1" ]; then
	echo "[3/4] Korrektur-LLM installieren"
	./06_install_correction_llm_dgx_spark.sh
else
	echo "[3/4] Korrektur-LLM übersprungen (INSTALL_CORRECTION_LLM!=1)"
fi

if [ "${INSTALL_SPARK_ADMIN:-1}" = "1" ]; then
	echo "[4/4] Spark Admin installieren"
	./08_install_spark_admin_dgx_spark.sh
else
	echo "[4/4] Spark Admin übersprungen (INSTALL_SPARK_ADMIN=0)"
fi

echo ""
echo "Alle Services wurden eingerichtet. Starten mit:"
echo "  sudo systemctl start voxtral-vllm whisperx"
echo "Optional zusätzlich: sudo systemctl start correction-llm"
echo "Optional zusätzlich: sudo systemctl start spark-admin"
echo "Prüfen mit:"
echo "  sudo systemctl status voxtral-vllm whisperx spark-admin --no-pager -l"
