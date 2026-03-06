#!/bin/bash
# ============================================================
# Voxtral Installation in WSL2 - Schritt 2
# Dieses Skript in WSL2/Ubuntu ausführen!
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}======================================"
echo " Voxtral Installation in WSL2"
echo -e "======================================${NC}"
echo ""

# 1. System-Pakete
echo -e "${YELLOW}[1/5] System-Pakete installieren...${NC}"
sudo apt update && sudo apt install -y python3-venv python3-pip curl
echo -e "${GREEN}  Fertig.${NC}"

# 2. Python venv erstellen
echo -e "${YELLOW}[2/5] Python Virtual Environment erstellen...${NC}"
if [ -d "$HOME/voxtral-env" ]; then
    echo -e "${GREEN}  voxtral-env existiert bereits.${NC}"
else
    python3 -m venv "$HOME/voxtral-env"
    echo -e "${GREEN}  voxtral-env erstellt.${NC}"
fi

# Aktivieren
source "$HOME/voxtral-env/bin/activate"
echo -e "${GREEN}  Environment aktiviert: $(python3 --version)${NC}"

# 3. vLLM und Mistral-Audio installieren
echo -e "${YELLOW}[3/5] vLLM und mistral-common[audio] installieren...${NC}"
echo -e "${YELLOW}  (Dies dauert einige Minuten - PyTorch, CUDA-Binaries etc.)${NC}"
pip install --upgrade pip
pip install vllm "mistral-common[audio]"
echo -e "${GREEN}  vLLM installiert.${NC}"

# 4. HuggingFace Hub installieren
echo -e "${YELLOW}[4/5] HuggingFace Hub installieren...${NC}"
pip install huggingface_hub
echo -e "${GREEN}  huggingface_hub installiert.${NC}"

# 5. HuggingFace Login
echo -e "${YELLOW}[5/5] HuggingFace Login...${NC}"
echo ""
echo -e "${CYAN}WICHTIG - Vor dem Login:${NC}"
echo -e "  1. Öffne ${CYAN}https://huggingface.co/mistralai/Voxtral-Mini-3B-2507${NC}"
echo -e "  2. Klicke auf ${GREEN}'Agree and access repository'${NC}"
echo -e "  3. Erstelle ein Token: ${CYAN}https://huggingface.co/settings/tokens${NC}"
echo ""

# Prüfe ob bereits eingeloggt
if huggingface-cli whoami &>/dev/null; then
    WHOAMI=$(huggingface-cli whoami 2>/dev/null | head -1)
    echo -e "${GREEN}  Bereits eingeloggt als: ${WHOAMI}${NC}"
    echo -n "  Erneut einloggen? (j/N): "
    read -r RELOGIN
    if [[ "$RELOGIN" == "j" || "$RELOGIN" == "J" ]]; then
        huggingface-cli login
    fi
else
    huggingface-cli login
fi

# GPU Check
echo ""
echo -e "${YELLOW}GPU-Check:${NC}"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    python3 -c "import torch; print(f'  PyTorch CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
else
    echo -e "${RED}  nvidia-smi nicht gefunden! GPU-Treiber prüfen.${NC}"
fi

echo ""
echo -e "${CYAN}======================================"
echo " Installation abgeschlossen!"
echo -e "======================================${NC}"
echo ""
echo -e "Nächster Schritt - Server starten:"
echo -e "  ${GREEN}bash /mnt/d/GitHub/Mistral/HTML/scripts/start_voxtral_batch.sh${NC}     # Batch (3B)"
echo -e "  ${GREEN}bash /mnt/d/GitHub/Mistral/HTML/scripts/start_voxtral_realtime.sh${NC}  # Realtime (4B)"
echo ""
