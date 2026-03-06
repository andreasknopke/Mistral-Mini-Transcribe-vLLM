#!/bin/bash
# ============================================================
# Voxtral Server testen
# ============================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SERVER_URL="${VOXTRAL_LOCAL_URL:-http://localhost:8000}"

echo -e "${CYAN}======================================"
echo " Voxtral Server Test"
echo -e "======================================${NC}"
echo ""
echo "Server URL: $SERVER_URL"
echo ""

# 1. Health-Check
echo -e "${YELLOW}[1/3] Health-Check...${NC}"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/health" 2>/dev/null)
if [ "$HEALTH" = "200" ]; then
    echo -e "${GREEN}  ✅ Server ist erreichbar (HTTP $HEALTH)${NC}"
else
    echo -e "${RED}  ❌ Server nicht erreichbar (HTTP $HEALTH)${NC}"
    echo -e "${YELLOW}  Ist der vLLM Server gestartet?${NC}"
    echo -e "  Starte mit: bash /mnt/d/GitHub/Mistral/HTML/scripts/start_voxtral_batch.sh"
    exit 1
fi

# 2. Modell-Info
echo -e "${YELLOW}[2/3] Geladenes Modell prüfen...${NC}"
MODELS=$(curl -s "$SERVER_URL/v1/models" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  $MODELS${NC}" | python3 -m json.tool 2>/dev/null || echo "  $MODELS"
else
    echo -e "${RED}  Konnte Modell-Info nicht abrufen${NC}"
fi

# 3. Transkriptions-Test (nur wenn Audio-Datei vorhanden)
echo -e "${YELLOW}[3/3] Transkriptions-Test...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$TEST_DIR/test-audio.wav" ]; then
    echo "  Teste mit test-audio.wav..."
    RESULT=$(curl -s "$SERVER_URL/v1/audio/transcriptions" \
      -F "file=@$TEST_DIR/test-audio.wav" \
      -F "model=mistralai/Voxtral-Mini-3B-2507" \
      -F "language=de" \
      -F "response_format=verbose_json")
    echo -e "${GREEN}  Ergebnis:${NC}"
    echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "  $RESULT"
else
    echo -e "${YELLOW}  Keine test-audio.wav gefunden.${NC}"
    echo "  Lege eine WAV-Datei als 'test-audio.wav' im Projektverzeichnis ab zum Testen."
    echo ""
    echo "  Manueller Test:"
    echo "  curl $SERVER_URL/v1/audio/transcriptions \\"
    echo "    -F file=@deine-datei.wav \\"
    echo "    -F model=mistralai/Voxtral-Mini-3B-2507 \\"
    echo "    -F language=de \\"
    echo "    -F response_format=verbose_json"
fi

# 4. WebSocket-Check (für Realtime)
echo ""
echo -e "${YELLOW}[Bonus] WebSocket-Endpoint prüfen...${NC}"
WS_CHECK=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: dGVzdA==" \
  "$SERVER_URL/v1/realtime" 2>/dev/null)
if [ "$WS_CHECK" = "101" ]; then
    echo -e "${GREEN}  ✅ WebSocket-Endpoint verfügbar (Realtime-Modell geladen)${NC}"
elif [ "$WS_CHECK" = "404" ]; then
    echo -e "${YELLOW}  ℹ️  Kein WebSocket-Endpoint (Batch-Modell geladen — das ist OK)${NC}"
else
    echo -e "${YELLOW}  WebSocket-Status: HTTP $WS_CHECK${NC}"
fi

echo ""
echo -e "${GREEN}Test abgeschlossen.${NC}"
