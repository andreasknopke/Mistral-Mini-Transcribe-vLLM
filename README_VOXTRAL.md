# Voxtral Lokal – Mistral STT auf eigener GPU (Windows 11 + WSL2)

Mistral's Voxtral ist ein Open-Weight Speech-to-Text-Modell, das lokal auf NVIDIA GPUs läuft.
Es wird über **vLLM** als OpenAI-kompatibler Server bereitgestellt.

## Links
- Modell: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- Realtime-Modell: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- Docs: https://docs.mistral.ai/capabilities/audio_transcription/
- Blog: https://mistral.ai/news/voxtral-transcribe-2

---

## Verfügbare Modelle

| Modell | Parameter | VRAM (fp16) | Geschwindigkeit | Sprachen |
|--------|-----------|-------------|-----------------|----------|
| `mistralai/Voxtral-Mini-3B-2507` | 3B | ~6-8 GB | Schnell | 9 (inkl. DE) |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | 4B | ~8-10 GB | Echtzeit-Streaming | 9 (inkl. DE) |
| `mistralai/Voxtral-Small-24B-2507` | 24B | ~48 GB | Langsamer, höchste Qualität | 9 (inkl. DE) |

**Empfehlung:** `Voxtral-Mini-3B-2507` für Batch-Transkription, `Voxtral-Mini-4B-Realtime-2602` für Live-Diktat.

### GPU-Anforderungen

| GPU | VRAM | Mini 3B | Mini 4B Realtime | Small 24B |
|-----|------|---------|------------------|-----------|
| RTX 3060 | 12 GB | ✅ | ✅ | ❌ |
| RTX 3090 / 4090 | 24 GB | ✅ | ✅ | ❌ |
| V100 | 32 GB | ✅ | ✅ | ❌ |
| A100 | 40/80 GB | ✅ | ✅ | ✅ (40GB) |

---

## Voraussetzungen
- **NVIDIA GPU** mit mindestens 12 GB VRAM
- **NVIDIA Treiber** ≥ 525 (für CUDA 12.x) — Windows Game-Ready oder Studio-Treiber
- **WSL2** mit Ubuntu (wird automatisch eingerichtet)
- **Python** ≥ 3.10 (wird in WSL2 installiert)
- **HuggingFace Account** (Modelle erfordern Lizenzzustimmung)

---

## Schnellstart (Windows 11)

### 1. WSL2 einrichten (einmalig)
```powershell
# PowerShell als Administrator ausführen:
.\scripts\01_setup_wsl2.ps1
```

### 2. In WSL2 installieren (einmalig)
```bash
# In WSL2/Ubuntu Terminal:
bash /mnt/d/GitHub/Mistral/HTML/scripts/02_install_voxtral.sh
```

### 3. Server starten
```bash
# Batch-Modell (3B) — für Datei-Transkription:
bash /mnt/d/GitHub/Mistral/HTML/scripts/start_voxtral_batch.sh

# ODER Realtime-Modell (4B) — für Live-Diktat:
bash /mnt/d/GitHub/Mistral/HTML/scripts/start_voxtral_realtime.sh
```

### 4. Testen
```bash
bash /mnt/d/GitHub/Mistral/HTML/scripts/test_voxtral.sh
```

---

## Env-Variablen

| Variable | Default | Beschreibung |
|----------|---------|-------------|
| `VOXTRAL_LOCAL_URL` | `http://localhost:8000` | URL des vLLM-Servers (HTTP + WS) |
| `VOXTRAL_LOCAL_MODEL` | `mistralai/Voxtral-Mini-3B-2507` | HuggingFace Modell-ID (Batch) |

> **WebSocket-URL:** Wird automatisch aus `VOXTRAL_LOCAL_URL` abgeleitet:
> `http://localhost:8000` → `ws://localhost:8000/v1/realtime`

---

## Troubleshooting

### "CUDA out of memory"
```bash
--max-model-len 4096    # Kleinere max-model-len
--max-num-seqs 1        # Weniger parallele Anfragen
```

### "Model not found" / 403
- HuggingFace Login prüfen: `huggingface-cli whoami`
- Lizenz auf der Modell-Seite akzeptiert?
- Token hat Read-Berechtigung?

### Server startet, aber GPU wird nicht genutzt
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### WebSocket verbindet nicht (Realtime-Modus)
- **Falsches Modell?** `/v1/realtime` existiert nur mit dem Realtime-Modell
- **Port blockiert?** `lsof -i :8000` prüfen
- **Firewall?** In PowerShell: `netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$(wsl hostname -I)`
