"""
Voxtral Local Server - Windows Native
OpenAI-kompatibler Audio-Transkriptions-Server mit Voxtral auf lokaler GPU.

Speicher-Management:
  - response_format=verbose_json  → Offline-Batch → Modell wird ENTLADEN danach
  - response_format=json/text     → Online-Chunks → Modell bleibt GELADEN
  - Idle-Timer: Modell wird nach 60s ohne Anfrage automatisch entladen
  - POST /v1/unload               → Manuelles Entladen (z.B. bevor LLM startet)

API-Endpunkte:
  GET  /health                      - Health-Check (zeigt ob Modell geladen)
  GET  /v1/models                   - Geladenes Modell anzeigen
  POST /v1/audio/transcriptions     - Audio transkribieren (OpenAI-kompatibel)
  POST /v1/unload                   - Modell manuell aus GPU entladen
"""

import os
import gc
import time
import logging
import tempfile
import threading

# CUDA Speicher-Fragmentierung vermeiden
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voxtral")

# HuggingFace HTTP-Spam unterdruecken
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
logging.getLogger("mistral_common").setLevel(logging.WARNING)

# ============================================================
# Konfiguration
# ============================================================
MODEL_ID = os.environ.get("VOXTRAL_LOCAL_MODEL", "mistralai/Voxtral-Mini-3B-2507")
HOST = os.environ.get("VOXTRAL_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOXTRAL_PORT", "8000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDLE_TIMEOUT_SEC = 60  # Modell nach 60s Inaktivitaet entladen

# ============================================================
# Processor laden (klein, bleibt immer im RAM)
# ============================================================
logger.info(f"Device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from transformers import VoxtralForConditionalGeneration, AutoProcessor

logger.info(f"Lade Processor fuer {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
logger.info("Processor geladen! (Modell wird on-demand geladen)")

# ============================================================
# Modell-State: wird on-demand geladen und bei Bedarf entladen
# ============================================================
_model_lock = threading.Lock()
_ml_model = None          # None = nicht geladen
_idle_timer = None        # Timer fuer automatisches Entladen
_last_request_time = 0.0


def _get_or_load_model():
    """Modell laden falls noch nicht geladen. Thread-safe."""
    global _ml_model, _last_request_time
    with _model_lock:
        _last_request_time = time.time()
        _cancel_idle_timer()
        if _ml_model is not None:
            logger.info("Modell bereits geladen, verwende cached Version")
            return _ml_model
    # Laden ausserhalb des Locks (dauert lange, soll nicht blockieren)
    logger.info("Lade Voxtral-Modell auf GPU...")
    t0 = time.time()
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(DEVICE)
    model.eval()
    dur = time.time() - t0
    if DEVICE == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Modell geladen in {dur:.1f}s ({vram_used:.1f} GB VRAM belegt)")
    else:
        logger.info(f"Modell geladen in {dur:.1f}s")
    with _model_lock:
        _ml_model = model
    return model


def _do_unload():
    """Modell komplett aus GPU-Speicher entladen. Thread-safe."""
    global _ml_model
    with _model_lock:
        if _ml_model is None:
            return
        model = _ml_model
        _ml_model = None
        _cancel_idle_timer()
    logger.info("Entlade Modell aus GPU-Speicher...")
    model.to("cpu")
    for param in model.parameters():
        param.data = torch.empty(0)
        if param.grad is not None:
            param.grad = None
    for buf in model.buffers():
        buf.data = torch.empty(0)
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Modell entladen. VRAM belegt: {vram_used:.2f} GB")


def _cancel_idle_timer():
    """Laufenden Idle-Timer abbrechen."""
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _start_idle_timer():
    """Idle-Timer starten: entlaedt Modell nach IDLE_TIMEOUT_SEC Sekunden."""
    global _idle_timer
    _cancel_idle_timer()
    _idle_timer = threading.Timer(IDLE_TIMEOUT_SEC, _on_idle_timeout)
    _idle_timer.daemon = True
    _idle_timer.start()
    logger.info(f"Idle-Timer gestartet ({IDLE_TIMEOUT_SEC}s)")


def _on_idle_timeout():
    """Callback wenn Idle-Timer ablaeuft."""
    logger.info(f"Idle-Timeout ({IDLE_TIMEOUT_SEC}s) - entlade Modell automatisch")
    _do_unload()


def is_model_loaded() -> bool:
    return _ml_model is not None


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Voxtral Local Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    info = {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "model_loaded": is_model_loaded(),
    }
    if DEVICE == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        info["vram_used_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
    return info


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/unload")
async def unload_endpoint():
    """Modell manuell aus GPU entladen (z.B. bevor ein LLM gestartet wird)."""
    if not is_model_loaded():
        return {"status": "already_unloaded", "vram_used_gb": 0.0}
    _do_unload()
    vram = round(torch.cuda.memory_allocated() / 1024**3, 2) if DEVICE == "cuda" else 0.0
    return {"status": "unloaded", "vram_used_gb": vram}


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Audio laden und auf 16kHz Mono resampled."""
    import librosa
    audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
    return audio


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=None),
    language: str = Form(default="de"),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """
    OpenAI-kompatibler Transkriptions-Endpunkt.

    Speicher-Verhalten abhaengig von response_format:
      - verbose_json → Offline-Batch: Modell wird danach ENTLADEN
      - json/text    → Online-Chunks: Modell bleibt geladen + Idle-Timer
    """
    start_time = time.time()
    is_offline = (response_format == "verbose_json")

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    chunk_paths = []

    try:
        mode_label = "OFFLINE" if is_offline else "ONLINE"
        logger.info(f"[{mode_label}] Transkribiere: {file.filename} ({len(content)} bytes)")

        audio_data = load_audio(tmp_path)
        audio_duration = len(audio_data) / 16000

        # Chunking fuer lange Audios (120s pro Chunk)
        CHUNK_MAX_SECONDS = 120

        if audio_duration > CHUNK_MAX_SECONDS:
            samples_per_chunk = CHUNK_MAX_SECONDS * 16000
            num_chunks = int(np.ceil(len(audio_data) / samples_per_chunk))
            logger.info(f"Audio zu lang ({audio_duration:.0f}s) - teile in {num_chunks} Chunks")

            for i in range(num_chunks):
                chunk_start = i * samples_per_chunk
                chunk_end = min((i + 1) * samples_per_chunk, len(audio_data))
                chunk_data = audio_data[chunk_start:chunk_end]

                chunk_path = tmp_path + f".chunk{i}.wav"
                sf.write(chunk_path, chunk_data, 16000)
                chunk_paths.append(chunk_path)
        else:
            chunk_paths = [tmp_path]

        # Modell laden (oder cached verwenden)
        ml_model = _get_or_load_model()

        # Transkription aller Chunks
        all_texts = []
        for chunk_idx, chunk_path in enumerate(chunk_paths):
            if len(chunk_paths) > 1:
                logger.info(f"  Chunk {chunk_idx + 1}/{len(chunk_paths)}...")

            inputs = processor.apply_transcription_request(
                language=language,
                audio=chunk_path,
                model_id=MODEL_ID,
            )
            inputs = inputs.to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)

            chunk_audio = load_audio(chunk_path)
            chunk_dur = len(chunk_audio) / 16000
            max_tokens = max(512, min(4096, int(chunk_dur * 3.5)))

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                outputs = ml_model.generate(**inputs, max_new_tokens=max_tokens)

            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            chunk_text = decoded[0].strip() if decoded else ""
            all_texts.append(chunk_text)

            del inputs, outputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        text = " ".join(all_texts).strip()

        # Speicher-Management je nach Modus
        if is_offline:
            # Offline-Batch: Modell komplett entladen (danach kommt LLM etc.)
            _do_unload()
            logger.info("[OFFLINE] Modell entladen")
        else:
            # Online-Chunks: Modell behalten, Idle-Timer starten
            _start_idle_timer()
            logger.info(f"[ONLINE] Modell bleibt geladen (auto-entladen nach {IDLE_TIMEOUT_SEC}s Idle)")

        duration = time.time() - start_time
        logger.info(
            f"Transkription fertig: {duration:.1f}s "
            f"fuer {audio_duration:.1f}s Audio "
            f"({len(text.split())} Woerter)"
        )

        if response_format == "verbose_json":
            return JSONResponse({
                "text": text,
                "language": language,
                "duration": audio_duration,
                "segments": [],
            })
        elif response_format == "text":
            return JSONResponse(content=text, media_type="text/plain")
        else:
            return JSONResponse({"text": text})

    except Exception as e:
        logger.error(f"Fehler bei Transkription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Temp-Dateien aufraeumen
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        for cp in chunk_paths:
            if cp != tmp_path and os.path.exists(cp):
                os.unlink(cp)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info(f"Starte Server auf {HOST}:{PORT}")
    logger.info(f"Speicher-Modus: verbose_json=Offline(entladen), json/text=Online(behalten+{IDLE_TIMEOUT_SEC}s Idle)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
