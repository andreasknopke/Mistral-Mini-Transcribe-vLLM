"""Qwen3-ForcedAligner API Server – DGX Spark Edition.

Provides /v1/align endpoint: Audio + Text → Word-level timestamps.
Used by the Voxtral proxy to add timestamps to transcriptions.

Port: 7862 (same as the old VibeVoice service)
"""

import gc
import os
import shutil
import signal
import tempfile
import uuid

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.model_manager import aligner_manager
from src.transcriber import align_audio

load_dotenv()

AUTH_USER = os.getenv("VIBEVOICE_AUTH_USERNAME", "root")
AUTH_PASS = os.getenv("VIBEVOICE_AUTH_PASSWORD", "change-me")

TEMP_DIR = "/tmp/forced-aligner"
os.makedirs(TEMP_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

app = FastAPI(title="Qwen3-ForcedAligner", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "qwen3-forced-aligner",
        "model": "Qwen/Qwen3-ForcedAligner-0.6B",
        "loaded": aligner_manager.is_loaded,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "loaded": aligner_manager.is_loaded}


@app.post("/v1/align")
async def align(
    file: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(default="German"),
):
    """Forced-align text to audio. Returns word + segment timestamps.

    Args:
        file: Audio file (wav, mp3, flac, etc.)
        text: Transcription text to align to the audio
        language: Language name (German, English, French, etc.)

    Returns:
        JSON with segments (each with start, end, text, words[])
    """
    # Save upload to temp file
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}{suffix}")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        segments_json, status = align_audio(
            file_path=tmp_path,
            text=text,
            language=language,
        )

        import json
        segments = json.loads(segments_json)

        return JSONResponse({
            "segments": segments,
            "status": status,
            "language": language,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/v1/system/cleanup")
async def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "ok", "message": "GPU memory cleared"}


@app.post("/v1/system/restart")
async def restart():
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "restarting"}


if __name__ == "__main__":
    # ForcedAligner always runs on CPU – on DGX Spark the memory is unified
    # anyway, and this keeps GPU compute cores free for Voxtral / Correction-LLM.
    device = "cpu"
    print(f"--- ForcedAligner: 🟡 CPU (unified memory) ---")

    # Eager-load the model at startup
    print("--- ForcedAligner: Pre-loading model... ---")
    aligner_manager.get_model(device=device)

    host = os.getenv("VIBEVOICE_HOST", "0.0.0.0")
    port = int(os.getenv("VIBEVOICE_PORT", "7862"))
    print(f"--- ForcedAligner: Starting on {host}:{port} ---")
    uvicorn.run(app, host=host, port=port, log_level="info")
