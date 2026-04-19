"""
Voxtral vLLM compatibility proxy.

Stellt eine OpenAI-kompatible Audio-API vor einem vLLM-Server bereit und
nutzt den Qwen3-ForcedAligner Service (Port 7862) fuer echte Timestamps
im `response_format=verbose_json` Format.
"""

import logging
import os

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voxtral-vllm-proxy")

UPSTREAM_URL = os.environ.get("VOXTRAL_VLLM_UPSTREAM_URL", "http://127.0.0.1:8001").rstrip("/")
ALIGNER_URL = os.environ.get("VOXTRAL_ALIGNER_URL", "http://127.0.0.1:7862").rstrip("/")
HOST = os.environ.get("VOXTRAL_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOXTRAL_PROXY_PORT", "8000"))
REQUEST_TIMEOUT = float(os.environ.get("VOXTRAL_PROXY_TIMEOUT_SEC", "900"))
ALIGNER_TIMEOUT = float(os.environ.get("VOXTRAL_ALIGNER_TIMEOUT_SEC", "120"))
# Audio files smaller than this skip the aligner (live chunks are typically < 1 MB)
ALIGNER_MIN_BYTES = int(os.environ.get("VOXTRAL_ALIGNER_MIN_BYTES", str(1 * 1024 * 1024)))  # 1 MB

app = FastAPI(title="Voxtral vLLM Compat Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LANGUAGE_MAP = {
    "de": "German", "en": "English", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
}

# Debug: store last response for comparison (disabled in production)
# Set VOXTRAL_DEBUG=1 to enable
import json as _json
_LAST_RESPONSE_FILE = "/tmp/last_voxtral_timestamps.json"
_DEBUG_ENABLED = os.environ.get("VOXTRAL_DEBUG", "0") == "1"

def _save_debug_response(data: dict) -> None:
    """Save last verbose_json response for debugging (only when VOXTRAL_DEBUG=1)."""
    if not _DEBUG_ENABLED:
        return
    try:
        import datetime
        debug = {"saved_at": datetime.datetime.now().isoformat(), "response": data}
        with open(_LAST_RESPONSE_FILE, "w") as f:
            _json.dump(debug, f, ensure_ascii=False, indent=2)
        logger.info("Debug: saved last response to %s (%d segments)", _LAST_RESPONSE_FILE, len(data.get("segments", [])))
    except Exception as exc:
        logger.warning("Debug: failed to save response: %s", exc)


async def _align_with_timestamps(
    file_bytes: bytes,
    file_name: str,
    text: str,
    language: str,
    temperature: float,
) -> dict:
    """Call ForcedAligner service to get real timestamps, with fallback to dummy."""
    normalized_text = text.strip()
    if not normalized_text:
        return _build_fallback_verbose_json("", language, temperature)

    lang_full = LANGUAGE_MAP.get(language, language)

    try:
        async with httpx.AsyncClient(timeout=ALIGNER_TIMEOUT) as client:
            resp = await client.post(
                f"{ALIGNER_URL}/v1/align",
                data={"text": normalized_text, "language": lang_full},
                files={"file": (file_name, file_bytes, "application/octet-stream")},
            )
        if resp.status_code != 200:
            logger.warning("Aligner returned %s: %s – falling back to dummy", resp.status_code, resp.text[:200])
            return _build_fallback_verbose_json(normalized_text, language, temperature)

        result = resp.json()
        segments_raw = result.get("segments", [])
    except Exception as exc:
        logger.warning("Aligner call failed: %s – falling back to dummy", exc)
        return _build_fallback_verbose_json(normalized_text, language, temperature)

    # Convert aligner segments to WhisperX-compatible format
    # WhisperX segments: { start, end, text, words: [{word, start, end, score}] }
    segments = []
    for seg in segments_raw:
        words = [
            {"word": w["word"], "start": w["start"], "end": w["end"], "score": 1.0}
            for w in seg.get("words", [])
        ]
        segments.append({
            "start": round(seg.get("start", 0.0), 3),
            "end": round(seg.get("end", 0.0), 3),
            "text": seg.get("text", ""),
            "words": words,
        })

    duration = segments[-1]["end"] if segments else 0.0

    return {
        "task": "transcribe",
        "language": language,
        "duration": duration,
        "text": normalized_text,
        "segments": segments,
    }


def _build_fallback_verbose_json(text: str, language: str, temperature: float) -> dict:
    """Fallback when aligner is unavailable."""
    segment = {
        "start": 0.0,
        "end": 0.0,
        "text": text,
        "words": [],
    }
    return {
        "task": "transcribe", "language": language, "duration": 0.0,
        "text": text, "segments": [segment] if text else [],
    }


@app.get("/debug/last_response")
async def debug_last_response():
    """Return the last verbose_json response for debugging (requires VOXTRAL_DEBUG=1)."""
    if not _DEBUG_ENABLED:
        return JSONResponse({"error": "Debug disabled. Set VOXTRAL_DEBUG=1 to enable."}, status_code=403)
    try:
        with open(_LAST_RESPONSE_FILE, "r") as f:
            return JSONResponse(_json.load(f))
    except FileNotFoundError:
        return JSONResponse({"error": "No response saved yet"}, status_code=404)


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "voxtral-vllm-compat-proxy", "upstream": UPSTREAM_URL}


@app.get("/health")
async def health() -> Response:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            upstream = await client.get(f"{UPSTREAM_URL}/health")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Upstream health check failed: {exc}") from exc

    content_type = upstream.headers.get("content-type", "application/json")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=content_type.split(";")[0])


@app.get("/v1/models")
async def list_models() -> Response:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            upstream = await client.get(f"{UPSTREAM_URL}/v1/models")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Upstream model list failed: {exc}") from exc

    content_type = upstream.headers.get("content-type", "application/json")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=content_type.split(";")[0])


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=None),
    language: str = Form(default="de"),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    requested_format = response_format
    upstream_format = "json" if requested_format == "verbose_json" else requested_format

    file_bytes = await file.read()
    file_name = file.filename or "audio.wav"
    content_type = file.content_type or "application/octet-stream"

    logger.info(
        "Transcription request: file=%s size=%s format=%s upstream_format=%s",
        file_name,
        len(file_bytes),
        requested_format,
        upstream_format,
    )

    data = {
        "language": language,
        "response_format": upstream_format,
        "temperature": str(temperature),
    }
    if model:
        data["model"] = model

    files = {
        "file": (file_name, file_bytes, content_type),
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            upstream = await client.post(
                f"{UPSTREAM_URL}/v1/audio/transcriptions",
                data=data,
                files=files,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Upstream transcription failed: {exc}") from exc

    if upstream.status_code >= 400:
        raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    if requested_format == "text":
        return PlainTextResponse(upstream.text)

    if requested_format == "verbose_json":
        try:
            payload = upstream.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Invalid upstream JSON: {exc}") from exc

        text = payload.get("text", "") if isinstance(payload, dict) else ""

        # Skip aligner for small audio (live/realtime chunks) – not worth the overhead
        if len(file_bytes) < ALIGNER_MIN_BYTES:
            logger.info("Skipping aligner for small audio (%s bytes < %s)", len(file_bytes), ALIGNER_MIN_BYTES)
            verbose = _build_fallback_verbose_json(text, language, temperature)
        else:
            verbose = await _align_with_timestamps(
                file_bytes=file_bytes,
                file_name=file_name,
                text=text,
                language=language,
                temperature=temperature,
            )
        _save_debug_response(verbose)
        return JSONResponse(verbose)

    content_type = upstream.headers.get("content-type", "application/json")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=content_type.split(";")[0])


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting proxy on %s:%s -> %s", HOST, PORT, UPSTREAM_URL)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")