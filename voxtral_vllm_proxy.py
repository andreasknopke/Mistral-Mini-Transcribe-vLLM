"""
Voxtral vLLM compatibility proxy.

Stellt eine OpenAI-kompatible Audio-API vor einem vLLM-Server bereit und
emuliert `response_format=verbose_json` mit Dummy-Segmenten, da Voxtral via
vLLM dieses Format aktuell nicht nativ unterstuetzt.
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
HOST = os.environ.get("VOXTRAL_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOXTRAL_PROXY_PORT", "8000"))
REQUEST_TIMEOUT = float(os.environ.get("VOXTRAL_PROXY_TIMEOUT_SEC", "900"))

app = FastAPI(title="Voxtral vLLM Compat Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_dummy_verbose_json(text: str, language: str, temperature: float) -> dict:
    normalized_text = text.strip()
    segment = {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": 0.0,
        "text": normalized_text,
        "tokens": [],
        "temperature": temperature,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
        "words": [],
    }
    return {
        "task": "transcribe",
        "language": language,
        "duration": 0.0,
        "text": normalized_text,
        "segments": [segment] if normalized_text else [],
    }


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
        return JSONResponse(_build_dummy_verbose_json(text=text, language=language, temperature=temperature))

    content_type = upstream.headers.get("content-type", "application/json")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=content_type.split(";")[0])


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting proxy on %s:%s -> %s", HOST, PORT, UPSTREAM_URL)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")