import mimetypes
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List

SUPPORTED_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aac",
    ".m4a",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".webm",
    ".mov",
    ".mkv",
)

TEMP_DIR = Path(os.getenv("WHISPERX_TEMP_DIR", "/tmp/whisperx-spark"))


def ensure_temp_dir() -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


def temp_copy_path(source_path: str) -> Path:
    ensure_temp_dir()
    extension = Path(source_path).suffix.lower() or ".wav"
    return TEMP_DIR / f"{uuid.uuid4().hex}{extension}"


def materialize_input_file(file_path: str) -> str:
    source = Path(file_path)
    if not source.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    destination = temp_copy_path(file_path)
    shutil.copy2(source, destination)
    return str(destination)


def is_valid_multimedia_file(file_path: str) -> bool:
    normalized_path = os.path.normpath(file_path)
    mime_type, _ = mimetypes.guess_type(normalized_path)
    is_supported_mime = mime_type and (
        mime_type.startswith("audio") or mime_type.startswith("video")
    )
    return bool(is_supported_mime or normalized_path.lower().endswith(SUPPORTED_EXTENSIONS))


def validate_multimedia_file(file_path: str) -> str:
    if not is_valid_multimedia_file(file_path):
        raise ValueError(
            "Nicht unterstütztes Dateiformat. Unterstützt werden Audio- und Video-Dateien wie WAV, MP3, M4A, MP4 und WEBM."
        )
    return file_path


def convert_to_wav(file_path: str) -> str:
    input_path = Path(file_path)
    output_path = input_path.with_suffix(".wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg-Konvertierung fehlgeschlagen: {result.stderr.strip()}")
    return str(output_path)


def save_transcription(base_name: str, text: str, segments_json: str) -> str:
    ensure_temp_dir()
    target = TEMP_DIR / f"{base_name}.json"
    target.write_text(segments_json, encoding="utf-8")
    transcript_file = TEMP_DIR / f"{base_name}.txt"
    transcript_file.write_text(text, encoding="utf-8")
    return str(target)


def cleanup(paths: Iterable[str]) -> None:
    for path in paths:
        if not path:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def cleanup_temp_root() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
