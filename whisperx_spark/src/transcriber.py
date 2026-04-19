import gc
import json
import os
import threading
import time
import traceback
from pathlib import Path

import torch
import whisperx

from src.model_manager import model_pool
from src.utils import cleanup, convert_to_wav, materialize_input_file, save_transcription, validate_multimedia_file

FORMAT_PROMPT = (
    "Klammern (so wie diese) und Satzzeichen wie Punkt, Komma, Doppelpunkt und Semikolon sind wichtig."
)

LANGUAGE_OPTIONS = {
    "Identify": None,
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
}

_ALIGNMENT_CACHE: dict[tuple[str, str, str], tuple[object, object]] = {}
_ALIGNMENT_LOCK = threading.Lock()
GERMAN_ALIGNMENT_MODEL = os.getenv(
    "WHISPERX_GERMAN_ALIGNMENT_MODEL",
    "jonatasgrosman/wav2vec2-large-xlsr-53-german",
)


def clear_memory(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_alignment_resources(language_code: str, alignment_device: str):
    align_model_id = GERMAN_ALIGNMENT_MODEL if language_code == "de" else None
    cache_key = (language_code, alignment_device, align_model_id or "default")
    with _ALIGNMENT_LOCK:
        if cache_key in _ALIGNMENT_CACHE:
            return _ALIGNMENT_CACHE[cache_key]

        print(
            f"--- ALIGN: Lade Alignment-Modell für Sprache '{language_code}' auf '{alignment_device}'"
            f" ({align_model_id or 'whisperx-default'}) ---"
        )
        align_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=alignment_device,
            model_name=align_model_id,
        )
        _ALIGNMENT_CACHE[cache_key] = (align_model, metadata)
        return align_model, metadata


def transcribe_audio(
    file_path: str,
    language: str,
    model_name: str,
    device: str,
    initial_prompt_user: str = "",
    speed_mode: bool = False,
):
    worker = None
    temp_paths: list[str] = []
    try:
        start_time = time.time()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("--- SYSTEM: Device 'cuda' → 'cpu' (kein CUDA-Backend verfügbar) ---")

        combined_prompt = f"{FORMAT_PROMPT} {initial_prompt_user}".strip()
        worker = model_pool.acquire(model_name, device, timeout=180)
        model = worker.model

        beam_size = 1 if speed_mode else 5
        if hasattr(model, "options"):
            try:
                if hasattr(model.options, "_replace"):
                    model.options = model.options._replace(
                        initial_prompt=combined_prompt,
                        beam_size=beam_size,
                        best_of=beam_size,
                        temperatures=[0.0] if speed_mode else [0.0, 0.2, 0.4],
                        condition_on_previous_text=True,
                        prompt_reset_on_temperature=0.5,
                    )
                print(
                    f"--- WORKER {worker.worker_id}: {'Speed-Mode' if speed_mode else 'Deep-Context'} aktiviert (Beam: {beam_size}) ---"
                )
            except Exception as exc:
                print(f"--- WORKER {worker.worker_id}: Optionen teilweise nicht gesetzt: {exc} ---")

        if hasattr(model, "asr_options"):
            model.asr_options["initial_prompt"] = combined_prompt

        local_input = materialize_input_file(file_path)
        temp_paths.append(local_input)
        validated_file_path = validate_multimedia_file(local_input)
        if not validated_file_path.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a")):
            converted = convert_to_wav(validated_file_path)
            temp_paths.append(converted)
            validated_file_path = converted

        audio = whisperx.load_audio(validated_file_path)
        transcribe_options = {"batch_size": 16 if speed_mode else 4}
        if language != "Identify":
            transcribe_options["language"] = LANGUAGE_OPTIONS.get(language, "de")

        print(
            f"--- WORKER {worker.worker_id}: Starte Transkription für {Path(validated_file_path).name} ---"
        )
        result = model.transcribe(audio, **transcribe_options)
        final_segments = result.get("segments", [])
        detected_lang = result.get("language", LANGUAGE_OPTIONS.get(language, "de") or "de")

        if final_segments and not speed_mode:
            alignment_device = os.getenv(
                "WHISPERX_ALIGNMENT_DEVICE",
                device,
            )
            align_model, metadata = _get_alignment_resources(detected_lang, alignment_device)
            aligned = whisperx.align(
                final_segments,
                align_model,
                metadata,
                audio,
                alignment_device,
                return_char_alignments=False,
            )
            final_segments = aligned.get("segments", final_segments)

        full_text = " ".join(segment.get("text", "").strip() for segment in final_segments).strip()
        if not full_text:
            full_text = result.get("text", "").strip()

        segment_payload = []
        for segment in final_segments:
            normalized_segment = {
                "start": round(float(segment.get("start", 0.0)), 3),
                "end": round(float(segment.get("end", 0.0)), 3),
                "text": segment.get("text", "").strip(),
            }
            if "words" in segment and isinstance(segment.get("words"), list):
                normalized_segment["words"] = [
                    {
                        key: round(float(value), 3) if key in {"start", "end", "score"} and isinstance(value, (int, float)) else value
                        for key, value in word.items()
                    }
                    for word in segment["words"]
                    if isinstance(word, dict)
                ]
            segment_payload.append(normalized_segment)
        segments_json = json.dumps(segment_payload, ensure_ascii=False)
        save_transcription(Path(validated_file_path).stem, full_text, segments_json)

        # Debug: save last response for comparison (disabled in production)
        # Set WHISPERX_DEBUG=1 to enable
        if os.getenv("WHISPERX_DEBUG", "0") == "1":
            try:
                import datetime
                debug = {
                    "saved_at": datetime.datetime.now().isoformat(),
                    "response": {
                        "text": full_text,
                        "segments": segment_payload,
                        "language": detected_lang,
                        "mode": "speed" if speed_mode else "precision",
                    }
                }
                with open("/tmp/last_whisperx_timestamps.json", "w") as f:
                    json.dump(debug, f, ensure_ascii=False, indent=2)
                print(f"Debug: saved last WhisperX response ({len(segment_payload)} segments)")
            except Exception as dbg_exc:
                print(f"Debug: failed to save WhisperX response: {dbg_exc}")

        elapsed = time.time() - start_time
        status = (
            f"{'Speed' if speed_mode else 'Präzisions'}-Modus fertig in {elapsed:.2f}s "
            f"(Worker {worker.worker_id}, Sprache: {detected_lang})"
        )
        return [full_text, segments_json, status]
    except Exception as exc:
        print(
            f"--- TRANSCRIBE ERROR: model={model_name}, requested_device={device}, "
            f"language={language}, file={file_path}: {exc} ---"
        )
        traceback.print_exc()
        return ["", "[]", f"Fehler: {exc}"]
    finally:
        cleanup(temp_paths)
        clear_memory(device)
        if worker is not None:
            model_pool.release(worker)
