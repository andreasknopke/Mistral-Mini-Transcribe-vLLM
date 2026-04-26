import gc
import json
import os
import shutil
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch
import whisperx

from src.model_manager import MODELS, model_pool
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
MODEL_INDEX_BASE = 1 if os.getenv("WHISPERX_MODEL_INDEX_BASE", "1") != "0" else 0
MODEL_ALIASES = {
    "cstr/whisper-large-v3-turbo-german-int8_float32": os.getenv(
        "WHISPERX_TURBO_ALIAS_TARGET",
        "/home/ksai0001_local/models/primeline-turbo-de-int8_bf16",
    ),
}
DEBUG_CAPTURE_DIR = os.getenv("WHISPERX_DEBUG_CAPTURE_DIR", "/tmp/whisperx-debug")


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


def _audio_duration_seconds(audio) -> float:
    try:
        return round(len(audio) / 16000.0, 3)
    except Exception:
        return 0.0


def _audio_signal_stats(audio) -> tuple[float, float]:
    try:
        if len(audio) == 0:
            return 0.0, 0.0
        rms = float((audio**2).mean() ** 0.5)
        peak = float(abs(audio).max())
        return round(rms, 6), round(peak, 6)
    except Exception:
        return 0.0, 0.0


def _capture_debug_audio(source_path: str, reason: str, worker_id: int) -> str | None:
    try:
        capture_dir = Path(DEBUG_CAPTURE_DIR)
        capture_dir.mkdir(parents=True, exist_ok=True)
        source = Path(source_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target = capture_dir / f"{timestamp}_worker{worker_id}_{reason}_{source.name}"
        shutil.copy2(source, target)
        return str(target)
    except Exception as exc:
        print(f"--- WORKER {worker_id}: Debug-Capture fehlgeschlagen: {exc} ---")
        return None


def _transcribe_full_audio_fallback(model, audio, language_code: str | None, audio_duration: float):
    task = "transcribe"
    previous_suppress_tokens = None

    if model.tokenizer is None:
        language_code = language_code or model.detect_language(audio)
        model.tokenizer = whisperx.asr.Tokenizer(
            model.model.hf_tokenizer,
            model.model.model.is_multilingual,
            task=task,
            language=language_code,
        )
    else:
        language_code = language_code or model.tokenizer.language_code
        task = model.tokenizer.task
        if task != model.tokenizer.task or language_code != model.tokenizer.language_code:
            model.tokenizer = whisperx.asr.Tokenizer(
                model.model.hf_tokenizer,
                model.model.model.is_multilingual,
                task=task,
                language=language_code,
            )

    if getattr(model, "suppress_numerals", False):
        previous_suppress_tokens = model.options.suppress_tokens
        numeral_symbol_tokens = whisperx.asr.find_numeral_symbol_tokens(model.tokenizer)
        new_suppressed_tokens = list(set(numeral_symbol_tokens + model.options.suppress_tokens))
        model.options = replace(model.options, suppress_tokens=new_suppressed_tokens)

    try:
        outputs = list(model.__call__([{"inputs": audio}], batch_size=1, num_workers=0))
        if not outputs:
            return {"segments": [], "language": language_code}

        text = outputs[0].get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        text = str(text).strip()
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": round(audio_duration, 3),
                    "text": text,
                }
            ],
            "language": language_code,
            "text": text,
        }
    finally:
        if model.preset_language is None:
            model.tokenizer = None
        if getattr(model, "suppress_numerals", False) and previous_suppress_tokens is not None:
            model.options = replace(model.options, suppress_tokens=previous_suppress_tokens)


def _empty_transcription_response(reason: str, elapsed: float, worker_id: int, language_code: str):
    status = (
        f"Keine aktive Sprache erkannt ({reason}) nach {elapsed:.2f}s "
        f"(Worker {worker_id}, Sprache: {language_code})"
    )
    return ["", "[]", status]


def _looks_like_empty_vad_failure(exc: Exception) -> bool:
    return isinstance(exc, IndexError) and "list index out of range" in str(exc)


def _should_retry_full_audio(audio_duration: float, audio_rms: float, audio_peak: float) -> bool:
    return audio_duration >= 1.0 and (audio_rms >= 0.003 or audio_peak >= 0.03)


def _resolve_model_index(index: int) -> tuple[str, str]:
    candidates: list[tuple[int, str]] = []
    if MODEL_INDEX_BASE == 1:
        candidates.append((index - 1, f"index-1-based:{index}"))
        candidates.append((index, f"index-0-based-fallback:{index}"))
    else:
        candidates.append((index, f"index-0-based:{index}"))
        candidates.append((index - 1, f"index-1-based-fallback:{index}"))

    for candidate_index, reason in candidates:
        if 0 <= candidate_index < len(MODELS):
            return MODELS[candidate_index], reason

    raise ValueError(
        f"Ungültiger Modellindex {index}. Gültige Modelle: 0..{len(MODELS) - 1} oder 1..{len(MODELS)}"
    )


def _normalize_model_name(raw_model_name) -> tuple[str, str]:
    if raw_model_name is None or str(raw_model_name).strip() == "":
        model_name = os.getenv("WHISPERX_MODEL", MODELS[0])
        if model_name in MODEL_ALIASES:
            return MODEL_ALIASES[model_name], f"env-default-alias:{model_name}"
        return model_name, "env-default"

    if isinstance(raw_model_name, str):
        candidate = raw_model_name.strip()
        if candidate in MODEL_ALIASES:
            return MODEL_ALIASES[candidate], f"alias:{candidate}"
        if candidate in MODELS:
            return candidate, "exact-string"
        if candidate.lstrip("+-").isdigit():
            return _resolve_model_index(int(candidate))
        return candidate, "custom-string"

    if isinstance(raw_model_name, int):
        return _resolve_model_index(raw_model_name)

    if isinstance(raw_model_name, float) and raw_model_name.is_integer():
        return _resolve_model_index(int(raw_model_name))

    candidate = str(raw_model_name).strip()
    if candidate in MODEL_ALIASES:
        return MODEL_ALIASES[candidate], f"alias:{candidate}"
    if candidate in MODELS:
        return candidate, f"stringified-{type(raw_model_name).__name__}"
    if candidate.lstrip("+-").isdigit():
        return _resolve_model_index(int(candidate))
    return candidate, f"stringified-{type(raw_model_name).__name__}"


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
    resolved_model_name = model_name
    try:
        start_time = time.time()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("--- SYSTEM: Device 'cuda' → 'cpu' (kein CUDA-Backend verfügbar) ---")

        combined_prompt = f"{FORMAT_PROMPT} {initial_prompt_user}".strip()
        requested_language_code = LANGUAGE_OPTIONS.get(language, "de") or "de"
        resolved_model_name, model_resolution = _normalize_model_name(model_name)
        print(
            "--- TRANSCRIBE REQUEST: "
            f"raw_model={model_name!r} ({type(model_name).__name__}), "
            f"resolved_model={resolved_model_name!r}, resolution={model_resolution}, "
            f"language={language!r}/{requested_language_code}, device={device!r}, speed_mode={speed_mode}, "
            f"file={file_path!r} ---"
        )
        worker = model_pool.acquire(resolved_model_name, device, timeout=180)
        model = worker.model

        beam_size = 2 if speed_mode else 5
        if hasattr(model, "options"):
            try:
                if hasattr(model.options, "_replace"):
                    model.options = model.options._replace(
                        initial_prompt=combined_prompt,
                        beam_size=beam_size,
                        best_of=beam_size,
                        temperatures=[0.0] if speed_mode else [0.0, 0.2, 0.4],
                        condition_on_previous_text=not speed_mode,
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
            transcribe_options["language"] = requested_language_code

        audio_duration = _audio_duration_seconds(audio)
        audio_rms, audio_peak = _audio_signal_stats(audio)
        print(
            f"--- WORKER {worker.worker_id}: Starte Transkription für {Path(validated_file_path).name} "
            f"(dauer={audio_duration:.2f}s, rms={audio_rms:.6f}, peak={audio_peak:.6f}) ---"
        )
        try:
            result = model.transcribe(audio, **transcribe_options)
        except Exception as exc:
            if _looks_like_empty_vad_failure(exc):
                elapsed = time.time() - start_time
                debug_copy = _capture_debug_audio(validated_file_path, "empty_vad", worker.worker_id)
                should_retry = _should_retry_full_audio(audio_duration, audio_rms, audio_peak)
                print(
                    f"--- WORKER {worker.worker_id}: Keine Sprache erkannt "
                    f"({audio_duration:.2f}s Audio, rms={audio_rms:.6f}, peak={audio_peak:.6f}, "
                    f"Datei: {Path(validated_file_path).name}, debug_copy={debug_copy!r}, "
                    f"full_audio_retry={should_retry}) ---"
                )
                if should_retry:
                    print(
                        f"--- WORKER {worker.worker_id}: Starte Full-Audio-Fallback ohne VAD-Segmente "
                        f"für {Path(validated_file_path).name} ---"
                    )
                    result = _transcribe_full_audio_fallback(
                        model,
                        audio,
                        transcribe_options.get("language"),
                        audio_duration,
                    )
                else:
                    return _empty_transcription_response(
                        "VAD ohne Sprachsegmente",
                        elapsed,
                        worker.worker_id,
                        requested_language_code,
                    )
            else:
                raise

        final_segments = result.get("segments", [])
        detected_lang = result.get("language", requested_language_code)

        if not final_segments and not result.get("text", "").strip():
            elapsed = time.time() - start_time
            print(
                f"--- WORKER {worker.worker_id}: WhisperX lieferte keine Segmente "
                f"({audio_duration:.2f}s Audio, Datei: {Path(validated_file_path).name}) ---"
            )
            return _empty_transcription_response(
                "keine Segmente erkannt",
                elapsed,
                worker.worker_id,
                detected_lang,
            )

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
            f"--- TRANSCRIBE ERROR: raw_model={model_name!r}, resolved_model={resolved_model_name!r}, requested_device={device}, "
            f"language={language}, file={file_path}: {exc} ---"
        )
        traceback.print_exc()
        return ["", "[]", f"Fehler: {exc}"]
    finally:
        cleanup(temp_paths)
        clear_memory(device)
        if worker is not None:
            model_pool.release(worker)
