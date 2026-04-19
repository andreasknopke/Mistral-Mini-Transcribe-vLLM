"""Qwen3-ForcedAligner Service – Adds timestamps to existing transcriptions.

Takes audio + text, returns word/segment-level timestamps.
Designed to post-process Voxtral transcriptions.
Handles long audio by chunking (Qwen3-ForcedAligner has ~240s internal limit).
"""

import gc
import json
import math
import os
import shutil
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import torch

from src.model_manager import aligner_manager

TEMP_DIR = Path(os.getenv("ALIGNER_TEMP_DIR", "/tmp/forced-aligner"))
# Maximum audio chunk duration in seconds (aligner fails beyond ~240s)
MAX_CHUNK_SECS = int(os.getenv("ALIGNER_MAX_CHUNK_SECS", "180"))


def _ensure_temp() -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


def clear_memory(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _materialize(file_path: str) -> str:
    """Copy uploaded file to temp dir."""
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
    _ensure_temp()
    ext = src.suffix.lower() or ".wav"
    dst = TEMP_DIR / f"{uuid.uuid4().hex}{ext}"
    shutil.copy2(src, dst)
    return str(dst)


def _get_audio_duration(path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _split_audio_chunks(audio_path: str, chunk_secs: int) -> list[str]:
    """Split audio into chunks of chunk_secs using ffmpeg. Returns list of chunk paths."""
    duration = _get_audio_duration(audio_path)
    if duration <= 0 or duration <= chunk_secs + 10:
        return [audio_path]  # No splitting needed

    chunks = []
    n_chunks = math.ceil(duration / chunk_secs)
    base = Path(audio_path)
    for i in range(n_chunks):
        start = i * chunk_secs
        chunk_path = str(base.parent / f"{base.stem}_chunk{i}{base.suffix}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-ss", str(start), "-t", str(chunk_secs),
                 "-acodec", "copy", chunk_path],
                capture_output=True, timeout=30,
            )
            if Path(chunk_path).exists() and Path(chunk_path).stat().st_size > 100:
                chunks.append(chunk_path)
        except Exception as e:
            print(f"--- ALIGNER: Chunk {i} split error: {e} ---")

    return chunks if chunks else [audio_path]


def _split_text_for_chunks(text: str, n_chunks: int) -> list[str]:
    """Split text into n roughly equal parts at sentence boundaries.
    (Kept for reference but no longer used in sequential alignment.)
    """
    if n_chunks <= 1:
        return [text]
    words = text.split()
    words_per_chunk = max(1, len(words) // n_chunks)
    parts = []
    for i in range(n_chunks):
        start_idx = i * words_per_chunk
        if i == n_chunks - 1:
            part = " ".join(words[start_idx:])
        else:
            part = " ".join(words[start_idx:start_idx + words_per_chunk])
        if part.strip():
            parts.append(part.strip())
    return parts


def _count_good_words(chunk_words, chunk_duration: float) -> int:
    """Count how many words from the aligner output have valid timestamps.

    The aligner collapses remaining words when it runs out of audio.
    We scan FORWARD and detect the first region where multiple consecutive
    words have near-zero duration – that's where the aligner lost sync.
    """
    if not chunk_words:
        return 0

    n = len(chunk_words)
    # Scan forward looking for a run of 3+ zero-duration words
    ZERO_DUR_THRESHOLD = 0.025  # words shorter than 25ms are "zero"
    MIN_COLLAPSE_RUN = 3  # 3+ in a row = collapse detected

    run_start = None
    run_count = 0

    for i in range(n):
        w = chunk_words[i]
        dur = w.end_time - w.start_time
        if dur < ZERO_DUR_THRESHOLD:
            if run_start is None:
                run_start = i
            run_count += 1
        else:
            if run_count >= MIN_COLLAPSE_RUN:
                # Found a collapse region – everything from run_start onwards is bad
                print(f"--- ALIGNER: Collapse detected at word {run_start} "
                      f"('{chunk_words[run_start].text}' @ {chunk_words[run_start].start_time:.2f}s), "
                      f"run of {run_count} zero-dur words ---")
                return run_start
            run_start = None
            run_count = 0

    # Check trailing run
    if run_count >= MIN_COLLAPSE_RUN and run_start is not None:
        print(f"--- ALIGNER: Collapse at end, word {run_start} "
              f"('{chunk_words[run_start].text}' @ {chunk_words[run_start].start_time:.2f}s), "
              f"{run_count} zero-dur words ---")
        return run_start

    return n  # All words are good


def _remove_n_words_from_text(text: str, n: int, aligned_words) -> str:
    """Remove the first n words from text, matching against aligned words.

    Uses fuzzy matching to handle punctuation differences between
    original text and aligner output.
    """
    import re

    orig_tokens = text.split()
    if n <= 0:
        return text
    if n >= len(orig_tokens):
        return ""

    # Try to match aligned words to original tokens to find the exact split point
    oi = 0
    matched = 0
    for aw in aligned_words:
        aw_clean = re.sub(r'[^\w]', '', aw.text.lower()) if hasattr(aw, 'text') else re.sub(r'[^\w]', '', str(aw.get('word', '')).lower())
        if not aw_clean:
            matched += 1
            continue

        # Search forward
        search_limit = min(oi + 5, len(orig_tokens))
        for j in range(oi, search_limit):
            ot_clean = re.sub(r'[^\w]', '', orig_tokens[j].lower())
            if ot_clean == aw_clean:
                oi = j + 1
                matched += 1
                break
        else:
            oi += 1
            matched += 1

        if matched >= n:
            break

    # oi now points to the first unmatched token
    return " ".join(orig_tokens[oi:])


def _reinject_punctuation(aligner_words: list[dict], original_text: str) -> list[dict]:
    """Re-inject punctuation from original text into aligner word results.

    The aligner strips punctuation from words. We match words back to the
    original text and restore trailing punctuation.
    """
    import re

    # Extract original words with their punctuation
    orig_tokens = original_text.split()
    if not orig_tokens or not aligner_words:
        return aligner_words

    # Build a mapping: lowercase stripped word -> list of original tokens
    # We'll do a simple sequential matching
    oi = 0  # index into orig_tokens
    for aw in aligner_words:
        aw_clean = re.sub(r'[^\w]', '', aw["word"].lower())
        if not aw_clean:
            continue

        # Search forward in original tokens for a match
        search_limit = min(oi + 5, len(orig_tokens))
        for j in range(oi, search_limit):
            ot_clean = re.sub(r'[^\w]', '', orig_tokens[j].lower())
            if ot_clean == aw_clean:
                # Found match - use original token (with punctuation)
                aw["word"] = orig_tokens[j]
                oi = j + 1
                break
        else:
            # No match found nearby, skip ahead
            if oi < len(orig_tokens):
                oi += 1

    return aligner_words


def _extract_audio_chunk(audio_path: str, start_sec: float, duration_sec: float) -> str:
    """Extract a chunk from an audio file using ffmpeg. Returns path to chunk."""
    base = Path(audio_path)
    chunk_path = str(base.parent / f"{base.stem}_dyn_{start_sec:.0f}{base.suffix}")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(start_sec), "-t", str(duration_sec),
             "-acodec", "copy", chunk_path],
            capture_output=True, timeout=30,
        )
        if Path(chunk_path).exists() and Path(chunk_path).stat().st_size > 100:
            return chunk_path
    except Exception as e:
        print(f"--- ALIGNER: Chunk extract error: {e} ---")
    return ""


def align_audio(
    file_path: str,
    text: str,
    language: str = "German",
    device: str = "cuda",
):
    """Align text to audio using Qwen3-ForcedAligner.

    For long audio (>MAX_CHUNK_SECS), uses dynamic chunking:
    1. Extract chunk starting at current position
    2. Align with remaining text
    3. Detect where timestamps collapse
    4. Next chunk starts where collapse happened (with small overlap)
    5. Repeat until all text is aligned or audio is exhausted

    Args:
        file_path: Path to audio file
        text: Transcription text to align
        language: Language name (e.g. "German", "English")
        device: cuda or cpu

    Returns:
        [segments_json, status_string]
        Where segments_json contains word-level timestamps.
    """
    temp_path = None
    chunk_paths_to_clean = []
    try:
        start_time = time.time()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        model = aligner_manager.get_model(device=device)
        temp_path = _materialize(file_path)

        duration = _get_audio_duration(temp_path)
        print(f"--- ALIGNER: Audio={duration:.1f}s, Text={len(text)} chars, lang={language} ---")

        all_words = []
        remaining_text = text

        if duration <= MAX_CHUNK_SECS + 10:
            # Single chunk - simple alignment
            print(f"--- ALIGNER: Single chunk, words={len(text.split())} ---")
            results = model.align(audio=temp_path, text=text, language=language)
            if results and len(results) > 0:
                for w in results[0]:
                    all_words.append({
                        "word": w.text,
                        "start": round(w.start_time, 3),
                        "end": round(w.end_time, 3),
                    })
            n_chunks = 1
        else:
            # Dynamic chunking
            audio_pos = 0.0  # current position in audio (seconds)
            chunk_num = 0
            max_chunks = 20  # safety limit
            OVERLAP_SEC = 5.0  # overlap to avoid boundary artifacts

            while remaining_text.strip() and audio_pos < duration and chunk_num < max_chunks:
                chunk_num += 1
                chunk_len = min(MAX_CHUNK_SECS, duration - audio_pos)

                if chunk_len < 5:
                    print(f"--- ALIGNER: Chunk {chunk_num} too short ({chunk_len:.1f}s), stopping ---")
                    break

                # Extract audio chunk from current position
                chunk_path = _extract_audio_chunk(temp_path, audio_pos, chunk_len)
                if not chunk_path:
                    print(f"--- ALIGNER: Chunk {chunk_num} extraction failed ---")
                    break
                chunk_paths_to_clean.append(chunk_path)

                remaining_words = remaining_text.split()
                print(f"--- ALIGNER: Chunk {chunk_num}, audio={audio_pos:.1f}-{audio_pos+chunk_len:.1f}s, "
                      f"remaining_words={len(remaining_words)} ---")

                try:
                    results = model.align(
                        audio=chunk_path,
                        text=remaining_text,
                        language=language,
                    )

                    if results and len(results) > 0:
                        chunk_words = results[0]
                        chunk_dur = _get_audio_duration(chunk_path)
                        good_count = _count_good_words(chunk_words, chunk_dur)

                        if good_count == 0:
                            print(f"--- ALIGNER: Chunk {chunk_num} aligned 0 words, "
                                  f"skipping ahead {MAX_CHUNK_SECS//3:.0f}s ---")
                            audio_pos += MAX_CHUNK_SECS // 3
                            continue

                        # Get the timestamp of the last good word (in chunk-local time)
                        last_good_time = chunk_words[good_count - 1].end_time

                        for w in chunk_words[:good_count]:
                            all_words.append({
                                "word": w.text,
                                "start": round(w.start_time + audio_pos, 3),
                                "end": round(w.end_time + audio_pos, 3),
                            })

                        # Remove aligned words from remaining text
                        remaining_text = _remove_n_words_from_text(
                            remaining_text, good_count, chunk_words[:good_count]
                        )

                        print(f"--- ALIGNER: Chunk {chunk_num} aligned {good_count}/{len(chunk_words)} words "
                              f"up to {last_good_time:.1f}s (abs {audio_pos+last_good_time:.1f}s), "
                              f"remaining={len(remaining_text.split())} ---")

                        # Next chunk starts where this one's good words ended
                        # Subtract small overlap for safety
                        audio_pos += max(last_good_time - OVERLAP_SEC, last_good_time * 0.9)

                    else:
                        print(f"--- ALIGNER: Chunk {chunk_num} no results, skipping ---")
                        audio_pos += MAX_CHUNK_SECS // 3

                except Exception as chunk_err:
                    print(f"--- ALIGNER: Chunk {chunk_num} error: {chunk_err} ---")
                    audio_pos += MAX_CHUNK_SECS // 3

            n_chunks = chunk_num

            if remaining_text.strip():
                leftover = remaining_text.split()
                print(f"--- ALIGNER: {len(leftover)} words unaligned after {n_chunks} chunks ---")

        # Re-inject punctuation from original text
        all_words = _reinject_punctuation(all_words, text)

        # Group words into sentence-level segments
        segments = _words_to_segments(all_words)
        segments_json = json.dumps(segments, ensure_ascii=False)

        elapsed = time.time() - start_time
        status = (
            f"ForcedAligner fertig in {elapsed:.2f}s "
            f"(Wörter: {len(all_words)}, Segmente: {len(segments)}, Chunks: {n_chunks})"
        )
        print(f"--- ALIGNER: {status} ---")
        return [segments_json, status]

    except Exception as exc:
        print(f"--- ALIGNER ERROR: {exc} ---")
        traceback.print_exc()
        return ["[]", f"Fehler: {exc}"]
    finally:
        # Cleanup all temp files
        for cp in chunk_paths_to_clean:
            try:
                os.remove(cp)
            except OSError:
                pass
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        clear_memory(device)


def _words_to_segments(words: list[dict]) -> list[dict]:
    """Group word timestamps into sentence-level segments.

    Splits at sentence-ending punctuation (.!?) or every ~30 words.
    """
    if not words:
        return []

    segments = []
    current_words = []
    segment_id = 0

    for w in words:
        current_words.append(w)
        text = w["word"].rstrip()

        is_sentence_end = text and text[-1] in ".!?:;"
        is_long = len(current_words) >= 30

        if is_sentence_end or is_long:
            seg_text = " ".join(cw["word"] for cw in current_words).strip()
            segments.append({
                "id": segment_id,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": seg_text,
                "words": current_words.copy(),
            })
            segment_id += 1
            current_words = []

    # Remaining words
    if current_words:
        seg_text = " ".join(cw["word"] for cw in current_words).strip()
        segments.append({
            "id": segment_id,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "text": seg_text,
            "words": current_words.copy(),
        })

    return segments
