#!/usr/bin/env python3
"""
WhisperX Parallel-Latenz-Benchmark für DGX Spark

Misst die tatsächliche Antwortzeit bei verschiedenen Parallelitätsstufen
und zeigt GPU-Auslastung während des Benchmarks.

Usage:
    python3 tests/benchmark_whisperx_parallel.py --endpoint http://127.0.0.1:7860
    python3 tests/benchmark_whisperx_parallel.py --concurrency 1,2,3,4 --duration 60
    python3 tests/benchmark_whisperx_parallel.py --audio-file tests/audio_test.mp3
"""

import argparse
import concurrent.futures
import io
import json
import math
import os
import random
import requests
import statistics
import struct
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkResult:
    concurrency: int
    latencies: list[float] = field(default_factory=list)
    errors: int = 0
    total_requests: int = 0
    gpu_samples: list[dict] = field(default_factory=list)
    wall_time_sec: float = 0.0
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0
    
    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def throughput_rps(self) -> float:
        if self.wall_time_sec <= 0:
            return 0.0
        return self.total_requests / self.wall_time_sec
    
    @property
    def avg_gpu_util(self) -> float:
        if not self.gpu_samples:
            return 0.0
        return statistics.mean([s.get("gpu_util", 0) for s in self.gpu_samples])
    
    @property
    def avg_gpu_mem(self) -> float:
        if not self.gpu_samples:
            return 0.0
        return statistics.mean([s.get("gpu_mem_mb", 0) for s in self.gpu_samples])


def create_test_audio(duration_sec: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Erzeugt ein synthetisches Test-Audio (Sinus-Welle + Rauschen) ohne numpy."""
    num_samples = int(sample_rate * duration_sec)
    
    freq1 = 200 + random.uniform(-50, 50)
    freq2 = 400 + random.uniform(-50, 50)
    freq3 = 800 + random.uniform(-100, 100)
    
    signal = []
    for i in range(num_samples):
        t = i / sample_rate
        sample = (
            0.5 * math.sin(2 * math.pi * freq1 * t) +
            0.3 * math.sin(2 * math.pi * freq2 * t) +
            0.2 * math.sin(2 * math.pi * freq3 * t)
        )
        signal.append(sample)
    
    for i in range(num_samples):
        signal[i] += random.gauss(0, 0.05)
    
    max_val = max(abs(s) for s in signal)
    if max_val > 0:
        signal = [int(s / max_val * 32767) for s in signal]
    else:
        signal = [0] * num_samples
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(signal)}h", *signal))
    
    return buffer.getvalue()


def get_gpu_stats() -> dict:
    """Ermittelt aktuelle GPU-Auslastung via nvidia-smi, inkl. GB10-Fallbacks."""
    timestamp = time.time()

    def _parse_number(value: str) -> float:
        value = value.strip()
        if not value or value.lower() in {"n/a", "[not supported]", "not supported"}:
            return 0.0
        return float(value)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            return {
                "gpu_util": _parse_number(parts[0]),
                "gpu_mem_mb": _parse_number(parts[1]),
                "gpu_mem_total_mb": _parse_number(parts[2]) if len(parts) >= 3 else 0.0,
                "timestamp": timestamp,
            }
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        util_line = result.stdout.strip().split("\n")[0].strip()
        gpu_util = _parse_number(util_line)

        proc_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_mem_mb = 0.0
        for line in proc_result.stdout.strip().splitlines():
            try:
                gpu_mem_mb += _parse_number(line)
            except Exception:
                continue

        return {
            "gpu_util": gpu_util,
            "gpu_mem_mb": gpu_mem_mb,
            "gpu_mem_total_mb": 0.0,
            "timestamp": timestamp,
        }
    except Exception:
        pass

    return {"gpu_util": 0.0, "gpu_mem_mb": 0.0, "gpu_mem_total_mb": 0.0, "timestamp": timestamp}


def get_audio_mime_type(file_path: str) -> str:
    """Ermittelt den MIME-Type basierend auf der Dateiendung."""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    return mime_types.get(ext, "audio/wav")


def send_whisperx_request(endpoint: str, audio_data: bytes, file_name: str = "test.wav",
                          language: str = "de", model: str = "large-v3",
                          username: str = "root", password: str = "Goethestrasse28!",
                          timeout: float = 300.0) -> float:
    """Sendet eine Transkriptions-Anfrage an WhisperX (Gradio 5 Queue API) und misst die Latenz."""
    
    base = endpoint.rstrip("/")
    
    # 1. Login
    session = requests.Session()
    login = session.post(
        f"{base}/login",
        data={"username": username, "password": password},
        timeout=10,
    )
    if login.status_code != 200:
        raise Exception(f"WhisperX Login fehlgeschlagen: {login.status_code}")
    
    # 2. Datei hochladen
    ext = os.path.splitext(file_name)[1].lower() or ".wav"
    mime = get_audio_mime_type(file_name)
    upload = session.post(
        f"{base}/gradio_api/upload",
        files={"files": (file_name, audio_data, mime)},
        timeout=30,
    )
    if upload.status_code != 200:
        raise Exception(f"Upload fehlgeschlagen: {upload.status_code} {upload.text[:200]}")
    
    upload_result = upload.json()
    if isinstance(upload_result, list) and len(upload_result) > 0:
        file_path = upload_result[0]
    elif isinstance(upload_result, dict) and "data" in upload_result and upload_result["data"]:
        file_path = upload_result["data"][0]
    else:
        raise Exception(f"Unerwartete Upload-Antwort: {upload_result}")
    
    # 3. Transkription in die Queue einstellen
    lang_value = language
    if language == "de":
        lang_value = "German"
    elif language == "en":
        lang_value = "English"

    file_data = {
        "path": file_path,
        "url": f"{base}/gradio_api/file={file_path}",
        "size": len(audio_data),
        "orig_name": os.path.basename(file_name),
        "mime_type": mime,
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }
    
    payload = {
        "fn_index": 4,
        "data": [
            file_data,
            lang_value,
            model,
            "cuda",
            "",
            False,
        ],
        "session_hash": f"benchmark_{random.randint(10000, 99999)}",
    }
    
    start_time = time.time()
    queue_resp = session.post(f"{base}/gradio_api/queue/join", json=payload, timeout=30)
    
    if queue_resp.status_code != 200:
        raise Exception(f"Queue Join fehlgeschlagen: {queue_resp.status_code} {queue_resp.text[:200]}")
    
    queue_result = queue_resp.json()
    event_id = queue_result.get("event_id") if isinstance(queue_result, dict) else None
    
    if not event_id:
        raise Exception(f"Kein event_id in Queue-Antwort: {queue_result}")

    # 4. Queue-Eventstream lesen
    result_url = f"{base}/gradio_api/queue/data"
    with session.get(
        result_url,
        params={"session_hash": payload["session_hash"]},
        stream=True,
        timeout=(10, timeout + 30),
    ) as result_resp:
        if result_resp.status_code != 200:
            raise Exception(
                f"Event-Stream fehlgeschlagen: {result_resp.status_code} {result_resp.text[:200]}"
            )

        for line in result_resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if not data_str:
                continue

            try:
                data_obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if not isinstance(data_obj, dict):
                continue

            if data_obj.get("event_id") not in (None, event_id):
                continue

            msg = data_obj.get("msg", "")
            if msg == "process_completed":
                output = data_obj.get("output", {})
                if isinstance(output, dict) and output.get("error"):
                    raise Exception(f"WhisperX Verarbeitung fehlgeschlagen: {output['error']}")
                if data_obj.get("success") is False and not output.get("data"):
                    raise Exception(f"WhisperX Verarbeitung fehlgeschlagen: {data_obj}")
                elapsed = time.time() - start_time
                return elapsed
            if msg in {"process_starts", "process_generating", "estimation", "heartbeat"}:
                continue
            if msg == "process_failed":
                raise Exception(f"WhisperX Verarbeitung fehlgeschlagen: {data_obj}")

    raise Exception("WhisperX Event-Stream endete ohne process_completed")


def worker_task(args, audio_data: bytes, worker_id: int, file_name: str = "test.wav") -> float:
    """Einzelne Worker-Aufgabe."""
    time.sleep(random.uniform(0, 0.1))
    latency = send_whisperx_request(
        args.endpoint,
        audio_data,
        file_name,
        args.language,
        args.model,
        args.username,
        args.password,
        args.timeout,
    )
    return latency


def gpu_monitor(stop_event: threading.Event, gpu_samples: list, interval: float = 1.0):
    """Hintergrund-Thread zur GPU-Überwachung."""
    while not stop_event.is_set():
        stats = get_gpu_stats()
        gpu_samples.append(stats)
        time.sleep(interval)


def run_benchmark(args, concurrency: int, audio_data: bytes, file_name: str = "test.wav") -> BenchmarkResult:
    """Führt einen Benchmark mit definierter Parallelität durch."""
    result = BenchmarkResult(concurrency=concurrency)
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {concurrency} parallele Nutzer")
    print(f"Audio-Datei: {file_name}")
    print(f"Modell: {args.model}")
    print(f"Dauer: ~{args.duration} Sekunden")
    print(f"{'='*60}")
    
    # GPU-Monitoring starten
    stop_event = threading.Event()
    gpu_thread = threading.Thread(target=gpu_monitor, args=(stop_event, result.gpu_samples))
    gpu_thread.start()
    
    start_time = time.time()
    request_count = 0
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []

            def _drain_completed() -> None:
                nonlocal futures
                still_running = []
                for future in futures:
                    if future.done():
                        try:
                            latency = future.result()
                            result.latencies.append(latency)
                            result.total_requests += 1
                        except Exception as e:
                            result.errors += 1
                            print(f"[ERROR] Anfrage fehlgeschlagen: {e}")
                    else:
                        still_running.append(future)
                futures = still_running
            
            while time.time() - start_time < args.duration:
                _drain_completed()

                if len(futures) < concurrency:
                    future = executor.submit(worker_task, args, audio_data, request_count, file_name)
                    futures.append(future)
                    request_count += 1
                else:
                    time.sleep(0.05)
            
            print(f"Warte auf {len(futures)} laufende Anfragen...")
            for future in concurrent.futures.as_completed(list(futures)):
                try:
                    latency = future.result()
                    result.latencies.append(latency)
                    result.total_requests += 1
                except Exception as e:
                    result.errors += 1
                    print(f"[ERROR] Anfrage fehlgeschlagen: {e}")
    
    finally:
        stop_event.set()
        gpu_thread.join(timeout=5)

    result.wall_time_sec = time.time() - start_time
    
    return result


def print_results(results: list[BenchmarkResult]):
    """Gibt die Benchmark-Ergebnisse in einer Tabelle aus."""
    print("\n" + "=" * 100)
    print("WHISPERX BENCHMARK ERGEBNISSE")
    print("=" * 100)
    
    header = (
        f"{'Nutzer':>8} | "
        f"{'Anfragen':>10} | "
        f"{'Fehler':>8} | "
        f"{'Ø Latenz':>10} | "
        f"{'Median':>10} | "
        f"{'P95':>10} | "
        f"{'P99':>10} | "
        f"{'Throughput':>12} | "
        f"{'Ø GPU %':>8} | "
        f"{'Ø GPU MB':>10}"
    )
    print(header)
    print("-" * len(header))
    
    for r in results:
        print(
            f"{r.concurrency:>8} | "
            f"{r.total_requests:>10} | "
            f"{r.errors:>8} | "
            f"{r.avg_latency:>9.2f}s | "
            f"{r.median_latency:>9.2f}s | "
            f"{r.p95_latency:>9.2f}s | "
            f"{r.p99_latency:>9.2f}s | "
            f"{r.throughput_rps:>11.2f} r/s | "
            f"{r.avg_gpu_util:>7.1f}% | "
            f"{r.avg_gpu_mem:>9.0f} MB"
        )
    
    print("=" * 100)
    
    # Empfehlung
    print("\n📊 EMPFEHLUNG:")
    
    best_concurrency = None
    best_score = 0
    
    for r in results:
        if r.errors > r.total_requests * 0.1:
            continue
        
        if r.avg_latency > 0:
            score = r.throughput_rps / r.avg_latency
            if score > best_score:
                best_score = score
                best_concurrency = r.concurrency
    
    if best_concurrency:
        print(f"   Optimaler Parallelitätsgrad: {best_concurrency} Nutzer")
        
        for r in results:
            if r.concurrency == best_concurrency:
                if r.avg_latency < 5.0:
                    print(f"   Latenz: {r.avg_latency:.2f}s (gut für Batch)")
                elif r.avg_latency < 15.0:
                    print(f"   Latenz: {r.avg_latency:.2f}s (okay für Offline)")
                else:
                    print(f"   Latenz: {r.avg_latency:.2f}s (sehr langsam)")
    else:
        print("   Kein optimaler Parallelitätsgrad gefunden (zu viele Fehler).")
    
    print("\n💡 Hinweise:")
    print("   - WhisperX nutzt CTranslate2 (weniger GPU-intensiv als vLLM)")
    print("   - Worker-Pool limitiert Parallelität auf Pool-Größe")
    print("   - Ø Latenz < 5s: Gut für Batch-Verarbeitung")
    print("   - Ø Latenz > 15s: Nur für Offline geeignet")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark für WhisperX Parallel-Latenz auf DGX Spark"
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:7860",
        help="WhisperX Gradio API Endpoint (default: http://127.0.0.1:7860)",
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,3,4",
        help="Komma-getrennte Liste der Parallelitätsstufen (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark-Dauer pro Stufe in Sekunden (default: 60)",
    )
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=10.0,
        help="Länge des Test-Audios in Sekunden (default: 10.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout pro Anfrage in Sekunden (default: 300)",
    )
    parser.add_argument(
        "--audio-file",
        help="Pfad zu einer echten Audio-Datei (optional, sonst synthetisch)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="WhisperX Modell (default: large-v3)",
    )
    parser.add_argument(
        "--language",
        default="de",
        help="Sprache (default: de)",
    )
    parser.add_argument(
        "--username",
        default="root",
        help="WhisperX Login-Username (default: root)",
    )
    parser.add_argument(
        "--password",
        default="Goethestrasse28!",
        help="WhisperX Login-Passwort (default: Goethestrasse28!)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Ausgabe als JSON",
    )
    
    args = parser.parse_args()
    
    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]
    
    # Test-Audio laden oder erstellen
    audio_file_name = "test.wav"
    if args.audio_file:
        with open(args.audio_file, "rb") as f:
            audio_data = f.read()
        audio_file_name = os.path.basename(args.audio_file)
        print(f"[INFO] Lade Test-Audio von {args.audio_file} ({len(audio_data)} bytes)")
    else:
        print(f"[INFO] Erzeuge synthetisches Test-Audio ({args.audio_duration}s)...")
        audio_data = create_test_audio(args.audio_duration)
        print(f"[INFO] Test-Audio erzeugt: {len(audio_data)} bytes")
    
    # Prüfe, ob der Server erreichbar ist
    print(f"[INFO] Prüfe Verbindung zu {args.endpoint}...")
    try:
        resp = requests.get(args.endpoint, timeout=10)
        if resp.status_code == 200:
            print(f"[INFO] WhisperX ist erreichbar! Status: {resp.status_code}")
        else:
            print(f"[WARN] WhisperX antwortet mit Status {resp.status_code}")
    except Exception as e:
        print(f"[WARN] Konnte WhisperX nicht erreichen: {e}")
        print(f"[WARN] Benchmark wird trotzdem gestartet...")
    
    # Benchmarks durchführen
    results = []
    for concurrency in concurrency_levels:
        result = run_benchmark(args, concurrency, audio_data, audio_file_name)
        results.append(result)
        
        if concurrency != concurrency_levels[-1]:
            print(f"[INFO] Pause von 10 Sekunden vor nächster Stufe...")
            time.sleep(10)
    
    # Ergebnisse ausgeben
    if args.json:
        output = []
        for r in results:
            output.append({
                "concurrency": r.concurrency,
                "total_requests": r.total_requests,
                "errors": r.errors,
                "avg_latency_sec": r.avg_latency,
                "median_latency_sec": r.median_latency,
                "p95_latency_sec": r.p95_latency,
                "p99_latency_sec": r.p99_latency,
                "throughput_rps": r.throughput_rps,
                "avg_gpu_util_percent": r.avg_gpu_util,
                "avg_gpu_mem_mb": r.avg_gpu_mem,
            })
        print(json.dumps(output, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
