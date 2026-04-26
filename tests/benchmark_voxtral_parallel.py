#!/usr/bin/env python3
"""
Voxtral Parallel-Latenz-Benchmark für DGX Spark

Misst die tatsächliche Antwortzeit bei verschiedenen Parallelitätsstufen
und zeigt GPU-Auslastung während des Benchmarks.

Usage:
    python3 tests/benchmark_voxtral_parallel.py --endpoint http://127.0.0.1:8000/v1/audio/transcriptions
    python3 tests/benchmark_voxtral_parallel.py --concurrency 1,2,4,8 --duration 60
    python3 tests/benchmark_voxtral_parallel.py --audio-file tests/audio_test.mp3
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
import tempfile
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
    
    # Mehrere Sinus-Töne für "Sprach-ähnliches" Signal
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
    
    # Leichtes Rauschen hinzufügen
    for i in range(num_samples):
        signal[i] += random.gauss(0, 0.05)
    
    # Normalisieren auf 16-bit Bereich
    max_val = max(abs(s) for s in signal)
    if max_val > 0:
        signal = [int(s / max_val * 32767) for s in signal]
    else:
        signal = [0] * num_samples
    
    # Als WAV speichern
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


def resolve_voxtral_endpoint(endpoint: str) -> str:
    """Prüft den gewünschten Endpoint und fällt bei Bedarf auf 8001 zurück."""
    candidates = [endpoint]

    if ":8000/" in endpoint:
        candidates.append(endpoint.replace(":8000/", ":8001/"))
    elif ":8001/" in endpoint:
        candidates.append(endpoint.replace(":8001/", ":8000/"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        health_url = candidate.replace("/v1/audio/transcriptions", "/health")
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                if candidate != endpoint:
                    print(f"[INFO] Nutze erreichbaren Fallback-Endpoint: {candidate}")
                return candidate
        except Exception:
            continue

    raise RuntimeError(
        "Weder Voxtral-Proxy auf Port 8000 noch direkter vLLM auf Port 8001 sind erreichbar. "
        "Prüfe den Service mit: sudo systemctl status voxtral-vllm"
    )


def send_transcription_request(endpoint: str, audio_data: bytes, file_name: str = "test.wav", timeout: float = 120.0, model_name: str = "voxtral") -> float:
    """Sendet eine Transkriptions-Anfrage und misst die Latenz."""
    mime_type = get_audio_mime_type(file_name)
    files = {
        "file": (file_name, audio_data, mime_type),
    }
    data = {
        "model": model_name,
        "language": "de",
        "response_format": "json",
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            endpoint,
            files=files,
            data=data,
            timeout=timeout,
        )
        response.raise_for_status()
        return time.time() - start_time
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"[ERROR] Endpunkt nicht gefunden (404).")
            print(f"[ERROR] URL: {endpoint}")
            print(f"[ERROR] Prüfe, ob der voxtral-vllm Service läuft: sudo systemctl status voxtral-vllm")
            # Versuche Fallback auf direkten vLLM Port
            fallback_url = endpoint.replace(":8000/", ":8001/")
            print(f"[INFO] Versuche Fallback auf direkten vLLM: {fallback_url}")
            try:
                response = requests.post(
                    fallback_url,
                    files=files,
                    data=data,
                    timeout=timeout,
                )
                response.raise_for_status()
                print(f"[INFO] Fallback erfolgreich!")
                return time.time() - start_time
            except Exception as fallback_e:
                print(f"[ERROR] Fallback auch fehlgeschlagen: {fallback_e}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        raise


def discover_model_name(endpoint: str) -> str:
    """Versucht, den korrekten Modellnamen vom Server zu ermitteln."""
    models_url = endpoint.replace("/v1/audio/transcriptions", "/v1/models")
    try:
        resp = requests.get(models_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                model_id = data["data"][0].get("id", "voxtral")
                print(f"[INFO] Verfügbares Modell gefunden: {model_id}")
                return model_id
    except Exception as e:
        print(f"[WARN] Konnte Modelle nicht abrufen: {e}")
    return "voxtral"


def worker_task(args, audio_data: bytes, worker_id: int, file_name: str = "test.wav", model_name: str = "voxtral") -> float:
    """Einzelne Worker-Aufgabe."""
    # Kleiner zufälliger Offset, damit nicht alle exakt gleichzeitig starten
    time.sleep(random.uniform(0, 0.1))
    latency = send_transcription_request(args.endpoint, audio_data, file_name, args.timeout, model_name)
    return latency


def gpu_monitor(stop_event: threading.Event, gpu_samples: list, interval: float = 1.0):
    """Hintergrund-Thread zur GPU-Überwachung."""
    while not stop_event.is_set():
        stats = get_gpu_stats()
        gpu_samples.append(stats)
        time.sleep(interval)


def run_benchmark(args, concurrency: int, audio_data: bytes, file_name: str = "test.wav", model_name: str = "voxtral") -> BenchmarkResult:
    """Führt einen Benchmark mit definierter Parallelität durch."""
    result = BenchmarkResult(concurrency=concurrency)
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {concurrency} parallele Nutzer")
    print(f"Audio-Datei: {file_name}")
    print(f"Modell: {model_name}")
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
            
            # Starte so viele Anfragen wie möglich in der gegebenen Zeit
            while time.time() - start_time < args.duration:
                _drain_completed()

                if result.total_requests == 0 and result.errors > 0:
                    print("[WARN] Breche Benchmark früh ab, da aktuell kein erfolgreicher Voxtral-Request möglich ist.")
                    break

                # Stelle sicher, dass nicht mehr als concurrency gleichzeitig laufen
                if len(futures) < concurrency:
                    future = executor.submit(worker_task, args, audio_data, request_count, file_name, model_name)
                    futures.append(future)
                    request_count += 1
                else:
                    time.sleep(0.05)  # Kurz warten
            
            # Warte auf alle laufenden Futures
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
    print("BENCHMARK ERGEBNISSE")
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
    
    # Finde den besten Sweet Spot
    best_concurrency = None
    best_score = 0
    
    for r in results:
        if r.errors > r.total_requests * 0.1:  # Mehr als 10% Fehler
            continue
        
        # Score: Durchsatz / Latenz (höher ist besser)
        if r.avg_latency > 0:
            score = r.throughput_rps / r.avg_latency
            if score > best_score:
                best_score = score
                best_concurrency = r.concurrency
    
    if best_concurrency:
        print(f"   Optimaler Parallelitätsgrad: {best_concurrency} Nutzer")
        
        # Finde Latenz-Schwellen
        for r in results:
            if r.concurrency == best_concurrency:
                if r.avg_latency < 2.0:
                    print(f"   Latenz: {r.avg_latency:.2f}s (sehr gut für Echtzeit)")
                elif r.avg_latency < 5.0:
                    print(f"   Latenz: {r.avg_latency:.2f}s (okay für Batch)")
                else:
                    print(f"   Latenz: {r.avg_latency:.2f}s (nur für Offline geeignet)")
    else:
        print("   Kein optimaler Parallelitätsgrad gefunden (zu viele Fehler).")
    
    print("\n💡 Hinweise:")
    print("   - Ø Latenz < 2s: Gut für interaktive Anwendungen")
    print("   - Ø Latenz 2-5s: Akzeptabel für Batch-Verarbeitung")
    print("   - Ø Latenz > 5s: Nur für Offline-Transkription geeignet")
    print("   - GPU % sollte nicht dauerhaft 100% sein (Thermalthrottling)")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark für Voxtral Parallel-Latenz auf DGX Spark"
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/audio/transcriptions",
        help="Voxtral API Endpoint (default: http://127.0.0.1:8000/v1/audio/transcriptions)",
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="Komma-getrennte Liste der Parallelitätsstufen (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Benchmark-Dauer pro Stufe in Sekunden (default: 30)",
    )
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=5.0,
        help="Länge des Test-Audios in Sekunden (default: 5.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout pro Anfrage in Sekunden (default: 120)",
    )
    parser.add_argument(
        "--audio-file",
        help="Pfad zu einer echten WAV-Datei (optional, sonst synthetisch)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Ausgabe als JSON",
    )
    
    args = parser.parse_args()
    
    # Konvertiere Concurrency-String zu Liste
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
        args.endpoint = resolve_voxtral_endpoint(args.endpoint)
        health_url = args.endpoint.replace("/v1/audio/transcriptions", "/health")
        resp = requests.get(health_url, timeout=10)
        if resp.status_code == 200:
            print(f"[INFO] Server ist erreichbar! Status: {resp.status_code}")
        else:
            print(f"[WARN] Server antwortet mit Status {resp.status_code}")
    except RuntimeError as e:
        print(f"[FEHLER] {e}")
        return
    except Exception as e:
        print(f"[WARN] Konnte Server nicht erreichen: {e}")
        print(f"[WARN] Benchmark wird trotzdem gestartet...")
    
    # Versuche, den korrekten Modellnamen zu ermitteln
    model_name = discover_model_name(args.endpoint)
    
    # Benchmarks durchführen
    results = []
    for concurrency in concurrency_levels:
        result = run_benchmark(args, concurrency, audio_data, audio_file_name, model_name)
        results.append(result)
        
        # Kurze Pause zwischen den Stufen
        if concurrency != concurrency_levels[-1]:
            print(f"[INFO] Pause von 5 Sekunden vor nächster Stufe...")
            time.sleep(5)
    
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
