#!/usr/bin/env python3
"""
vLLM Kapazitäts-Rechner für DGX Spark

Analysiert vLLM-Logs oder API-Ausgaben und berechnet die tatsächliche
maximale Parallelität basierend auf KV Cache Pool, Block-Größe und Modell-Parametern.

Usage:
    python3 check_vllm_capacity.py
    python3 check_vllm_capacity.py --service voxtral-vllm
    python3 check_vllm_capacity.py --gpu-mem-util 0.45 --max-model-len 8192 --max-num-seqs 20
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from typing import Optional, Tuple


def run_command(cmd: list[str]) -> str:
    """Führt einen Shell-Befehl aus und gibt stdout zurück."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[WARN] Befehl fehlgeschlagen: {' '.join(cmd)} -> {e}")
        return ""


def get_gpu_memory_mb() -> Optional[int]:
    """Ermittelt den gesamten GPU-Speicher in MB via nvidia-smi oder Fallbacks."""
    # Versuch 1: Standard nvidia-smi Query
    output = run_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    try:
        val = output.strip().split("\n")[0].strip()
        if val and val.lower() not in ("not supported", "[not supported]", "n/a", ""):
            return int(val)
    except (ValueError, IndexError):
        pass

    # Versuch 2: Volle nvidia-smi Ausgabe parsen (für GB10 Unified Memory)
    full_output = run_command(["nvidia-smi"])
    if "GB10" in full_output:
        # GB10 hat ~128 GB Unified Memory (variiert je nach Systemkonfiguration)
        # Wir versuchen zuerst, den verfügbaren Speicher aus den vLLM Logs zu ermitteln
        # oder nutzen einen konservativen Default
        print("[INFO] NVIDIA GB10 erkannt (Unified Memory). Nutze Fallback-Methode...")
        
        # Versuch 2a: Aus /proc/meminfo den System-RAM lesen
        meminfo = run_command(["cat", "/proc/meminfo"])
        mem_match = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
        if mem_match:
            total_kb = int(mem_match.group(1))
            total_mb = total_kb // 1024
            # Auf dem GB10 ist der GPU-Speicher = System-RAM (Unified Memory)
            print(f"[INFO] System-RAM ermittelt: {total_mb} MB ({total_mb/1024:.1f} GB)")
            return total_mb
        
        # Versuch 2b: Konservativer Default für GB10
        print("[WARN] Konnte System-RAM nicht ermitteln. Nutze GB10-Default von 128 GB.")
        return 128 * 1024  # 128 GB in MB

    # Versuch 3: nvidia-smi mit --query-gpu=memory.total als Fallback
    # manchmal liefert es den Wert in einer anderen Zeile
    for line in output.strip().split("\n"):
        line = line.strip()
        if line and line.lower() not in ("not supported", "[not supported]", "n/a", ""):
            try:
                return int(line)
            except ValueError:
                continue

    return None


def get_vllm_logs(service_name: str, lines: int = 200) -> str:
    """Holt die letzten N Zeilen des systemd-Journals für den vLLM-Service."""
    return run_command([
        "journalctl", "-u", service_name, "--no-pager", "-n", str(lines)
    ])


def parse_kv_cache_blocks(logs: str) -> Optional[Tuple[int, int]]:
    """
    Extrahiert aus vLLM-Logs:
      - Anzahl der KV Cache Blöcke
      - Block-Größe in Tokens
    
    Returns: (num_blocks, block_size) oder None
    """
    # Muster 1: "Number of blocks: 2048"
    num_blocks_match = re.search(
        r"Number of blocks:\s*(\d+)",
        logs,
        re.IGNORECASE,
    )
    
    # Muster 2: "Maximum concurrency: 8 sequences"
    concurrency_match = re.search(
        r"Maximum concurrency:\s*(\d+)\s*sequences?",
        logs,
        re.IGNORECASE,
    )
    
    # Muster 3: "block_size=16" oder "block size: 16"
    block_size_match = re.search(
        r"block[_\s]?size[=:]\s*(\d+)",
        logs,
        re.IGNORECASE,
    )
    
    num_blocks = int(num_blocks_match.group(1)) if num_blocks_match else None
    concurrency = int(concurrency_match.group(1)) if concurrency_match else None
    block_size = int(block_size_match.group(1)) if block_size_match else 16  # Default in vLLM
    
    if num_blocks is not None:
        return num_blocks, block_size
    
    # Fallback: Wenn nur "Maximum concurrency" gefunden wurde, können wir
    # num_blocks ausrechnen, wenn wir max_model_len kennen
    if concurrency is not None:
        # Annahme: num_blocks ≈ concurrency * (max_model_len / block_size)
        # Das ist ungenau, aber besser als nichts
        return None, block_size
    
    return None, block_size


def parse_model_info(logs: str) -> dict:
    """Extrahiert Modell-Informationen aus vLLM-Logs."""
    info = {
        "model_id": None,
        "hidden_size": None,
        "num_layers": None,
        "num_heads": None,
        "num_kv_heads": None,
        "dtype": "float16",
    }
    
    # Modell-ID
    model_match = re.search(r"Loading model '([^']+)'", logs) or \
                  re.search(r"model='([^']+)'", logs) or \
                  re.search(r"Using model '([^']+)'", logs)
    if model_match:
        info["model_id"] = model_match.group(1)
    
    # Hidden size
    hidden_match = re.search(r"hidden_size[=:]\s*(\d+)", logs, re.IGNORECASE)
    if hidden_match:
        info["hidden_size"] = int(hidden_match.group(1))
    
    # Num layers
    layers_match = re.search(r"num_hidden_layers[=:]\s*(\d+)", logs, re.IGNORECASE)
    if layers_match:
        info["num_layers"] = int(layers_match.group(1))
    
    # Num heads
    heads_match = re.search(r"num_attention_heads[=:]\s*(\d+)", logs, re.IGNORECASE)
    if heads_match:
        info["num_heads"] = int(heads_match.group(1))
    
    # Num KV heads
    kv_heads_match = re.search(r"num_key_value_heads[=:]\s*(\d+)", logs, re.IGNORECASE)
    if kv_heads_match:
        info["num_kv_heads"] = int(kv_heads_match.group(1))
    
    # Dtype
    dtype_match = re.search(r"dtype[=:]\s*(\w+)", logs, re.IGNORECASE)
    if dtype_match:
        info["dtype"] = dtype_match.group(1).lower()
    
    return info


def parse_gpu_memory_utilization(logs: str) -> Optional[float]:
    """Extrahiert gpu-memory-utilization aus Logs oder Start-Kommando."""
    match = re.search(r"gpu-memory-utilization[=\s]+(\d+\.?\d*)", logs, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def parse_max_model_len(logs: str) -> Optional[int]:
    """Extrahiert max-model-len aus Logs."""
    match = re.search(r"max-model-len[=\s]+(\d+)", logs, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def parse_max_num_seqs(logs: str) -> Optional[int]:
    """Extrahiert max-num-seqs aus Logs."""
    match = re.search(r"max-num-seqs[=\s]+(\d+)", logs, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def calculate_kv_cache_size_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype_bytes: int = 2,  # fp16 = 2 bytes
) -> int:
    """
    Berechnet die KV Cache Größe pro Token in Bytes.
    
    KV Cache = Key + Value pro Layer
    Pro Token: 2 (K+V) * num_layers * num_kv_heads * head_size * dtype_bytes
    """
    bytes_per_token = 2 * num_layers * num_kv_heads * head_size * dtype_bytes
    return bytes_per_token


def estimate_model_weights_mb(model_id: str, dtype: str = "float16") -> int:
    """
    Schätzt die Modell-Weights-Größe in MB basierend auf Modell-ID.
    """
    # Bekannte Modelle
    if "Voxtral-Mini-3B" in model_id:
        params = 3_000_000_000
    elif "Voxtral-Mini-4B" in model_id:
        params = 4_000_000_000
    elif "Voxtral-Small-24B" in model_id:
        params = 24_000_000_000
    else:
        # Extrahiere Zahl aus Modellnamen (z.B. "7B", "13B")
        size_match = re.search(r"(\d+)[Bb]", model_id)
        if size_match:
            params = int(size_match.group(1)) * 1_000_000_000
        else:
            params = 3_000_000_000  # Default
    
    bytes_per_param = 2 if dtype in ("float16", "half") else 4 if dtype == "float32" else 1
    return (params * bytes_per_param) // (1024 * 1024)


def calculate_real_capacity(
    gpu_memory_mb: int,
    gpu_memory_utilization: float,
    model_weights_mb: int,
    max_model_len: int,
    bytes_per_token: int,
    block_size: int = 16,
    overhead_mb: int = 512,  # CUDA Overhead, Framework, etc.
) -> dict:
    """
    Berechnet die tatsächliche maximale Parallelität.
    
    Formel:
    - Total Memory = gpu_memory_mb * gpu_memory_utilization
    - Available for KV Cache = Total Memory - Model Weights - Overhead
    - Tokens in Cache = Available Memory / bytes_per_token
    - Num Blocks = Tokens in Cache / block_size
    - Max Sequences = Num Blocks / (max_model_len / block_size)
    """
    total_memory_mb = gpu_memory_mb * gpu_memory_utilization
    available_kv_mb = total_memory_mb - model_weights_mb - overhead_mb
    
    if available_kv_mb <= 0:
        return {
            "error": "Nicht genug Speicher für KV Cache! Erhöhe gpu-memory-utilization.",
            "total_memory_mb": total_memory_mb,
            "model_weights_mb": model_weights_mb,
            "overhead_mb": overhead_mb,
        }
    
    available_kv_bytes = available_kv_mb * 1024 * 1024
    total_tokens_in_cache = int(available_kv_bytes / bytes_per_token)
    num_blocks = total_tokens_in_cache // block_size
    blocks_per_sequence = math.ceil(max_model_len / block_size)
    max_real_sequences = num_blocks // blocks_per_sequence
    
    return {
        "total_memory_mb": round(total_memory_mb, 1),
        "model_weights_mb": model_weights_mb,
        "overhead_mb": overhead_mb,
        "available_kv_mb": round(available_kv_mb, 1),
        "bytes_per_token": bytes_per_token,
        "total_tokens_in_cache": total_tokens_in_cache,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "blocks_per_sequence": blocks_per_sequence,
        "max_real_sequences": max_real_sequences,
    }


def get_config_from_systemd(service_name: str) -> dict:
    """Versucht, die aktuellen Parameter aus dem systemd Service zu extrahieren."""
    config = {}
    
    # Lese die Service-Datei
    service_file = f"/etc/systemd/system/{service_name}.service"
    if os.path.exists(service_file):
        with open(service_file, "r") as f:
            content = f.read()
        
        # Extrahiere das Start-Kommando
        exec_match = re.search(r"ExecStart=.*vllm serve.*", content)
        if exec_match:
            cmd = exec_match.group(0)
            config["gpu_memory_utilization"] = parse_gpu_memory_utilization(cmd)
            config["max_model_len"] = parse_max_model_len(cmd)
            config["max_num_seqs"] = parse_max_num_seqs(cmd)
            
            model_match = re.search(r"vllm serve\s+(\S+)", cmd)
            if model_match:
                config["model_id"] = model_match.group(1)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Berechnet die tatsächliche vLLM-Kapazität basierend auf Logs und Konfiguration."
    )
    parser.add_argument(
        "--service",
        default="voxtral-vllm",
        help="Name des systemd Services (default: voxtral-vllm)",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        help="GPU Memory Utilization (z.B. 0.45). Wenn nicht gesetzt, wird aus Logs gelesen.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Maximale Modell-Länge (z.B. 8192). Wenn nicht gesetzt, wird aus Logs gelesen.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Konfigurierte max-num-seqs (z.B. 20). Wenn nicht gesetzt, wird aus Logs gelesen.",
    )
    parser.add_argument(
        "--model-id",
        help="Modell-ID (z.B. mistralai/Voxtral-Mini-3B-2507). Wenn nicht gesetzt, wird aus Logs gelesen.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV Cache Block-Größe in Tokens (default: 16)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Ausgabe als JSON",
    )
    
    args = parser.parse_args()
    
    # GPU-Speicher ermitteln
    gpu_memory_mb = get_gpu_memory_mb()
    if gpu_memory_mb is None:
        print("[FEHLER] Konnte GPU-Speicher nicht ermitteln. Ist nvidia-smi verfügbar?")
        sys.exit(1)
    
    # Logs holen
    print(f"[INFO] Lese Logs für Service '{args.service}'...")
    logs = get_vllm_logs(args.service)
    
    # Konfiguration aus Logs oder systemd extrahieren
    config = get_config_from_systemd(args.service)
    
    # Überschreibe mit CLI-Argumenten
    gpu_mem_util = args.gpu_mem_util or config.get("gpu_memory_utilization") or parse_gpu_memory_utilization(logs) or 0.28
    max_model_len = args.max_model_len or config.get("max_model_len") or parse_max_model_len(logs) or 8192
    max_num_seqs_configured = args.max_num_seqs or config.get("max_num_seqs") or parse_max_num_seqs(logs) or 4
    model_id = args.model_id or config.get("model_id") or parse_model_info(logs).get("model_id") or "mistralai/Voxtral-Mini-3B-2507"
    
    # Modell-Info
    model_info = parse_model_info(logs)
    if model_info["model_id"] is None:
        model_info["model_id"] = model_id
    
    # KV Cache Blocks aus Logs
    blocks_info = parse_kv_cache_blocks(logs)
    num_blocks_from_logs = blocks_info[0] if blocks_info[0] else None
    block_size = blocks_info[1] if blocks_info[1] else args.block_size
    
    # Modell-Weights schätzen
    model_weights_mb = estimate_model_weights_mb(model_id, model_info.get("dtype", "float16"))
    
    # Wenn wir Modell-Architektur aus Logs haben, berechne bytes_per_token genau
    if model_info.get("num_layers") and model_info.get("num_kv_heads") and model_info.get("hidden_size") and model_info.get("num_heads"):
        head_size = model_info["hidden_size"] // model_info["num_heads"]
        bytes_per_token = calculate_kv_cache_size_per_token(
            model_info["num_layers"],
            model_info["num_kv_heads"],
            head_size,
            2 if model_info["dtype"] in ("float16", "half") else 4,
        )
    else:
        # Fallback: Schätzung für Voxtral 3B
        # ~3B Modelle: typisch 24 layers, 8 KV heads, 128 head_dim
        bytes_per_token = calculate_kv_cache_size_per_token(24, 8, 128, 2)
    
    # Kapazität berechnen
    capacity = calculate_real_capacity(
        gpu_memory_mb=gpu_memory_mb,
        gpu_memory_utilization=gpu_mem_util,
        model_weights_mb=model_weights_mb,
        max_model_len=max_model_len,
        bytes_per_token=bytes_per_token,
        block_size=block_size,
    )
    
    if "error" in capacity:
        print(f"[FEHLER] {capacity['error']}")
        sys.exit(1)
    
    # Ergebnis zusammenstellen
    result = {
        "gpu": {
            "total_memory_mb": gpu_memory_mb,
            "total_memory_gb": round(gpu_memory_mb / 1024, 2),
        },
        "configuration": {
            "model_id": model_id,
            "gpu_memory_utilization": gpu_mem_util,
            "max_model_len": max_model_len,
            "max_num_seqs_configured": max_num_seqs_configured,
            "block_size": block_size,
        },
        "model": {
            "estimated_weights_mb": model_weights_mb,
            "estimated_weights_gb": round(model_weights_mb / 1024, 2),
            "dtype": model_info.get("dtype", "float16"),
        },
        "kv_cache": {
            "available_mb": capacity["available_kv_mb"],
            "available_gb": round(capacity["available_kv_mb"] / 1024, 2),
            "bytes_per_token": bytes_per_token,
            "total_tokens": capacity["total_tokens_in_cache"],
            "num_blocks": capacity["num_blocks"],
            "blocks_per_sequence": capacity["blocks_per_sequence"],
        },
        "capacity": {
            "max_real_sequences": capacity["max_real_sequences"],
            "configured_max_num_seqs": max_num_seqs_configured,
            "is_configured_realistic": capacity["max_real_sequences"] >= max_num_seqs_configured,
            "utilization_at_configured": round(
                (max_num_seqs_configured / capacity["max_real_sequences"] * 100), 1
            ) if capacity["max_real_sequences"] > 0 else float('inf'),
        },
    }
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print("  vLLM KAPAZITÄTS-ANALYSE")
        print("=" * 60)
        print(f"\n📊 GPU-Speicher:")
        print(f"   Gesamt:           {gpu_memory_mb} MB ({gpu_memory_mb/1024:.1f} GB)")
        print(f"\n⚙️  Konfiguration:")
        print(f"   Modell:           {model_id}")
        print(f"   GPU Mem Util:     {gpu_mem_util*100:.0f}%")
        print(f"   Max Model Len:    {max_model_len} Tokens")
        print(f"   Max Num Seqs:     {max_num_seqs_configured} (konfiguriert)")
        print(f"   Block Size:       {block_size} Tokens")
        print(f"\n🧠 Modell-Weights (geschätzt):")
        print(f"   Größe:            {model_weights_mb} MB ({model_weights_mb/1024:.1f} GB)")
        print(f"   Datentyp:         {model_info.get('dtype', 'float16')}")
        print(f"\n💾 KV Cache Pool:")
        print(f"   Verfügbar:        {capacity['available_kv_mb']:.0f} MB ({capacity['available_kv_mb']/1024:.1f} GB)")
        print(f"   Bytes/Token:      {bytes_per_token}")
        print(f"   Gesamt-Tokens:    {capacity['total_tokens_in_cache']:,}")
        print(f"   Anzahl Blöcke:    {capacity['num_blocks']:,}")
        print(f"   Blöcke/Seq:       {capacity['blocks_per_sequence']}")
        print(f"\n🚀 PARALLELITÄT:")
        print(f"   Max REAL möglich: {capacity['max_real_sequences']} Sequenzen")
        print(f"   Konfiguriert:     {max_num_seqs_configured} Sequenzen")
        
        if capacity["max_real_sequences"] >= max_num_seqs_configured:
            util = (max_num_seqs_configured / capacity["max_real_sequences"] * 100)
            print(f"   Status:           ✅ REALISTISCH")
            print(f"   Auslastung:       {util:.1f}% des theoretischen Maximums")
        else:
            deficit = max_num_seqs_configured - capacity["max_real_sequences"]
            print(f"   Status:           ❌ NICHT REALISTISCH")
            print(f"   Defizit:          {deficit} Sequenzen zu viel konfiguriert!")
            print(f"\n💡 Empfehlung:")
            print(f"   Setze --max-num-seqs auf max. {capacity['max_real_sequences']}")
            print(f"   Oder erhöhe --gpu-memory-utilization auf min. "
                  f"{((model_weights_mb + 512 + (max_num_seqs_configured * max_model_len * bytes_per_token / (1024*1024))) / gpu_memory_mb):.2f}")
        
        print("\n" + "=" * 60)
        print("\n📋 Formel:")
        print("   Max Sequences = KV Cache Pool / (Max Model Len × Bytes pro Token)")
        print("   KV Cache Pool = (GPU Gesamt × Utilization) - Model Weights - Overhead")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
