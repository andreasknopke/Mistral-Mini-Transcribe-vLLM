"""Canonically merge a LoRA adapter into the ORIGINAL Mistral-format
consolidated.safetensors of Voxtral-Mini-3B-2507.

Output format is byte-compatible with the original model:
- LM weights:  layers.N.attention.{wq,wk,wv,wo}.weight, layers.N.feed_forward.{w1,w2,w3}.weight
- AUDIO weights (with prefix): mm_whisper_embeddings.whisper_encoder.transformer.layers.N.attention.{wq,wk,wv,wo}.{weight,bias}, ...feed_forward.{w1,w2}.{weight,bias}
- All other weights pass through unchanged.

LoRA adapter key format (HF/PEFT):
  base_model.model.model.language_model.layers.<N>.self_attn.<mod>.lora_{A,B}.weight
  base_model.model.model.language_model.layers.<N>.mlp.<mod>.lora_{A,B}.weight
  base_model.model.model.audio_tower.layers.<N>.self_attn.<mod>.lora_{A,B}.weight

Mapping tables:
  LM:        q_proj->attention.wq   k_proj->attention.wk
             v_proj->attention.wv   o_proj->attention.wo
             gate_proj->feed_forward.w1   down_proj->feed_forward.w2
             up_proj->feed_forward.w3

  AUDIO:     q_proj->attention.wq   k_proj->attention.wk   v_proj->attention.wv
             (out_proj not trained)
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("merge_native")

# ──── LM module name mapping ────
LM_MODULE_MAP = {
    "q_proj":     "attention.wq",
    "k_proj":     "attention.wk",
    "v_proj":     "attention.wv",
    "o_proj":     "attention.wo",
    "gate_proj":  "feed_forward.w1",
    "down_proj":  "feed_forward.w2",
    "up_proj":    "feed_forward.w3",
}

# ──── Audio module name mapping (only self_attn, no out_proj trained) ────
AUDIO_MODULE_MAP = {
    "q_proj": "attention.wq",
    "k_proj": "attention.wk",
    "v_proj": "attention.wv",
}

LM_PREFIX    = "base_model.model.model.language_model.layers."
AUDIO_PREFIX = "base_model.model.model.audio_tower.layers."


def parse_lora_key(key):
    """Return dict with {'domain':'lm'/'audio', 'layer':N, 'mistral':'...', 'ab':'A'/'B'}.
    Or None if not a recognised LoRA target.
    """
    if key.startswith(LM_PREFIX):
        suffix = key[len(LM_PREFIX):]
        parts = suffix.split(".")
        # 0.self_attn.q_proj.lora_A.weight   (5 parts)
        # 5.mlp.gate_proj.lora_B.weight      (5 parts)
        if len(parts) != 5:
            return None
        try:
            layer = int(parts[0])
        except ValueError:
            return None
        parent = parts[1]             # 'self_attn' or 'mlp'
        hf_mod = parts[2]
        mistral_mod = LM_MODULE_MAP.get(hf_mod)
        if mistral_mod is None:
            return None
        if parts[3] not in ("lora_A", "lora_B") or parts[4] != "weight":
            return None
        return {"domain": "lm", "layer": layer, "mistral": mistral_mod, "ab": parts[3]}

    if key.startswith(AUDIO_PREFIX):
        suffix = key[len(AUDIO_PREFIX):]
        parts = suffix.split(".")
        # 0.self_attn.q_proj.lora_A.weight
        if len(parts) != 5:
            return None
        try:
            layer = int(parts[0])
        except ValueError:
            return None
        hf_mod = parts[2]
        mistral_mod = AUDIO_MODULE_MAP.get(hf_mod)
        if mistral_mod is None:
            return None
        if parts[3] not in ("lora_A", "lora_B") or parts[4] != "weight":
            return None
        return {"domain": "audio", "layer": layer, "mistral": mistral_mod, "ab": parts[3]}

    return None


def mistral_key_for(parsed):
    """Convert parsed LoRA info to the canonical Mistral consolidated.safetensors
    key where W_orig lives.
    """
    if parsed["domain"] == "lm":
        return f"layers.{parsed['layer']}.{parsed['mistral']}.weight"
    if parsed["domain"] == "audio":
        return (f"mm_whisper_embeddings.whisper_encoder.transformer.layers"
                f".{parsed['layer']}.{parsed['mistral']}.weight")
    raise ValueError("unknown domain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original",
                    default="/root/.cache/huggingface/hub/models--mistralai--Voxtral-Mini-3B-2507/snapshots/3060fe34b35ba5d44202ce9ff3c097642914f8f3/consolidated.safetensors")
    ap.add_argument("--lora", required=True,
                    help="Path to adapter_model.safetensors")
    ap.add_argument("--adapter-config", default=None)
    ap.add_argument("--output", required=True,
                    help="Output consolidated.safetensors path")
    args = ap.parse_args()

    # ──── LoRA alpha/r ────
    cfg_path = Path(args.adapter_config or (Path(args.lora).parent / "adapter_config.json"))
    with open(cfg_path, "r") as f:
        adapter_cfg = json.load(f)
    lora_alpha = float(adapter_cfg.get("lora_alpha", 64))
    lora_r = float(adapter_cfg.get("r", 32))
    scaling = lora_alpha / lora_r
    log.info(f"LoRA scaling = alpha/r = {lora_alpha}/{lora_r} = {scaling}")

    # ──── Load LoRA ────
    log.info(f"Loading LoRA adapter: {args.lora}")
    A = {}
    B = {}
    with safe_open(args.lora, framework="pt") as f:
        for key in f.keys():
            parsed = parse_lora_key(key)
            if parsed is None:
                continue
            mistral_k = mistral_key_for(parsed)
            tensor = f.get_tensor(key)
            store = A if parsed["ab"] == "lora_A" else B
            store[mistral_k] = tensor
    log.info(f"Loaded LoRA-A: {len(A)} matrices  LoRA-B: {len(B)} matrices")
    pairs = set(A.keys()) & set(B.keys())
    log.info(f"{len(pairs)} complete (A,B) pairs → applying LoRA delta")

    # ──── Apply delta ────
    log.info(f"Loading original weights from: {args.original}")
    final = {}
    touched_lm = 0
    touched_audio = 0
    with safe_open(args.original, framework="pt") as f:
        for k in f.keys():
            tensor = f.get_tensor(k).clone()
            if k in pairs:
                delta = scaling * (B[k].to(tensor.dtype) @ A[k].to(tensor.dtype))
                if delta.shape != tensor.shape:
                    log.warning(f"shape mismatch {k}: W={tuple(tensor.shape)} "
                                f"delta={tuple(delta.shape)} — skipping")
                    final[k] = tensor
                    continue
                tensor.add_(delta)
                if k.startswith("layers."):
                    touched_lm += 1
                elif k.startswith("mm_whisper_embeddings"):
                    touched_audio += 1
            final[k] = tensor
    log.info(f"Applied LoRA delta: {touched_lm} LM, {touched_audio} audio matrices")

    # ──── Save ────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing: {out}")
    save_file(final, str(out), metadata={"format": "pt"})
    log.info(f"Done. Tensors: {len(final)}, size: {out.stat().st_size/1024**3:.2f} GB")

    n_audio = sum(1 for k in final if k.startswith("mm_whisper_embeddings"))
    n_lm = sum(1 for k in final if k.startswith("layers.") or k in ("norm.weight", "output.weight"))
    log.info(f"Output key buckets: {n_audio} audio + {n_lm} LM = {len(final)} total")


if __name__ == "__main__":
    main()
