"""Qwen3-ForcedAligner Model Manager – Thread-safe singleton.

Loads Qwen3-ForcedAligner-0.6B for word-level timestamp alignment.
Used to add timestamps to Voxtral transcriptions.
"""

import os
import threading
import time

import torch

DEFAULT_ALIGNER_MODEL = os.getenv("QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")


class AlignerManager:
    """Lazy-loading singleton for Qwen3-ForcedAligner."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._loading = False

    def get_model(
        self,
        model_name: str = DEFAULT_ALIGNER_MODEL,
        device: str = "cpu",
    ):
        """Return the loaded Qwen3ForcedAligner (lazy init, thread-safe)."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            self._loading = True
            print(f"--- ForcedAligner: Lade {model_name} auf {device} ---")
            t0 = time.time()

            from qwen_asr import Qwen3ForcedAligner

            dtype = torch.bfloat16 if device != "cpu" else torch.float32
            kwargs = {
                "dtype": dtype,
                "device_map": device,
            }
            # Only set attn_implementation on CUDA
            if device != "cpu":
                attn_impl = "sdpa"
                try:
                    from flash_attn import flash_attn_func  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    pass
                kwargs["attn_implementation"] = attn_impl

            model = Qwen3ForcedAligner.from_pretrained(model_name, **kwargs)

            elapsed = time.time() - t0
            print(f"--- ForcedAligner: Geladen in {elapsed:.1f}s (device={device}) ---")

            self._model = model
            self._loading = False
            return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading


# Global singleton
aligner_manager = AlignerManager()
