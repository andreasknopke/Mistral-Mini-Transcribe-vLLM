"""Thread-sicherer WhisperX-Worker-Pool für CUDA/Linux."""

import os
import queue
import threading
import time
import traceback

try:
    import torch

    _original_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
except Exception:
    pass

try:
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, sample_rate: int = 0, num_frames: int = 0, num_channels: int = 0, bits_per_sample: int = 0, encoding: str = "PCM_S"):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None

    if not hasattr(torchaudio, "info"):
        import soundfile as sf

        def _compat_info(path_or_file, backend=None):
            info = sf.info(path_or_file)
            return torchaudio.AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=info.subtype_info,
                encoding=info.format,
            )

        torchaudio.info = _compat_info
except Exception:
    pass

import whisperx

MODELS = [
    "large-v3",
    "guillaumekln/faster-whisper-large-v2",
    "large-v2",
    "cstr/whisper-large-v3-turbo-german-int8_float32",
]

DEFAULT_POOL_SIZE = int(os.getenv("WHISPERX_POOL_SIZE", "2"))
if DEFAULT_POOL_SIZE < 1:
    DEFAULT_POOL_SIZE = 1
    print("--- POOL: WHISPERX_POOL_SIZE muss >= 1 sein, setze auf 1 ---")

print(f"--- POOL: Konfigurierte Größe: {DEFAULT_POOL_SIZE} Worker ---")


class ModelWorker:
    """Repräsentiert einen einzelnen WhisperX-Worker mit vorgeladenem Modell."""

    def __init__(self, worker_id: int, model_name: str, device: str):
        self.worker_id = worker_id
        self.model_name = model_name
        self.device = device
        self.model = None
        self.created_at = time.time()
        self.jobs_completed = 0
        self._load_model()

    def _load_model(self) -> None:
        requested_device = self.device
        if requested_device == "cuda":
            compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", "float16")
        else:
            compute_type = os.getenv("WHISPERX_CPU_COMPUTE_TYPE", "int8")

        print(
            f"--- POOL: Worker {self.worker_id} lädt Modell "
            f"'{self.model_name}' auf '{requested_device}' (compute: {compute_type}) ---"
        )
        try:
            self.model = whisperx.load_model(
                self.model_name,
                device=requested_device,
                compute_type=compute_type,
            )
        except ValueError as exc:
            if requested_device == "cuda" and "not compiled with CUDA support" in str(exc):
                fallback_device = "cpu"
                fallback_compute_type = os.getenv("WHISPERX_CPU_COMPUTE_TYPE", "int8")
                print(
                    f"--- POOL: CUDA für CTranslate2 nicht verfügbar, "
                    f"falle auf '{fallback_device}' zurück (compute: {fallback_compute_type}) ---"
                )
                self.device = fallback_device
                self.model = whisperx.load_model(
                    self.model_name,
                    device=fallback_device,
                    compute_type=fallback_compute_type,
                )
            else:
                raise
        print(f"--- POOL: Worker {self.worker_id} bereit auf '{self.device}' ---")

    def increment_jobs(self) -> None:
        self.jobs_completed += 1


class ModelPool:
    """Pool aus WhisperX-Modellinstanzen für parallele Transkription."""

    def __init__(self, pool_size: int = DEFAULT_POOL_SIZE):
        self.pool_size = pool_size
        self._pool: queue.Queue[ModelWorker] = queue.Queue(maxsize=pool_size)
        self._workers: list[ModelWorker] = []
        self._lock = threading.Lock()
        self._initialized = False
        self._current_model_name = None
        self._current_device = None
        self._pending_requests = 0
        self._pending_lock = threading.Lock()

    def initialize(self, model_name: str = "large-v3", device: str = "cuda") -> None:
        with self._lock:
            if self._initialized and model_name == self._current_model_name and device == self._current_device:
                return

            self._cleanup_workers()
            last_error: Exception | None = None
            estimated_usage_gb = self.pool_size * float(os.getenv("WHISPERX_ESTIMATED_WORKER_GB", "6"))
            try:
                import psutil

                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                print(f"\n{'=' * 60}")
                print(f"POOL: System RAM: {total_ram_gb:.0f} GB")
                print(
                    f"POOL: Geschätzte Nutzung: {estimated_usage_gb:.1f} GB "
                    f"({self.pool_size} Worker)"
                )
                if estimated_usage_gb > total_ram_gb * 0.7:
                    print(
                        f"⚠️  WARNUNG: Worker-Anzahl könnte zu hoch sein. "
                        f"Empfohlen: max {max(1, int(total_ram_gb * 0.7 / max(estimated_usage_gb / self.pool_size, 1)))} Worker"
                    )
                print(f"{'=' * 60}\n")
            except ImportError:
                pass

            print(f"POOL: Initialisiere {self.pool_size} Worker mit '{model_name}' auf '{device}'")
            for index in range(self.pool_size):
                try:
                    worker = ModelWorker(index, model_name, device)
                    self._workers.append(worker)
                    self._pool.put(worker)
                except Exception as exc:
                    last_error = exc
                    print(f"--- POOL: Worker {index} konnte nicht erstellt werden: {exc} ---")
                    traceback.print_exc()
                    break

            if not self._workers:
                message = "Kein WhisperX-Worker konnte initialisiert werden."
                if last_error is not None:
                    message = f"{message} Ursache: {last_error}"
                raise RuntimeError(message)

            self._current_model_name = model_name
            self._current_device = device
            self._initialized = True
            self.pool_size = len(self._workers)
            print(f"--- POOL: {len(self._workers)} Worker erfolgreich initialisiert ---")

    def acquire(self, model_name: str, device: str, timeout: float = 120) -> ModelWorker:
        if not self._initialized or model_name != self._current_model_name or device != self._current_device:
            self.initialize(model_name, device)

        with self._pending_lock:
            self._pending_requests += 1

        try:
            print(
                f"--- POOL: Worker angefordert ({self.get_available_count()}/{self.pool_size} frei, "
                f"{self.get_queue_size()} wartend) ---"
            )
            return self._pool.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Kein Worker innerhalb von {timeout}s verfügbar. Alle {self.pool_size} Worker sind belegt."
            ) from exc
        finally:
            with self._pending_lock:
                self._pending_requests -= 1

    def release(self, worker: ModelWorker) -> None:
        worker.increment_jobs()
        self._pool.put(worker)
        print(
            f"--- POOL: Worker {worker.worker_id} freigegeben (Jobs: {worker.jobs_completed}) ---"
        )

    def get_available_count(self) -> int:
        return self._pool.qsize()

    def get_queue_size(self) -> int:
        with self._pending_lock:
            return self._pending_requests

    def _cleanup_workers(self) -> None:
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
        self._workers.clear()

        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


model_pool = ModelPool(pool_size=DEFAULT_POOL_SIZE)
