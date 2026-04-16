"""Patch voxtral_orig.py -> voxtral_patched.py: CPU STFT for GB10 cuFFT workaround."""
import shutil

src = "/home/ksai0001_local/voxtral-vllm/voxtral_orig.py"
dst = "/home/ksai0001_local/voxtral-vllm/voxtral_patched.py"

shutil.copy2(src, dst)

with open(dst) as f:
    text = f.read()

OLD = """        window = torch.hann_window(
            self.config.window_size, device=audio_waveforms.device
        )
        stft = torch.stft(
            audio_waveforms,
            self.config.window_size,
            self.config.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.T @ magnitudes"""

NEW = """        # cuFFT workaround for GB10: run STFT on CPU
        window = torch.hann_window(
            self.config.window_size, device="cpu"
        )
        stft = torch.stft(
            audio_waveforms.float().cpu(),
            self.config.window_size,
            self.config.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        magnitudes = magnitudes.to(
            device=audio_waveforms.device, dtype=self.mel_filters.dtype
        )
        mel_spec = self.mel_filters.T @ magnitudes"""

assert OLD in text, "Pattern not found!"
text = text.replace(OLD, NEW)

with open(dst, "w") as f:
    f.write(text)

# Verify syntax
import py_compile
py_compile.compile(dst, doraise=True)
print("Patched and syntax-verified OK!")
