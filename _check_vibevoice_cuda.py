"""Check VibeVoice CUDA readiness on DGX Spark."""
import sys
sys.path.insert(0, "scripts")
from ssh_helper import get_client

SCRIPT = """\
import torch, sys
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    print('cuDNN:', torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A')
    print('BF16 support:', torch.cuda.is_bf16_supported())

try:
    from flash_attn import flash_attn_func
    print('FlashAttention2: INSTALLED')
except ImportError:
    print('FlashAttention2: NOT INSTALLED')

print('SDPA available:', hasattr(torch.nn.functional, 'scaled_dot_product_attention'))

try:
    sys.path.insert(0, '/home/ksai0001_local/vibevoice-spark')
    from src.model_manager import model_pool
    w = model_pool.acquire('microsoft/VibeVoice-ASR', 'cuda', timeout=120)
    print()
    print('=== Model loaded ===')
    print('Model device:', next(w.model.parameters()).device)
    print('Model dtype:', next(w.model.parameters()).dtype)
    config = w.model.config
    attn = getattr(config, '_attn_implementation', 'unknown')
    print('Attention impl:', attn)
    model_pool.release(w)
except Exception as e:
    print(f'Model check error: {e}')
"""

c = get_client()
sftp = c.open_sftp()
with sftp.file("/tmp/_vv_diag.py", "w") as f:
    f.write(SCRIPT)
sftp.close()

stdin, stdout, stderr = c.exec_command(
    "/home/ksai0001_local/vibevoice-env/bin/python3 /tmp/_vv_diag.py",
    timeout=300,
)
print(stdout.read().decode(errors="replace"))
err = stderr.read().decode(errors="replace")
if err:
    print("STDERR:", err)
c.close()
