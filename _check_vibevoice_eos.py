"""Check VibeVoice EOS token config on DGX Spark."""
import sys
sys.path.insert(0, "scripts")
from ssh_helper import get_client

SCRIPT = """\
import sys
sys.path.insert(0, '/home/ksai0001_local/vibevoice-spark')
from src.model_manager import model_pool

w = model_pool.acquire('microsoft/VibeVoice-ASR', 'cuda', timeout=120)
p = w.processor

print('=== Processor attributes ===')
print('Has pad_id:', hasattr(p, 'pad_id'))
if hasattr(p, 'pad_id'):
    print('  pad_id:', p.pad_id)

print('Has tokenizer:', hasattr(p, 'tokenizer'))
if hasattr(p, 'tokenizer'):
    t = p.tokenizer
    print('  eos_token_id:', t.eos_token_id)
    print('  eos_token:', repr(t.eos_token) if hasattr(t, 'eos_token') else 'N/A')
    print('  pad_token_id:', t.pad_token_id)

print()
print('=== Model generation_config ===')
gc = w.model.generation_config
print('  eos_token_id:', gc.eos_token_id)
print('  pad_token_id:', gc.pad_token_id)
print('  max_new_tokens:', gc.max_new_tokens)
print('  max_length:', gc.max_length)

# Check all special tokens
if hasattr(p, 'tokenizer'):
    t = p.tokenizer
    print()
    print('=== Special tokens ===')
    for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id']:
        print(f'  {attr}: {getattr(t, attr, "N/A")}')
    if hasattr(t, 'additional_special_tokens'):
        print(f'  additional_special_tokens: {t.additional_special_tokens[:10]}')

model_pool.release(w)
"""

c = get_client()
sftp = c.open_sftp()
with sftp.file("/tmp/_vv_eos.py", "w") as f:
    f.write(SCRIPT)
sftp.close()

stdin, stdout, stderr = c.exec_command(
    "/home/ksai0001_local/vibevoice-env/bin/python3 /tmp/_vv_eos.py",
    timeout=300,
)
print(stdout.read().decode(errors="replace"))
err = stderr.read().decode(errors="replace")
if err:
    # Filter out known warnings
    for line in err.splitlines():
        if any(x in line for x in ['preprocessor_config', 'tokenizer class', 'torch_dtype', 'Loading checkpoint', '100%']):
            continue
        print("STDERR:", line)
c.close()
