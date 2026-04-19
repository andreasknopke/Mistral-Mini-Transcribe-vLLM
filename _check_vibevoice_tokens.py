"""Check VibeVoice special tokens and correct EOS."""
import sys
sys.path.insert(0, "scripts")
from ssh_helper import get_client

SCRIPT = """\
import sys
sys.path.insert(0, '/home/ksai0001_local/vibevoice-spark')
from src.model_manager import model_pool

w = model_pool.acquire('microsoft/VibeVoice-ASR', 'cuda', timeout=120)
t = w.processor.tokenizer

# Find all EOS-like tokens
print('=== Looking for EOS-like special tokens ===')
for name, tid in sorted(t.get_added_vocab().items()):
    if any(x in name.lower() for x in ['end', 'eos', 'im_end', 'stop', 'eot']):
        print(f'  {name!r} -> {tid}')

print()
print('=== Key token IDs ===')
for name in ['<|endoftext|>', '<|im_end|>', '<|im_start|>', '<|end|>', '<|eot_id|>']:
    try:
        tid = t.convert_tokens_to_ids(name)
        print(f'  {name!r} -> {tid}')
    except:
        print(f'  {name!r} -> NOT FOUND')

# Check what VibeVoice processor itself recommends
p = w.processor
print()
print('=== Processor generate-related attrs ===')
for attr in dir(p):
    if any(x in attr.lower() for x in ['eos', 'end', 'stop', 'gen', 'decode']):
        val = getattr(p, attr, None)
        if not callable(val):
            print(f'  {attr}: {val}')

model_pool.release(w)
"""

c = get_client()
sftp = c.open_sftp()
with sftp.file("/tmp/_vv_tokens.py", "w") as f:
    f.write(SCRIPT)
sftp.close()

stdin, stdout, stderr = c.exec_command(
    "/home/ksai0001_local/vibevoice-env/bin/python3 /tmp/_vv_tokens.py",
    timeout=300,
)
print(stdout.read().decode(errors="replace"))
c.close()
