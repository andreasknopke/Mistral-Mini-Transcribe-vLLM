import os
from huggingface_hub import HfApi, hf_hub_download
token = os.environ.get("HF_TOKEN", "")
print(f"Token present: {bool(token)}, length: {len(token)}")
api = HfApi(token=token)
try:
    info = api.whoami()
    print(f"Logged in as: {info.get('name', info)}")
except Exception as e:
    print(f"Auth error: {e}")

# Test if we can access the gated model
try:
    info = api.model_info("mistralai/Mistral-Small-24B-Instruct-2501", token=token)
    print(f"Model access OK: {info.id}, gated={info.gated}")
except Exception as e:
    print(f"Model access FAILED: {e}")
