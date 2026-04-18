#!/usr/bin/env python3
"""Quick test of the correction LLM API on the DGX Spark."""
import os, json, paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.188.173", username="ksai0001_local", password=os.environ["GAITA_PWD"])

payload = json.dumps({
    "model": "nvidia/Qwen3-8B-FP8",
    "messages": [{"role": "user", "content": "Korrigiere folgenden Text und gib NUR den korrigierten Text zurück: Ich habe gestern ein treffen mit meine Kolegen gehabt"}],
    "max_tokens": 200
})

cmd = f"curl -s http://localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{payload}'"
stdin, stdout, stderr = c.exec_command(cmd, timeout=120)
result = stdout.read().decode()
c.close()

try:
    data = json.loads(result)
    content = data["choices"][0]["message"]["content"]
    print(f"✅ Antwort: {content}")
except Exception as e:
    print(f"Raw: {result}")
    print(f"Error: {e}")
