#!/usr/bin/env python3
"""Check admin panel service status."""
import os, json, paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.188.173", username="ksai0001_local", password=os.environ["GAITA_PWD"])
stdin, stdout, stderr = c.exec_command("curl -s http://localhost:7000/api/overview", timeout=30)
data = json.loads(stdout.read().decode())
c.close()

print(json.dumps(data, indent=2)[:2000])
