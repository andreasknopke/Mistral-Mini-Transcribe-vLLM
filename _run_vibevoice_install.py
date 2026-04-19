#!/usr/bin/env python3
"""Run vibevoice install script on DGX Spark remotely."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from ssh_helper import get_client

pwd = os.environ["GAITA_PWD"]
c = get_client()

cmd = (
    f"export SUDO_PASSWORD='{pwd}' && "
    "cd ~/voxtral-setup && "
    "tr -d '\\r' < 10_install_vibevoice_dgx_spark.sh > /tmp/vibevoice_install.sh && "
    "bash /tmp/vibevoice_install.sh 2>&1"
)

stdin, stdout, stderr = c.exec_command(cmd, timeout=1200)
for line in iter(stdout.readline, ""):
    print(line, end="", flush=True)
rc = stdout.channel.recv_exit_status()
err = stderr.read().decode(errors="replace")
if err:
    print(err, end="")
c.close()
print(f"\nExit code: {rc}")
sys.exit(rc)
