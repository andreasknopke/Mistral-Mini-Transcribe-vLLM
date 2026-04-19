"""Restart vibevoice service – always sync staging → service dir first.

Paths:
  Staging:  ~/voxtral-setup/vibevoice_spark/
  Service:  ~/vibevoice-spark/
"""
import os, sys
sys.path.insert(0, "scripts")
from ssh_helper import get_client

STAGING = "/home/ksai0001_local/voxtral-setup/vibevoice_spark"
SERVICE_DIR = "/home/ksai0001_local/vibevoice-spark"

pwd = os.environ["GAITA_PWD"]
c = get_client()
cmd = (
    f"cp -r {STAGING}/* {SERVICE_DIR}/ 2>&1"
    f" && echo '✅ Files synced to {SERVICE_DIR}'"
    f" && printf '%s\n' '{pwd}' | sudo -S systemctl restart vibevoice 2>&1"
    f" ; sleep 2; systemctl status vibevoice --no-pager -l 2>&1"
)
stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
print(stdout.read().decode(errors="replace"))
c.close()
