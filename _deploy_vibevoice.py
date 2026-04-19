#!/usr/bin/env python3
"""Deploy vibevoice: upload → copy to service dir → restart.

Paths:
  Staging:  ~/voxtral-setup/vibevoice_spark/
  Service:  ~/vibevoice-spark/          (systemd vibevoice.service)
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from ssh_helper import get_client

HOME = "/home/ksai0001_local"
STAGING = f"{HOME}/voxtral-setup/vibevoice_spark"
SERVICE_DIR = f"{HOME}/vibevoice-spark"

APP_FILES = [
    "app.py",
    "requirements.txt",
    "src/__init__.py",
    "src/model_manager.py",
    "src/transcriber.py",
]


def upload(c):
    """SFTP upload local vibevoice_spark/ → staging dir."""
    sftp = c.open_sftp()
    for d in [STAGING, f"{STAGING}/src"]:
        try:
            sftp.mkdir(d)
        except OSError:
            pass
    for f in APP_FILES:
        local = os.path.join("vibevoice_spark", f)
        remote = f"{STAGING}/{f}"
        sftp.put(local, remote)
        print(f"  ↑ {local}")
    # Also upload install script
    sftp.put(
        os.path.join("scripts", "10_install_vibevoice_dgx_spark.sh"),
        f"{HOME}/voxtral-setup/10_install_vibevoice_dgx_spark.sh",
    )
    sftp.close()
    print("Upload done.")


def copy_and_restart(c):
    """Copy staging → service dir, restart systemd unit."""
    pwd = os.environ["GAITA_PWD"]
    cmd = (
        f"cp -r {STAGING}/* {SERVICE_DIR}/ 2>&1"
        f" && echo 'Copied to {SERVICE_DIR}'"
        f" && printf '%s\n' '{pwd}' | sudo -S systemctl restart vibevoice 2>&1"
        f" ; sleep 2; systemctl status vibevoice --no-pager -l 2>&1"
    )
    stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
    print(stdout.read().decode(errors="replace"))


def main():
    c = get_client()
    upload(c)
    c.close()

    c = get_client()
    copy_and_restart(c)
    c.close()
    print("✅ Deploy + Restart done.")


if __name__ == "__main__":
    main()
