"""Start vibevoice + deploy admin panel to DGX Spark.

Workflow:
  1. Sync staging → service dir, then start/restart vibevoice
  2. Upload admin panel files, sudo cp to /opt, restart spark-admin

Paths:
  vibevoice staging:  ~/voxtral-setup/vibevoice_spark/
  vibevoice service:  ~/vibevoice-spark/
  admin staging:      ~/voxtral-setup/spark_admin/
  admin service:      /opt/spark-admin/
"""
import os, sys, time
sys.path.insert(0, "scripts")
from ssh_helper import get_client

HOME = "/home/ksai0001_local"
pwd = os.environ["GAITA_PWD"]


def run(cmd, timeout=60):
    c = get_client()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    print(out)
    if err:
        print(err)
    return rc


# ── 1) VibeVoice: sync + start ──────────────────────────────────
print("=== Syncing & starting VibeVoice ===")
run(
    f"cp -r {HOME}/voxtral-setup/vibevoice_spark/* {HOME}/vibevoice-spark/ 2>&1"
    f" && echo '✅ Files synced'"
    f" && printf '%s\n' '{pwd}' | sudo -S systemctl restart vibevoice 2>&1"
)
time.sleep(3)
run("systemctl status vibevoice --no-pager -l 2>&1")

# ── 2) Admin panel: upload + copy + restart ─────────────────────
print("\n=== Deploying Spark Admin Panel ===")
c = get_client()
sftp = c.open_sftp()
staging = f"{HOME}/voxtral-setup/spark_admin"
local_base = os.path.join(os.path.dirname(__file__), "spark_admin")
for sub in ["app.py", "requirements.txt",
            "templates/dashboard.html", "templates/login.html",
            "static/app.js", "static/style.css"]:
    local = os.path.join(local_base, sub)
    if os.path.exists(local):
        sftp.put(local, f"{staging}/{sub}")
        print(f"  ↑ {sub}")
sftp.close()
c.close()

print("\n=== Copying to /opt & restarting spark-admin ===")
run(
    f"printf '%s\n' '{pwd}' | sudo -S cp -r {staging}/* /opt/spark-admin/ 2>&1"
    f" && printf '%s\n' '{pwd}' | sudo -S systemctl restart spark-admin 2>&1"
)
time.sleep(2)
run("systemctl status spark-admin --no-pager -l 2>&1")

print("\n✅ Done!")
print("   Admin panel: http://192.168.188.173:7000")
print("   VibeVoice:   http://192.168.188.173:7862")
