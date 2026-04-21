"""Deploy admin panel: upload → sudo cp to /opt → restart.

Paths:
  Staging:  ~/voxtral-setup/spark_admin/
  Service:  /opt/spark-admin/           (systemd spark-admin.service)
"""
import os, sys
sys.path.insert(0, "scripts")
from ssh_helper import get_client

HOME = "/home/ksai0001_local"
STAGING = f"{HOME}/voxtral-setup/spark_admin"
SERVICE_DIR = "/opt/spark-admin"
pwd = os.environ["GAITA_PWD"]

ADMIN_FILES = [
    "app.py", "requirements.txt",
    "templates/dashboard.html", "templates/login.html",
    "static/app.js", "static/password-editor.js", "static/style.css",
]

# 1) Upload
c = get_client()
sftp = c.open_sftp()
for f in ADMIN_FILES:
    local = os.path.join("spark_admin", f)
    if os.path.exists(local):
        sftp.put(local, f"{STAGING}/{f}")
        print(f"  ↑ {f}")
sftp.close()
c.close()

# 2) sudo cp + restart
c = get_client()
cmd = (
    f"printf '%s\\n' '{pwd}' | sudo -S cp -r {STAGING}/* {SERVICE_DIR}/ 2>&1"
    f" && echo '✅ Copied to {SERVICE_DIR}'"
    f" && printf '%s\\n' '{pwd}' | sudo -S systemctl restart spark-admin 2>&1"
    f"; sleep 2; systemctl status spark-admin --no-pager -l 2>&1"
)
stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
print(stdout.read().decode(errors="replace"))
c.close()
print("✅ Admin deploy done.")
