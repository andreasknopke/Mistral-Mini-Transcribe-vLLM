"""Upload vibevoice_spark files to DGX Spark via SFTP."""
import os, sys, paramiko, stat

HOST = os.environ.get("GAITA_HOST", "192.168.188.173")
USER = os.environ.get("GAITA_LOGIN", "ksai0001_local").split("@")[0]
PWD  = os.environ.get("GAITA_PWD", "")
REMOTE_BASE = "/home/ksai0001_local/voxtral-setup"

LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    ("vibevoice_spark/app.py", "vibevoice_spark/app.py"),
    ("vibevoice_spark/requirements.txt", "vibevoice_spark/requirements.txt"),
    ("vibevoice_spark/src/__init__.py", "vibevoice_spark/src/__init__.py"),
    ("vibevoice_spark/src/model_manager.py", "vibevoice_spark/src/model_manager.py"),
    ("vibevoice_spark/src/transcriber.py", "vibevoice_spark/src/transcriber.py"),
    ("scripts/10_install_vibevoice_dgx_spark.sh", "10_install_vibevoice_dgx_spark.sh"),
]

def mkdir_p(sftp, path):
    dirs = []
    while path not in ('/', ''):
        dirs.append(path)
        path = os.path.dirname(path)
    dirs.reverse()
    for d in dirs:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PWD, timeout=15)
sftp = c.open_sftp()

for local_rel, remote_rel in FILES:
    local_path = os.path.join(LOCAL_BASE, local_rel)
    remote_path = REMOTE_BASE + "/" + remote_rel
    remote_dir = os.path.dirname(remote_path)
    mkdir_p(sftp, remote_dir)
    sftp.put(local_path, remote_path)
    print(f"✅ {local_rel} -> {remote_path}")

sftp.close()
c.close()
print("\nAlle Dateien hochgeladen.")
