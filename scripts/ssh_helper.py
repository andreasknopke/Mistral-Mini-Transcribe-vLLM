#!/usr/bin/env python3
"""SSH helper for automated DGX Spark access via paramiko."""
import os, sys, paramiko, time

def get_client():
    login = os.environ.get("GAITA_LOGIN", "ksai0001_Local").split("@")[0]
    pwd = os.environ.get("GAITA_PWD", "")
    host = os.environ.get("GAITA_HOST", "192.168.188.173")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=login, password=pwd, timeout=15)
    return c

def run(cmd, timeout=300, sudo=False):
    c = get_client()
    pwd = os.environ.get("GAITA_PWD", "")
    if sudo and not cmd.strip().startswith("sudo"):
        cmd = f"echo '{pwd}' | sudo -S bash -c '{cmd}'"
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=sudo)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    if out: print(out, end="")
    if err: print(err, end="", file=sys.stderr)
    return rc

def scp_upload(local_path, remote_path):
    c = get_client()
    sftp = c.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    c.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ssh_helper.py [--sudo] <command>")
        sys.exit(1)
    use_sudo = "--sudo" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--sudo"]
    cmd = " ".join(args)
    rc = run(cmd, sudo=use_sudo)
    sys.exit(rc)
