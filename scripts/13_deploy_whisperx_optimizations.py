#!/usr/bin/env python3
"""Rollback-sicherer Deploy/Restore für WhisperX-Turbo-Optimierungen auf dem DGX Spark."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import stat
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
REMOTE_HOME = "/home/ksai0001_local"
REMOTE_APP_DIR = f"{REMOTE_HOME}/whisperx-spark"
REMOTE_STAGE_DIR = f"{REMOTE_HOME}/voxtral-setup/whisperx_spark"
REMOTE_SCRIPTS_DIR = f"{REMOTE_HOME}/voxtral-setup/scripts"
REMOTE_BACKUP_ROOT = f"{REMOTE_HOME}/whisperx-rollback"
SERVICE_NAME = "whisperx"
DEFAULT_HEALTH_URL = "http://127.0.0.1:7860/"

LOCAL_MODEL_MANAGER = ROOT / "whisperx_spark" / "src" / "model_manager.py"
LOCAL_APP = ROOT / "whisperx_spark" / "app.py"
LOCAL_TRANSCRIBER = ROOT / "whisperx_spark" / "src" / "transcriber.py"
LOCAL_QUANT_SCRIPT = ROOT / "scripts" / "11_quantize_whisper_turbo_de.sh"
LOCAL_CT2_SCRIPT = ROOT / "scripts" / "12_rebuild_ctranslate2_flashattn.sh"

REMOTE_MANAGED_FILES = {
    LOCAL_APP: [
        f"{REMOTE_APP_DIR}/app.py",
        f"{REMOTE_STAGE_DIR}/app.py",
    ],
    LOCAL_MODEL_MANAGER: [
        f"{REMOTE_APP_DIR}/src/model_manager.py",
        f"{REMOTE_STAGE_DIR}/src/model_manager.py",
    ],
    LOCAL_TRANSCRIBER: [
        f"{REMOTE_APP_DIR}/src/transcriber.py",
        f"{REMOTE_STAGE_DIR}/src/transcriber.py",
    ],
    LOCAL_QUANT_SCRIPT: [f"{REMOTE_SCRIPTS_DIR}/{LOCAL_QUANT_SCRIPT.name}"],
    LOCAL_CT2_SCRIPT: [f"{REMOTE_SCRIPTS_DIR}/{LOCAL_CT2_SCRIPT.name}"],
}
ENV_TARGET = f"{REMOTE_APP_DIR}/.env"

OPTIMIZED_ENV = {
    "WHISPERX_MODEL": "/home/ksai0001_local/models/primeline-turbo-de-int8_bf16",
    "WHISPERX_COMPUTE_TYPE": "int8_bfloat16",
    "WHISPERX_DEVICE": "cuda",
    "WHISPERX_POOL_SIZE": "6",
    "WHISPERX_ESTIMATED_WORKER_GB": "2",
    "WHISPERX_BEAM_SIZE": "1",
    "WHISPERX_ALIGNMENT_DEVICE": "cuda",
}


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


def get_client() -> paramiko.SSHClient:
    login = os.environ.get("GAITA_LOGIN", "ksai0001_local").split("@")[0]
    password = os.environ.get("GAITA_PWD", "")
    host = os.environ.get("GAITA_HOST", "192.168.188.173")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=login, password=password, timeout=20)
    return client


def run_remote(client: paramiko.SSHClient, command: str, timeout: int = 300, sudo: bool = False) -> CommandResult:
    if sudo:
        password = os.environ.get("GAITA_PWD", "")
        command = f"echo {shlex.quote(password)} | sudo -S bash -lc {shlex.quote(command)}"
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=sudo)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    return CommandResult(out, err, exit_code)


def ensure_success(result: CommandResult, context: str) -> None:
    if result.exit_code != 0:
        raise RuntimeError(
            f"{context} fehlgeschlagen (rc={result.exit_code})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def remote_exists(client: paramiko.SSHClient, remote_path: str) -> bool:
    sftp = client.open_sftp()
    try:
        sftp.stat(remote_path)
        return True
    except FileNotFoundError:
        return False
    finally:
        sftp.close()


def mkdir_p(client: paramiko.SSHClient, remote_dir: str) -> None:
    result = run_remote(client, f"mkdir -p {shlex.quote(remote_dir)}")
    ensure_success(result, f"mkdir -p {remote_dir}")


def read_remote_text(client: paramiko.SSHClient, remote_path: str) -> str:
    sftp = client.open_sftp()
    try:
        with sftp.open(remote_path, "r") as handle:
            return handle.read().decode("utf-8")
    finally:
        sftp.close()


def write_remote_text(client: paramiko.SSHClient, remote_path: str, content: str, mode: int | None = None) -> None:
    remote_dir = posixpath.dirname(remote_path)
    mkdir_p(client, remote_dir)
    sftp = client.open_sftp()
    try:
        with sftp.open(remote_path, "w") as handle:
            handle.write(content)
        if mode is not None:
            sftp.chmod(remote_path, mode)
    finally:
        sftp.close()


def upload_file(client: paramiko.SSHClient, local_path: Path, remote_path: str) -> None:
    mkdir_p(client, posixpath.dirname(remote_path))
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
        local_mode = local_path.stat().st_mode
        sftp.chmod(remote_path, stat.S_IMODE(local_mode))
    finally:
        sftp.close()


def backup_file(client: paramiko.SSHClient, remote_path: str, backup_root: str) -> str | None:
    if not remote_exists(client, remote_path):
        return None
    safe_name = remote_path.strip("/").replace("/", "__")
    backup_path = f"{backup_root}/files/{safe_name}"
    mkdir_p(client, posixpath.dirname(backup_path))
    result = run_remote(client, f"cp -a {shlex.quote(remote_path)} {shlex.quote(backup_path)}")
    ensure_success(result, f"Backup von {remote_path}")
    return backup_path


def parse_env(text: str) -> tuple[list[str], dict[str, int]]:
    lines = text.splitlines()
    mapping: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key and key not in mapping:
            mapping[key] = index
    return lines, mapping


def merge_env(existing: str, overrides: dict[str, str]) -> str:
    lines, mapping = parse_env(existing)
    if not lines and existing == "":
        lines = []
    for key, value in overrides.items():
        line = f"{key}={value}"
        if key in mapping:
            lines[mapping[key]] = line
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def save_manifest(client: paramiko.SSHClient, backup_root: str, manifest: dict) -> str:
    manifest_path = f"{backup_root}/manifest.json"
    write_remote_text(client, manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest_path


def restart_service(client: paramiko.SSHClient) -> None:
    result = run_remote(client, f"systemctl restart {SERVICE_NAME}", timeout=180, sudo=True)
    ensure_success(result, f"systemctl restart {SERVICE_NAME}")


def stop_service(client: paramiko.SSHClient) -> None:
    result = run_remote(client, f"systemctl stop {SERVICE_NAME}", timeout=180, sudo=True)
    ensure_success(result, f"systemctl stop {SERVICE_NAME}")


def get_service_state(client: paramiko.SSHClient) -> str:
    result = run_remote(client, f"systemctl is-active {SERVICE_NAME} || true")
    return result.stdout.strip() or "unknown"


def healthcheck(client: paramiko.SSHClient, url: str, timeout_seconds: int = 120) -> None:
    escaped_url = shlex.quote(url)
    script = textwrap.dedent(
        f"""
        python3 - <<'PY'
        import sys, time, urllib.request
        deadline = time.time() + {timeout_seconds}
        url = {url!r}
        last_error = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    code = getattr(response, 'status', response.getcode())
                    if 200 <= code < 500:
                        print(f'healthcheck ok: {{code}}')
                        raise SystemExit(0)
            except Exception as exc:
                last_error = exc
                time.sleep(3)
        print(f'healthcheck failed: {{last_error}}', file=sys.stderr)
        raise SystemExit(1)
        PY
        """
    ).strip()
    result = run_remote(client, script, timeout=timeout_seconds + 30)
    ensure_success(result, f"Healthcheck {escaped_url}")


def get_latest_manifest_path(client: paramiko.SSHClient) -> str:
    result = run_remote(
        client,
        f"python3 - <<'PY'\n"
        f"from pathlib import Path\n"
        f"root = Path({REMOTE_BACKUP_ROOT!r})\n"
        f"manifests = sorted(root.glob('whisperx_opt_*/manifest.json'))\n"
        f"print(manifests[-1] if manifests else '')\n"
        f"PY",
    )
    ensure_success(result, "Manifest-Suche")
    manifest_path = result.stdout.strip()
    if not manifest_path:
        raise RuntimeError("Keine WhisperX-Optimierungs-Backups gefunden.")
    return manifest_path


def restore_from_manifest(client: paramiko.SSHClient, manifest_path: str) -> dict:
    manifest = json.loads(read_remote_text(client, manifest_path))
    backups: dict[str, str | None] = manifest.get("backups", {})
    for remote_path, backup_path in backups.items():
        if backup_path:
            mkdir_p(client, posixpath.dirname(remote_path))
            result = run_remote(client, f"cp -a {shlex.quote(backup_path)} {shlex.quote(remote_path)}")
            ensure_success(result, f"Restore von {remote_path}")
    prior_state = manifest.get("previous_service_state", "active")
    if prior_state == "active":
        restart_service(client)
    else:
        stop_service(client)
    return manifest


def deploy(args: argparse.Namespace) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = f"{REMOTE_BACKUP_ROOT}/whisperx_opt_{timestamp}"
    client = get_client()
    try:
        mkdir_p(client, backup_root)
        mkdir_p(client, f"{backup_root}/files")
        previous_service_state = get_service_state(client)
        backups: dict[str, str | None] = {}

        backup_targets = [ENV_TARGET]
        for remote_paths in REMOTE_MANAGED_FILES.values():
            backup_targets.extend(remote_paths)

        for remote_target in backup_targets:
            backups[remote_target] = backup_file(client, remote_target, backup_root)

        for local_path, remote_targets in REMOTE_MANAGED_FILES.items():
            for remote_path in remote_targets:
                upload_file(client, local_path, remote_path)

        env_overrides = None
        if not args.code_only:
            existing_env = read_remote_text(client, ENV_TARGET) if remote_exists(client, ENV_TARGET) else ""
            updated_env = merge_env(existing_env, OPTIMIZED_ENV)
            if args.enable_flash_attention:
                updated_env = merge_env(updated_env, {"WHISPERX_FLASH_ATTENTION": "1"})
            else:
                updated_env = merge_env(updated_env, {"WHISPERX_FLASH_ATTENTION": "0"})
            write_remote_text(client, ENV_TARGET, updated_env)
            env_overrides = dict(
                OPTIMIZED_ENV,
                WHISPERX_FLASH_ATTENTION="1" if args.enable_flash_attention else "0",
            )

        manifest = {
            "timestamp": timestamp,
            "backup_root": backup_root,
            "previous_service_state": previous_service_state,
            "backups": backups,
            "deployed_files": {str(path): targets for path, targets in REMOTE_MANAGED_FILES.items()},
            "env_overrides": env_overrides,
            "code_only": args.code_only,
        }
        manifest_path = save_manifest(client, backup_root, manifest)

        if not args.code_only:
            try:
                restart_service(client)
                healthcheck(client, args.health_url, timeout_seconds=args.health_timeout)
            except Exception:
                restore_from_manifest(client, manifest_path)
                raise

        print(f"Deploy erfolgreich. Backup: {backup_root}")
        print(f"Rollback: python scripts/13_deploy_whisperx_optimizations.py rollback --manifest {manifest_path}")
    finally:
        client.close()


def rollback(args: argparse.Namespace) -> None:
    client = get_client()
    try:
        manifest_path = args.manifest or get_latest_manifest_path(client)
        manifest = restore_from_manifest(client, manifest_path)
        healthcheck(client, args.health_url, timeout_seconds=args.health_timeout)
        print(f"Rollback erfolgreich: {manifest_path}")
        print(f"Wiederhergestellt auf Stand: {manifest.get('timestamp', 'unbekannt')}")
    finally:
        client.close()


def status(args: argparse.Namespace) -> None:
    client = get_client()
    try:
        state = get_service_state(client)
        env_text = read_remote_text(client, ENV_TARGET) if remote_exists(client, ENV_TARGET) else ""
        interesting_keys = [
            "WHISPERX_MODEL",
            "WHISPERX_COMPUTE_TYPE",
            "WHISPERX_POOL_SIZE",
            "WHISPERX_FLASH_ATTENTION",
            "WHISPERX_BEAM_SIZE",
        ]
        lines, mapping = parse_env(env_text)
        print(f"Service: {state}")
        for key in interesting_keys:
            if key in mapping:
                print(lines[mapping[key]])
    finally:
        client.close()


def run_remote_script(args: argparse.Namespace) -> None:
    client = get_client()
    try:
        command = args.command
        if args.env:
            env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in args.env)
            command = f"env {env_prefix} {command}"
        result = run_remote(client, command, timeout=args.timeout, sudo=args.sudo)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        ensure_success(result, f"Remote-Kommando {command}")
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Optimierten WhisperX-Stand deployen")
    deploy_parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    deploy_parser.add_argument("--health-timeout", type=int, default=120)
    deploy_parser.add_argument("--enable-flash-attention", action="store_true")
    deploy_parser.add_argument("--code-only", action="store_true", help="Nur Dateien hochladen, .env und Service unverändert lassen")
    deploy_parser.set_defaults(func=deploy)

    rollback_parser = subparsers.add_parser("rollback", help="Letztes Optimierungs-Backup wiederherstellen")
    rollback_parser.add_argument("--manifest", help="Expliziter Remote-Pfad zu manifest.json")
    rollback_parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    rollback_parser.add_argument("--health-timeout", type=int, default=120)
    rollback_parser.set_defaults(func=rollback)

    status_parser = subparsers.add_parser("status", help="Aktive WhisperX-Optimierungs-Parameter anzeigen")
    status_parser.set_defaults(func=status)

    quantize_parser = subparsers.add_parser("run-quantize", help="Turbo-DE-Quantisierung auf dem Spark ausführen")
    quantize_parser.add_argument("--timeout", type=int, default=3600)
    quantize_parser.add_argument("--force", action="store_true")
    quantize_parser.set_defaults(
        func=run_remote_script,
        command=f"bash {shlex.quote(REMOTE_SCRIPTS_DIR + '/' + LOCAL_QUANT_SCRIPT.name)}",
        sudo=False,
    )

    rebuild_parser = subparsers.add_parser("run-ct2-rebuild", help="CTranslate2 mit Flash-Attention auf dem Spark neu bauen")
    rebuild_parser.add_argument("--timeout", type=int, default=14400)
    rebuild_parser.add_argument("--jobs")
    rebuild_parser.add_argument("--arch", default="121")
    rebuild_parser.set_defaults(
        func=run_remote_script,
        command=f"bash {shlex.quote(REMOTE_SCRIPTS_DIR + '/' + LOCAL_CT2_SCRIPT.name)}",
        sudo=True,
    )

    exec_parser = subparsers.add_parser("exec", help="Beliebiges Remote-Kommando auf dem Spark ausführen")
    exec_parser.add_argument("command")
    exec_parser.add_argument("--timeout", type=int, default=600)
    exec_parser.add_argument("--sudo", action="store_true")
    exec_parser.set_defaults(func=run_remote_script)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "func", None) is run_remote_script:
        env_vars: list[tuple[str, str]] = list(getattr(args, "env", []))
        if getattr(args, "force", False):
            env_vars.append(("FORCE", "1"))
        if getattr(args, "jobs", None):
            env_vars.append(("CT2_BUILD_JOBS", args.jobs))
        if getattr(args, "arch", None):
            env_vars.append(("CT2_CUDA_ARCH", args.arch))
        args.env = env_vars
    try:
        args.func(args)
        return 0
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())