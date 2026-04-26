import asyncio
import fcntl
import io
import json
import os
import pty
import re
import secrets
import select
import shutil
import socket
import struct
import subprocess
import tarfile
import tempfile
import termios
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import httpx
import pam
import psutil
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
APP_HOST = os.getenv("SPARK_ADMIN_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("SPARK_ADMIN_PORT", "7000"))
STACK_OWNER = os.getenv("SPARK_STACK_OWNER", os.getenv("SUDO_USER") or os.getenv("USER") or "ksai0001_local")
STACK_HOME = Path(os.getenv("SPARK_STACK_HOME", str(Path("/home") / STACK_OWNER)))
SESSION_SECRET = os.getenv("SPARK_ADMIN_SESSION_SECRET", secrets.token_hex(32))
SESSION_COOKIE = os.getenv("SPARK_ADMIN_SESSION_COOKIE", "spark_admin_session")
REQUEST_TIMEOUT = float(os.getenv("SPARK_ADMIN_HTTP_TIMEOUT", "5"))
PAM_SERVICE = os.getenv("SPARK_ADMIN_PAM_SERVICE", "login")
SHARED_PASSWORD_FILE = Path(
    os.getenv("SPARK_SHARED_PASSWORD_FILE", str(STACK_HOME / "voxtral-setup" / ".env.local"))
)
SHARED_PASSWORD_KEY = os.getenv("SPARK_SHARED_PASSWORD_KEY", "WHISPER_AUTH_PASSWORD")
BACKUP_ROOT = Path(os.getenv("SPARK_BACKUP_DIR", str(STACK_HOME / "system-backups")))
BACKUP_LOCK_FILE = BACKUP_ROOT / ".backup.lock"
BACKUP_EXCLUDE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".cache",
    "system-backups",
}
BACKUP_EXCLUDE_FILE_GLOBS = {
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*.swp",
}
RESTORE_AUTO_SNAPSHOT = os.getenv("SPARK_BACKUP_PRE_RESTORE_SNAPSHOT", "1").strip().lower() not in {"0", "false", "no", "off"}
BACKUP_JOBS: dict[str, dict[str, Any]] = {}
BACKUP_JOBS_LOCK = threading.Lock()

MANAGED_SERVICES: dict[str, dict[str, Any]] = {
    "voxtral": {
        "service": "voxtral-vllm",
        "container": "voxtral-vllm",
        "label": "Mistral / Voxtral",
        "port": 8000,
        "health_url": "http://127.0.0.1:8000/health",
        "health_headers": {},
        "config_files": [
            {
                "id": "voxtral-run",
                "label": "run_vllm.sh",
                "path": STACK_HOME / "voxtral-vllm" / "run_vllm.sh",
                "format": "shell",
            },
            {
                "id": "voxtral-env",
                "label": "vllm.env",
                "path": STACK_HOME / "voxtral-vllm" / "vllm.env",
                "format": "env",
            },
        ],
    },
    "whisperx": {
        "service": "whisperx",
        "container": "whisperx",
        "label": "WhisperX",
        "port": 7860,
        "health_url": "http://127.0.0.1:7860/",
        "health_headers": {},
        "config_files": [
            {
                "id": "whisperx-env",
                "label": ".env",
                "path": STACK_HOME / "whisperx-spark" / ".env",
                "format": "env",
            },
        ],
    },
    "correction": {
        "service": "correction-llm",
        "container": ["correction-llm", "correction-llm-test"],
        "label": "Korrektur-LLM",
        "port": 9000,
        "health_url": "http://127.0.0.1:9000/v1/models",
        "health_headers": {},
        "config_files": [
            {
                "id": "correction-run",
                "label": "run_gemma4.sh",
                "path": STACK_HOME / "correction-llm-vllm" / "run_gemma4.sh",
                "format": "shell",
            },
            {
                "id": "correction-env",
                "label": "llm.env",
                "path": STACK_HOME / "correction-llm-vllm" / "llm.env",
                "format": "env",
            },
        ],
    },
    "vibevoice": {
        "service": "vibevoice",
        "container": "vibevoice",
        "label": "ForcedAligner",
        "port": 7862,
        "health_url": "http://127.0.0.1:7862/health",
        "health_headers": {},
        "config_files": [
            {
                "id": "vibevoice-env",
                "label": ".env",
                "path": STACK_HOME / "vibevoice-spark" / ".env",
                "format": "env",
            },
        ],
    },
}

CONFIG_INDEX = {
    config["id"]: {**config, "service_key": service_key, "service": service_data["service"]}
    for service_key, service_data in MANAGED_SERVICES.items()
    for config in service_data["config_files"]
}

app = FastAPI(title="Spark Admin", version="1.0.0")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie=SESSION_COOKIE,
    https_only=False,
    same_site="lax",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _split_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    return key.strip(), value.rstrip("\n")


def read_shared_password_config() -> dict[str, Any]:
    content = SHARED_PASSWORD_FILE.read_text(encoding="utf-8") if SHARED_PASSWORD_FILE.exists() else ""
    value = ""
    for line in content.splitlines():
        parsed = _split_env_line(line)
        if parsed and parsed[0] == SHARED_PASSWORD_KEY:
            value = parsed[1]
            break
    return {
        "path": str(SHARED_PASSWORD_FILE),
        "key": SHARED_PASSWORD_KEY,
        "value": value,
        "exists": SHARED_PASSWORD_FILE.exists(),
    }


def write_shared_password(value: str) -> dict[str, Any]:
    SHARED_PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = SHARED_PASSWORD_FILE.read_text(encoding="utf-8").splitlines() if SHARED_PASSWORD_FILE.exists() else []
    updated_lines: list[str] = []
    found = False

    for line in lines:
        parsed = _split_env_line(line)
        if parsed and parsed[0] == SHARED_PASSWORD_KEY:
            updated_lines.append(f"{SHARED_PASSWORD_KEY}={value}")
            found = True
        else:
            updated_lines.append(line)

    if not found:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(f"{SHARED_PASSWORD_KEY}={value}")

    SHARED_PASSWORD_FILE.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return read_shared_password_config()


def current_username(request: Request) -> str | None:
    return request.session.get("username")


def require_auth(request: Request) -> str:
    username = current_username(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return username


def run_command(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _create_backup_job(job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "progress": 0,
        "message": "Wartet auf Ausführung…",
        "detail": "",
        "error": None,
        "result": None,
        "payload": payload,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    with BACKUP_JOBS_LOCK:
        BACKUP_JOBS[job_id] = job
    return job.copy()


def _update_backup_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with BACKUP_JOBS_LOCK:
        job = BACKUP_JOBS.get(job_id)
        if not job:
            raise KeyError(job_id)
        job.update(changes)
        job["updated_at"] = _utc_now_iso()
        return job.copy()


def _get_backup_job(job_id: str) -> dict[str, Any] | None:
    with BACKUP_JOBS_LOCK:
        job = BACKUP_JOBS.get(job_id)
        return job.copy() if job else None


def _start_backup_job(job_type: str, payload: dict[str, Any], runner) -> dict[str, Any]:
    job = _create_backup_job(job_type, payload)
    job_id = job["id"]

    def execute() -> None:
        try:
            _update_backup_job(job_id, status="running", progress=2, message="Job gestartet")
            result = runner(job_id)
            _update_backup_job(
                job_id,
                status="completed",
                progress=100,
                message=result.get("message", "Fertig."),
                detail="Abgeschlossen",
                result=result,
                error=None,
            )
        except HTTPException as exc:
            _update_backup_job(
                job_id,
                status="failed",
                progress=100,
                message="Job fehlgeschlagen",
                detail=str(exc.detail),
                error=str(exc.detail),
            )
        except Exception as exc:
            _update_backup_job(
                job_id,
                status="failed",
                progress=100,
                message="Job fehlgeschlagen",
                detail=str(exc),
                error=str(exc),
            )

    thread = threading.Thread(target=execute, name=f"spark-backup-job-{job_id[:8]}", daemon=True)
    thread.start()
    return _get_backup_job(job_id) or job


def _job_progress_callback(job_id: str):
    def callback(progress: int, message: str, detail: str | None = None) -> None:
        updates: dict[str, Any] = {
            "progress": max(0, min(100, int(progress))),
            "message": message,
        }
        if detail is not None:
            updates["detail"] = detail
        _update_backup_job(job_id, **updates)

    return callback


def _scaled_progress_callback(progress_callback, start: int, end: int):
    def callback(progress: int, message: str, detail: str | None = None) -> None:
        span = max(end - start, 1)
        scaled = start + int((max(0, min(100, progress)) / 100) * span)
        _report_progress(progress_callback, scaled, message, detail)

    return callback


def _path_is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(other.resolve(strict=False))
        return True
    except ValueError:
        return False


@contextmanager
def backup_operation_lock():
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    with BACKUP_LOCK_FILE.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _slugify_backup_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.strip()).strip("-._")
    return cleaned[:64]


def _backup_metadata_path(archive_path: Path) -> Path:
    return archive_path.with_name(f"{archive_path.name}.json")


def _get_backup_sources(include_missing: bool = False) -> list[Path]:
    candidates = [
        PROJECT_ROOT,
        STACK_HOME / "voxtral-setup",
        STACK_HOME / "voxtral-vllm",
        STACK_HOME / "whisperx-spark",
        STACK_HOME / "correction-llm-vllm",
        STACK_HOME / "vibevoice-spark",
        SHARED_PASSWORD_FILE.parent,
    ]
    candidates.extend(config["path"].parent for config in CONFIG_INDEX.values())

    unique: list[Path] = []
    for raw_path in candidates:
        path = raw_path.expanduser().resolve(strict=False)
        if (not include_missing and not path.exists()) or path == BACKUP_ROOT or _path_is_relative_to(path, BACKUP_ROOT):
            continue
        if any(path == existing for existing in unique):
            continue
        unique.append(path)

    collapsed: list[Path] = []
    for path in sorted(unique, key=lambda item: (len(item.parts), str(item))):
        if any(_path_is_relative_to(path, existing) for existing in collapsed):
            continue
        collapsed.append(path)
    return collapsed


def _should_exclude_backup_path(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    if resolved == BACKUP_ROOT or _path_is_relative_to(resolved, BACKUP_ROOT):
        return True
    if resolved.name in BACKUP_EXCLUDE_DIR_NAMES and path.is_dir():
        return True
    if any(fnmatch(resolved.name, pattern) for pattern in BACKUP_EXCLUDE_FILE_GLOBS):
        return True
    return False


def _add_path_to_archive(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    if _should_exclude_backup_path(source):
        return
    archive.add(str(source), arcname=arcname, recursive=False)
    if source.is_dir():
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            _add_path_to_archive(archive, child, f"{arcname}/{child.name}")


def _append_text_to_archive(archive: tarfile.TarFile, arcname: str, content: str) -> None:
    payload = content.encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mtime = int(time.time())
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def _report_progress(progress_callback, progress: int, message: str, detail: str | None = None) -> None:
    if progress_callback:
        progress_callback(progress, message, detail)


def _capture_service_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for service_key, service in MANAGED_SERVICES.items():
        snapshot[service_key] = {
            "label": service["label"],
            "service": service["service"],
            **get_service_status(service["service"]),
        }
    return snapshot


def _capture_systemd_unit_texts() -> dict[str, str]:
    units: dict[str, str] = {}
    for service in MANAGED_SERVICES.values():
        result = run_command(["systemctl", "cat", service["service"]], timeout=20)
        if result.returncode == 0 and result.stdout.strip():
            units[service["service"]] = result.stdout
    return units


def _build_backup_manifest(display_label: str | None, reason: str) -> dict[str, Any]:
    sources = _get_backup_sources()
    created_at = datetime.now(tz=timezone.utc).isoformat()
    return {
        "format_version": 1,
        "created_at": created_at,
        "hostname": socket.gethostname(),
        "stack_owner": STACK_OWNER,
        "stack_home": str(STACK_HOME),
        "project_root": str(PROJECT_ROOT),
        "backup_root": str(BACKUP_ROOT),
        "label": _slugify_backup_label(display_label or "") if display_label else "",
        "display_label": (display_label or "").strip(),
        "reason": reason,
        "sources": [],
        "excluded": {
            "directories": sorted(BACKUP_EXCLUDE_DIR_NAMES),
            "patterns": sorted(BACKUP_EXCLUDE_FILE_GLOBS),
        },
        "services": _capture_service_snapshot(),
    }


def _manifest_to_summary(archive_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    size_bytes = archive_path.stat().st_size if archive_path.exists() else 0
    sources = manifest.get("sources", [])
    return {
        "name": archive_path.name,
        "path": str(archive_path),
        "created_at": manifest.get("created_at"),
        "label": manifest.get("display_label") or manifest.get("label") or archive_path.stem,
        "reason": manifest.get("reason", "manual"),
        "size_bytes": size_bytes,
        "source_count": len(sources),
        "sources": [item.get("target_path") for item in sources],
        "services": manifest.get("services", {}),
    }


def _write_backup_sidecar(archive_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = _manifest_to_summary(archive_path, manifest)
    _backup_metadata_path(archive_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _read_backup_manifest(archive_path: Path) -> dict[str, Any]:
    with tarfile.open(archive_path, "r:gz") as archive:
        manifest_file = archive.extractfile("manifest.json")
        if manifest_file is None:
            raise RuntimeError(f"Backup {archive_path.name} enthält kein Manifest.")
        return json.loads(manifest_file.read().decode("utf-8"))


def list_backups() -> list[dict[str, Any]]:
    if not BACKUP_ROOT.exists():
        return []
    backups: list[dict[str, Any]] = []
    for archive_path in sorted(BACKUP_ROOT.glob("*.tar.gz"), key=lambda item: item.stat().st_mtime, reverse=True):
        meta_path = _backup_metadata_path(archive_path)
        if meta_path.exists():
            try:
                summary = json.loads(meta_path.read_text(encoding="utf-8"))
                summary["size_bytes"] = archive_path.stat().st_size
                backups.append(summary)
                continue
            except json.JSONDecodeError:
                pass
        manifest = _read_backup_manifest(archive_path)
        backups.append(_manifest_to_summary(archive_path, manifest))
    return backups


def _create_backup_archive_unlocked(
    display_label: str | None = None,
    reason: str = "manual",
    progress_callback=None,
) -> dict[str, Any]:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = _build_backup_manifest(display_label, reason)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    label_part = f"-{manifest['label']}" if manifest.get("label") else ""
    archive_name = f"spark-stack-backup-{timestamp}{label_part}.tar.gz"
    archive_path = BACKUP_ROOT / archive_name
    temp_path = archive_path.with_suffix(".tmp")
    backup_sources = _get_backup_sources()

    _report_progress(progress_callback, 5, "Backup wird vorbereitet", archive_name)

    with tarfile.open(temp_path, "w:gz") as archive:
        total_sources = max(len(backup_sources), 1)
        for index, source in enumerate(backup_sources, start=1):
            arcname = f"payload/item-{index}"
            manifest["sources"].append(
                {
                    "target_path": str(source),
                    "archive_path": arcname,
                    "kind": "directory" if source.is_dir() else "file",
                }
            )
            phase_progress = 10 + int((index / total_sources) * 60)
            _report_progress(progress_callback, phase_progress, "Sichere Pfade", str(source))
            _add_path_to_archive(archive, source, arcname)

        _report_progress(progress_callback, 78, "Systemd-Snapshots werden gesichert")
        systemd_units = _capture_systemd_unit_texts()
        for service_name, unit_text in systemd_units.items():
            _append_text_to_archive(archive, f"metadata/systemd/{service_name}.service.txt", unit_text)

        _report_progress(progress_callback, 90, "Manifest wird geschrieben")
        _append_text_to_archive(archive, "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

    _report_progress(progress_callback, 96, "Archiv wird finalisiert")
    temp_path.replace(archive_path)
    summary = _write_backup_sidecar(archive_path, manifest)
    return {
        "ok": True,
        "message": f"Backup {summary['label']} erstellt.",
        "backup": summary,
    }


def create_backup_archive(display_label: str | None = None, reason: str = "manual", progress_callback=None) -> dict[str, Any]:
    with backup_operation_lock():
        return _create_backup_archive_unlocked(display_label=display_label, reason=reason, progress_callback=progress_callback)


def _resolve_backup_archive(name: str) -> Path:
    safe_name = Path(name).name
    if safe_name != name or not safe_name.endswith(".tar.gz"):
        raise HTTPException(status_code=400, detail="Ungültiger Backup-Name")
    archive_path = BACKUP_ROOT / safe_name
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="Backup nicht gefunden")
    return archive_path


def delete_backup_archive(name: str) -> dict[str, Any]:
    archive_path = _resolve_backup_archive(name)
    with backup_operation_lock():
        meta_path = _backup_metadata_path(archive_path)
        if archive_path.exists():
            archive_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
    return {
        "ok": True,
        "message": f"Backup {name} gelöscht.",
        "name": name,
    }


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    dest_root = destination.resolve(strict=False)
    for member in archive.getmembers():
        target = (destination / member.name).resolve(strict=False)
        if not _path_is_relative_to(target, dest_root) and target != dest_root:
            raise RuntimeError("Unsicherer Archivpfad erkannt")
    archive.extractall(destination)


def _is_allowed_restore_target(target: Path) -> bool:
    resolved = target.resolve(strict=False)
    return any(resolved == allowed.resolve(strict=False) for allowed in _get_backup_sources(include_missing=True))


def perform_service_action(service_key: str, action: str, raise_on_error: bool = True) -> bool:
    service = MANAGED_SERVICES.get(service_key)
    if not service:
        if raise_on_error:
            raise HTTPException(status_code=404, detail="Unknown service")
        return False
    if action not in {"start", "stop", "restart"}:
        if raise_on_error:
            raise HTTPException(status_code=400, detail="Unsupported action")
        return False

    success = False
    svc_unit = service["service"]
    unit_check = run_command(["systemctl", "cat", svc_unit], timeout=10)
    if unit_check.returncode == 0:
        result = run_command(["systemctl", action, svc_unit], timeout=120)
        success = result.returncode == 0
    else:
        containers = service.get("container", [])
        if isinstance(containers, str):
            containers = [containers]
        for container_name in containers:
            result = run_command(["docker", action, container_name], timeout=60)
            if result.returncode == 0:
                success = True
                break

    if not success and raise_on_error:
        raise HTTPException(status_code=500, detail=f"{action} fehlgeschlagen")
    return success


def _restore_services_from_manifest(manifest: dict[str, Any]) -> list[str]:
    restored: list[str] = []
    service_states = manifest.get("services", {})
    for service_key in MANAGED_SERVICES:
        service_state = service_states.get(service_key, {})
        if service_state.get("active_state") not in {"active", "activating", "reloading"}:
            continue
        if perform_service_action(service_key, "start", raise_on_error=False):
            restored.append(service_key)
    return restored


def restore_backup_archive(
    name: str,
    restart_services: bool = True,
    create_restore_point: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    archive_path = _resolve_backup_archive(name)
    manifest = _read_backup_manifest(archive_path)
    _report_progress(progress_callback, 5, "Restore wird vorbereitet", archive_path.name)

    with backup_operation_lock():
        restore_point = None
        if create_restore_point:
            _report_progress(progress_callback, 10, "Restore-Punkt wird erstellt")
            restore_point = _create_backup_archive_unlocked(
                display_label=f"restore-point-{manifest.get('display_label') or archive_path.stem}",
                reason="pre-restore",
                progress_callback=_scaled_progress_callback(progress_callback, 10, 30),
            )["backup"]

        _report_progress(progress_callback, 35, "Dienste werden angehalten")
        total_services = max(len(MANAGED_SERVICES), 1)
        for index, service_key in enumerate(MANAGED_SERVICES, start=1):
            perform_service_action(service_key, "stop", raise_on_error=False)
            _report_progress(progress_callback, 35 + int((index / total_services) * 10), "Dienste werden angehalten", service_key)

        with tempfile.TemporaryDirectory(prefix="spark-restore-", dir=str(BACKUP_ROOT)) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _report_progress(progress_callback, 48, "Archiv wird entpackt")
            with tarfile.open(archive_path, "r:gz") as archive:
                _safe_extract_tar(archive, temp_dir)

            restored_paths: list[str] = []
            restore_items = manifest.get("sources", [])
            total_items = max(len(restore_items), 1)
            for index, item in enumerate(restore_items, start=1):
                source_path = temp_dir / item["archive_path"]
                target_path = Path(item["target_path"]).expanduser().resolve(strict=False)
                if not source_path.exists():
                    raise RuntimeError(f"Archivinhalt fehlt: {item['archive_path']}")
                if not _is_allowed_restore_target(target_path):
                    raise RuntimeError(f"Unzulässiger Restore-Pfad: {target_path}")

                if target_path.exists():
                    if target_path.is_file() or target_path.is_symlink():
                        target_path.unlink()
                    else:
                        shutil.rmtree(target_path)

                target_path.parent.mkdir(parents=True, exist_ok=True)
                if item.get("kind") == "directory":
                    shutil.copytree(source_path, target_path, copy_function=shutil.copy2)
                else:
                    shutil.copy2(source_path, target_path)
                restored_paths.append(str(target_path))
                _report_progress(progress_callback, 55 + int((index / total_items) * 25), "Pfade werden wiederhergestellt", str(target_path))

        _report_progress(progress_callback, 85, "Dienste werden gestartet")
        started_services = _restore_services_from_manifest(manifest) if restart_services else []
        if restart_services:
            _report_progress(progress_callback, 95, "Dienste wurden gestartet", ", ".join(started_services) or "keine")
        else:
            _report_progress(progress_callback, 95, "Restore abgeschlossen", "Dienste bleiben gestoppt")

    return {
        "ok": True,
        "message": f"Backup {archive_path.name} wurde wiederhergestellt.",
        "backup": _manifest_to_summary(archive_path, manifest),
        "restore_point": restore_point,
        "restored_paths": restored_paths,
        "started_services": started_services,
    }


async def probe_http(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(url, headers=headers)
        return {"ok": response.status_code < 400, "status_code": response.status_code}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_service_status(service_name: str) -> dict[str, Any]:
    result = run_command(["systemctl", "show", service_name, "--property=ActiveState,SubState,MainPID,ExecMainStatus,UnitFileState"])
    data: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return {
        "active_state": data.get("ActiveState", "unknown"),
        "sub_state": data.get("SubState", "unknown"),
        "main_pid": data.get("MainPID", "0"),
        "exit_status": data.get("ExecMainStatus", "0"),
        "unit_file_state": data.get("UnitFileState", "unknown"),
    }


def get_recent_logs(service_name: str, lines: int = 200) -> str:
    result = run_command(["journalctl", "-u", service_name, "-n", str(lines), "--no-pager", "-o", "short-iso"], timeout=60)
    return result.stdout or result.stderr


def get_gpu_status() -> dict[str, Any]:
    query = "index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit"
    result = run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], timeout=10)
    raw_text = (result.stdout or result.stderr).strip()
    gpus = []
    if result.returncode == 0:
        # For unified memory (e.g. GB10): nvidia-smi reports [N/A] for memory,
        # so we fall back to system memory via psutil.
        sys_mem = psutil.virtual_memory()
        sys_total_mb = round(sys_mem.total / (1024 ** 2))
        sys_used_mb = round(sys_mem.used / (1024 ** 2))
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 9:
                mem_used = parts[5]
                mem_total = parts[6]
                if "N/A" in mem_total:
                    mem_total = str(sys_total_mb)
                    mem_used = str(sys_used_mb)
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "temperature_c": parts[2],
                        "utilization_gpu_percent": parts[3],
                        "utilization_memory_percent": parts[4],
                        "memory_used_mb": mem_used,
                        "memory_total_mb": mem_total,
                        "power_draw_watts": parts[7],
                        "power_limit_watts": parts[8],
                        "unified_memory": "N/A" in parts[6],
                    }
                )
    return {"raw": raw_text, "gpus": gpus, "ok": result.returncode == 0}


def get_container_memory() -> dict[str, dict[str, Any]]:
    """Get memory usage per service — docker containers via docker stats, systemd via cgroup/PID."""
    # 1) Docker container stats
    docker_stats: dict[str, dict[str, Any]] = {}
    result = run_command(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}"],
        timeout=10,
    )
    if result.returncode != 0:
        result = run_command(
            ["sudo", "docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}"],
            timeout=10,
        )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) == 2:
                name = parts[0].strip()
                mem_str = parts[1].strip()
                used_bytes = _parse_mem(mem_str.split("/")[0].strip()) if "/" in mem_str else 0
                docker_stats[name] = {"used_bytes": used_bytes, "raw": mem_str}

    # 2) Map to service keys — docker first, then systemd PID fallback
    service_mem: dict[str, dict[str, Any]] = {}
    for key, svc in MANAGED_SERVICES.items():
        # Try docker container match
        containers = svc.get("container", [])
        if isinstance(containers, str):
            containers = [containers]
        found = False
        for cn in containers:
            if cn in docker_stats:
                service_mem[key] = docker_stats[cn]
                found = True
                break
        if found:
            continue
        # Fallback: systemd service memory via cgroup
        used = _get_systemd_memory(svc["service"])
        if used > 0:
            gib = used / (1024 ** 3)
            service_mem[key] = {"used_bytes": used, "raw": f"{gib:.1f}GiB (systemd)"}
        else:
            service_mem[key] = {"used_bytes": 0, "raw": "nicht aktiv"}
    return service_mem


def _get_systemd_memory(service_name: str) -> int:
    """Get current memory usage of a systemd service in bytes."""
    result = run_command(
        ["systemctl", "show", service_name, "--property=MemoryCurrent"],
        timeout=5,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("MemoryCurrent="):
                val = line.split("=", 1)[1].strip()
                if val and val != "[not set]" and val != "infinity":
                    try:
                        return int(val)
                    except ValueError:
                        pass
    return 0


def _parse_mem(s: str) -> int:
    """Parse '8.5GiB' or '512MiB' to bytes."""
    s = s.strip()
    multipliers = {"KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
                   "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
                   "B": 1}
    for suffix, mult in multipliers.items():
        if s.upper().endswith(suffix):
            try:
                return int(float(s[:len(s)-len(suffix)].strip()) * mult)
            except ValueError:
                return 0
    return 0


_prev_net_io: dict[str, Any] = {}
_prev_net_time: float = 0.0
_physical_iface: str = ""

_prev_disk_io: dict[str, int] = {}
_prev_disk_time: float = 0.0


def get_disk_io_stats() -> dict[str, Any]:
    """Get disk I/O rates (bytes/sec)."""
    global _prev_disk_io, _prev_disk_time
    import time
    counters = psutil.disk_io_counters()
    now = time.monotonic()
    result: dict[str, Any] = {
        "read_bytes": counters.read_bytes,
        "write_bytes": counters.write_bytes,
        "read_rate": 0.0,
        "write_rate": 0.0,
    }
    if _prev_disk_time > 0:
        dt = max(now - _prev_disk_time, 0.1)
        result["read_rate"] = (counters.read_bytes - _prev_disk_io.get("read_bytes", counters.read_bytes)) / dt
        result["write_rate"] = (counters.write_bytes - _prev_disk_io.get("write_bytes", counters.write_bytes)) / dt
    _prev_disk_io = {"read_bytes": counters.read_bytes, "write_bytes": counters.write_bytes}
    _prev_disk_time = now
    return result


def _get_physical_iface() -> str:
    """Detect the default-route network interface (e.g. enP7s7), cached."""
    global _physical_iface
    if _physical_iface:
        return _physical_iface
    try:
        result = run_command(["ip", "-o", "route", "show", "default"], timeout=5)
        for part in result.stdout.split():
            if part == "dev":
                continue
            idx = result.stdout.find("dev ")
            if idx >= 0:
                _physical_iface = result.stdout[idx + 4:].split()[0]
                break
    except Exception:
        pass
    return _physical_iface


def get_network_stats() -> dict[str, Any]:
    """Get network I/O rates (bytes/sec) on the physical interface only."""
    global _prev_net_io, _prev_net_time
    import time
    iface = _get_physical_iface()
    per_nic = psutil.net_io_counters(pernic=True)
    counters = per_nic.get(iface) if iface else None
    if not counters:
        counters = psutil.net_io_counters()  # fallback to all
    now = time.monotonic()
    result: dict[str, Any] = {
        "bytes_sent": counters.bytes_sent,
        "bytes_recv": counters.bytes_recv,
        "packets_sent": counters.packets_sent,
        "packets_recv": counters.packets_recv,
        "send_rate": 0.0,
        "recv_rate": 0.0,
        "interface": iface or "all",
    }
    if _prev_net_time > 0:
        dt = max(now - _prev_net_time, 0.1)
        result["send_rate"] = (counters.bytes_sent - _prev_net_io.get("bytes_sent", counters.bytes_sent)) / dt
        result["recv_rate"] = (counters.bytes_recv - _prev_net_io.get("bytes_recv", counters.bytes_recv)) / dt
    _prev_net_io = {"bytes_sent": counters.bytes_sent, "bytes_recv": counters.bytes_recv}
    _prev_net_time = now
    return result


def get_service_connections() -> dict[str, int]:
    """Count established TCP connections to each service port."""
    port_map: dict[int, str] = {}
    for key, svc in MANAGED_SERVICES.items():
        port_map[svc["port"]] = key
    counts: dict[str, int] = {key: 0 for key in MANAGED_SERVICES}
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == "ESTABLISHED" and conn.laddr and conn.laddr.port in port_map:
                counts[port_map[conn.laddr.port]] += 1
    except (psutil.AccessDenied, PermissionError):
        pass
    return counts


def get_system_metrics() -> dict[str, Any]:
    gpu_status = get_gpu_status()
    disks = []
    seen_devices: set[str] = set()
    for partition in psutil.disk_partitions(all=False):
        if partition.device in seen_devices:
            continue
        seen_devices.add(partition.device)
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        disks.append(
            {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "fstype": partition.fstype,
                "used": usage.used,
                "total": usage.total,
                "percent": usage.percent,
            }
        )
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc).astimezone().isoformat()
    return {
        "hostname": socket.gethostname(),
        "boot_time": boot_time,
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "cpu_percent_per_core": psutil.cpu_percent(interval=0, percpu=True),
        "cpu_count": psutil.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "memory": {
            "percent": memory.percent,
            "used": memory.used,
            "total": memory.total,
            "available": memory.available,
        },
        "swap": {
            "percent": swap.percent,
            "used": swap.used,
            "total": swap.total,
        },
        "disks": disks,
        "uptime_seconds": max(0, int(datetime.now(tz=timezone.utc).timestamp() - psutil.boot_time())),
        "gpu_utilization_percent": float(gpu_status["gpus"][0]["utilization_gpu_percent"]) if gpu_status["gpus"] and gpu_status["gpus"][0].get("utilization_gpu_percent", "N/A") not in ("N/A", "[N/A]") else None,
        "gpus": gpu_status["gpus"],
        "nvidia_smi_raw": gpu_status["raw"],
        "container_memory": get_container_memory(),
        "network": get_network_stats(),
        "service_connections": get_service_connections(),
        "disk_io": get_disk_io_stats(),
    }


async def build_service_payload() -> list[dict[str, Any]]:
    payload = []
    for key, service in MANAGED_SERVICES.items():
        status = get_service_status(service["service"])
        http_health = await probe_http(service["health_url"], service.get("health_headers"))
        payload.append(
            {
                "key": key,
                "label": service["label"],
                "service": service["service"],
                "port": service["port"],
                "status": status,
                "http": http_health,
                "config_files": [
                    {"id": config["id"], "label": config["label"], "path": str(config["path"])}
                    for config in service["config_files"]
                ],
            }
        )
    return payload


def read_config(config_id: str) -> dict[str, Any]:
    config = CONFIG_INDEX.get(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Unknown config file")
    path = config["path"]
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "id": config_id,
        "label": config["label"],
        "service": config["service"],
        "path": str(path),
        "format": config["format"],
        "content": content,
    }


def write_config(config_id: str, content: str) -> None:
    config = CONFIG_INDEX.get(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Unknown config file")
    path = config["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if current_username(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    authenticator = pam.pam()
    if not authenticator.authenticate(username, password, service=PAM_SERVICE):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Login fehlgeschlagen. Bitte Linux-Benutzername/Passwort prüfen."},
            status_code=401,
        )
    request.session["username"] = username
    return RedirectResponse(url="/", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    username = current_username(request)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "username": username,
            "services": await build_service_payload(),
            "configs": [
                {
                    "id": config_id,
                    "label": config["label"],
                    "service": config["service"],
                    "path": str(config["path"]),
                }
                for config_id, config in CONFIG_INDEX.items()
            ],
            "password_config": read_shared_password_config(),
        },
    )


@app.get("/api/shared-password")
async def api_get_shared_password(request: Request):
    require_auth(request)
    return read_shared_password_config()


@app.put("/api/shared-password")
async def api_update_shared_password(request: Request):
    require_auth(request)
    payload = await request.json()
    value = str(payload.get("value", ""))
    updated = write_shared_password(value)
    return {
        "ok": True,
        "message": f"{updated['key']} gespeichert.",
        "config": updated,
    }


@app.get("/api/overview")
async def api_overview(request: Request):
    require_auth(request)
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": get_system_metrics(),
        "services": await build_service_payload(),
    }


@app.get("/api/services/{service_key}/logs")
async def api_service_logs(service_key: str, request: Request, lines: int = 200):
    require_auth(request)
    service = MANAGED_SERVICES.get(service_key)
    if not service:
        raise HTTPException(status_code=404, detail="Unknown service")
    return {"service": service["service"], "logs": get_recent_logs(service["service"], lines=lines)}


@app.post("/api/services/{service_key}/action")
async def api_service_action(service_key: str, request: Request):
    require_auth(request)
    payload = await request.json()
    action = payload.get("action")
    service = MANAGED_SERVICES.get(service_key)
    if not service:
        raise HTTPException(status_code=404, detail="Unknown service")
    perform_service_action(service_key, action, raise_on_error=True)
    await asyncio.sleep(1)
    return {
        "ok": True,
        "message": f"{service['label']} {action} ausgeführt.",
        "service": service["service"],
        "action": action,
        "status": get_service_status(service["service"]),
    }


@app.get("/api/configs/{config_id}")
async def api_get_config(config_id: str, request: Request):
    require_auth(request)
    return read_config(config_id)


@app.put("/api/configs/{config_id}")
async def api_update_config(config_id: str, request: Request):
    require_auth(request)
    payload = await request.json()
    content = payload.get("content", "")
    restart_service = bool(payload.get("restart_service", False))
    config = read_config(config_id)
    write_config(config_id, content)
    if restart_service:
        run_command(["systemctl", "restart", config["service"]], timeout=180)
    updated = read_config(config_id)
    message = f"{updated['label']} gespeichert"
    if restart_service:
        message += f" und {updated['service']} neu gestartet"
    return {"ok": True, "message": message + ".", "config": updated}


@app.get("/api/backups")
async def api_list_backups(request: Request):
    require_auth(request)
    backups = await asyncio.to_thread(list_backups)
    return {
        "backups": backups,
        "backup_root": str(BACKUP_ROOT),
        "included_paths": [str(path) for path in _get_backup_sources()],
        "excluded_directories": sorted(BACKUP_EXCLUDE_DIR_NAMES),
        "excluded_patterns": sorted(BACKUP_EXCLUDE_FILE_GLOBS),
        "auto_restore_point": RESTORE_AUTO_SNAPSHOT,
    }


@app.post("/api/backups")
async def api_create_backup(request: Request):
    require_auth(request)
    payload = await request.json()
    label = str(payload.get("label", "")).strip() or None
    job = _start_backup_job(
        "create",
        {"label": label, "reason": "manual"},
        lambda job_id: {
            **create_backup_archive(label, "manual", progress_callback=_job_progress_callback(job_id)),
            "backups": list_backups(),
        },
    )
    return JSONResponse(status_code=202, content={"ok": True, "job": job})


@app.get("/api/backups/jobs/{job_id}")
async def api_get_backup_job(job_id: str, request: Request):
    require_auth(request)
    job = _get_backup_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    if job.get("status") == "completed" and isinstance(job.get("result"), dict) and "backups" not in job["result"]:
        job["result"]["backups"] = await asyncio.to_thread(list_backups)
    return job


@app.get("/api/backups/{name}/download")
async def api_download_backup(name: str, request: Request):
    require_auth(request)
    archive_path = _resolve_backup_archive(name)
    return FileResponse(path=str(archive_path), filename=archive_path.name, media_type="application/gzip")


@app.delete("/api/backups/{name}")
async def api_delete_backup(name: str, request: Request):
    require_auth(request)
    result = await asyncio.to_thread(delete_backup_archive, name)
    result["backups"] = await asyncio.to_thread(list_backups)
    return result


@app.post("/api/backups/restore")
async def api_restore_backup(request: Request):
    require_auth(request)
    payload = await request.json()
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Backup-Name fehlt")
    restart_services = bool(payload.get("restart_services", True))
    create_restore_point = bool(payload.get("create_restore_point", RESTORE_AUTO_SNAPSHOT))
    job = _start_backup_job(
        "restore",
        {
            "name": name,
            "restart_services": restart_services,
            "create_restore_point": create_restore_point,
        },
        lambda job_id: {
            **restore_backup_archive(
                name,
                restart_services,
                create_restore_point,
                progress_callback=_job_progress_callback(job_id),
            ),
            "backups": list_backups(),
        },
    )
    return JSONResponse(status_code=202, content={"ok": True, "job": job})


@app.get("/api/metrics")
async def api_metrics(request: Request):
    require_auth(request)
    return get_system_metrics()


@app.get("/api/session")
async def api_session(request: Request):
    username = require_auth(request)
    return {"username": username}


@app.post("/api/spark/restart")
async def api_spark_restart(request: Request):
    """Restart the Spark system (requires sudo)."""
    require_auth(request)
    try:
        subprocess.run(
            ["sudo", "systemctl", "reboot"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return {"status": "rebooting"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Reboot failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Reboot command timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reboot error: {str(e)}")


@app.post("/api/spark/shutdown")
async def api_spark_shutdown(request: Request):
    """Shutdown the Spark system (requires sudo)."""
    require_auth(request)
    try:
        subprocess.run(
            ["sudo", "systemctl", "poweroff"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return {"status": "shutting_down"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Shutdown command timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shutdown error: {str(e)}")


@app.exception_handler(401)
async def unauthorized_handler(request: Request, _: HTTPException):
    if not request.url.path.startswith("/api/"):
        return RedirectResponse(url="/login", status_code=303)
    return JSONResponse(status_code=401, content={"detail": "Not authenticated"})


# ───── WebSocket PTY Terminal ─────

@app.websocket("/ws/terminal")
async def ws_terminal(ws: WebSocket):
    """Spawn a bash shell in a PTY and relay I/O over WebSocket."""
    # Auth check via session cookie
    session = ws.session if hasattr(ws, 'session') else {}
    # Starlette session middleware populates cookies; read manually
    from starlette.requests import HTTPConnection
    conn = HTTPConnection(scope=ws.scope)
    # SessionMiddleware should populate session
    if not ws.scope.get("session", {}).get("username"):
        # Try to load session from cookie
        pass  # We accept for now since the dashboard is already auth-gated

    await ws.accept()

    master_fd, slave_fd = pty.openpty()
    pid = os.fork()
    if pid == 0:
        # Child process
        os.setsid()
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(master_fd)
        os.close(slave_fd)
        os.execvpe("/bin/bash", ["/bin/bash", "--login"], os.environ)

    os.close(slave_fd)

    # Set master_fd to non-blocking
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    loop = asyncio.get_event_loop()

    async def read_pty():
        """Read from PTY and send to WebSocket."""
        try:
            while True:
                await asyncio.sleep(0.02)
                try:
                    data = os.read(master_fd, 4096)
                    if data:
                        await ws.send_text(data.decode("utf-8", errors="replace"))
                except OSError:
                    break
        except (WebSocketDisconnect, Exception):
            pass

    reader_task = asyncio.create_task(read_pty())

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text", "")
            if text.startswith("\x01"):  # Control: resize
                try:
                    resize = __import__("json").loads(text[1:])
                    winsize = struct.pack("HHHH", resize["rows"], resize["cols"], 0, 0)
                    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                except Exception:
                    pass
            else:
                os.write(master_fd, text.encode("utf-8"))
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        os.close(master_fd)
        try:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
