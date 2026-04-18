import asyncio
import os
import secrets
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pam
import psutil
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).resolve().parent
APP_HOST = os.getenv("SPARK_ADMIN_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("SPARK_ADMIN_PORT", "7000"))
STACK_OWNER = os.getenv("SPARK_STACK_OWNER", os.getenv("SUDO_USER") or os.getenv("USER") or "ksai0001_local")
STACK_HOME = Path(os.getenv("SPARK_STACK_HOME", str(Path("/home") / STACK_OWNER)))
SESSION_SECRET = os.getenv("SPARK_ADMIN_SESSION_SECRET", secrets.token_hex(32))
SESSION_COOKIE = os.getenv("SPARK_ADMIN_SESSION_COOKIE", "spark_admin_session")
REQUEST_TIMEOUT = float(os.getenv("SPARK_ADMIN_HTTP_TIMEOUT", "5"))
PAM_SERVICE = os.getenv("SPARK_ADMIN_PAM_SERVICE", "login")

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
        "label": "Correction LLM",
        "port": 9000,
        "health_url": "http://127.0.0.1:9000/v1/models",
        "health_headers": {"Authorization": "Bearer local-correction-llm"},
        "config_files": [
            {
                "id": "correction-run",
                "label": "run_llm.sh",
                "path": STACK_HOME / "correction-llm-vllm" / "run_llm.sh",
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


def current_username(request: Request) -> str | None:
    return request.session.get("username")


def require_auth(request: Request) -> str:
    username = current_username(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return username


def run_command(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


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
        },
    )


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
    service = MANAGED_SERVICES.get(service_key)
    if not service:
        raise HTTPException(status_code=404, detail="Unknown service")
    payload = await request.json()
    action = payload.get("action")
    if action not in {"start", "stop", "restart"}:
        raise HTTPException(status_code=400, detail="Unsupported action")
    # Always prefer systemctl — it handles docker containers AND respects
    # Restart=always policies (plain 'docker stop' would be undone by systemd).
    success = False
    svc_unit = service["service"]
    # Check if a systemd unit exists for this service
    unit_check = run_command(["systemctl", "cat", svc_unit], timeout=10)
    if unit_check.returncode == 0:
        result = run_command(["systemctl", action, svc_unit], timeout=120)
        success = result.returncode == 0
    else:
        # No systemd unit — fall back to docker commands
        containers = service.get("container", [])
        if isinstance(containers, str):
            containers = [containers]
        for cn in containers:
            cmd = ["docker", action if action != "restart" else "restart", cn]
            dr = run_command(cmd, timeout=60)
            if dr.returncode == 0:
                success = True
                break
    if not success:
        raise HTTPException(status_code=500, detail=f"{action} fehlgeschlagen")
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


@app.get("/api/metrics")
async def api_metrics(request: Request):
    require_auth(request)
    return get_system_metrics()


@app.get("/api/session")
async def api_session(request: Request):
    username = require_auth(request)
    return {"username": username}


@app.exception_handler(401)
async def unauthorized_handler(request: Request, _: HTTPException):
    if not request.url.path.startswith("/api/"):
        return RedirectResponse(url="/login", status_code=303)
    return JSONResponse(status_code=401, content={"detail": "Not authenticated"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
