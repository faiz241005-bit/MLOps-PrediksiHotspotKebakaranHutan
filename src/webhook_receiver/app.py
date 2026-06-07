"""
FireGuard LK12 — Grafana Webhook Receiver.

Endpoint:
    POST /webhook/grafana — terima alert dari Grafana Alerting
                            → exec auto_retrain.py sebagai subprocess.
    GET  /health          — health check (200 jika receiver hidup)
    GET  /history         — list 50 trigger terakhir (memory-only, capped)
    GET  /                — info

Architecture:
    Grafana Alert Rule FIRING
        │
        ▼ POST /webhook/grafana  (JSON body)
    [webhook_receiver]
        │
        ├── Validate token (X-Auth-Token header)
        ├── Parse alert payload → tentukan reason
        ├── Cek concurrency lock (max 1 retrain berjalan)
        ├── Spawn detached subprocess: auto_retrain.py
        └── Return 202 Accepted
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
REPO_ROOT = Path(os.getenv("FIREGUARD_REPO_ROOT", "/app"))
PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(1 * 1024 * 1024)))  # 1 MiB
HISTORY_SIZE = int(os.getenv("HISTORY_SIZE", "50"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
RETRAIN_DRY_RUN = os.getenv("RETRAIN_DRY_RUN", "false").lower() == "true"
SKIP_DVC = os.getenv("SKIP_DVC", "true").lower() == "true"  # default skip di Codespace

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [webhook_receiver] %(message)s",
)
LOG = logging.getLogger(__name__)

# Ring buffer history — bounded (maxlen)
HISTORY: deque = deque(maxlen=HISTORY_SIZE)

# Concurrency lock — at most 1 retrain berjalan
RETRAIN_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info("Webhook receiver started.")
    LOG.info("  Repo root      : %s", REPO_ROOT)
    LOG.info("  Python bin     : %s", PYTHON_BIN)
    LOG.info("  Token enforced : %s", bool(WEBHOOK_TOKEN))
    LOG.info("  Dry-run mode   : %s", RETRAIN_DRY_RUN)
    LOG.info("  Skip DVC       : %s", SKIP_DVC)
    if not WEBHOOK_TOKEN:
        LOG.warning("WEBHOOK_TOKEN not set — receiver dapat di-trigger siapa saja!")
    yield
    LOG.info("Webhook receiver stopped.")


app = FastAPI(
    title="FireGuard CT Webhook Receiver",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,        # Swagger UI dimatikan
    redoc_url=None,
    openapi_url=None,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_grafana_alert(payload: dict[str, Any]) -> tuple[str, str]:
    """
    Parse Grafana Unified Alerting webhook payload.

    Returns (alert_name, reason_string)
    """
    # Grafana sends "alerts" array dengan status, labels, annotations
    alerts = payload.get("alerts", [])
    firing = [a for a in alerts if a.get("status") == "firing"]
    if not firing:
        return ("unknown", "no firing alerts in payload")

    a = firing[0]
    labels = a.get("labels", {})
    name = labels.get("alertname", "unknown")
    summary = (a.get("annotations") or {}).get("summary", "")

    # Construct reason yang informative
    reason = f"grafana_alert:{name}"
    if summary:
        # Strip ke 80 char untuk hindari log bloat
        reason += f" | {summary[:80]}"
    return (name, reason)


def spawn_retrain(reason: str, extra_args: Optional[list] = None) -> int:
    """
    Fire-and-forget subprocess untuk auto_retrain.py.

    Return: PID of spawned process.
    Tidak block — caller akan kembalikan 202 ke Grafana.
    """
    cmd = [
        PYTHON_BIN, "-m", "src.scripts.auto_retrain",
        "--reason", reason,
    ]
    if SKIP_DVC:
        cmd.append("--skip-dvc")
    if RETRAIN_DRY_RUN:
        cmd.append("--dry-run")
    if extra_args:
        cmd.extend(extra_args)

    LOG.info("Spawning: %s", " ".join(cmd))

    # Detach — biar kalau receiver mati, training tetap jalan.
    # stdout/stderr ditulis ke file di /tmp (tmpfs, bounded).
    log_path = f"/tmp/ct_run_{int(time.time())}.log"
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # detach dari parent
            )
        LOG.info("Retrain spawned: PID=%d, log=%s", proc.pid, log_path)
        return proc.pid
    except FileNotFoundError as e:
        LOG.error("Python bin tidak ada: %s", e)
        raise


def record_history(entry: dict) -> None:
    HISTORY.append(entry)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> PlainTextResponse:
    return PlainTextResponse(
        "FireGuard CT Webhook Receiver. See /health, /history, "
        "POST /webhook/grafana.",
        status_code=200,
    )


@app.get("/health")
async def health() -> dict:
    """Health check sederhana."""
    return {
        "status": "ok",
        "dry_run": RETRAIN_DRY_RUN,
        "skip_dvc": SKIP_DVC,
        "retrain_busy": RETRAIN_LOCK.locked(),
        "history_count": len(HISTORY),
    }


@app.get("/history")
async def history() -> dict:
    """List trigger terakhir (memory-only, capped HISTORY_SIZE)."""
    return {"count": len(HISTORY), "entries": list(HISTORY)}


def _check_token(request: Request, x_auth_token: Optional[str]) -> None:
    """Validate token via header ATAU query param ?token=..."""
    if not WEBHOOK_TOKEN:
        return  # Auth disabled
    # Coba header dulu, fallback ke query
    qp_token = request.query_params.get("token")
    if x_auth_token == WEBHOOK_TOKEN or qp_token == WEBHOOK_TOKEN:
        return
    raise HTTPException(status_code=401, detail="invalid token")


@app.post("/webhook/grafana", status_code=status.HTTP_202_ACCEPTED)
async def grafana_webhook(
    request: Request,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
) -> JSONResponse:
    """Terima alert dari Grafana Alerting."""
    # ---- Token auth (header ATAU query ?token=...) ----
    _check_token(request, x_auth_token)

    # ---- Body size limit ----
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="payload too large")

    # ---- Parse JSON ----
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON")

    alert_name, reason = parse_grafana_alert(payload)
    LOG.info("Received alert: name=%s reason=%s", alert_name, reason)

    # ---- Concurrency check ----
    if RETRAIN_LOCK.locked():
        LOG.warning("Retrain already in progress — skipping new trigger.")
        record_history({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "alert": alert_name,
            "reason": reason,
            "action": "skipped_busy",
        })
        return JSONResponse(
            content={"status": "skipped", "reason": "retrain in progress"},
            status_code=429,
        )

    # ---- Spawn (non-blocking) ----
    # Kita ambil lock untuk SHORT duration: hanya untuk track "started"
    # Lock di-release segera; training jalan di subprocess detached.
    async with RETRAIN_LOCK:
        try:
            pid = spawn_retrain(reason)
        except Exception as e:  # noqa: BLE001
            LOG.error("Spawn failed: %s", e)
            record_history({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "alert": alert_name,
                "reason": reason,
                "action": "spawn_failed",
                "error": str(e),
            })
            raise HTTPException(status_code=500, detail=f"spawn failed: {e}")

    record_history({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alert": alert_name,
        "reason": reason,
        "action": "spawned",
        "pid": pid,
    })

    return JSONResponse(
        content={"status": "accepted", "pid": pid, "reason": reason},
        status_code=202,
    )


@app.post("/webhook/manual", status_code=status.HTTP_202_ACCEPTED)
async def manual_trigger(
    request: Request,
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
) -> JSONResponse:
    """
    Manual trigger endpoint — untuk testing/demo tanpa Grafana.

    Body opsional: {"reason": "<text>", "extra_args": ["--dry-run"]}
    """
    _check_token(request, x_auth_token)

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="payload too large")

    payload = {}
    if body:
        try:
            payload = await request.json()
        except ValueError:
            payload = {}

    reason = str(payload.get("reason", "manual_webhook"))[:200]

    # Whitelist extra_args yang diizinkan
    allowed_flags = {"--dry-run", "--skip-dvc"}
    extra_args_raw = payload.get("extra_args", [])
    extra_args = [a for a in extra_args_raw if a in allowed_flags]

    if RETRAIN_LOCK.locked():
        return JSONResponse(
            content={"status": "skipped", "reason": "retrain in progress"},
            status_code=429,
        )

    async with RETRAIN_LOCK:
        try:
            pid = spawn_retrain(reason, extra_args=extra_args)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"spawn failed: {e}")

    record_history({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alert": "manual",
        "reason": reason,
        "action": "spawned",
        "pid": pid,
        "extra_args": extra_args,
    })

    return JSONResponse(
        content={"status": "accepted", "pid": pid, "reason": reason},
        status_code=202,
    )
