"""
FireGuard Metrics Proxy — FastAPI sidecar untuk observability (LK11).

Tugas service ini:
  1. Forward POST /invocations ke mlflow-model-server (Docker DNS round-robin
     antar 3 replicas → seimbang otomatis).
  2. Forward GET /ping (health check passthrough).
  3. Expose GET /metrics dengan format Prometheus berisi:
       - mlflow_request_duration_seconds (Histogram, label: endpoint, status)
       - mlflow_requests_total           (Counter,   label: endpoint, status)
       - mlflow_prediction_value         (Histogram, no label — untuk drift)
       - mlflow_active_requests          (Gauge,     no label)
       - mlflow_upstream_errors_total    (Counter,   label: reason)

Catatan keamanan & resource:
  - Body size dibatasi MAX_BODY_BYTES (default 1 MiB) → mitigasi DoS.
  - Upstream timeout ditetapkan via httpx.Timeout (no hanging request).
  - httpx.AsyncClient direuse via lifespan agar koneksi tidak bocor
    (NO per-request client allocation → mencegah memory leak).
  - Label cardinality TERIKAT (endpoint ∈ {invocations, ping}; status ∈ kode
    HTTP standar). Tidak ada label dinamis dari user input.

Endpoint internal:
  POST /invocations   → forwarded
  GET  /ping          → forwarded
  GET  /health        → local (cek apakah upstream reachable)
  GET  /metrics       → Prometheus exposition
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Config (env-overridable, semua punya default aman)
# ---------------------------------------------------------------------------
UPSTREAM_URL = os.getenv("MLFLOW_UPSTREAM_URL", "http://mlflow-model-server:8080")
UPSTREAM_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "15"))
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(10 * 1024 * 1024)))  # 10 MiB
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
LOG = logging.getLogger("fireguard.metrics_proxy")

# ---------------------------------------------------------------------------
# Prometheus metrics — registry dedicated (hindari konflik default registry)
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry()

REQ_DURATION = Histogram(
    "mlflow_request_duration_seconds",
    "Latensi end-to-end proxy → MLflow → response (detik).",
    labelnames=("endpoint", "status"),
    # Buckets cocok untuk inference 10ms-10s
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

REQ_TOTAL = Counter(
    "mlflow_requests_total",
    "Jumlah total request yang diproses proxy.",
    labelnames=("endpoint", "status"),
    registry=REGISTRY,
)

PRED_VALUE = Histogram(
    "mlflow_prediction_value",
    "Distribusi nilai prediksi model (hotspot_count_tomorrow). "
    "Pergeseran distribusi = sinyal data drift.",
    # Buckets 0 → 200+ hotspots
    buckets=(0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    registry=REGISTRY,
)

ACTIVE_REQ = Gauge(
    "mlflow_active_requests",
    "Jumlah request yang sedang in-flight di proxy.",
    registry=REGISTRY,
)

UPSTREAM_ERR = Counter(
    "mlflow_upstream_errors_total",
    "Error saat memanggil upstream MLflow.",
    labelnames=("reason",),  # cardinality terkontrol: timeout|network|http_5xx|invalid_json
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Lifespan — async HTTP client lifecycle (CEGAH memory leak)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Buka httpx.AsyncClient sekali, tutup saat shutdown."""
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=5.0)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    app.state.http = httpx.AsyncClient(
        base_url=UPSTREAM_URL,
        timeout=timeout,
        limits=limits,
    )
    LOG.info("Metrics proxy started. Upstream=%s timeout=%.1fs",
             UPSTREAM_URL, UPSTREAM_TIMEOUT_S)
    try:
        yield
    finally:
        await app.state.http.aclose()
        LOG.info("Metrics proxy stopped. HTTP client closed.")


app = FastAPI(
    title="FireGuard Metrics Proxy",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,        # disable Swagger UI di production (security)
    redoc_url=None,
    openapi_url=None,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bucket_status(code: int) -> str:
    """Reduce HTTP status code menjadi label berbutir kasar (cardinality control)."""
    if 200 <= code < 300:
        return "2xx"
    if 300 <= code < 400:
        return "3xx"
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return "other"


def _record_predictions(body: Any) -> None:
    """Ekstrak nilai prediksi dari body MLflow → observe ke histogram (drift)."""
    if not isinstance(body, dict):
        return
    preds = body.get("predictions")
    if not isinstance(preds, list):
        return
    for v in preds:
        try:
            PRED_VALUE.observe(float(v))
        except (TypeError, ValueError):
            # Tidak crash kalau model return non-numeric; cukup skip.
            continue


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape target."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Local health: ok kalau bisa hubungi upstream /ping."""
    client: httpx.AsyncClient = request.app.state.http
    try:
        r = await client.get("/ping", timeout=3.0)
        if r.status_code == 200:
            return JSONResponse({"status": "ok", "upstream": "ok"})
        return JSONResponse(
            {"status": "degraded", "upstream_status": r.status_code},
            status_code=503,
        )
    except httpx.HTTPError as e:
        return JSONResponse(
            {"status": "degraded", "error": type(e).__name__},
            status_code=503,
        )


@app.get("/ping")
async def ping(request: Request) -> Response:
    """Passthrough MLflow /ping (untuk healthcheck compatibility)."""
    endpoint = "ping"
    ACTIVE_REQ.inc()
    started = time.perf_counter()
    status = "5xx"
    try:
        client: httpx.AsyncClient = request.app.state.http
        try:
            r = await client.get("/ping")
            status = _bucket_status(r.status_code)
            return Response(content=r.content, status_code=r.status_code)
        except httpx.TimeoutException:
            UPSTREAM_ERR.labels(reason="timeout").inc()
            raise HTTPException(status_code=504, detail="upstream timeout")
        except httpx.HTTPError:
            UPSTREAM_ERR.labels(reason="network").inc()
            raise HTTPException(status_code=502, detail="upstream unreachable")
    finally:
        REQ_DURATION.labels(endpoint=endpoint, status=status).observe(
            time.perf_counter() - started
        )
        REQ_TOTAL.labels(endpoint=endpoint, status=status).inc()
        ACTIVE_REQ.dec()


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    """Forward MLflow /invocations + record metrik (latency, count, prediction)."""
    endpoint = "invocations"
    ACTIVE_REQ.inc()
    started = time.perf_counter()
    status = "5xx"
    try:
        # ---- Validasi body size (mitigasi DoS) ----
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            status = "4xx"
            raise HTTPException(status_code=413, detail="payload too large")

        client: httpx.AsyncClient = request.app.state.http
        try:
            r = await client.post(
                "/invocations",
                content=body,
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException:
            UPSTREAM_ERR.labels(reason="timeout").inc()
            raise HTTPException(status_code=504, detail="upstream timeout")
        except httpx.HTTPError as e:
            UPSTREAM_ERR.labels(reason="network").inc()
            LOG.warning("Upstream network error: %s", type(e).__name__)
            raise HTTPException(status_code=502, detail="upstream unreachable")

        status = _bucket_status(r.status_code)
        if 500 <= r.status_code < 600:
            UPSTREAM_ERR.labels(reason="http_5xx").inc()

        # ---- Record prediction values utk drift detection ----
        if r.status_code == 200:
            try:
                _record_predictions(r.json())
            except ValueError:
                UPSTREAM_ERR.labels(reason="invalid_json").inc()

        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )
    finally:
        REQ_DURATION.labels(endpoint=endpoint, status=status).observe(
            time.perf_counter() - started
        )
        REQ_TOTAL.labels(endpoint=endpoint, status=status).inc()
        ACTIVE_REQ.dec()


# ---------------------------------------------------------------------------
# Root — biar curl ke "/" tidak return 404 jelek
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> PlainTextResponse:
    return PlainTextResponse(
        "FireGuard metrics-proxy OK. See /metrics, /health, /invocations.",
        status_code=200,
    )
