# syntax=docker/dockerfile:1.7
# =============================================================================
# FireGuard Metrics Proxy — multi-stage build (LK11)
# =============================================================================
# Stage 1: builder — install deps di venv
# Stage 2: runtime — slim image, non-root user (uid 10004)
# =============================================================================

# ---------- Stage 1: builder -------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements/metrics-proxy.txt /tmp/requirements-metrics-proxy.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements-metrics-proxy.txt

# ---------- Stage 2: runtime -------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    MLFLOW_UPSTREAM_URL="http://mlflow-model-server:8080" \
    UPSTREAM_TIMEOUT_S="15" \
    MAX_BODY_BYTES="10485760" \
    LOG_LEVEL="INFO"

# curl untuk HEALTHCHECK Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 10004, terpisah dari mlflow=10002, dashboard=10003)
RUN groupadd --system --gid 10004 mproxy && \
    useradd  --system --uid 10004 --gid mproxy --home-dir /app --shell /sbin/nologin mproxy

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=mproxy:mproxy src/metrics_proxy/ /app/src/metrics_proxy/

USER mproxy

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --silent --fail http://localhost:9000/health || exit 1

# Single uvicorn worker. Untuk scale-up, naikkan replicas di compose (bukan workers)
# supaya tiap Prometheus scrape dapat metric registry yang terpisah & jelas.
CMD ["uvicorn", "src.metrics_proxy.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--workers", "1", \
     "--no-access-log"]
