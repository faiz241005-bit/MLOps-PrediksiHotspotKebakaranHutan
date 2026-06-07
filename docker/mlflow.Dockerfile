# syntax=docker/dockerfile:1.7
# =============================================================================
# FireGuard MLflow Tracking Server (LK09)
# =============================================================================
# Backend store: SQLite (file di volume) — cukup untuk demo & dev.
# Artifact store: filesystem di volume terpisah.
#
# Production upgrade path: ganti backend ke Postgres + artifact ke S3/R2 dengan
# env var MLFLOW_BACKEND_STORE_URI dan MLFLOW_DEFAULT_ARTIFACT_ROOT.
# =============================================================================
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_HOME=/mlflow

# curl untuk HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Pin versi mlflow sama dengan client di service lain (avoid protocol drift)
RUN pip install --upgrade pip && \
    pip install mlflow==2.13.0

# Non-root user
RUN groupadd --system --gid 10002 mlflow && \
    useradd  --system --uid 10002 --gid mlflow --home-dir /mlflow --shell /sbin/nologin mlflow && \
    mkdir -p /mlflow/db /mlflow/artifacts && \
    chown -R mlflow:mlflow /mlflow

WORKDIR /mlflow
USER mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --silent --fail http://localhost:5000/ || exit 1

# --serve-artifacts                : server menerima upload/download artifact via HTTP.
# --artifacts-destination /mlflow/artifacts : path SERVER-side untuk simpan artifact.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts"]
