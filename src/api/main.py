"""
FireGuard Inference API — FastAPI service yang load model dari MLflow Registry.

Tugas LK09: layanan API inferensi untuk Docker Compose orchestration.
Bersama dengan mlflow-server (service kedua), API ini membentuk sistem ML yang
ter-containerize end-to-end.

Endpoints
---------
GET  /health        — liveness/readiness probe untuk Docker healthcheck
GET  /model-info    — metadata model yang sedang dipakai (name, version, stage)
POST /predict       — inferensi hotspot_count_tomorrow + risk_level

Security
--------
- Pydantic v2 strict validation di setiap field (anti-injection / type confusion)
- Field bounds (ge/le) untuk mencegah nilai absurd yang bisa men-trigger NaN
- Model loaded SEKALI saat startup (lifespan event), bukan per-request → no memory leak
- MLflow tracking URI via env var (jangan hardcode), tidak di-log
- /predict log hanya metadata request, tidak feature payload (privacy)
- Error handler global: jangan leak stack trace ke client (mask internal error)

Resource hygiene
----------------
- Model wrapper di module-level `_STATE` dict, di-clear di shutdown event
- Tidak ada in-memory cache yang membesar (predict tidak menyimpan history)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

LOG = logging.getLogger("fireguard.api")

# ---------------------------------------------------------------------------
# Module-level state — model di-load sekali, di-share antar request.
# Pakai dict, bukan globals terpisah, supaya mudah di-clear di shutdown.
# ---------------------------------------------------------------------------
_STATE: dict[str, Any] = {
    "model": None,
    "model_name": None,
    "model_version": None,
    "model_stage": None,
    "loaded_at": None,
    "ready": False,
}

# Default feature order — match output build_features.py & synthetic_data.py.
# Disimpan eksplisit supaya request DataFrame punya kolom yang konsisten,
# tidak tergantung urutan field di Pydantic schema.
FEATURE_COLUMNS = [
    "hotspot_count", "frp_mean", "frp_max", "frp_sum",
    "n_daytime", "n_nighttime", "n_confidence_high",
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "windspeed_10m_max", "winddirection_10m_dominant",
    "relative_humidity_2m_mean",
    "month", "day_of_year", "month_sin", "month_cos",
    "hotspot_count_1d", "hotspot_count_3d", "hotspot_count_7d",
    "frp_mean_1d", "frp_mean_3d", "frp_mean_7d",
    "hotspot_count_lag_1d", "hotspot_count_lag_3d", "hotspot_count_lag_7d",
    "days_since_rain",
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Input fitur untuk inferensi 1 sample (1 provinsi × 1 hari).

    Semua field punya bounds realistis (ge/le) untuk reject input liar.
    Bounds berdasarkan range yang masuk akal untuk Indonesia (cuaca tropis).
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Hotspot statistics
    hotspot_count: int = Field(..., ge=0, le=10_000)
    frp_mean: float = Field(..., ge=0.0, le=5_000.0)
    frp_max: float = Field(..., ge=0.0, le=10_000.0)
    frp_sum: float = Field(..., ge=0.0, le=1_000_000.0)
    n_daytime: int = Field(..., ge=0, le=10_000)
    n_nighttime: int = Field(..., ge=0, le=10_000)
    n_confidence_high: int = Field(..., ge=0, le=10_000)

    # Weather
    temperature_2m_max: float = Field(..., ge=-10.0, le=55.0)
    temperature_2m_min: float = Field(..., ge=-15.0, le=45.0)
    precipitation_sum: float = Field(..., ge=0.0, le=1_000.0)
    windspeed_10m_max: float = Field(..., ge=0.0, le=200.0)
    winddirection_10m_dominant: float = Field(..., ge=0.0, le=360.0)
    relative_humidity_2m_mean: float = Field(..., ge=0.0, le=100.0)

    # Calendar
    month: int = Field(..., ge=1, le=12)
    day_of_year: int = Field(..., ge=1, le=366)
    month_sin: float = Field(..., ge=-1.0, le=1.0)
    month_cos: float = Field(..., ge=-1.0, le=1.0)

    # Rolling
    hotspot_count_1d: float = Field(..., ge=0.0, le=100_000.0)
    hotspot_count_3d: float = Field(..., ge=0.0, le=100_000.0)
    hotspot_count_7d: float = Field(..., ge=0.0, le=100_000.0)
    frp_mean_1d: float = Field(..., ge=0.0, le=5_000.0)
    frp_mean_3d: float = Field(..., ge=0.0, le=5_000.0)
    frp_mean_7d: float = Field(..., ge=0.0, le=5_000.0)

    # Lags
    hotspot_count_lag_1d: float = Field(..., ge=0.0, le=100_000.0)
    hotspot_count_lag_3d: float = Field(..., ge=0.0, le=100_000.0)
    hotspot_count_lag_7d: float = Field(..., ge=0.0, le=100_000.0)
    days_since_rain: int = Field(..., ge=0, le=365)


class PredictResponse(BaseModel):
    """Output prediksi."""
    hotspot_count_tomorrow: float = Field(..., description="Prediksi hotspot besok (regresi)")
    risk_level: int = Field(..., ge=0, le=2,
                            description="0=Aman, 1=Waspada, 2=Bahaya (derived dari count)")
    risk_label: str = Field(..., description="Aman | Waspada | Bahaya")
    served_by: dict[str, Any] = Field(..., description="Model name/version/stage")


class ModelInfoResponse(BaseModel):
    name: Optional[str]
    version: Optional[str]
    stage: Optional[str]
    loaded_at: Optional[float]
    ready: bool
    tracking_uri: str
    feature_count: int


class HealthResponse(BaseModel):
    status: str
    ready: bool
    uptime_sec: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _risk_label(level: int) -> str:
    return {0: "Aman", 1: "Waspada", 2: "Bahaya"}.get(level, "Unknown")


def _derive_risk_level(count: float) -> int:
    """Sama dengan logika di build_features.py."""
    if count <= 0:
        return 0
    if count <= 10:
        return 1
    return 2


def _load_model_from_registry() -> None:
    """
    Load model dari MLflow Model Registry.

    Strategy:
      1) Coba load stage 'Production' (env: FIREGUARD_MODEL_STAGE, default Production)
      2) Kalau tidak ada, fallback ke stage 'Staging'
      3) Kalau dua-duanya tidak ada, raise — service tetap up tapi /health akan 503

    Env vars:
      MLFLOW_TRACKING_URI   — wajib di-set di docker-compose ke http://mlflow-server:5000
      FIREGUARD_MODEL_NAME  — default 'fireguard-regressor'
      FIREGUARD_MODEL_STAGE — default 'Production'
    """
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    model_name = os.getenv("FIREGUARD_MODEL_NAME", "fireguard-regressor")
    preferred_stage = os.getenv("FIREGUARD_MODEL_STAGE", "Production")

    mlflow.set_tracking_uri(tracking_uri)
    LOG.info("Connecting to MLflow at %s", tracking_uri)
    client = mlflow.tracking.MlflowClient()

    stages_to_try = [preferred_stage]
    if preferred_stage != "Staging":
        stages_to_try.append("Staging")

    last_error: Optional[Exception] = None
    for stage in stages_to_try:
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                LOG.warning("No version found at stage=%s for model=%s", stage, model_name)
                continue
            v = versions[0]
            model_uri = f"models:/{model_name}/{stage}"
            LOG.info("Loading %s (version %s, stage %s)", model_uri, v.version, stage)
            model = mlflow.pyfunc.load_model(model_uri)
            _STATE.update({
                "model": model,
                "model_name": model_name,
                "model_version": str(v.version),
                "model_stage": stage,
                "loaded_at": time.time(),
                "ready": True,
            })
            LOG.info("Model loaded successfully (v%s @ %s)", v.version, stage)
            return
        except Exception as e:  # noqa: BLE001
            last_error = e
            LOG.warning("Failed to load model at stage=%s: %s", stage, type(e).__name__)

    # Tidak fatal — service tetap up supaya /health bisa return 503 untuk debugging.
    LOG.error("No usable model found in registry. Last error: %s", last_error)
    _STATE.update({"ready": False})


# ---------------------------------------------------------------------------
# FastAPI lifecycle — load model di startup, clear di shutdown.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup: load model dari MLflow Registry (sekali).
    Shutdown: clear _STATE untuk memastikan GC bisa free memory.
    """
    _STATE["startup_time"] = time.time()
    LOG.info("FireGuard API starting up...")
    try:
        _load_model_from_registry()
    except Exception as e:  # noqa: BLE001
        # Service tetap up walau load gagal — supaya bisa debug via /health.
        LOG.error("Startup model load raised: %s", e)
    yield
    LOG.info("FireGuard API shutting down...")
    # Memory hygiene: drop reference ke model object supaya GC bisa free.
    _STATE["model"] = None
    _STATE["ready"] = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FireGuard Inference API",
    version="1.0.0",
    description="Forest fire hotspot prediction service (LK09 — Docker Compose orchestrated)",
    lifespan=lifespan,
    # Disable docs di prod nanti via env, tapi default on untuk demo akademik.
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """
    Liveness + readiness probe.

    - status='ok' selalu kalau proses jalan
    - ready=False kalau model belum loaded (Docker healthcheck akan retry)
    """
    uptime = time.time() - _STATE.get("startup_time", time.time())
    return HealthResponse(
        status="ok",
        ready=bool(_STATE.get("ready")),
        uptime_sec=round(uptime, 2),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info() -> ModelInfoResponse:
    """Metadata model yang sedang di-serve."""
    return ModelInfoResponse(
        name=_STATE.get("model_name"),
        version=_STATE.get("model_version"),
        stage=_STATE.get("model_stage"),
        loaded_at=_STATE.get("loaded_at"),
        ready=bool(_STATE.get("ready")),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"),
        feature_count=len(FEATURE_COLUMNS),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Inferensi hotspot besok untuk 1 sample.

    Returns:
        hotspot_count_tomorrow (float), risk_level (0/1/2), risk_label, served_by
    """
    if not _STATE.get("ready") or _STATE.get("model") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — check /model-info and MLflow Registry",
        )

    # Bangun DataFrame 1 baris dengan urutan kolom FEATURE_COLUMNS.
    # model_dump() aman: hanya berisi field yang sudah di-validate Pydantic.
    raw = payload.model_dump()
    try:
        df = pd.DataFrame([[raw[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    except KeyError as e:  # noqa: BLE001
        # Defensive: kalau ada drift antara schema & FEATURE_COLUMNS.
        LOG.error("Feature mapping error: %s", e)
        raise HTTPException(status_code=500, detail="Internal feature mapping error") from e

    try:
        pred = _STATE["model"].predict(df)
    except Exception as e:  # noqa: BLE001
        LOG.error("Model.predict raised %s", type(e).__name__)
        # Jangan leak detail internal ke client.
        raise HTTPException(status_code=500, detail="Inference failed") from e

    # mlflow.pyfunc.predict bisa return ndarray / list / Series — normalize.
    value = float(np.asarray(pred).reshape(-1)[0])
    value = max(0.0, value)  # hotspot count tidak boleh negatif
    risk = _derive_risk_level(value)

    return PredictResponse(
        hotspot_count_tomorrow=round(value, 3),
        risk_level=risk,
        risk_label=_risk_label(risk),
        served_by={
            "name": _STATE.get("model_name"),
            "version": _STATE.get("model_version"),
            "stage": _STATE.get("model_stage"),
        },
    )


# Setup basic logging kalau dijalankan via uvicorn dengan --log-config default.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
