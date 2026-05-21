"""
API client untuk Streamlit dashboard — call MLflow native /invocations.

Default URL: replica 1 di port 8010 (sesuai docker-compose.yaml).
Override via env FIREGUARD_API_URL kalau perlu test replica lain.

Format MLflow native (LK10):
    POST /invocations
    Body: {"dataframe_split": {"columns": [...], "data": [[...]]}}
    Response: {"predictions": [N]}

Risk level derivation di client (model native return raw count only).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger(__name__)

API_BASE = os.getenv("FIREGUARD_API_URL", "http://mlflow-model-server:8080")
_TIMEOUT_S = 10

# 27 fitur — match dengan model signature
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

_INT_COLS = {
    "month", "day_of_year",
    "hotspot_count", "n_daytime", "n_nighttime", "n_confidence_high",
    "days_since_rain",
}


def get_health() -> Optional[dict]:
    """GET /ping — health check MLflow native (return empty body, status 200)."""
    try:
        r = requests.get(f"{API_BASE}/ping", timeout=_TIMEOUT_S)
        return {"status": "ok"} if r.ok else None
    except Exception as e:  # noqa: BLE001
        LOG.warning("Ping failed: %s", type(e).__name__)
        return None


def predict_one(features: dict[str, Any]) -> Optional[float]:
    """POST /invocations — kirim 1 sample, dapat 1 prediksi (raw count)."""
    values: list[Any] = []
    for col in FEATURE_COLUMNS:
        v = features.get(col, 0)
        if pd.isna(v):
            v = 0
        if col in _INT_COLS:
            values.append(int(v))
        else:
            values.append(float(v))

    payload = {
        "dataframe_split": {
            "columns": FEATURE_COLUMNS,
            "data": [values],
        }
    }

    try:
        r = requests.post(
            f"{API_BASE}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=_TIMEOUT_S,
        )
        if not r.ok:
            LOG.warning("Predict failed HTTP %s: %s", r.status_code, r.text[:200])
            return None
        body = r.json()
        preds = body.get("predictions")
        return float(preds[0]) if preds else None
    except Exception as e:  # noqa: BLE001
        LOG.warning("Predict request error: %s", type(e).__name__)
        return None


def derive_risk(count: float) -> tuple[int, str]:
    """Konversi prediksi count → risk level/label (sama dengan build_features.py)."""
    if count <= 0:
        return 0, "Aman"
    if count <= 10:
        return 1, "Waspada"
    return 2, "Bahaya"


def build_features_from_today(
    df_today: pd.DataFrame,
    df_history: pd.DataFrame,
) -> dict[str, Any]:
    """
    Bangun 27-feature dict dari data 1 provinsi pada 1 hari + history 7 hari.

    df_today: rows untuk satu (province, date) hari ini
    df_history: rows historis untuk provinsi yang sama (7 hari ke belakang)
    """
    if df_today.empty:
        return {}

    n = len(df_today)
    frp = df_today["frp"].dropna()
    frp_mean = float(frp.mean()) if not frp.empty else 0.0
    frp_max = float(frp.max()) if not frp.empty else 0.0
    frp_sum = float(frp.sum()) if not frp.empty else 0.0

    daytime = df_today.get("daynight", pd.Series([])).astype(str).str.upper()
    n_daytime = int((daytime == "D").sum())
    n_nighttime = int((daytime == "N").sum())

    conf = df_today.get("confidence", pd.Series([])).astype(str).str.lower()
    n_conf_high = int((conf == "h").sum())

    today = pd.to_datetime(df_today["acq_date"].iloc[0])
    month = int(today.month)
    day_of_year = int(today.dayofyear)
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))

    def rolling_count(days_back: int) -> float:
        cutoff = today - pd.Timedelta(days=days_back)
        return float(df_history[
            (df_history["acq_date"] > cutoff) &
            (df_history["acq_date"] <= today)
        ].shape[0])

    def rolling_frp_mean(days_back: int) -> float:
        cutoff = today - pd.Timedelta(days=days_back)
        sub_frp = df_history.loc[
            (df_history["acq_date"] > cutoff) &
            (df_history["acq_date"] <= today),
            "frp"
        ].dropna()
        return float(sub_frp.mean()) if not sub_frp.empty else 0.0

    def lag_count(days_back: int) -> float:
        target = today - pd.Timedelta(days=days_back)
        return float(df_history[
            df_history["acq_date"].dt.date == target.date()
        ].shape[0])

    # Weather default (tropical Indonesia rata-rata) — dashboard tidak punya weather real-time
    return {
        "hotspot_count": n,
        "frp_mean": frp_mean,
        "frp_max": frp_max,
        "frp_sum": frp_sum,
        "n_daytime": n_daytime,
        "n_nighttime": n_nighttime,
        "n_confidence_high": n_conf_high,
        "temperature_2m_max": 32.0,
        "temperature_2m_min": 24.0,
        "precipitation_sum": 0.0,
        "windspeed_10m_max": 10.0,
        "winddirection_10m_dominant": 180.0,
        "relative_humidity_2m_mean": 75.0,
        "month": month,
        "day_of_year": day_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "hotspot_count_1d": rolling_count(1),
        "hotspot_count_3d": rolling_count(3),
        "hotspot_count_7d": rolling_count(7),
        "frp_mean_1d": rolling_frp_mean(1),
        "frp_mean_3d": rolling_frp_mean(3),
        "frp_mean_7d": rolling_frp_mean(7),
        "hotspot_count_lag_1d": lag_count(1),
        "hotspot_count_lag_3d": lag_count(3),
        "hotspot_count_lag_7d": lag_count(7),
        "days_since_rain": 7,
    }
