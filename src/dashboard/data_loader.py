"""
Data loader untuk Streamlit dashboard.

Baca semua CSV FIRMS di /app/data/raw/firms (mount via docker-compose volume).
Cache TTL 5 menit supaya tidak baca disk berulang saat user filter.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

LOG = logging.getLogger(__name__)
_FIRMS_DIR = Path("/app/data/raw/firms")


@st.cache_data(ttl=300)
def load_all_hotspots() -> pd.DataFrame:
    """Concat semua CSV FIRMS jadi 1 DataFrame, dengan province_id dari nama file."""
    if not _FIRMS_DIR.exists():
        LOG.warning("FIRMS dir not found: %s", _FIRMS_DIR)
        return pd.DataFrame()

    files = sorted(_FIRMS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["province_id"] = p.stem.split("_")[0]
            dfs.append(df)
        except Exception as e:  # noqa: BLE001
            LOG.error("Failed to read %s: %s", p.name, e)

    if not dfs:
        return pd.DataFrame()

    full = pd.concat(dfs, ignore_index=True, sort=False)
    full["acq_date"] = pd.to_datetime(full["acq_date"], errors="coerce")
    full["latitude"] = pd.to_numeric(full["latitude"], errors="coerce")
    full["longitude"] = pd.to_numeric(full["longitude"], errors="coerce")
    full["frp"] = pd.to_numeric(full["frp"], errors="coerce")
    if "confidence" in full.columns:
        full["confidence"] = full["confidence"].astype(str).str.lower().str.strip()

    full = full.dropna(subset=["acq_date", "latitude", "longitude"])
    LOG.info("Loaded %d hotspots from %d files", len(full), len(files))
    return full


def filter_hotspots(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    provinces: list[str],
    min_confidence: str = "low",
) -> pd.DataFrame:
    """Filter berdasarkan date range, provinsi, dan minimum confidence."""
    if df.empty:
        return df

    mask = (df["acq_date"] >= start_date) & (df["acq_date"] <= end_date)
    if provinces:
        mask &= df["province_id"].isin(provinces)

    conf_order = {"l": 0, "n": 1, "h": 2}
    min_lvl = {"low": 0, "nominal": 1, "high": 2}.get(min_confidence, 0)
    if "confidence" in df.columns:
        df_conf = df["confidence"].map(conf_order).fillna(0)
        mask &= df_conf >= min_lvl

    return df.loc[mask].reset_index(drop=True)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hotspot count per (province × date) untuk time-series chart."""
    if df.empty:
        return pd.DataFrame(columns=["acq_date", "province_id", "hotspot_count"])

    grp = (df.groupby([df["acq_date"].dt.date.rename("acq_date"), "province_id"])
             .size().reset_index(name="hotspot_count"))
    grp["acq_date"] = pd.to_datetime(grp["acq_date"])
    return grp
