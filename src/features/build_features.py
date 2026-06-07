"""
Feature Engineering — bangun training dataset dari data/processed/.

Tugas LK06: agregasi per (province_id, date), tambah rolling features,
days_since_rain, cyclical month encoding, lag features, dan target label
(hotspot_count_tomorrow + risk_level 3-kelas).

Pipeline:
    data/processed/firms_weather_joined_*.parquet  (per-detection)
        ↓
    Agregasi per (province_id, date_wib)
        ↓
    Rolling + lag + derived features
        ↓
    Target label generation (regresi + klasifikasi)
        ↓
    data/features/training_dataset_*.parquet  (siap konsumsi train.py)
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Threshold risk_level (kelas Aman/Waspada/Bahaya)
# Berbasis count hotspot harian per provinsi
_RISK_THRESHOLDS = {
    "aman_max": 0,      # 0 hotspot → Aman
    "waspada_max": 10,  # 1..10 → Waspada; > 10 → Bahaya
}

# Threshold precipitation untuk "days_since_rain" (mm/hari)
_RAIN_THRESHOLD_MM = 1.0


@dataclass(frozen=True)
class FeatureConfig:
    rolling_windows_days: tuple[int, ...] = (1, 3, 7)  # 1d ≈ 24h, 3d ≈ 72h, 7d
    lag_days: tuple[int, ...] = (1, 3, 7)
    forecast_horizon_days: int = 1  # prediksi 1 hari ke depan


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_per_province_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-detection DataFrame ke per-(province, day) summary.

    Input columns (dari preprocess.py):
        province_id, acq_date_wib, frp, confidence, daynight,
        temperature_2m_max, ..., relative_humidity_2m_mean

    Output columns:
        province_id, date, hotspot_count, frp_mean, frp_max, frp_sum,
        n_daytime, n_nighttime, n_confidence_high,
        temperature_2m_max, ..., relative_humidity_2m_mean
    """
    if df.empty:
        return pd.DataFrame()

    required = {"province_id", "acq_date_wib", "frp"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Aggregate input missing required columns: {sorted(missing)}")

    df = df.copy()
    df["acq_date_wib"] = pd.to_datetime(df["acq_date_wib"], errors="coerce")
    df = df.dropna(subset=["acq_date_wib"])

    # Hotspot agregate
    grp = df.groupby(["province_id", "acq_date_wib"], as_index=False)
    agg_hotspot = grp.agg(
        hotspot_count=("frp", "size"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),
        frp_sum=("frp", "sum"),
    )

    # Daynight count
    if "daynight" in df.columns:
        dn = df.assign(
            is_day=(df["daynight"].astype(str).str.upper() == "D").astype(int),
            is_night=(df["daynight"].astype(str).str.upper() == "N").astype(int),
        ).groupby(["province_id", "acq_date_wib"], as_index=False).agg(
            n_daytime=("is_day", "sum"),
            n_nighttime=("is_night", "sum"),
        )
        agg_hotspot = agg_hotspot.merge(dn, on=["province_id", "acq_date_wib"], how="left")

    # Confidence high count (FIRMS VIIRS confidence: 'l', 'n', 'h')
    if "confidence" in df.columns:
        conf = df.assign(
            is_high=(df["confidence"].astype(str).str.lower() == "h").astype(int),
        ).groupby(["province_id", "acq_date_wib"], as_index=False).agg(
            n_confidence_high=("is_high", "sum"),
        )
        agg_hotspot = agg_hotspot.merge(conf, on=["province_id", "acq_date_wib"], how="left")

    # Cuaca: ambil 1 baris pertama per (province, date) — cuaca harian sama untuk semua deteksi
    weather_cols = [
        c for c in [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "windspeed_10m_max", "winddirection_10m_dominant", "relative_humidity_2m_mean",
        ] if c in df.columns
    ]
    if weather_cols:
        wagg = df.groupby(["province_id", "acq_date_wib"], as_index=False)[weather_cols].first()
        agg_hotspot = agg_hotspot.merge(
            wagg, on=["province_id", "acq_date_wib"], how="left"
        )

    agg_hotspot = agg_hotspot.rename(columns={"acq_date_wib": "date"})
    return agg_hotspot.sort_values(["province_id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur kalender: month (raw), month_sin, month_cos, day_of_year."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    # Cyclical encoding (12 bulan)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_rolling_features(df: pd.DataFrame, windows_days: tuple[int, ...]) -> pd.DataFrame:
    """Rolling sum/mean per provinsi untuk hotspot_count & frp_mean."""
    df = df.sort_values(["province_id", "date"]).copy()
    for w in windows_days:
        df[f"hotspot_count_{w}d"] = (
            df.groupby("province_id")["hotspot_count"]
              .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).sum())
        )
        df[f"frp_mean_{w}d"] = (
            df.groupby("province_id")["frp_mean"]
              .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )
    return df


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    """Lag features hotspot_count untuk capture time dependency."""
    df = df.sort_values(["province_id", "date"]).copy()
    for lag in lags:
        df[f"hotspot_count_lag_{lag}d"] = (
            df.groupby("province_id")["hotspot_count"].shift(lag)
        )
    return df


def _dsr_per_group(rain_flags: pd.Series) -> pd.Series:
    """Hitung days_since_rain dalam 1 grup (1 provinsi). Loop eksplisit, mudah di-trace."""
    result = []
    counter = -1   # sebelum rain pertama, NaN
    for has_rain in rain_flags:
        if has_rain == 1:
            counter = 0
            result.append(0)
        else:
            if counter < 0:
                # Belum pernah rain — pakai 0 (asumsi: anggap baseline)
                result.append(0)
            else:
                counter += 1
                result.append(counter)
    return pd.Series(result, index=rain_flags.index)


def add_days_since_rain(
    df: pd.DataFrame, rain_threshold_mm: float = _RAIN_THRESHOLD_MM
) -> pd.DataFrame:
    """
    Jumlah hari berturut-turut tanpa hujan signifikan.

    Semantik: 0 = hari ini hujan; 1 = kemarin hujan terakhir; dst.
    Reset setiap kali precipitation_sum >= threshold.
    """
    if "precipitation_sum" not in df.columns:
        out = df.copy()
        out["days_since_rain"] = np.nan
        return out

    out = df.sort_values(["province_id", "date"]).copy()
    rain_flag = (out["precipitation_sum"] >= rain_threshold_mm).astype(int)
    # Apply per provinsi
    out["days_since_rain"] = (
        rain_flag.groupby(out["province_id"], group_keys=False).apply(_dsr_per_group)
    )
    return out


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame, forecast_horizon_days: int = 1) -> pd.DataFrame:
    """
    Generate dua target:
      1. hotspot_count_tomorrow (regresi): shift(-horizon) dari hotspot_count
      2. risk_level (klasifikasi): 0=Aman, 1=Waspada, 2=Bahaya
    """
    df = df.sort_values(["province_id", "date"]).copy()
    df["hotspot_count_tomorrow"] = (
        df.groupby("province_id")["hotspot_count"].shift(-forecast_horizon_days)
    )

    def _risk(n: float) -> int:
        if pd.isna(n):
            return -1   # akan di-drop nanti
        if n <= _RISK_THRESHOLDS["aman_max"]:
            return 0
        if n <= _RISK_THRESHOLDS["waspada_max"]:
            return 1
        return 2

    df["risk_level"] = df["hotspot_count_tomorrow"].apply(_risk)
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
def build_features(processed_df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """End-to-end: agregasi → fitur → target."""
    agg = aggregate_per_province_day(processed_df)
    if agg.empty:
        return agg
    agg = add_calendar_features(agg)
    agg = add_rolling_features(agg, cfg.rolling_windows_days)
    agg = add_lag_features(agg, cfg.lag_days)
    agg = add_days_since_rain(agg)
    agg = add_targets(agg, cfg.forecast_horizon_days)

    # Drop baris dengan target NaN (hari terakhir tiap provinsi)
    before = len(agg)
    agg = agg.dropna(subset=["hotspot_count_tomorrow"])
    agg = agg[agg["risk_level"] != -1]
    LOG.info("Drop rows tanpa label: %d → %d", before, len(agg))
    return agg.reset_index(drop=True)


def _read_processed_files(folder: Path) -> pd.DataFrame:
    """Concat semua parquet di folder data/processed/."""
    files = sorted(folder.glob("*.parquet"))
    if not files:
        LOG.warning("No processed parquet found in %s", folder)
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True, sort=False)


def write_features(df: pd.DataFrame, out_dir: Path,
                   now_utc: Optional[datetime] = None) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    now = (now_utc or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"training_dataset_{now}_UTC.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    LOG.info("Wrote %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FireGuard feature engineering (LK06)")
    p.add_argument("--processed-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "processed")
    p.add_argument("--output-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "features")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    LOG.info("Loading processed data from %s", args.processed_dir)
    processed = _read_processed_files(args.processed_dir)
    if processed.empty:
        LOG.error("No processed data — run preprocess.py first (LK04)")
        return 1

    LOG.info("Loaded %d rows", len(processed))
    cfg = FeatureConfig()
    features = build_features(processed, cfg)

    if features.empty:
        LOG.error("Feature dataset empty after pipeline — periksa data input")
        return 1

    path = write_features(features, args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
