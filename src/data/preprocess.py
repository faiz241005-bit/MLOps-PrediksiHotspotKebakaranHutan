"""
Preprocessing — clean raw FIRMS + weather data, join, dan tulis ke data/processed/.

Tugas LK04: skrip preprocessing yang dapat dijalankan ulang (idempoten).
Membaca seluruh CSV di data/raw/{firms,weather}/, lakukan cleaning &
validasi, join FIRMS dengan cuaca harian per provinsi+tanggal, dan
tulis parquet ber-timestamp di data/processed/.

Output schema (per baris = 1 deteksi FIRMS, di-enrich cuaca harian):
    province_id, acq_datetime_wib, latitude, longitude, frp, confidence,
    daynight, temperature_2m_max, temperature_2m_min, precipitation_sum,
    windspeed_10m_max, winddirection_10m_dominant, relative_humidity_2m_mean

Resource hygiene:
    - pd.read_csv pakai path-string (pandas mengelola FD internal)
    - tidak ada global cache yang membesar lintas-call
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Kolom kritis dari FIRMS — wajib ada
_FIRMS_REQUIRED = {"latitude", "longitude", "acq_date", "acq_time", "frp"}

# Kolom yang akan kita pertahankan dari FIRMS
_FIRMS_KEEP = [
    "latitude", "longitude", "acq_date", "acq_time", "frp",
    "confidence", "daynight", "satellite",
]


@dataclass(frozen=True)
class PreprocessConfig:
    raw_firms_dir: Path
    raw_weather_dir: Path
    output_dir: Path
    timezone_offset_hours: int = 7  # WIB = UTC+7 untuk Kalimantan & Sumatera


# ---------------------------------------------------------------------------
# FIRMS cleaning
# ---------------------------------------------------------------------------
def _read_firms_files(folder: Path) -> pd.DataFrame:
    """Concat semua CSV FIRMS di folder. Tambahkan province_id dari nama file."""
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        LOG.warning("No FIRMS CSV found in %s", folder)
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            # Filename pattern: {province}_{YYYYMMDD_HHMMSS}_UTC.csv
            province_id = f.stem.split("_")[0]
            # Catatan: read_csv mengelola FD internal (auto-close).
            df = pd.read_csv(f, low_memory=False)
            df["province_id"] = province_id
            df["_source_file"] = f.name
            dfs.append(df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            LOG.warning("Skip malformed FIRMS file %s: %s", f.name, e)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def clean_firms(df: pd.DataFrame, tz_offset_hours: int = 7) -> pd.DataFrame:
    """Clean & normalize FIRMS DataFrame."""
    if df.empty:
        return df

    missing = _FIRMS_REQUIRED - set(df.columns)
    if missing:
        raise RuntimeError(f"FIRMS df missing required columns: {sorted(missing)}")

    # Drop rows dengan kolom kritis NA
    before = len(df)
    df = df.dropna(subset=list(_FIRMS_REQUIRED)).copy()
    LOG.info("FIRMS dropna critical: %d → %d rows", before, len(df))

    # Type coerce
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    if "confidence" in df.columns:
        # FIRMS VIIRS confidence bisa 'l'/'n'/'h' (string) atau numeric — normalize ke string lower
        df["confidence"] = df["confidence"].astype(str).str.lower().str.strip()

    # Validasi domain
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    df = df[df["frp"] >= 0]

    # Compose acq_datetime_utc dari acq_date + acq_time
    # acq_time format: HHMM integer (mis. 1418 = 14:18)
    df["acq_time_str"] = df["acq_time"].astype(str).str.zfill(4)
    df["acq_datetime_utc"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + df["acq_time_str"].str[:2] + ":" + df["acq_time_str"].str[2:4],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["acq_datetime_utc"]).copy()

    # Konversi ke WIB
    df["acq_datetime_wib"] = df["acq_datetime_utc"] + pd.Timedelta(hours=tz_offset_hours)
    df["acq_date_wib"] = df["acq_datetime_wib"].dt.date

    # Dedup: kombinasi koordinat + waktu + satellite
    dedup_keys = ["latitude", "longitude", "acq_datetime_utc"]
    if "satellite" in df.columns:
        dedup_keys.append("satellite")
    before_d = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    LOG.info("FIRMS dedup: %d → %d rows", before_d, len(df))

    # Subset kolom yang dipertahankan + meta
    keep = [c for c in _FIRMS_KEEP if c in df.columns]
    keep += ["province_id", "acq_datetime_utc", "acq_datetime_wib", "acq_date_wib"]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weather cleaning
# ---------------------------------------------------------------------------
def _read_weather_files(folder: Path) -> pd.DataFrame:
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        LOG.warning("No weather CSV found in %s", folder)
        return pd.DataFrame()
    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            LOG.warning("Skip malformed weather file %s: %s", f.name, e)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    required = {"province_id", "date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Weather df missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # Coerce numeric columns
    numeric_cols = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "windspeed_10m_max", "winddirection_10m_dominant", "relative_humidity_2m_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dedup: kalau ada multiple snapshot per (province, date), ambil yang terbaru
    if "fetched_at_utc" in df.columns:
        df = df.sort_values("fetched_at_utc", ascending=True)
    df = df.drop_duplicates(subset=["province_id", "date"], keep="last").reset_index(drop=True)
    LOG.info("Weather cleaned: %d rows (after dedup)", len(df))
    return df


# ---------------------------------------------------------------------------
# Join + write
# ---------------------------------------------------------------------------
def join_firms_weather(firms: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left join FIRMS (per deteksi) ↔ weather (harian) on (province_id, date)."""
    if firms.empty:
        LOG.warning("FIRMS df empty — join skipped")
        return firms
    if weather.empty:
        LOG.warning("Weather df empty — return FIRMS without enrichment")
        return firms

    weather_keep = ["province_id", "date"] + [
        c for c in [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "windspeed_10m_max", "winddirection_10m_dominant", "relative_humidity_2m_mean",
        ] if c in weather.columns
    ]
    w = weather[weather_keep].copy()

    merged = firms.merge(
        w,
        how="left",
        left_on=["province_id", "acq_date_wib"],
        right_on=["province_id", "date"],
        validate="many_to_one",
    )
    n_unjoined = merged["date"].isna().sum()
    if n_unjoined:
        LOG.warning("%d FIRMS rows without weather match (cuaca tanggal itu tidak tersedia)",
                    n_unjoined)
    return merged.drop(columns=["date"]).reset_index(drop=True)


def write_processed(df: pd.DataFrame, out_dir: Path,
                    now_utc: Optional[datetime] = None) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    now = (now_utc or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"firms_weather_joined_{now}_UTC.parquet"

    # pyarrow engine (default for to_parquet kalau pyarrow tersedia)
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    LOG.info("Wrote %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
    return out_path


# ---------------------------------------------------------------------------
# Public API + CLI
# ---------------------------------------------------------------------------
def run_preprocess(cfg: PreprocessConfig) -> Path:
    LOG.info("Loading FIRMS from %s", cfg.raw_firms_dir)
    firms_raw = _read_firms_files(cfg.raw_firms_dir)
    LOG.info("Loaded FIRMS: %d rows", len(firms_raw))

    LOG.info("Loading weather from %s", cfg.raw_weather_dir)
    weather_raw = _read_weather_files(cfg.raw_weather_dir)
    LOG.info("Loaded weather: %d rows", len(weather_raw))

    firms_clean = clean_firms(firms_raw, tz_offset_hours=cfg.timezone_offset_hours)
    weather_clean = clean_weather(weather_raw)
    merged = join_firms_weather(firms_clean, weather_clean)

    if merged.empty:
        raise RuntimeError("Processed dataset is empty — periksa raw data & log di atas")
    return write_processed(merged, cfg.output_dir)


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FireGuard data preprocessing (LK04)")
    p.add_argument("--raw-firms-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "raw" / "firms")
    p.add_argument("--raw-weather-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "raw" / "weather")
    p.add_argument("--output-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "processed")
    p.add_argument("--tz-offset-hours", type=int, default=7,
                   help="Offset jam dari UTC (default 7 untuk WIB)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = PreprocessConfig(
        raw_firms_dir=args.raw_firms_dir,
        raw_weather_dir=args.raw_weather_dir,
        output_dir=args.output_dir,
        timezone_offset_hours=args.tz_offset_hours,
    )
    try:
        path = run_preprocess(cfg)
        print(path)
        return 0
    except Exception as e:  # noqa: BLE001
        LOG.error("Preprocess failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
