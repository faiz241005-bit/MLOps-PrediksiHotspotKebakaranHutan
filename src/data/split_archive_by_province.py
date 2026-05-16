"""
Split FIRMS archive download (negara-level CSV) jadi per-provinsi files.

Input  : data/raw/firms_archive/fire_archive_*.csv + fire_nrt_*.csv
         (download dari https://firms.modaps.eosdis.nasa.gov/download/)
Output : data/raw/firms/{province_id}_{start_date}_{end_date}_archive.csv
         (langsung kompatibel dengan src/data/preprocess.py)

Strategi:
    1. Baca SEMUA CSV di folder firms_archive/ (archive + nrt) dan concat
    2. Filter per bbox provinsi (overlapping bbox → first match wins)
    3. Tulis 1 CSV per provinsi dengan filename yang preprocess.py expect
       (province_id ditarik dari prefix filename via split("_")[0])

Schema columns yang dipertahankan:
    latitude, longitude, brightness, scan, track, acq_date, acq_time,
    satellite, instrument, confidence, version, bright_t31, frp, daynight, type
    (preprocess.py cuma butuh: latitude, longitude, acq_date, acq_time, frp,
     confidence, daynight, satellite — sisanya disimpan untuk audit)

Security & memory:
    - Path traversal guard untuk filename output
    - Validate kolom required ada di input sebelum proses
    - Stream-write per province; tidak hold seluruh dataset > 1x di memori
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default province bboxes (sama dengan src/data/bulk_fetch.py)
_DEFAULT_PROVINCES = {
    "riau":    [0.0, 100.0, 4.5, 106.5],
    "kalteng": [-3.5, 110.5, 1.5, 116.5],
    "kalbar":  [-3.0, 108.0, 2.5, 118.0],
    "sumsel":  [-5.5, 102.0, -1.0, 108.5],
    "jambi":   [-3.0, 101.0, -0.5, 105.0],
}

_REQUIRED_COLS = {"latitude", "longitude", "acq_date", "acq_time", "frp"}


@dataclass(frozen=True)
class ProvinceBox:
    province_id: str
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @classmethod
    def from_bbox(cls, prov_id: str, bbox: Sequence[float]) -> "ProvinceBox":
        if len(bbox) != 4:
            raise ValueError(f"bbox must be 4 elements, got {len(bbox)}")
        return cls(prov_id, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    def contains(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat <= self.lat_max
                and self.lon_min <= lon <= self.lon_max)


def _safe_output_path(out_dir: Path, filename: str) -> Path:
    """Path traversal guard."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = (out_dir / filename).resolve()
    out_str = str(out_dir)
    if not (str(candidate) == out_str or str(candidate).startswith(out_str + os.sep)):
        raise ValueError(f"Unsafe filename {filename!r}")
    return candidate


def _load_provinces(config_path: Optional[Path]) -> list[ProvinceBox]:
    """Load province bboxes dari config/params.yaml atau fallback ke defaults."""
    if config_path and config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            provs = ((cfg.get("data") or {}).get("provinces") or [])
            boxes = [ProvinceBox.from_bbox(p["id"], p["bbox"]) for p in provs
                     if "id" in p and "bbox" in p]
            if boxes:
                return boxes
        except Exception as e:  # noqa: BLE001
            LOG.warning("Config parse error (%s); fallback ke defaults", e)
    return [ProvinceBox.from_bbox(k, v) for k, v in _DEFAULT_PROVINCES.items()]


def load_firms_csv(path: Path) -> pd.DataFrame:
    """Baca 1 CSV FIRMS, validate kolom required."""
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    LOG.info("  %s: %d rows, %d cols", path.name, len(df), len(df.columns))
    return df


def assign_province(df: pd.DataFrame, boxes: list[ProvinceBox]) -> pd.Series:
    """
    Untuk tiap baris, assign ke provinsi pertama yang match (first-match-wins).
    Return Series berisi province_id atau None kalau di luar semua bbox.
    """
    result = pd.Series([None] * len(df), index=df.index, dtype=object)
    lat = df["latitude"].astype(float)
    lon = df["longitude"].astype(float)
    for box in boxes:
        mask_unassigned = result.isna()
        if not mask_unassigned.any():
            break
        mask_in_box = (
            (lat >= box.lat_min) & (lat <= box.lat_max) &
            (lon >= box.lon_min) & (lon <= box.lon_max)
            & mask_unassigned
        )
        result.loc[mask_in_box] = box.province_id
    return result


def split_archive(input_dir: Path, output_dir: Path,
                  config_path: Optional[Path] = None) -> dict[str, int]:
    """
    Baca semua CSV di input_dir, split per provinsi, tulis ke output_dir.
    Return: dict {province_id: n_rows_written}.
    """
    boxes = _load_provinces(config_path)
    LOG.info("Provinces (bbox first-match-wins order): %s",
             [b.province_id for b in boxes])

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {input_dir}")
    LOG.info("Reading %d CSV files from %s", len(csv_files), input_dir)

    # Baca + concat semua CSV (archive + nrt). Hold di memory sekali, tapi
    # untuk 110k rows × 15 cols (~10 MB) ini aman di Codespace.
    dfs: list[pd.DataFrame] = []
    for p in csv_files:
        dfs.append(load_firms_csv(p))
    df = pd.concat(dfs, ignore_index=True, sort=False)
    LOG.info("Concatenated: %d total rows", len(df))

    # Drop baris dengan NaN lat/lon (defensive — beberapa baris di NRT bisa
    # punya nilai blank kalau scan partial)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    LOG.info("After NaN coord drop: %d rows", len(df))

    # Assign per provinsi
    df["_province_id"] = assign_province(df, boxes)
    n_in_provinces = df["_province_id"].notna().sum()
    LOG.info("Hotspots dalam bbox 5 provinsi: %d (%.1f%% dari total)",
             n_in_provinces, 100 * n_in_provinces / max(1, len(df)))

    # Determine date range untuk filename
    df["acq_date"] = df["acq_date"].astype(str)
    date_min = df.loc[df["_province_id"].notna(), "acq_date"].min()
    date_max = df.loc[df["_province_id"].notna(), "acq_date"].max()
    LOG.info("Date range across all provinces: %s → %s", date_min, date_max)

    # Tulis per provinsi
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for box in boxes:
        sub = df[df["_province_id"] == box.province_id].copy()
        sub = sub.drop(columns=["_province_id"])
        if sub.empty:
            LOG.warning("%s: 0 rows after filter — skipped", box.province_id)
            summary[box.province_id] = 0
            continue
        # Filename pattern: {province_id}_{date_min}_{date_max}_archive.csv
        # preprocess.py extract province_id dari prefix (split("_")[0])
        prov_dmin = sub["acq_date"].min()
        prov_dmax = sub["acq_date"].max()
        fname = f"{box.province_id}_{prov_dmin}_{prov_dmax}_archive.csv"
        path = _safe_output_path(output_dir, fname)
        sub.to_csv(path, index=False, encoding="utf-8")
        LOG.info("  wrote %s (%d rows, %s → %s)",
                 path.name, len(sub), prov_dmin, prov_dmax)
        summary[box.province_id] = len(sub)

    return summary


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split FIRMS archive download per-provinsi untuk preprocess pipeline"
    )
    p.add_argument("--input-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "raw" / "firms_archive",
                   help="Folder berisi CSV download dari FIRMS UI")
    p.add_argument("--output-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "raw" / "firms",
                   help="Folder output (langsung dipakai preprocess.py)")
    p.add_argument("--config", type=Path,
                   default=_PROJECT_ROOT / "config" / "params.yaml")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not args.input_dir.exists():
        LOG.error("Input dir not found: %s", args.input_dir)
        LOG.error("Pastikan CSV download dari FIRMS UI sudah di-extract ke folder ini.")
        return 2

    try:
        summary = split_archive(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config if args.config.exists() else None,
        )
    except Exception as e:  # noqa: BLE001
        LOG.exception("Split failed: %s", type(e).__name__)
        return 1

    LOG.info("=== Summary ===")
    for prov, n in summary.items():
        LOG.info("  %-10s : %d rows", prov, n)
    LOG.info("Total: %d rows in 5 provinces", sum(summary.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
