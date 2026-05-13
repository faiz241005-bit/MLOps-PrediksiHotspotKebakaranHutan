"""
Synthetic features dataset generator — fallback untuk CI/CD ketika real data
(dari DVC remote) tidak tersedia di GitHub Actions runner.

Tugas LK08: workflow Actions butuh dataset untuk training. Karena DVC remote
kita lokal (atau optional cloud), CI run kadang tidak punya akses ke real data.
Synthetic data men-demonstrate workflow end-to-end tanpa external dependency.

Output schema match dengan build_features.py — bisa langsung di-konsumsi train.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Province defaults — match config/params.example.yaml
_DEFAULT_PROVINCES = ["riau", "kalteng", "kalbar", "sumsel", "jambi"]


def generate(
    n_provinces: int = 5, n_days: int = 60,
    seed: int = 42, start_date: str = "2026-03-01"
) -> pd.DataFrame:
    """
    Generate synthetic training features dataset.

    Schema match dengan output build_features.py (27 fitur + 2 target).
    Distribusi:
        - hotspot_count_tomorrow ~ Poisson(lambda yang varies dengan musim)
        - risk_level dihitung dari count: 0 (aman), 1 (waspada 1-10), 2 (bahaya >10)
        - Cuaca: temperature, humidity, precipitation di range realistic Indonesia
    """
    rng = np.random.default_rng(seed)
    rows = []

    for p_idx in range(min(n_provinces, len(_DEFAULT_PROVINCES))):
        province = _DEFAULT_PROVINCES[p_idx]
        # Setiap provinsi punya "fire risk baseline" sendiri (Riau & Kalteng lebih tinggi)
        province_baseline = {"riau": 8, "kalteng": 10, "kalbar": 4, "sumsel": 6, "jambi": 5}[province]

        for day_offset in range(n_days):
            date = pd.Timestamp(start_date) + pd.Timedelta(days=day_offset)
            month = date.month

            # Seasonal factor: peak musim kemarau Juli-September (multiplier 2x)
            seasonal_mult = 2.0 if month in (7, 8, 9) else 0.5 if month in (12, 1, 2) else 1.0

            lam = max(0.5, province_baseline * seasonal_mult)
            hotspot_count = int(rng.poisson(lam))
            frp_mean = max(5.0, rng.normal(15.0, 5.0)) if hotspot_count > 0 else 0.0

            # Cuaca realistic
            temp_max = rng.normal(32 if seasonal_mult > 1 else 29, 2)
            temp_min = temp_max - rng.uniform(6, 10)
            precipitation = max(0, rng.normal(2 if seasonal_mult > 1 else 10, 4))
            humidity = max(40, min(95, rng.normal(75 if precipitation > 1 else 65, 8)))

            # Tomorrow's prediction — slightly correlated dengan today
            tomorrow_lam = max(0.5, lam * rng.uniform(0.7, 1.3))
            tomorrow_count = int(rng.poisson(tomorrow_lam))

            # Risk level berdasarkan tomorrow's count
            if tomorrow_count == 0:
                risk_level = 0
            elif tomorrow_count <= 10:
                risk_level = 1
            else:
                risk_level = 2

            rows.append({
                "province_id": province,
                "date": date,
                "hotspot_count": hotspot_count,
                "frp_mean": frp_mean,
                "frp_max": frp_mean * 1.5 if hotspot_count > 0 else 0.0,
                "frp_sum": frp_mean * hotspot_count,
                "n_daytime": int(hotspot_count * rng.uniform(0.4, 0.7)),
                "n_nighttime": hotspot_count - int(hotspot_count * rng.uniform(0.4, 0.7)),
                "n_confidence_high": int(hotspot_count * rng.uniform(0.2, 0.5)),
                "temperature_2m_max": temp_max,
                "temperature_2m_min": temp_min,
                "precipitation_sum": precipitation,
                "windspeed_10m_max": rng.uniform(5, 18),
                "winddirection_10m_dominant": rng.uniform(0, 360),
                "relative_humidity_2m_mean": humidity,
                "month": month,
                "day_of_year": date.dayofyear,
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                # Rolling features (approximated)
                "hotspot_count_1d": float(hotspot_count),
                "hotspot_count_3d": float(hotspot_count * 3),
                "hotspot_count_7d": float(hotspot_count * 7),
                "frp_mean_1d": frp_mean,
                "frp_mean_3d": frp_mean,
                "frp_mean_7d": frp_mean,
                # Lag features
                "hotspot_count_lag_1d": float(rng.integers(0, max(1, lam * 2))),
                "hotspot_count_lag_3d": float(rng.integers(0, max(1, lam * 2))),
                "hotspot_count_lag_7d": float(rng.integers(0, max(1, lam * 2))),
                "days_since_rain": int(0 if precipitation > 1 else rng.integers(1, 8)),
                # Targets
                "hotspot_count_tomorrow": float(tomorrow_count),
                "risk_level": risk_level,
            })

    df = pd.DataFrame(rows)
    LOG.info("Generated synthetic features: %d rows × %d cols", len(df), len(df.columns))
    LOG.info("Risk distribution:\n%s", df["risk_level"].value_counts().sort_index().to_string())
    return df


def write_synthetic(out_dir: Path,
                    n_provinces: int = 5, n_days: int = 60, seed: int = 42,
                    now_utc: Optional[datetime] = None) -> Path:
    """Generate + tulis ke data/features/training_dataset_*.parquet."""
    df = generate(n_provinces=n_provinces, n_days=n_days, seed=seed)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    now = (now_utc or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"training_dataset_{now}_UTC.parquet"

    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    LOG.info("Wrote synthetic dataset: %s (%d rows)", out_path, len(df))
    return out_path


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic FireGuard features dataset (LK08 CI fallback)"
    )
    p.add_argument("--n-provinces", type=int, default=5)
    p.add_argument("--n-days", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=_PROJECT_ROOT / "data" / "features")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    path = write_synthetic(
        args.output_dir, n_provinces=args.n_provinces,
        n_days=args.n_days, seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
