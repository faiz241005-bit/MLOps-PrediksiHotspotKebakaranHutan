"""
FireGuard LK12 — Drift simulation script.

Tujuan:
    Generate dataset training "shifted" untuk mendemonstrasikan
    Continuous Training loop: ketika data berubah distribusinya, model
    yang dilatih ulang akan punya prediksi berbeda → akhirnya terlihat
    sebagai drift di Grafana heatmap.

3 skenario shift yang tersedia:
    - drought    : musim kering panjang (precip↓, temp↑, days_since_rain↑)
                   → model akan prediksi hotspot LEBIH banyak
    - wet        : musim hujan deras (precip↑, hum↑, days_since_rain↓)
                   → model akan prediksi hotspot LEBIH sedikit
    - mixed      : random noise ±15% di semua fitur weather + frp
                   → uji robustness

Usage:
    # Generate shifted features dari latest parquet
    python -m src.scripts.inject_shifted_data --scenario drought

    # Custom magnitude
    python -m src.scripts.inject_shifted_data --scenario drought --magnitude 1.5

    # Specify input file
    python -m src.scripts.inject_shifted_data \\
        --input data/features/training_dataset_2026-04-01.parquet \\
        --scenario wet

    # Dry-run (preview perubahan, jangan write file)
    python -m src.scripts.inject_shifted_data --scenario drought --dry-run

Output:
    data/features/training_dataset_<scenario>_<timestamp>.parquet

Setelah generate, jalankan untuk track via DVC:
    dvc add data/features/training_dataset_*.parquet
    git add data/features/*.dvc
    git commit -m "data(lk12): inject <scenario> drift for CT demo"
    dvc push

Security/robustness:
    - Path validation: input file harus di bawah data/features/
    - Numeric clamping: cegah nilai out-of-range (mis. negative humidity)
    - Output filename deterministic — tidak overwrite input
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"FATAL: pandas/numpy not installed → {e}", file=sys.stderr)
    sys.exit(2)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [inject_shift] %(message)s",
)
LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data" / "features"

# Kolom yang boleh di-shift per skenario
WEATHER_COLS = [
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "relative_humidity_2m_mean",
    "windspeed_10m_max",
]
FIRE_COLS = ["frp_mean", "frp_max", "frp_sum", "hotspot_count"]
ROLLING_COLS = [
    "hotspot_count_1d", "hotspot_count_3d", "hotspot_count_7d",
    "frp_mean_1d", "frp_mean_3d", "frp_mean_7d",
    "hotspot_count_lag_1d", "hotspot_count_lag_3d", "hotspot_count_lag_7d",
]

# Clamping bounds — physical reality
CLAMP_BOUNDS = {
    "precipitation_sum": (0.0, 500.0),
    "relative_humidity_2m_mean": (0.0, 100.0),
    "temperature_2m_max": (-10.0, 50.0),
    "temperature_2m_min": (-15.0, 40.0),
    "windspeed_10m_max": (0.0, 100.0),
    "days_since_rain": (0, 90),
}


# ---------------------------------------------------------------------------
# Scenario applicators
# ---------------------------------------------------------------------------
def apply_drought(df: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    """Musim kering panjang — model akan prediksi hotspot MENINGKAT."""
    out = df.copy()
    if "precipitation_sum" in out.columns:
        out["precipitation_sum"] *= max(0.0, 1.0 - 0.7 * magnitude)
    if "relative_humidity_2m_mean" in out.columns:
        out["relative_humidity_2m_mean"] *= max(0.0, 1.0 - 0.25 * magnitude)
    if "temperature_2m_max" in out.columns:
        out["temperature_2m_max"] += 3.5 * magnitude
    if "temperature_2m_min" in out.columns:
        out["temperature_2m_min"] += 2.5 * magnitude
    if "days_since_rain" in out.columns:
        # tambah hari kering
        bumped = out["days_since_rain"].astype(float) + 14 * magnitude
        out["days_since_rain"] = bumped.round().astype(int)
    # Sedikit boost ke fire features (recent fire activity meningkat saat kemarau)
    for c in FIRE_COLS:
        if c in out.columns:
            out[c] = out[c] * (1.0 + 0.35 * magnitude)
    for c in ROLLING_COLS:
        if c in out.columns:
            out[c] = out[c] * (1.0 + 0.25 * magnitude)
    return out


def apply_wet(df: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    """Musim hujan deras — model akan prediksi hotspot MENURUN."""
    out = df.copy()
    if "precipitation_sum" in out.columns:
        out["precipitation_sum"] += 25.0 * magnitude
    if "relative_humidity_2m_mean" in out.columns:
        out["relative_humidity_2m_mean"] += 15.0 * magnitude
    if "temperature_2m_max" in out.columns:
        out["temperature_2m_max"] -= 2.5 * magnitude
    if "temperature_2m_min" in out.columns:
        out["temperature_2m_min"] -= 1.5 * magnitude
    if "days_since_rain" in out.columns:
        out["days_since_rain"] = 0
    # Tekan fire activity
    for c in FIRE_COLS:
        if c in out.columns:
            out[c] = out[c] * max(0.0, 1.0 - 0.5 * magnitude)
    for c in ROLLING_COLS:
        if c in out.columns:
            out[c] = out[c] * max(0.0, 1.0 - 0.4 * magnitude)
    return out


def apply_mixed(df: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    """Random noise ±15% × magnitude di weather+fire features."""
    out = df.copy()
    noise_range = 0.15 * magnitude
    for c in WEATHER_COLS + FIRE_COLS + ROLLING_COLS:
        if c in out.columns:
            mult = 1.0 + rng.uniform(-noise_range, noise_range, size=len(out))
            out[c] = out[c] * mult
    return out


SCENARIO_FUNCS = {
    "drought": apply_drought,
    "wet": apply_wet,
    "mixed": apply_mixed,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_latest_features() -> Optional[Path]:
    if not FEATURES_DIR.is_dir():
        return None
    files = sorted(
        FEATURES_DIR.glob("training_dataset_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    # Skip kalau itu sendiri output shifted (jangan loop)
    for f in files:
        if "_drought_" in f.name or "_wet_" in f.name or "_mixed_" in f.name:
            continue
        return f
    return files[0]


def validate_input_path(path: Path) -> Path:
    """Pastikan path di bawah FEATURES_DIR (security: prevent path traversal)."""
    resolved = path.resolve()
    if not str(resolved).startswith(str(FEATURES_DIR.resolve())):
        raise ValueError(
            f"Input path harus di bawah {FEATURES_DIR}, dapat: {resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"File tidak ada: {resolved}")
    return resolved


def clamp_physical(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan nilai tetap dalam range fisik realistik."""
    for col, (lo, hi) in CLAMP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    # No negative for fire features
    for c in FIRE_COLS + ROLLING_COLS:
        if c in df.columns:
            df[c] = df[c].clip(lower=0.0)
    return df


def summarize_shift(before: pd.DataFrame, after: pd.DataFrame, cols: list[str]) -> None:
    """Print perbandingan mean per kolom (debugging visibility)."""
    LOG.info("Shift summary (mean values):")
    LOG.info("  %-32s %12s %12s %10s", "column", "before", "after", "delta%")
    for c in cols:
        if c in before.columns and c in after.columns:
            b = before[c].mean()
            a = after[c].mean()
            d = (a - b) / b * 100 if abs(b) > 1e-9 else 0.0
            LOG.info("  %-32s %12.3f %12.3f %+9.1f%%", c, b, a, d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject shifted features for CT demo")
    p.add_argument("--scenario", choices=list(SCENARIO_FUNCS.keys()),
                   default="drought", help="Tipe shift (default: drought)")
    p.add_argument("--magnitude", type=float, default=1.0,
                   help="Intensitas shift (0.5=ringan, 1.0=normal, 2.0=ekstrem)")
    p.add_argument("--input", type=Path, default=None,
                   help="Path input parquet (default: latest di data/features/)")
    p.add_argument("--output", type=Path, default=None,
                   help="Path output (default: auto-named)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview perubahan, jangan write file")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Cari input file
    if args.input:
        try:
            input_path = validate_input_path(args.input)
        except (ValueError, FileNotFoundError) as e:
            LOG.error("Invalid input: %s", e)
            return 1
    else:
        input_path = find_latest_features()
        if not input_path:
            LOG.error(
                "Tidak ada feature parquet di %s. Jalankan build_features dulu "
                "atau pass --input.", FEATURES_DIR
            )
            return 1
    LOG.info("Input  : %s", input_path)

    # Load
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:  # noqa: BLE001
        LOG.error("Gagal baca parquet: %s", e)
        return 1
    LOG.info("Loaded : %d rows × %d cols", len(df), len(df.columns))

    # Apply shift
    rng = np.random.default_rng(args.seed)
    apply_fn = SCENARIO_FUNCS[args.scenario]
    LOG.info("Apply  : scenario=%s magnitude=%.2f", args.scenario, args.magnitude)
    shifted = apply_fn(df, args.magnitude, rng)
    shifted = clamp_physical(shifted)

    # Show comparison
    summarize_shift(df, shifted, WEATHER_COLS + FIRE_COLS[:2])

    if args.dry_run:
        LOG.info("Dry-run mode — file TIDAK ditulis.")
        return 0

    # Output path
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = FEATURES_DIR / (
            f"training_dataset_{args.scenario}_{ts}.parquet"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shifted.to_parquet(output_path, index=False)
    except Exception as e:  # noqa: BLE001
        LOG.error("Gagal write parquet: %s", e)
        return 1
    LOG.info("Output : %s (%.1f MB)",
             output_path,
             output_path.stat().st_size / 1e6)

    # Print next steps
    print()
    print("=" * 64)
    print("Next steps untuk track via DVC:")
    print(f"  dvc add {output_path.relative_to(REPO_ROOT)}")
    print(f"  git add {output_path.relative_to(REPO_ROOT)}.dvc")
    print(f"  git commit -m 'data(lk12): inject {args.scenario} drift'")
    print(f"  dvc push")
    print()
    print("Lalu trigger retraining:")
    print(f"  python -m src.scripts.auto_retrain "
          f"--reason 'drift_simulation:{args.scenario}'")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
