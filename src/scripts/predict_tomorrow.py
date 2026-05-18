"""
Predict Tomorrow — Demo end-to-end FireGuard prediction per provinsi.

Workflow:
    1. Fetch real data terbaru (FIRMS NRT last 7 days + Open-Meteo Archive 14 days)
    2. Run preprocess (join FIRMS + weather)
    3. Run build_features (rolling, lag, target)
    4. For each provinsi, ambil row latest available
    5. Call FastAPI /predict (api-service di Docker Compose)
    6. Print hasil dalam tabel rapi

Usage:
    # Run dengan fetch fresh (3-5 menit, butuh NASA_FIRMS_API_KEY di .env)
    python -m src.scripts.predict_tomorrow

    # Skip fetch — pakai data existing di data/features/ (cepat, untuk demo)
    python -m src.scripts.predict_tomorrow --skip-fetch

    # Custom api URL (default: http://localhost:8000)
    python -m src.scripts.predict_tomorrow --api-url http://localhost:8000

Security & memory:
    - Tidak ada credential di-log
    - API timeout 10 detik per request
    - Stream-write per subprocess call (tidak hold output di RAM)
    - Pakai existing src/data dan src/features modules — single source of truth
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Province display names — match dengan training data
_PROVINCE_DISPLAY = {
    "riau":    "Riau",
    "kalteng": "Kalimantan Tengah",
    "kalbar":  "Kalimantan Barat",
    "sumsel":  "Sumatera Selatan",
    "jambi":   "Jambi",
}

# Features yang dikirim ke /predict (27 fitur, sesuai schema FastAPI)
_FEATURE_COLUMNS = [
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

# Kolom yang harus di-cast ke int (sesuai signature model)
_INT32_COLS = {"month", "day_of_year"}
_INT64_COLS = {
    "hotspot_count", "n_daytime", "n_nighttime", "n_confidence_high",
    "days_since_rain",
}


# ---------------------------------------------------------------------------
# Pipeline subprocess wrappers
# ---------------------------------------------------------------------------
def _run_step(cmd: list[str], description: str) -> bool:
    """Run subprocess + log progress. Return True kalau sukses."""
    LOG.info("▶ %s", description)
    try:
        result = subprocess.run(
            cmd, cwd=_PROJECT_ROOT,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            LOG.error("  ✗ FAILED")
            LOG.error("  stdout: %s", result.stdout[-500:])
            LOG.error("  stderr: %s", result.stderr[-500:])
            return False
        LOG.info("  ✓ done")
        return True
    except subprocess.TimeoutExpired:
        LOG.error("  ✗ TIMEOUT (>10 menit)")
        return False
    except Exception as e:  # noqa: BLE001
        LOG.error("  ✗ EXCEPTION: %s", type(e).__name__)
        return False


def fetch_latest_data(days_back: int = 7) -> bool:
    """Fetch FIRMS NRT + Open-Meteo Archive untuk last N days."""
    today = date.today()
    # Open-Meteo Archive punya lag ~5 hari; pakai cutoff hari ini - 2
    end_date = today - timedelta(days=2)
    start_date = end_date - timedelta(days=days_back)

    LOG.info("Fetch window: %s → %s (%d hari)", start_date, end_date, days_back + 1)

    # Bulk fetch FIRMS + weather sekaligus
    return _run_step(
        [
            sys.executable, "-m", "src.data.bulk_fetch",
            "--start-date", start_date.strftime("%Y-%m-%d"),
            "--end-date", end_date.strftime("%Y-%m-%d"),
            "--provinces", "all",
            "--sleep-between-chunks", "0.5",
        ],
        f"Fetch FIRMS + Weather ({days_back} hari)",
    )


def run_preprocess() -> bool:
    return _run_step(
        [sys.executable, "-m", "src.data.preprocess"],
        "Preprocess (join FIRMS × weather)",
    )


def run_build_features() -> bool:
    return _run_step(
        [sys.executable, "-m", "src.features.build_features"],
        "Build features (rolling, lag, target)",
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def load_latest_features() -> Optional[pd.DataFrame]:
    """Load training_dataset_*.parquet terbaru di data/features/."""
    features_dir = _PROJECT_ROOT / "data" / "features"
    files = sorted(features_dir.glob("training_dataset_*.parquet"))
    if not files:
        LOG.error("Tidak ada training_dataset_*.parquet di %s", features_dir)
        return None
    latest = files[-1]
    LOG.info("Pakai features file: %s", latest.name)
    df = pd.read_parquet(latest)
    return df


def get_latest_row_per_province(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Untuk tiap provinsi, ambil row dengan date terbaru yang ada hotspot."""
    result: dict[str, pd.Series] = {}
    for prov_id in df["province_id"].unique():
        sub = df[df["province_id"] == prov_id].sort_values("date")
        if sub.empty:
            continue
        # Ambil row TERAKHIR (latest date) per provinsi
        result[prov_id] = sub.iloc[-1]
    return result


def features_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert pandas Series jadi dict siap-kirim ke /predict (dengan type casting)."""
    payload: dict[str, Any] = {}
    for col in _FEATURE_COLUMNS:
        if col not in row.index:
            LOG.warning("Missing feature %s di row, default 0", col)
            payload[col] = 0
            continue
        v = row[col]
        if pd.isna(v):
            v = 0
        # Cast sesuai expected type model
        if col in _INT32_COLS or col in _INT64_COLS:
            payload[col] = int(v)
        else:
            payload[col] = float(v)
    return payload


def call_predict(api_url: str, payload: dict[str, Any], timeout: int = 10) -> Optional[dict]:
    """POST /predict ke api-service. Return JSON response atau None kalau error."""
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=timeout)
        if not r.ok:
            LOG.error("HTTP %s: %s", r.status_code, r.text[:200])
            return None
        return r.json()
    except Exception as e:  # noqa: BLE001
        LOG.error("Request failed: %s", type(e).__name__)
        return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_header(api_url: str, model_info: Optional[dict]) -> None:
    print()
    print("=" * 70)
    print("🔥 FireGuard — Prediksi Hotspot Besok per Provinsi")
    print("=" * 70)
    print(f"  API endpoint     : {api_url}")
    if model_info:
        print(f"  Model            : {model_info.get('name')} v{model_info.get('version')} "
              f"({model_info.get('stage')})")
    print(f"  Waktu eksekusi   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}")
    print("=" * 70)
    print()


def print_results(results: dict[str, dict]) -> None:
    if not results:
        print("\n⚠️  Tidak ada hasil prediksi.\n")
        return

    # Header tabel
    print(f"{'Provinsi':<22} {'Today':>10} {'Tomorrow':>12} {'Risk':<12}")
    print("-" * 70)

    total_today = 0
    total_tomorrow = 0.0

    for prov_id, data in results.items():
        prov_name = _PROVINCE_DISPLAY.get(prov_id, prov_id)
        today_count = data.get("today_count", 0)
        pred = data.get("prediction")
        if pred is None:
            print(f"{prov_name:<22} {today_count:>10} {'—':>12} {'❌ ERROR':<12}")
            continue
        tomorrow = pred.get("hotspot_count_tomorrow", 0)
        risk_label = pred.get("risk_label", "—")
        risk_emoji = {"Aman": "🟢", "Waspada": "🟡", "Bahaya": "🔴"}.get(risk_label, "⚪")
        print(f"{prov_name:<22} {today_count:>10} {tomorrow:>12.1f} "
              f"{risk_emoji} {risk_label:<10}")
        total_today += today_count
        total_tomorrow += tomorrow

    print("-" * 70)
    print(f"{'TOTAL Indonesia':<22} {total_today:>10} {total_tomorrow:>12.1f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict tomorrow's hotspots per province (demo end-to-end)"
    )
    parser.add_argument("--api-url", default=os.getenv("FIREGUARD_API_URL", "http://localhost:8000"),
                        help="FastAPI api-service URL (default: http://localhost:8000)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetch — pakai existing data/features/")
    parser.add_argument("--days-back", type=int, default=7,
                        help="Berapa hari data ke belakang yang di-fetch (default 7)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Step 1-3: Pipeline data (opsional skip)
    if not args.skip_fetch:
        if not fetch_latest_data(days_back=args.days_back):
            LOG.error("Data fetch gagal — coba --skip-fetch kalau mau pakai data lama")
            return 1
        if not run_preprocess():
            LOG.error("Preprocess gagal")
            return 1
        if not run_build_features():
            LOG.error("Build features gagal")
            return 1
    else:
        LOG.info("Skip fetch — pakai data existing di data/features/")

    # Step 4: Load features + ambil latest row per provinsi
    df = load_latest_features()
    if df is None or df.empty:
        LOG.error("Features dataset kosong / tidak ada")
        return 1

    latest_rows = get_latest_row_per_province(df)
    LOG.info("Provinces dengan data: %s", list(latest_rows.keys()))

    # Step 5: Cek api-service reachable
    try:
        info_resp = requests.get(f"{args.api_url}/model-info", timeout=5)
        model_info = info_resp.json() if info_resp.ok else None
        if not model_info or not model_info.get("ready"):
            LOG.error("api-service tidak reachable atau model belum loaded di %s", args.api_url)
            LOG.error("Pastikan: docker compose ps → fireguard-api status (healthy)")
            return 1
    except Exception as e:  # noqa: BLE001
        LOG.error("Gagal connect ke api-service: %s", type(e).__name__)
        return 1

    print_header(args.api_url, model_info)

    # Step 6: Loop predict per provinsi
    results: dict[str, dict] = {}
    for prov_id, row in latest_rows.items():
        payload = features_to_payload(row)
        today_count = int(row.get("hotspot_count", 0))
        LOG.info("Predict %s (today_count=%d, date=%s)...",
                 prov_id, today_count, row.get("date"))
        pred = call_predict(args.api_url, payload)
        results[prov_id] = {
            "today_count": today_count,
            "prediction": pred,
            "date_today": row.get("date"),
        }

    # Step 7: Tampilkan hasil
    print_results(results)

    # Optional: write JSON output untuk consumption lain
    output_file = _PROJECT_ROOT / "data" / "predictions_latest.json"
    output_data = {
        "executed_at": datetime.now().isoformat(),
        "api_url": args.api_url,
        "model": model_info,
        "results": {
            prov: {
                "today_count": r["today_count"],
                "date_today": str(r["date_today"]),
                "prediction": r["prediction"],
            } for prov, r in results.items()
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, indent=2, default=str))
    LOG.info("Hasil ter-simpan ke %s", output_file.relative_to(_PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
