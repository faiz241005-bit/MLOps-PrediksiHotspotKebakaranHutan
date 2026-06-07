"""
FireGuard LK11 — Load test untuk demo observability.

Tujuan:
  - Generate trafik konstan ke metrics-proxy /invocations
  - Variasikan feature input → variasi nilai prediksi → drift terlihat di heatmap
  - Tampilkan progress di terminal

Usage:
    # Default: 50 request, concurrency 5, target localhost:9000
    python -m src.scripts.load_test_lk11

    # Custom: 500 request, 20 concurrent, 30 detik durasi maksimum
    python -m src.scripts.load_test_lk11 --requests 500 --concurrency 20 --max-seconds 30

    # Target eksternal (mis. Codespace forwarded port)
    python -m src.scripts.load_test_lk11 --url http://localhost:9000

Tip: jalankan 2 fase berbeda untuk demonstrasi drift detection:
    # Fase 1: low-fire scenario (prediksi kecil)
    python -m src.scripts.load_test_lk11 --requests 100 --scenario low

    # Fase 2: high-fire scenario (prediksi besar) — heatmap akan shift!
    python -m src.scripts.load_test_lk11 --requests 100 --scenario high
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s :: %(message)s",
)
LOG = logging.getLogger("loadtest")

# ---------------------------------------------------------------------------
# Feature schema (27 features — match dengan model FireGuard)
# ---------------------------------------------------------------------------
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

# Untuk skema int yang strict — match model signature
_INT_COLS = {
    "month", "day_of_year",
    "hotspot_count", "n_daytime", "n_nighttime", "n_confidence_high",
    "days_since_rain",
}


# ---------------------------------------------------------------------------
# Scenario generators — variasikan input agar prediksi punya distribusi
# ---------------------------------------------------------------------------
def gen_features(scenario: str) -> list:
    """Generate 27-feature row dengan distribusi berbeda per scenario."""
    if scenario == "low":
        # Skenario kemarau ringan / awal musim
        h = random.randint(0, 5)
        frp = random.uniform(0, 20)
        precip = random.uniform(2, 10)
        rain = random.randint(0, 3)
    elif scenario == "high":
        # Skenario puncak kebakaran (musim kering panjang)
        h = random.randint(30, 100)
        frp = random.uniform(50, 200)
        precip = 0.0
        rain = random.randint(10, 30)
    else:  # mixed (default)
        h = random.randint(0, 50)
        frp = random.uniform(5, 100)
        precip = random.uniform(0, 15)
        rain = random.randint(0, 15)

    month = random.randint(1, 12)
    doy = random.randint(1, 365)

    row = {
        "hotspot_count": h,
        "frp_mean": frp,
        "frp_max": frp * random.uniform(1.0, 2.5),
        "frp_sum": frp * h,
        "n_daytime": int(h * 0.6),
        "n_nighttime": int(h * 0.4),
        "n_confidence_high": int(h * 0.3),
        "temperature_2m_max": random.uniform(28, 36),
        "temperature_2m_min": random.uniform(20, 26),
        "precipitation_sum": precip,
        "windspeed_10m_max": random.uniform(5, 25),
        "winddirection_10m_dominant": random.uniform(0, 360),
        "relative_humidity_2m_mean": random.uniform(50, 90),
        "month": month,
        "day_of_year": doy,
        "month_sin": 0.0,  # sederhanakan
        "month_cos": 0.0,
        "hotspot_count_1d": h * random.uniform(0.5, 1.5),
        "hotspot_count_3d": h * random.uniform(1.5, 3.5),
        "hotspot_count_7d": h * random.uniform(3, 7),
        "frp_mean_1d": frp * random.uniform(0.7, 1.3),
        "frp_mean_3d": frp * random.uniform(0.7, 1.3),
        "frp_mean_7d": frp * random.uniform(0.7, 1.3),
        "hotspot_count_lag_1d": h * random.uniform(0.5, 1.5),
        "hotspot_count_lag_3d": h * random.uniform(0.5, 1.5),
        "hotspot_count_lag_7d": h * random.uniform(0.5, 1.5),
        "days_since_rain": rain,
    }

    # Ensure int columns are int
    values = []
    for col in FEATURE_COLUMNS:
        v = row[col]
        if col in _INT_COLS:
            values.append(int(v))
        else:
            values.append(float(v))
    return values


def build_payload(scenario: str) -> dict:
    """MLflow dataframe_split format."""
    return {
        "dataframe_split": {
            "columns": FEATURE_COLUMNS,
            "data": [gen_features(scenario)],
        }
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class Result:
    status: int
    elapsed_ms: float
    prediction: Optional[float] = None
    error: Optional[str] = None


async def hit_one(client: httpx.AsyncClient, scenario: str) -> Result:
    started = time.perf_counter()
    try:
        r = await client.post("/invocations", json=build_payload(scenario))
        elapsed = (time.perf_counter() - started) * 1000
        pred = None
        if r.status_code == 200:
            try:
                pred = float(r.json().get("predictions", [None])[0])
            except (ValueError, TypeError, IndexError):
                pred = None
        return Result(status=r.status_code, elapsed_ms=elapsed, prediction=pred)
    except httpx.HTTPError as e:
        elapsed = (time.perf_counter() - started) * 1000
        return Result(status=0, elapsed_ms=elapsed, error=type(e).__name__)


async def run_load(
    url: str,
    total: int,
    concurrency: int,
    scenario: str,
    max_seconds: Optional[float],
) -> list[Result]:
    timeout = httpx.Timeout(20.0, connect=5.0)
    limits = httpx.Limits(
        max_connections=concurrency * 2,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(
        base_url=url, timeout=timeout, limits=limits
    ) as client:
        # Cek dulu apakah proxy hidup
        try:
            ping = await client.get("/health", timeout=5.0)
            LOG.info("Health check: HTTP %s — %s", ping.status_code, ping.text[:80])
        except httpx.HTTPError as e:
            LOG.error("Proxy unreachable di %s — %s", url, type(e).__name__)
            return []

        sem = asyncio.Semaphore(concurrency)
        results: list[Result] = []
        started_run = time.perf_counter()

        async def worker(_i: int):
            async with sem:
                # Stop kalau lewat max-seconds
                if max_seconds and (time.perf_counter() - started_run) > max_seconds:
                    return None
                res = await hit_one(client, scenario)
                results.append(res)
                # Live progress tiap 25 request
                if len(results) % 25 == 0:
                    succ = sum(1 for r in results if r.status == 200)
                    LOG.info("Progress: %d/%d (succ=%d)", len(results), total, succ)
                return res

        await asyncio.gather(*[worker(i) for i in range(total)])
        return results


def summarize(results: list[Result], elapsed_s: float) -> None:
    if not results:
        LOG.error("No results collected.")
        return

    n = len(results)
    succ = [r for r in results if r.status == 200]
    fail = [r for r in results if r.status != 200]
    latencies = sorted(r.elapsed_ms for r in succ)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(int(p * len(latencies)), len(latencies) - 1)
        return latencies[idx]

    preds = [r.prediction for r in succ if r.prediction is not None]

    print("\n" + "=" * 60)
    print("  LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"  Total requests        : {n}")
    print(f"  Successful (HTTP 200) : {len(succ)} ({100*len(succ)/n:.1f}%)")
    print(f"  Failed                : {len(fail)}")
    print(f"  Wallclock             : {elapsed_s:.2f}s")
    print(f"  Throughput            : {n/elapsed_s:.2f} req/s")
    if latencies:
        print(f"  Latency p50/p95/p99   : "
              f"{pct(0.50):.1f} / {pct(0.95):.1f} / {pct(0.99):.1f} ms")
        print(f"  Latency min/max       : {latencies[0]:.1f} / {latencies[-1]:.1f} ms")
    if preds:
        preds_sorted = sorted(preds)
        n_p = len(preds_sorted)
        print(f"  Predictions count     : {n_p}")
        print(f"  Prediction min/median/max : "
              f"{preds_sorted[0]:.2f} / "
              f"{preds_sorted[n_p // 2]:.2f} / "
              f"{preds_sorted[-1]:.2f}")
    print("=" * 60)
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FireGuard LK11 load test")
    p.add_argument("--url", default="http://localhost:9000",
                   help="URL metrics-proxy (default: %(default)s)")
    p.add_argument("--requests", type=int, default=50,
                   help="Total request (default: %(default)s)")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Concurrent in-flight (default: %(default)s)")
    p.add_argument("--scenario", choices=["low", "high", "mixed"], default="mixed",
                   help="Distribusi input features (default: %(default)s)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Hard time limit (detik)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    LOG.info(
        "Mulai load test: %d req, concurrency=%d, scenario=%s, target=%s",
        args.requests, args.concurrency, args.scenario, args.url,
    )
    started = time.perf_counter()
    results = asyncio.run(
        run_load(
            url=args.url,
            total=args.requests,
            concurrency=args.concurrency,
            scenario=args.scenario,
            max_seconds=args.max_seconds,
        )
    )
    elapsed = time.perf_counter() - started
    summarize(results, elapsed)

    # Exit non-zero kalau ada gagal — biar bisa dipakai di CI
    if not results or any(r.status != 200 for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
