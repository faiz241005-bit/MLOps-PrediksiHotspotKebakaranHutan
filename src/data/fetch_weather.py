"""
Open-Meteo Weather Fetcher — fetch cuaca harian per provinsi.

Tugas LK04: ambil data cuaca (Open-Meteo Forecast API) sebagai input
fitur untuk model FireGuard. Tidak butuh API key.

Strategi koordinat:
    Open-Meteo butuh satu titik (lat, lon), bukan bbox. Kita pakai
    CENTROID dari bbox provinsi sebagai approximation. Untuk produksi
    yang lebih akurat, bisa di-extend ke multi-point sampling.

Security & resource hygiene sama seperti fetch_firms.py:
    - URL allow-list (cegah SSRF)
    - Timeout + retry exponential backoff
    - Session dengan context manager
    - Path traversal guard
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOG = logging.getLogger(__name__)

_ALLOWED_HOSTS = frozenset({"api.open-meteo.com"})
_DEFAULT_TIMEOUT_S = 30
_MAX_RETRIES = 3
_USER_AGENT = "FireGuard/0.1 (+education)"

_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "winddirection_10m_dominant",
    "relative_humidity_2m_mean",
]


@dataclass(frozen=True)
class WeatherFetchSpec:
    """Parameter satu kali fetch cuaca."""
    province_id: str
    bbox: tuple[float, float, float, float]
    past_days: int = 7
    forecast_days: int = 1


def _bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    lat_min, lon_min, lat_max, lon_max = bbox
    return ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)


def _validate_url(url: str) -> None:
    host = urlparse(url).hostname
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"Disallowed host: {host!r}; allowed={sorted(_ALLOWED_HOSTS)}")


def _build_url(lat: float, lon: float, past_days: int, forecast_days: int) -> str:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
    if not (0 <= past_days <= 92):
        raise ValueError(f"past_days must be 0..92 (Open-Meteo free tier), got {past_days}")
    if not (1 <= forecast_days <= 16):
        raise ValueError(f"forecast_days must be 1..16, got {forecast_days}")
    daily = ",".join(_DAILY_VARS)
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&daily={daily}"
        f"&timezone=Asia%2FJakarta"
        f"&past_days={past_days}&forecast_days={forecast_days}"
    )


def _safe_output_path(out_dir: Path, filename: str) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = (out_dir / filename).resolve()
    if not str(candidate).startswith(str(out_dir) + os.sep):
        raise RuntimeError(f"Refusing to write outside of {out_dir}: {candidate}")
    return candidate


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _http_get_json(url: str, timeout: int = _DEFAULT_TIMEOUT_S) -> dict:
    _validate_url(url)
    with requests.Session() as session:
        session.headers.update({"User-Agent": _USER_AGENT, "Accept": "application/json"})
        resp = session.get(url, timeout=timeout, allow_redirects=False)
        LOG.debug("Open-Meteo GET host=%s status=%s bytes=%s",
                  urlparse(url).hostname, resp.status_code, len(resp.content))
        resp.raise_for_status()
        return resp.json()


def _payload_to_dataframe(payload: dict, province_id: str) -> pd.DataFrame:
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        raise RuntimeError(f"Open-Meteo response malformed for {province_id}: missing 'daily.time'")
    df = pd.DataFrame(daily)
    df["province_id"] = province_id
    df["fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def fetch_one(spec: WeatherFetchSpec, out_dir: Path,
              now_utc: Optional[datetime] = None) -> Path:
    """Fetch cuaca satu provinsi, simpan CSV dengan nama ber-timestamp."""
    lat, lon = _bbox_centroid(spec.bbox)
    url = _build_url(lat, lon, spec.past_days, spec.forecast_days)

    LOG.info("Fetching weather province=%s lat=%.4f lon=%.4f past=%d forecast=%d",
             spec.province_id, lat, lon, spec.past_days, spec.forecast_days)
    try:
        payload = _http_get_json(url)
    except RetryError as e:
        raise RuntimeError(f"Weather fetch failed after retries for {spec.province_id}") from e

    df = _payload_to_dataframe(payload, spec.province_id)
    if df.empty:
        raise RuntimeError(f"Empty weather data for {spec.province_id}")

    now = (now_utc or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    fname = f"{spec.province_id}_{now}_UTC.csv"
    out_path = _safe_output_path(out_dir, fname)

    df.to_csv(out_path, index=False, encoding="utf-8")
    LOG.info("Wrote %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
    return out_path


def fetch_many(specs: Iterable[WeatherFetchSpec], out_dir: Path) -> list[Path]:
    written: list[Path] = []
    errors: list[tuple[str, str]] = []
    for spec in specs:
        try:
            written.append(fetch_one(spec, out_dir=out_dir))
        except Exception as e:  # noqa: BLE001
            LOG.error("Weather fetch failed for %s: %s", spec.province_id, e)
            errors.append((spec.province_id, str(e)))
    if errors:
        LOG.warning("Completed with %d failures: %s", len(errors), errors)
    return written


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch Open-Meteo daily weather per provinsi (FireGuard LK04)"
    )
    p.add_argument("--province", required=True, help="Province ID")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"))
    p.add_argument("--past-days", type=int, default=7)
    p.add_argument("--forecast-days", type=int, default=1)
    p.add_argument("--output-dir", default="data/raw/weather")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    spec = WeatherFetchSpec(
        province_id=args.province,
        bbox=tuple(args.bbox),
        past_days=args.past_days,
        forecast_days=args.forecast_days,
    )
    try:
        path = fetch_one(spec, out_dir=Path(args.output_dir))
        print(path)
        return 0
    except Exception as e:  # noqa: BLE001
        LOG.error("Weather fetch failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
