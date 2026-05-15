"""
Bulk Historical Fetcher — FIRMS + Open-Meteo Archive untuk training data.

Berbeda dengan fetch_firms.py/fetch_weather.py (LK04) yang hanya ambil 2-7 hari
data terkini, script ini meng-cover RANGE WAKTU SEMBARANG (default skenario:
1 tahun historical) untuk men-train model dengan dataset yang memadai.

Strategi:
    FIRMS: NASA FIRMS API hard-limit 10 hari per call → chunking otomatis,
           pakai parameter start_date di URL untuk historical fetch.
    Weather: Open-Meteo Archive endpoint (archive-api.open-meteo.com) yang
             support range tanggal tanpa batas → satu call per provinsi.

Security & memory hygiene (sama pattern dengan fetch_firms.py LK04):
    - URL allow-list (cegah SSRF)
    - Date format strict YYYY-MM-DD (cegah injection di URL path)
    - Path traversal guard di filename
    - Exponential backoff retry untuk transient API failure
    - Session dengan context manager (auto-close socket)
    - Tidak ada credential / API key yang di-log
    - Stream-write per chunk; tidak hold semua CSV di RAM sekaligus

CLI:
    python -m src.data.bulk_fetch \\
        --start-date 2025-05-15 --end-date 2026-05-15 \\
        --provinces all

    # Untuk window > 60 hari: NASA FIRMS NRT tidak punya data lama, ganti
    # sensor ke Standard Processing (SP):
    python -m src.data.bulk_fetch \\
        --start-date 2025-05-15 --end-date 2026-05-15 \\
        --provinces all --sensor VIIRS_SNPP_SP
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Security constants ------------------------------------------------------
_ALLOWED_HOSTS = frozenset({
    "firms.modaps.eosdis.nasa.gov",
    "api.open-meteo.com",          # Forecast API (kalau perlu fallback)
    "archive-api.open-meteo.com",  # Archive API untuk historical
})
_DEFAULT_TIMEOUT_S = 30
_MAX_RETRIES = 3
_USER_AGENT = "FireGuard/0.1 (+bulk-historical)"
_FIRMS_MAX_DAYS_PER_CALL = 10  # NASA FIRMS hard limit, tidak bisa dilampaui

# Daily weather vars — sama persis dengan fetch_weather.py supaya preprocess
# bisa join hasil bulk_fetch dengan hasil fetch_weather tanpa drift schema.
_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "winddirection_10m_dominant",
    "relative_humidity_2m_mean",
]

# Hardcoded default fallback kalau config/params.yaml tidak ada
_DEFAULT_PROVINCES = {
    "riau":    [0.0, 100.0, 4.5, 106.5],
    "kalteng": [-3.5, 110.5, 1.5, 116.5],
    "kalbar":  [-3.0, 108.0, 2.5, 118.0],
    "sumsel":  [-5.5, 102.0, -1.0, 108.5],
    "jambi":   [-3.0, 101.0, -0.5, 105.0],
}


# ---------------------------------------------------------------------------
# Date window utilities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DateWindow:
    """[start, end] inclusive, both YYYY-MM-DD dates."""
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) > end ({self.end})")
        if self.end > date.today() + timedelta(days=1):
            raise ValueError(f"end ({self.end}) is in the future")

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def _validate_date(s: str) -> date:
    """Parse YYYY-MM-DD strictly. Reject any other format (anti-injection di URL)."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date {s!r}; expected YYYY-MM-DD") from e


def _chunk_window(window: DateWindow, max_days_per_chunk: int) -> list[DateWindow]:
    """Split window jadi list of chunks, masing-masing maksimum max_days_per_chunk hari."""
    chunks: list[DateWindow] = []
    cur = window.start
    while cur <= window.end:
        end = min(cur + timedelta(days=max_days_per_chunk - 1), window.end)
        chunks.append(DateWindow(start=cur, end=end))
        cur = end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------
def _validate_url(url: str) -> None:
    host = urlparse(url).hostname
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"Disallowed host: {host!r}; allowed={sorted(_ALLOWED_HOSTS)}")


def _safe_output_path(out_dir: Path, filename: str) -> Path:
    """Cegah path traversal — pastikan resolved path tetap di dalam out_dir."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = (out_dir / filename).resolve()
    out_str = str(out_dir)
    if not (str(candidate) == out_str or str(candidate).startswith(out_str + os.sep)):
        raise ValueError(f"Unsafe filename {filename!r}")
    return candidate


# ---------------------------------------------------------------------------
# HTTP helpers — retry + allow-list + timeout
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, requests.Timeout)),
    reraise=False,
)
def _http_get_text(url: str, timeout: int = _DEFAULT_TIMEOUT_S) -> str:
    _validate_url(url)
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/csv"}
    with requests.Session() as session:
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        LOG.debug("FIRMS GET host=%s status=%s bytes=%s",
                  urlparse(url).hostname, resp.status_code, len(resp.content))
        return resp.text


@retry(
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, requests.Timeout)),
    reraise=False,
)
def _http_get_json(url: str, timeout: int = _DEFAULT_TIMEOUT_S) -> dict:
    _validate_url(url)
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    with requests.Session() as session:
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        LOG.debug("Archive GET host=%s status=%s bytes=%s",
                  urlparse(url).hostname, resp.status_code, len(resp.content))
        return resp.json()


# ---------------------------------------------------------------------------
# NASA FIRMS — chunked historical fetch
# ---------------------------------------------------------------------------
def _build_firms_url(api_key: str, sensor: str,
                     bbox: tuple[float, float, float, float],
                     day_range: int, start_date: date) -> str:
    """
    Build URL untuk NASA FIRMS Area API dengan parameter start_date (historical).
    Format: .../api/area/csv/{KEY}/{SENSOR}/{COORDS}/{DAYS}/{YYYY-MM-DD}
    """
    if not api_key.strip():
        raise ValueError("api_key is empty — set NASA_FIRMS_API_KEY env var")
    if not 1 <= day_range <= _FIRMS_MAX_DAYS_PER_CALL:
        raise ValueError(
            f"day_range must be 1..{_FIRMS_MAX_DAYS_PER_CALL}, got {day_range}"
        )
    lat_min, lon_min, lat_max, lon_max = bbox
    return (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{api_key}/{sensor}/{lon_min},{lat_min},{lon_max},{lat_max}/"
        f"{day_range}/{start_date.strftime('%Y-%m-%d')}"
    )


def fetch_firms_window(province_id: str,
                       bbox: tuple[float, float, float, float],
                       sensor: str,
                       window: DateWindow,
                       api_key: str,
                       out_dir: Path,
                       sleep_between_chunks_s: float = 1.0) -> list[Path]:
    """
    Fetch FIRMS hotspots untuk full window, di-chunk per 10 hari.

    Returns:
        List path file CSV yang berhasil ditulis. Chunk yang gagal di-skip
        (di-log), tidak meng-abort seluruh proses.
    """
    chunks = _chunk_window(window, _FIRMS_MAX_DAYS_PER_CALL)
    LOG.info("FIRMS province=%s sensor=%s: %d chunks (%s → %s)",
             province_id, sensor, len(chunks), window.start, window.end)

    written: list[Path] = []
    for i, ch in enumerate(chunks, 1):
        url = _build_firms_url(api_key, sensor, bbox, ch.days, ch.start)
        LOG.info("  chunk %d/%d: %s..%s (%d hari)",
                 i, len(chunks), ch.start, ch.end, ch.days)
        try:
            csv_text = _http_get_text(url)
        except RetryError as e:
            LOG.error("  chunk %d gagal setelah retry: %s", i, e)
            continue
        except Exception as e:  # noqa: BLE001
            LOG.error("  chunk %d gagal: %s", i, type(e).__name__)
            continue

        # Validate response: harus minimal punya CSV header (1 baris)
        lines = csv_text.strip().split("\n", 1)
        if not lines or "latitude" not in (lines[0].lower() if lines else ""):
            LOG.warning("  response tidak terlihat seperti FIRMS CSV (header anomaly)")
            continue

        fname = f"{province_id}_{ch.start}_{ch.end}_UTC.csv"
        path = _safe_output_path(out_dir, fname)
        path.write_text(csv_text, encoding="utf-8")
        n_rows = max(0, len(csv_text.strip().split("\n")) - 1)  # exclude header
        LOG.info("  wrote %s (%d rows)", path.name, n_rows)
        written.append(path)

        if i < len(chunks):
            time.sleep(sleep_between_chunks_s)  # courtesy rate-limiting

    LOG.info("FIRMS province=%s: %d/%d chunks sukses", province_id, len(written), len(chunks))
    return written


# ---------------------------------------------------------------------------
# Open-Meteo Archive — single-call historical fetch
# ---------------------------------------------------------------------------
def _bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    lat_min, lon_min, lat_max, lon_max = bbox
    return ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)


def _build_archive_url(lat: float, lon: float, window: DateWindow) -> str:
    """
    Open-Meteo Archive API URL — support start_date & end_date langsung,
    tidak butuh chunking (kecuali response sangat besar yang umumnya tidak terjadi
    untuk single province × 1 tahun harian).
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
    daily = ",".join(_DAILY_VARS)
    return (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={window.start.strftime('%Y-%m-%d')}"
        f"&end_date={window.end.strftime('%Y-%m-%d')}"
        f"&daily={daily}"
        f"&timezone=Asia%2FJakarta"
    )


def _archive_payload_to_df(payload: dict, province_id: str) -> pd.DataFrame:
    """Normalisasi response Archive jadi DataFrame, schema sama dengan fetch_weather.py."""
    daily = payload.get("daily") or {}
    times = daily.get("time")
    if not times:
        raise RuntimeError(f"Archive response missing 'daily.time' for {province_id}")
    cols = {"time": times}
    for var in _DAILY_VARS:
        cols[var] = daily.get(var) or [None] * len(times)
    df = pd.DataFrame(cols)
    df.insert(0, "province_id", province_id)
    df.rename(columns={"time": "date"}, inplace=True)
    return df


def fetch_weather_archive(province_id: str,
                          bbox: tuple[float, float, float, float],
                          window: DateWindow,
                          out_dir: Path) -> Optional[Path]:
    """
    Fetch entire window dalam satu call ke Open-Meteo Archive.

    Returns:
        Path file CSV yang ditulis, atau None kalau gagal.
    """
    lat, lon = _bbox_centroid(bbox)
    url = _build_archive_url(lat, lon, window)
    LOG.info("Weather Archive province=%s lat=%.4f lon=%.4f: %s → %s (%d hari, single call)",
             province_id, lat, lon, window.start, window.end, window.days)
    try:
        payload = _http_get_json(url)
    except RetryError as e:
        LOG.error("  gagal setelah retry: %s", e)
        return None
    except Exception as e:  # noqa: BLE001
        LOG.error("  gagal: %s", type(e).__name__)
        return None

    try:
        df = _archive_payload_to_df(payload, province_id)
    except RuntimeError as e:
        LOG.error("  payload malformed: %s", e)
        return None

    fname = f"{province_id}_{window.start}_{window.end}_UTC.csv"
    path = _safe_output_path(out_dir, fname)
    df.to_csv(path, index=False, encoding="utf-8")
    LOG.info("  wrote %s (%d rows)", path.name, len(df))
    return path


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_provinces(config_path: Optional[Path],
                    requested: Sequence[str]) -> list[tuple[str, list[float]]]:
    """
    Resolve daftar provinsi yang di-fetch. Pakai config/params.yaml kalau ada,
    fallback ke _DEFAULT_PROVINCES hard-coded.
    """
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        provs = ((cfg.get("data") or {}).get("provinces") or [])
        catalog = {p["id"]: list(p["bbox"]) for p in provs if "id" in p and "bbox" in p}
        if not catalog:
            LOG.warning("Config tidak punya provinces; fallback ke defaults")
            catalog = {k: list(v) for k, v in _DEFAULT_PROVINCES.items()}
    else:
        catalog = {k: list(v) for k, v in _DEFAULT_PROVINCES.items()}

    if list(requested) == ["all"]:
        return [(k, catalog[k]) for k in catalog]

    out: list[tuple[str, list[float]]] = []
    for r in requested:
        if r not in catalog:
            raise ValueError(f"Province {r!r} not in catalog {sorted(catalog)}")
        out.append((r, catalog[r]))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bulk historical fetcher untuk training data — "
                    "FIRMS (chunked) + Open-Meteo Archive (single call)."
    )
    p.add_argument("--start-date", type=_validate_date, required=True,
                   help="Tanggal mulai YYYY-MM-DD (inclusive)")
    p.add_argument("--end-date", type=_validate_date, required=True,
                   help="Tanggal akhir YYYY-MM-DD (inclusive)")
    p.add_argument("--provinces", nargs="+", required=True,
                   help="Province IDs atau 'all' (e.g. --provinces riau kalteng)")
    p.add_argument("--sources", default="firms,weather",
                   help="Comma-separated: firms, weather (default: keduanya)")
    p.add_argument("--sensor", default="VIIRS_SNPP_SP",
                   help="FIRMS sensor: VIIRS_SNPP_SP (archive, default) atau "
                        "VIIRS_SNPP_NRT (last 60 hari). MODIS_SP/MODIS_NRT juga valid.")
    p.add_argument("--config", type=Path, default=_PROJECT_ROOT / "config" / "params.yaml")
    p.add_argument("--raw-dir", type=Path, default=_PROJECT_ROOT / "data" / "raw")
    p.add_argument("--sleep-between-chunks", type=float, default=1.0,
                   help="Delay detik antar FIRMS chunk untuk hormati rate limit (default 1.0)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Load .env kalau ada (untuk NASA_FIRMS_API_KEY)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        window = DateWindow(start=args.start_date, end=args.end_date)
    except ValueError as e:
        LOG.error("Invalid date window: %s", e)
        return 2

    LOG.info("Window: %s → %s (%d hari)", window.start, window.end, window.days)

    try:
        provinces = _load_provinces(
            args.config if args.config.exists() else None, args.provinces
        )
    except ValueError as e:
        LOG.error("Province config error: %s", e)
        return 2
    LOG.info("Provinces: %s", [p[0] for p in provinces])

    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    invalid = sources - {"firms", "weather"}
    if invalid:
        LOG.error("Invalid sources: %s; expected 'firms' and/or 'weather'", invalid)
        return 2

    api_key = os.getenv("NASA_FIRMS_API_KEY", "")
    if "firms" in sources and not api_key:
        LOG.error("NASA_FIRMS_API_KEY not set; export atau isi .env terlebih dahulu")
        return 2

    firms_dir = args.raw_dir / "firms"
    weather_dir = args.raw_dir / "weather"

    n_ok = 0
    n_fail = 0
    for prov_id, bbox in provinces:
        bbox_t = tuple(bbox)  # type: ignore[assignment]

        if "firms" in sources:
            try:
                paths = fetch_firms_window(
                    prov_id, bbox_t, args.sensor, window, api_key,
                    firms_dir, args.sleep_between_chunks,
                )
                if paths:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:  # noqa: BLE001
                LOG.exception("FIRMS %s exception: %s", prov_id, type(e).__name__)
                n_fail += 1

        if "weather" in sources:
            try:
                path = fetch_weather_archive(prov_id, bbox_t, window, weather_dir)
                if path is not None:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:  # noqa: BLE001
                LOG.exception("Weather %s exception: %s", prov_id, type(e).__name__)
                n_fail += 1

    LOG.info("Done: %d sukses, %d gagal", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
