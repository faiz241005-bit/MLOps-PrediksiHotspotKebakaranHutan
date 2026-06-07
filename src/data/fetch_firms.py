"""
NASA FIRMS Hotspot Fetcher — production-ready ingestion module.

Tugas LK04: mengambil data hotspot satelit secara berkala dari NASA FIRMS
REST API, simpan ke data/raw/firms/ dengan nama ber-timestamp (append-only,
tidak overwrite data lama).
"""
from __future__ import annotations

import argparse
import csv
import io
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

# --- Konstanta ---------------------------------------------------------------
_ALLOWED_HOSTS = frozenset({"firms.modaps.eosdis.nasa.gov"})
_DEFAULT_TIMEOUT_S = 30
_MAX_RETRIES = 3
_USER_AGENT = "FireGuard/0.1 (+education)"

_EXPECTED_COLUMNS = {
    "latitude", "longitude", "bright_ti4", "scan", "track",
    "acq_date", "acq_time", "satellite", "instrument", "confidence",
    "version", "bright_ti5", "frp", "daynight",
}


@dataclass(frozen=True)
class FirmsFetchSpec:
    """Parameter satu kali fetch ke FIRMS API."""
    province_id: str
    bbox: tuple[float, float, float, float]
    sensor: str = "VIIRS_SNPP_NRT"
    day_range: int = 2


def _validate_url(url: str) -> None:
    host = urlparse(url).hostname
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"Disallowed host: {host!r}; allowed={sorted(_ALLOWED_HOSTS)}")


def _validate_bbox(bbox: Iterable[float]) -> tuple[float, float, float, float]:
    b = tuple(float(x) for x in bbox)
    if len(b) != 4:
        raise ValueError(f"bbox must have 4 numbers, got {len(b)}")
    lat_min, lon_min, lat_max, lon_max = b
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(f"Invalid latitude range: {lat_min} .. {lat_max}")
    if not (-180 <= lon_min < lon_max <= 180):
        raise ValueError(f"Invalid longitude range: {lon_min} .. {lon_max}")
    return b


def _build_url(api_key: str, sensor: str, bbox: tuple[float, float, float, float],
               day_range: int) -> str:
    if not api_key or not api_key.strip():
        raise ValueError("api_key is empty — set NASA_FIRMS_API_KEY env var")
    if not 1 <= day_range <= 10:
        raise ValueError(f"day_range must be 1..10, got {day_range}")
    lat_min, lon_min, lat_max, lon_max = bbox
    return (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{api_key}/{sensor}/{lon_min},{lat_min},{lon_max},{lat_max}/{day_range}"
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
def _http_get(url: str, timeout: int = _DEFAULT_TIMEOUT_S) -> str:
    _validate_url(url)
    with requests.Session() as session:
        session.headers.update({"User-Agent": _USER_AGENT, "Accept": "text/csv"})
        resp = session.get(url, timeout=timeout, allow_redirects=False)
        LOG.debug("FIRMS GET host=%s status=%s bytes=%s",
                  urlparse(url).hostname, resp.status_code, len(resp.content))
        resp.raise_for_status()
        return resp.text


def fetch_one(spec: FirmsFetchSpec, api_key: str, out_dir: Path,
              now_utc: Optional[datetime] = None) -> Path:
    """Fetch satu provinsi, simpan CSV dengan nama ber-timestamp."""
    bbox = _validate_bbox(spec.bbox)
    url = _build_url(api_key, spec.sensor, bbox, spec.day_range)

    LOG.info("Fetching FIRMS province=%s sensor=%s day_range=%d",
             spec.province_id, spec.sensor, spec.day_range)
    try:
        csv_text = _http_get(url)
    except RetryError as e:
        raise RuntimeError(f"FIRMS fetch failed after retries for {spec.province_id}") from e

    _validate_csv_header(csv_text, spec.province_id)

    now = (now_utc or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    fname = f"{spec.province_id}_{now}_UTC.csv"
    out_path = _safe_output_path(out_dir, fname)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(csv_text)

    n_lines = csv_text.count("\n")
    LOG.info("Wrote %s (%d lines, %d bytes)",
             out_path.name, n_lines, out_path.stat().st_size)
    return out_path


def _validate_csv_header(csv_text: str, province_id: str) -> None:
    with io.StringIO(csv_text) as buf:
        reader = csv.reader(buf)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(f"Empty CSV returned for {province_id}")
    header_set = {c.strip() for c in header}
    missing_critical = {"latitude", "longitude", "acq_date", "acq_time"} - header_set
    if missing_critical:
        raise RuntimeError(
            f"FIRMS response missing critical columns for {province_id}: "
            f"{sorted(missing_critical)}; got header={header}"
        )
    missing_extra = _EXPECTED_COLUMNS - header_set
    if missing_extra:
        LOG.warning("Non-critical missing columns for %s: %s",
                    province_id, sorted(missing_extra))


def fetch_many(specs: Iterable[FirmsFetchSpec], api_key: str, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    errors: list[tuple[str, str]] = []
    for spec in specs:
        try:
            p = fetch_one(spec, api_key=api_key, out_dir=out_dir)
            written.append(p)
        except Exception as e:  # noqa: BLE001
            LOG.error("Fetch failed for %s: %s", spec.province_id, e)
            errors.append((spec.province_id, str(e)))
    if errors:
        LOG.warning("Completed with %d failures: %s", len(errors), errors)
    return written


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch NASA FIRMS hotspot CSV per provinsi (FireGuard LK04)"
    )
    p.add_argument("--province", required=True,
                   help="Province ID (mis. riau, kalteng, kalbar, sumsel, jambi)")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
                   help="Bounding box geografis provinsi")
    p.add_argument("--sensor", default="VIIRS_SNPP_NRT",
                   help="FIRMS sensor (default: VIIRS_SNPP_NRT)")
    p.add_argument("--day-range", type=int, default=2,
                   help="Berapa hari ke belakang (1..10, default 2)")
    p.add_argument("--output-dir", default="data/raw/firms",
                   help="Folder output (default: data/raw/firms)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry-point. Return 0 on success, non-zero on failure."""
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Auto-load .env untuk dev lokal/Codespace.
    # Di CI GitHub Actions, env vars dari Secrets sudah real shell env — load_dotenv silent no-op.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("NASA_FIRMS_API_KEY", "").strip()
    if not api_key:
        LOG.error("NASA_FIRMS_API_KEY is not set (check .env / GitHub Secrets)")
        return 2

    spec = FirmsFetchSpec(
        province_id=args.province,
        bbox=tuple(args.bbox),
        sensor=args.sensor,
        day_range=args.day_range,
    )
    try:
        path = fetch_one(spec, api_key=api_key, out_dir=Path(args.output_dir))
        print(path)
        return 0
    except Exception as e:  # noqa: BLE001
        LOG.error("Fetch failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
