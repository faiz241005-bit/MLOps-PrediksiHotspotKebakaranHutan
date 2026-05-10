"""
NASA FIRMS Hotspot Fetcher (stub for LK03).

Security:
    - API key dibaca dari ENV var, tidak pernah hard-coded.
    - URL divalidasi terhadap allow-list (mencegah SSRF).
    - Timeout dan retry/backoff diatur agar runner CI tidak hang.

Resource hygiene:
    - HTTP session pakai context manager (auto-close).
    - File CSV ditulis via context manager.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOG = logging.getLogger(__name__)

# Allow-list domain — security: cegah SSRF
_ALLOWED_HOSTS = {"firms.modaps.eosdis.nasa.gov"}

# Default timeout (detik) — wajib di setiap network call agar tidak ngegantung
_DEFAULT_TIMEOUT = 30


def _validate_url(url: str) -> None:
    """Pastikan URL hanya mengarah ke domain yang diizinkan."""
    host = urlparse(url).hostname
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"Disallowed host: {host!r}")


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _fetch_csv(url: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Single GET dengan retry exponential backoff."""
    _validate_url(url)
    with requests.Session() as session:
        # Tidak set proxy/redirect agar tidak diarahkan ke domain lain
        session.headers.update({"User-Agent": "FireGuard/0.1 (+education)"})
        resp = session.get(url, timeout=timeout, allow_redirects=False)
        resp.raise_for_status()
        return resp.text


def fetch_province(
    api_key: str,
    sensor: str,
    bbox: Iterable[float],
    day_range: int,
    out_dir: Path,
    province_id: str,
) -> Path:
    """
    Fetch FIRMS hotspots untuk satu bbox provinsi, simpan ke CSV.

    Args:
        api_key: API key NASA FIRMS.
        sensor: e.g. "VIIRS_SNPP_NRT".
        bbox: [lat_min, lon_min, lat_max, lon_max].
        day_range: Berapa hari ke belakang (max 10 sesuai dokumentasi FIRMS).
        out_dir: Folder output (dibuat jika belum ada).
        province_id: ID provinsi untuk nama file.

    Returns:
        Path ke file CSV hasil fetch.
    """
    if not api_key:
        raise ValueError("api_key is required (set NASA_FIRMS_API_KEY env var)")

    lat_min, lon_min, lat_max, lon_max = bbox
    # Format URL sesuai docs FIRMS
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{api_key}/{sensor}/{lon_min},{lat_min},{lon_max},{lat_max}/{day_range}"
    )

    LOG.info("Fetching FIRMS for province=%s sensor=%s", province_id, sensor)
    csv_text = _fetch_csv(url)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Path traversal guard
    out_path = (out_dir / f"firms_{province_id}.csv").resolve()
    if not str(out_path).startswith(str(out_dir)):
        raise RuntimeError("Refusing to write outside of out_dir")

    with out_path.open("w", encoding="utf-8") as f:
        f.write(csv_text)

    LOG.info("Wrote %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


def main() -> None:
    """CLI entry-point — sederhana, baca config dari env. (Akan disempurnakan di LK04.)"""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    api_key = os.getenv("NASA_FIRMS_API_KEY")
    if not api_key:
        raise SystemExit("NASA_FIRMS_API_KEY is not set")

    LOG.info("Stub: fetch_firms.main() — implementasi penuh di LK04")
    # TODO(LK04): baca config/params.yaml, iterate provinces, panggil fetch_province().


if __name__ == "__main__":
    main()
