"""
Orchestrator Ingestion — entry-point untuk fetch semua sumber data.

Tugas LK04: skrip orchestrator yang dipanggil oleh GitHub Actions cron
(LK06) atau manual oleh developer. Loop semua provinsi yang
terdefinisi di config/params.yaml dan panggil fetch_firms +
fetch_weather.

Output:
    data/raw/firms/{province}_{timestamp}_UTC.csv
    data/raw/weather/{province}_{timestamp}_UTC.csv

Exit code:
    0 — semua provinsi sukses
    1 — satu atau lebih provinsi gagal (lihat log)
    2 — config invalid / env tidak siap

Run example:
    python -m src.data.ingest_data --provinces riau
    python -m src.data.ingest_data --provinces all --sources firms,weather
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from src.data.fetch_firms import FirmsFetchSpec, fetch_one as fetch_firms_one
from src.data.fetch_weather import WeatherFetchSpec, fetch_one as fetch_weather_one

LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "params.yaml"
_DEFAULT_CONFIG_EXAMPLE = _PROJECT_ROOT / "config" / "params.example.yaml"


@dataclass(frozen=True)
class ProvinceConfig:
    id: str
    name: str
    bbox: tuple[float, float, float, float]


def _load_config(path: Path) -> dict:
    """Load YAML config; fallback ke params.example.yaml kalau params.yaml tidak ada."""
    if not path.exists():
        if path == _DEFAULT_CONFIG and _DEFAULT_CONFIG_EXAMPLE.exists():
            LOG.warning("%s not found; using %s. Copy template & customize untuk production.",
                        path.name, _DEFAULT_CONFIG_EXAMPLE.name)
            path = _DEFAULT_CONFIG_EXAMPLE
        else:
            raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_provinces(cfg: dict, requested: Sequence[str]) -> list[ProvinceConfig]:
    """Filter daftar provinsi sesuai request (atau semua kalau 'all')."""
    all_provinces_raw = (cfg.get("data") or {}).get("provinces") or []
    if not all_provinces_raw:
        raise RuntimeError("config.data.provinces is empty")

    all_provinces = [
        ProvinceConfig(id=p["id"], name=p["name"], bbox=tuple(p["bbox"]))
        for p in all_provinces_raw
    ]
    if len(requested) == 1 and requested[0].lower() == "all":
        return all_provinces

    want = {r.lower() for r in requested}
    filtered = [p for p in all_provinces if p.id.lower() in want]
    missing = want - {p.id.lower() for p in filtered}
    if missing:
        raise RuntimeError(f"Unknown province IDs: {sorted(missing)}; "
                           f"available={[p.id for p in all_provinces]}")
    return filtered


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FireGuard data ingestion orchestrator (LK04)"
    )
    p.add_argument("--provinces", nargs="+", required=True,
                   help="Province IDs atau 'all' (mis. --provinces riau kalteng)")
    p.add_argument("--sources", default="firms,weather",
                   help="Comma-separated sumber: firms, weather (default: keduanya)")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                   help="Path config YAML (default: config/params.yaml)")
    p.add_argument("--raw-dir", type=Path, default=_PROJECT_ROOT / "data" / "raw",
                   help="Root folder data mentah (default: data/raw)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def run(provinces: list[ProvinceConfig], sources: set[str], cfg: dict,
        raw_dir: Path, api_key: str) -> tuple[int, int]:
    """Eksekusi orchestrator. Return (n_success, n_failure)."""
    sensor = (((cfg.get("data") or {}).get("api") or {})
              .get("firms") or {}).get("sensor", "VIIRS_SNPP_NRT")
    day_range = (((cfg.get("data") or {}).get("api") or {})
                 .get("firms") or {}).get("day_range", 2)

    firms_dir = raw_dir / "firms"
    weather_dir = raw_dir / "weather"

    success = 0
    failure = 0
    for prov in provinces:
        LOG.info("--- Province %s (%s) ---", prov.id, prov.name)

        if "firms" in sources:
            try:
                fetch_firms_one(
                    FirmsFetchSpec(province_id=prov.id, bbox=prov.bbox,
                                   sensor=sensor, day_range=day_range),
                    api_key=api_key, out_dir=firms_dir,
                )
                success += 1
            except Exception as e:  # noqa: BLE001
                LOG.error("FIRMS failed for %s: %s", prov.id, e)
                failure += 1

        if "weather" in sources:
            try:
                fetch_weather_one(
                    WeatherFetchSpec(province_id=prov.id, bbox=prov.bbox),
                    out_dir=weather_dir,
                )
                success += 1
            except Exception as e:  # noqa: BLE001
                LOG.error("Weather failed for %s: %s", prov.id, e)
                failure += 1

    return success, failure


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    invalid = sources - {"firms", "weather"}
    if invalid:
        LOG.error("Invalid sources: %s; valid={firms, weather}", sorted(invalid))
        return 2

    try:
        cfg = _load_config(args.config)
        provinces = _parse_provinces(cfg, args.provinces)
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        LOG.error("Config error: %s", e)
        return 2

    api_key = ""
    if "firms" in sources:
        api_key = os.getenv("NASA_FIRMS_API_KEY", "").strip()
        if not api_key:
            LOG.error("NASA_FIRMS_API_KEY is not set (check .env / Actions Secrets)")
            return 2

    LOG.info("Starting ingestion: %d provinces, sources=%s",
             len(provinces), sorted(sources))
    s, f = run(provinces, sources, cfg, args.raw_dir, api_key)
    LOG.info("Ingestion done: %d sukses, %d gagal", s, f)
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
