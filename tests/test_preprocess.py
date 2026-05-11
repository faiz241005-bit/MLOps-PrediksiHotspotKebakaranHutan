"""Unit tests untuk src.data.preprocess."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import preprocess as pp


# --- Helpers -----------------------------------------------------------------
def _firms_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "latitude":   [0.5, 1.0, 1.0, 0.7, 999.0],   # 1 invalid lat
        "longitude":  [101.2, 101.8, 101.8, 102.0, 101.0],
        "acq_date":   ["2026-05-10"] * 5,
        "acq_time":   [1418, 1422, 1422, 200, 530],
        "frp":        [12.3, 3.1, 3.1, -1.0, 5.0],   # 1 negative frp
        "confidence": ["n", "l", "l", "h", "n"],
        "daynight":   ["D", "D", "D", "N", "D"],
        "satellite":  ["N", "N", "N", "N", "N"],
        "province_id": ["riau"] * 5,
    })


def _weather_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-05-10"],
        "province_id": ["riau"],
        "temperature_2m_max": [32.1],
        "temperature_2m_min": [24.0],
        "precipitation_sum": [0.5],
        "windspeed_10m_max": [12.5],
        "winddirection_10m_dominant": [85.0],
        "relative_humidity_2m_mean": [78.0],
        "fetched_at_utc": ["2026-05-11T00:00:00+00:00"],
    })


# --- Tests -------------------------------------------------------------------
class TestCleanFirms:
    def test_drops_invalid_coordinates(self):
        df = _firms_sample()
        out = pp.clean_firms(df)
        # 5 rows in, 1 invalid lat dropped, 1 dedup-ed (rows 2 & 3 identik), 1 negative frp dropped
        assert (out["latitude"].between(-90, 90)).all()
        assert (out["frp"] >= 0).all()

    def test_dedup(self):
        df = _firms_sample()
        out = pp.clean_firms(df)
        keys = ["latitude", "longitude", "acq_datetime_utc"]
        assert not out.duplicated(subset=keys).any()

    def test_adds_wib_columns(self):
        df = _firms_sample()
        out = pp.clean_firms(df, tz_offset_hours=7)
        assert "acq_datetime_wib" in out.columns
        assert "acq_date_wib" in out.columns
        # 14:18 UTC + 7h = 21:18 WIB (sama tanggal)
        row = out[(out["latitude"] == 0.5) & (out["longitude"] == 101.2)].iloc[0]
        assert row["acq_datetime_wib"].hour == 21
        assert row["acq_datetime_wib"].minute == 18

    def test_empty_df(self):
        out = pp.clean_firms(pd.DataFrame())
        assert out.empty

    def test_missing_required_columns(self):
        bad = pd.DataFrame({"latitude": [0.5]})
        with pytest.raises(RuntimeError, match="missing required"):
            pp.clean_firms(bad)


class TestCleanWeather:
    def test_dedup_keep_latest(self):
        # Dua snapshot untuk (riau, 2026-05-10) — pakai yang fetched_at_utc terbaru
        df = pd.concat([_weather_sample(), _weather_sample()], ignore_index=True)
        df.loc[0, "temperature_2m_max"] = 30.0
        df.loc[1, "temperature_2m_max"] = 33.0
        df.loc[0, "fetched_at_utc"] = "2026-05-11T00:00:00+00:00"
        df.loc[1, "fetched_at_utc"] = "2026-05-11T12:00:00+00:00"
        out = pp.clean_weather(df)
        assert len(out) == 1
        # Yang dipertahankan yang terbaru
        assert out.iloc[0]["temperature_2m_max"] == 33.0

    def test_coerce_numeric(self):
        df = _weather_sample()
        df["temperature_2m_max"] = ["32.1"]  # string
        out = pp.clean_weather(df)
        assert out["temperature_2m_max"].dtype.kind == "f"

    def test_missing_required(self):
        with pytest.raises(RuntimeError, match="missing required"):
            pp.clean_weather(pd.DataFrame({"foo": [1]}))


class TestJoin:
    def test_join_enriches_with_weather(self):
        firms_clean = pp.clean_firms(_firms_sample())
        weather_clean = pp.clean_weather(_weather_sample())
        merged = pp.join_firms_weather(firms_clean, weather_clean)
        assert "temperature_2m_max" in merged.columns
        assert not merged.empty
        # Semua row dapat cuaca karena ada match riau / 2026-05-10
        assert merged["temperature_2m_max"].notna().all()

    def test_join_empty_weather(self):
        firms_clean = pp.clean_firms(_firms_sample())
        merged = pp.join_firms_weather(firms_clean, pd.DataFrame())
        assert len(merged) == len(firms_clean)  # no enrichment, no row loss


class TestWriteProcessed:
    def test_writes_parquet(self, tmp_path):
        df = pp.clean_firms(_firms_sample())
        out_path = pp.write_processed(df, tmp_path)
        assert out_path.exists()
        assert out_path.suffix == ".parquet"
        # Round-trip
        loaded = pd.read_parquet(out_path)
        assert len(loaded) == len(df)
