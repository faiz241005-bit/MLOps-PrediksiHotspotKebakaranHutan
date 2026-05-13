"""Unit tests untuk src.features.build_features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf


# --- Helpers -----------------------------------------------------------------
def _processed_sample() -> pd.DataFrame:
    """5 hari × 2 provinsi × beberapa deteksi per hari."""
    rows = []
    for prov in ["riau", "kalteng"]:
        for day_offset in range(5):
            date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=day_offset)
            # Variasi jumlah deteksi: hari 0=0, 1=3, 2=15, 3=5, 4=20
            n = [0, 3, 15, 5, 20][day_offset]
            for i in range(n):
                rows.append({
                    "province_id": prov,
                    "acq_date_wib": date,
                    "frp": 10.0 + i,
                    "confidence": "n",
                    "daynight": "D" if i % 2 == 0 else "N",
                    "satellite": "N",
                    "temperature_2m_max": 32.0,
                    "temperature_2m_min": 24.0,
                    "precipitation_sum": 0.0 if day_offset > 1 else 5.0,
                    "windspeed_10m_max": 10.0,
                    "winddirection_10m_dominant": 90.0,
                    "relative_humidity_2m_mean": 70.0,
                })
    return pd.DataFrame(rows)


# --- Aggregation tests -------------------------------------------------------
class TestAggregate:
    def test_basic_aggregation(self):
        df = _processed_sample()
        out = bf.aggregate_per_province_day(df)
        # 5 hari × 2 provinsi = 10 baris (kecuali hari 0 yang 0 deteksi → tidak ada di grouping)
        # Actually hari 0 punya 0 deteksi, jadi tidak ada baris di groupby
        assert len(out) > 0
        assert {"province_id", "date", "hotspot_count", "frp_mean"}.issubset(out.columns)

    def test_hotspot_count_matches(self):
        df = _processed_sample()
        out = bf.aggregate_per_province_day(df)
        # Riau hari ke-2 (May 3): harus 15 hotspot
        r = out[(out["province_id"] == "riau") &
                (out["date"] == pd.Timestamp("2026-05-03"))]
        assert r.iloc[0]["hotspot_count"] == 15

    def test_empty_input(self):
        out = bf.aggregate_per_province_day(pd.DataFrame())
        assert out.empty

    def test_missing_required_raises(self):
        bad = pd.DataFrame({"province_id": ["riau"]})
        with pytest.raises(RuntimeError, match="missing required"):
            bf.aggregate_per_province_day(bad)


# --- Calendar features -------------------------------------------------------
class TestCalendarFeatures:
    def test_cyclical_encoding_bounded(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-15", "2026-06-15", "2026-12-15"]),
        })
        out = bf.add_calendar_features(df)
        assert (out["month_sin"].between(-1, 1)).all()
        assert (out["month_cos"].between(-1, 1)).all()

    def test_month_extraction(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-05-15"])})
        out = bf.add_calendar_features(df)
        assert out.iloc[0]["month"] == 5


# --- Rolling features --------------------------------------------------------
class TestRollingFeatures:
    def test_rolling_no_leakage(self):
        # Rolling harus SHIFT 1 (yaitu tidak include hari current — cegah leakage)
        df = pd.DataFrame({
            "province_id": ["riau"] * 5,
            "date": pd.to_datetime([
                "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
            ]),
            "hotspot_count": [10, 20, 30, 40, 50],
            "frp_mean": [5.0, 10.0, 15.0, 20.0, 25.0],
        })
        out = bf.add_rolling_features(df, windows_days=(3,))
        # Baris pertama: rolling NaN (tidak ada history)
        assert pd.isna(out.iloc[0]["hotspot_count_3d"])
        # Baris ke-4 (index 3): rolling 3d harusnya 10+20+30 = 60 (sum 3 hari sebelumnya)
        assert out.iloc[3]["hotspot_count_3d"] == 60.0


# --- Lag features ------------------------------------------------------------
class TestLagFeatures:
    def test_lag_correctness(self):
        df = pd.DataFrame({
            "province_id": ["riau"] * 4,
            "date": pd.to_datetime([
                "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04",
            ]),
            "hotspot_count": [10, 20, 30, 40],
        })
        out = bf.add_lag_features(df, lags=(1, 2))
        assert out.iloc[2]["hotspot_count_lag_1d"] == 20  # shift(1)
        assert out.iloc[2]["hotspot_count_lag_2d"] == 10  # shift(2)


# --- Days since rain ---------------------------------------------------------
class TestDaysSinceRain:
    def test_consecutive_dry_days(self):
        df = pd.DataFrame({
            "province_id": ["riau"] * 5,
            "date": pd.to_datetime([
                "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
            ]),
            "precipitation_sum": [5.0, 0.0, 0.0, 0.0, 2.0],
        })
        out = bf.add_days_since_rain(df)
        # Hari 1 (1 May): hujan → 0 hari kering
        assert out.iloc[0]["days_since_rain"] == 0
        # Hari 2 (2 May): 1 hari sejak hujan
        assert out.iloc[1]["days_since_rain"] == 1
        # Hari 4 (4 May): 3 hari sejak hujan
        assert out.iloc[3]["days_since_rain"] == 3
        # Hari 5 (5 May): hujan lagi → reset
        assert out.iloc[4]["days_since_rain"] == 0


# --- Target generation -------------------------------------------------------
class TestTargets:
    def test_hotspot_tomorrow_shift(self):
        df = pd.DataFrame({
            "province_id": ["riau"] * 4,
            "date": pd.to_datetime([
                "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04",
            ]),
            "hotspot_count": [10, 20, 30, 40],
        })
        out = bf.add_targets(df)
        # Hari 1 → besok adalah hari 2 (count=20)
        assert out.iloc[0]["hotspot_count_tomorrow"] == 20
        # Hari terakhir tidak ada besok → NaN
        assert pd.isna(out.iloc[3]["hotspot_count_tomorrow"])

    def test_risk_level_thresholds(self):
        df = pd.DataFrame({
            "province_id": ["riau"] * 4,
            "date": pd.to_datetime([
                "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04",
            ]),
            "hotspot_count": [0, 5, 15, 50],  # last value will be dropped (no tomorrow)
        })
        out = bf.add_targets(df)
        # Hari 0 → besok=5 → Waspada (kelas 1)
        assert out.iloc[0]["risk_level"] == 1
        # Hari 1 → besok=15 → Bahaya (kelas 2)
        assert out.iloc[1]["risk_level"] == 2
        # Hari 2 → besok=50 → Bahaya (kelas 2)
        assert out.iloc[2]["risk_level"] == 2


# --- End-to-end pipeline -----------------------------------------------------
class TestEndToEnd:
    def test_full_pipeline(self):
        df = _processed_sample()
        cfg = bf.FeatureConfig()
        out = bf.build_features(df, cfg)
        # Harus punya target column dan beberapa baris
        assert "hotspot_count_tomorrow" in out.columns
        assert "risk_level" in out.columns
        assert "days_since_rain" in out.columns
        assert "month_sin" in out.columns
        assert (out["risk_level"].isin([0, 1, 2])).all()
        # Tidak ada NaN di target (sudah di-drop)
        assert out["hotspot_count_tomorrow"].notna().all()

    def test_write_parquet_roundtrip(self, tmp_path):
        df = _processed_sample()
        out = bf.build_features(df, bf.FeatureConfig())
        path = bf.write_features(out, tmp_path)
        assert path.exists()
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(out)
