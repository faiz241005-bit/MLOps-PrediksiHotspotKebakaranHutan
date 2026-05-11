"""Unit tests untuk src.data.fetch_firms (no real network)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.data import fetch_firms as ff

# --- Fixtures ----------------------------------------------------------------
SAMPLE_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "0.5,101.2,330.1,0.4,0.4,2026-05-10,1418,N,VIIRS,n,2.0NRT,290.1,12.3,D\n"
    "1.0,101.8,310.5,0.4,0.4,2026-05-10,1422,N,VIIRS,l,2.0NRT,280.1,3.1,D\n"
)


@pytest.fixture
def spec() -> ff.FirmsFetchSpec:
    return ff.FirmsFetchSpec(
        province_id="riau",
        bbox=(0.0, 100.0, 4.5, 106.5),
        sensor="VIIRS_SNPP_NRT",
        day_range=2,
    )


# --- Validation tests --------------------------------------------------------
class TestValidateBbox:
    def test_valid_bbox(self):
        assert ff._validate_bbox((0.0, 100.0, 4.5, 106.5)) == (0.0, 100.0, 4.5, 106.5)

    def test_inverted_lat(self):
        with pytest.raises(ValueError, match="latitude"):
            ff._validate_bbox((5.0, 100.0, 0.0, 106.0))

    def test_inverted_lon(self):
        with pytest.raises(ValueError, match="longitude"):
            ff._validate_bbox((0.0, 110.0, 4.5, 100.0))

    def test_wrong_count(self):
        with pytest.raises(ValueError, match="4 numbers"):
            ff._validate_bbox((0.0, 100.0, 4.5))

    def test_out_of_range_lat(self):
        with pytest.raises(ValueError, match="latitude"):
            ff._validate_bbox((-91.0, 100.0, 4.5, 106.0))


class TestValidateUrl:
    def test_allowed_host(self):
        # Tidak raise
        ff._validate_url("https://firms.modaps.eosdis.nasa.gov/api/area/csv/X/Y/0,0,1,1/2")

    def test_disallowed_host(self):
        with pytest.raises(ValueError, match="Disallowed host"):
            ff._validate_url("https://evil.example.com/api/area/csv/X")


class TestBuildUrl:
    def test_empty_key_rejected(self):
        with pytest.raises(ValueError, match="api_key is empty"):
            ff._build_url("", "VIIRS_SNPP_NRT", (0.0, 100.0, 4.5, 106.5), 2)

    def test_day_range_out_of_range(self):
        with pytest.raises(ValueError, match="day_range"):
            ff._build_url("abc", "VIIRS_SNPP_NRT", (0.0, 100.0, 4.5, 106.5), 11)

    def test_url_format(self):
        url = ff._build_url("MYKEY", "VIIRS_SNPP_NRT", (0.0, 100.0, 4.5, 106.5), 2)
        assert url.startswith("https://firms.modaps.eosdis.nasa.gov/api/area/csv/MYKEY/")
        # Format param: lon_min,lat_min,lon_max,lat_max
        assert "100.0,0.0,106.5,4.5" in url
        assert url.endswith("/2")


class TestSafeOutputPath:
    def test_path_traversal_blocked(self, tmp_path):
        with pytest.raises(RuntimeError, match="outside"):
            ff._safe_output_path(tmp_path, "../escape.csv")

    def test_normal_filename_ok(self, tmp_path):
        p = ff._safe_output_path(tmp_path, "ok.csv")
        assert str(p).startswith(str(tmp_path.resolve()))

    def test_makes_parent_dir(self, tmp_path):
        sub = tmp_path / "nested" / "dir"
        p = ff._safe_output_path(sub, "f.csv")
        assert sub.exists()


# --- Schema validation tests -------------------------------------------------
class TestValidateCsvHeader:
    def test_complete_header(self):
        ff._validate_csv_header(SAMPLE_CSV, "riau")  # no raise

    def test_missing_critical(self):
        bad_csv = "longitude,acq_date,acq_time\n101,2026-05-10,1418\n"
        with pytest.raises(RuntimeError, match="missing critical columns"):
            ff._validate_csv_header(bad_csv, "riau")

    def test_empty_csv_rejected(self):
        with pytest.raises(RuntimeError, match="Empty CSV"):
            ff._validate_csv_header("", "riau")


# --- End-to-end fetch (mocked HTTP) ------------------------------------------
class TestFetchOne:
    def test_happy_path(self, tmp_path, spec):
        fake_now = datetime(2026, 5, 11, 8, 0, 0, tzinfo=timezone.utc)
        with patch("src.data.fetch_firms._http_get", return_value=SAMPLE_CSV):
            out_path = ff.fetch_one(spec, api_key="FAKEKEY",
                                    out_dir=tmp_path, now_utc=fake_now)
        assert out_path.exists()
        assert out_path.name == "riau_20260511_080000_UTC.csv"
        assert out_path.read_text().startswith("latitude,longitude")

    def test_empty_api_key_rejected(self, tmp_path, spec):
        with pytest.raises(ValueError, match="api_key is empty"):
            ff.fetch_one(spec, api_key="", out_dir=tmp_path)

    def test_writes_inside_out_dir(self, tmp_path, spec):
        with patch("src.data.fetch_firms._http_get", return_value=SAMPLE_CSV):
            out_path = ff.fetch_one(spec, api_key="K", out_dir=tmp_path)
        assert str(out_path).startswith(str(tmp_path.resolve()))


class TestFetchMany:
    def test_partial_failure_continues(self, tmp_path):
        specs = [
            ff.FirmsFetchSpec(province_id="riau", bbox=(0.0, 100.0, 4.5, 106.5)),
            ff.FirmsFetchSpec(province_id="kalteng", bbox=(-3.5, 110.5, 1.5, 116.5)),
        ]

        call_count = {"n": 0}
        def fake_get(url, timeout=30):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated network failure")
            return SAMPLE_CSV

        with patch("src.data.fetch_firms._http_get", side_effect=fake_get):
            written = ff.fetch_many(specs, api_key="K", out_dir=tmp_path)
        assert len(written) == 1  # riau gagal, kalteng sukses
