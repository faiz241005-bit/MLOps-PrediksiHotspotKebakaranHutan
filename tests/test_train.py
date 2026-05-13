"""Unit tests untuk src.models.train (no MLflow server needed)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.models import train as tr


# --- Helpers -----------------------------------------------------------------
def _features_sample(n_provinces: int = 2, n_days: int = 30) -> pd.DataFrame:
    """Generate synthetic features dataset untuk testing training."""
    import numpy as np
    rng = np.random.default_rng(42)
    rows = []
    for p_idx, prov in enumerate(["riau", "kalteng"][:n_provinces]):
        for day_offset in range(n_days):
            date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=day_offset)
            count = int(rng.integers(0, 30))
            rows.append({
                "province_id": prov,
                "date": date,
                "hotspot_count": count,
                "frp_mean": rng.uniform(5, 30),
                "frp_max": rng.uniform(10, 50),
                "frp_sum": rng.uniform(50, 500),
                "n_daytime": rng.integers(0, count + 1) if count else 0,
                "n_nighttime": rng.integers(0, count + 1) if count else 0,
                "n_confidence_high": rng.integers(0, count + 1) if count else 0,
                "temperature_2m_max": rng.uniform(28, 35),
                "temperature_2m_min": rng.uniform(22, 26),
                "precipitation_sum": rng.uniform(0, 10),
                "windspeed_10m_max": rng.uniform(5, 20),
                "winddirection_10m_dominant": rng.uniform(0, 360),
                "relative_humidity_2m_mean": rng.uniform(60, 90),
                "month": date.month,
                "day_of_year": date.dayofyear,
                "month_sin": rng.uniform(-1, 1),
                "month_cos": rng.uniform(-1, 1),
                "hotspot_count_1d": rng.integers(0, 30),
                "hotspot_count_3d": rng.integers(0, 90),
                "hotspot_count_7d": rng.integers(0, 200),
                "frp_mean_1d": rng.uniform(5, 30),
                "frp_mean_3d": rng.uniform(5, 30),
                "frp_mean_7d": rng.uniform(5, 30),
                "hotspot_count_lag_1d": rng.integers(0, 30),
                "hotspot_count_lag_3d": rng.integers(0, 30),
                "hotspot_count_lag_7d": rng.integers(0, 30),
                "days_since_rain": rng.integers(0, 10),
                "hotspot_count_tomorrow": float(rng.integers(0, 30)),
                "risk_level": int(rng.integers(0, 3)),
            })
    return pd.DataFrame(rows)


# --- Split tests -------------------------------------------------------------
class TestTimeAwareSplit:
    def test_basic_split(self):
        df = _features_sample(n_provinces=2, n_days=30)
        train, test = tr._time_aware_split(df, holdout_days=7)
        assert len(train) > 0 and len(test) > 0
        # Train < Test pada axis date
        assert train["date"].max() <= test["date"].min()

    def test_too_few_dates_fallback(self):
        # Cuma 3 hari → fallback ke random split
        df = _features_sample(n_provinces=1, n_days=3)
        train, test = tr._time_aware_split(df, holdout_days=14)
        # Tidak boleh raise; random split kasih ~80/20
        assert len(train) >= 2
        assert len(test) >= 1


# --- xy extraction -----------------------------------------------------------
class TestXY:
    def test_drops_non_feature_columns(self):
        df = _features_sample()
        X, y = tr._xy(df, target="hotspot_count_tomorrow")
        assert "province_id" not in X.columns
        assert "date" not in X.columns
        assert "hotspot_count_tomorrow" not in X.columns
        assert "risk_level" not in X.columns

    def test_target_aligned(self):
        df = _features_sample()
        X, y = tr._xy(df, target="hotspot_count_tomorrow")
        assert len(X) == len(y)
        assert y.name == "hotspot_count_tomorrow"


# --- Training tests (skip kalau lightgbm/mlflow tidak available) -------------
def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _has_mlflow() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_lightgbm(), reason="lightgbm not installed")
class TestRegressorTraining:
    def test_train_returns_metrics(self):
        df = _features_sample(n_provinces=2, n_days=40)
        train, test = tr._time_aware_split(df, holdout_days=10)
        params = tr.TrainParams(algorithm="regressor",
                                n_estimators=20, max_depth=3)
        model, metrics = tr.train_regressor(train, test, params)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        # Model harus bisa predict
        from src.models.train import _xy, _TARGET_REGRESSOR
        X_test, _ = _xy(test, _TARGET_REGRESSOR)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)


@pytest.mark.skipif(not _has_lightgbm(), reason="lightgbm not installed")
class TestClassifierTraining:
    def test_train_returns_metrics(self):
        # Ensure all 3 classes ada di data
        df = _features_sample(n_provinces=2, n_days=40)
        # Force ada minimal 1 baris per kelas
        df.loc[df.index[:5], "risk_level"] = 0
        df.loc[df.index[5:10], "risk_level"] = 1
        df.loc[df.index[10:15], "risk_level"] = 2

        train, test = tr._time_aware_split(df, holdout_days=10)
        params = tr.TrainParams(algorithm="classifier",
                                n_estimators=20, max_depth=3)
        model, metrics = tr.train_classifier(train, test, params)
        assert "f1_macro" in metrics
        assert "recall_bahaya" in metrics
        assert 0 <= metrics["f1_macro"] <= 1


# --- MLflow integration (smoke test, no UI) ---------------------------------
@pytest.mark.skipif(not _has_lightgbm() or not _has_mlflow(),
                    reason="lightgbm and mlflow required")
class TestRunExperiment:
    def test_regressor_run(self, tmp_path):
        import mlflow
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        df = _features_sample(n_provinces=2, n_days=40)
        params = tr.TrainParams(algorithm="regressor",
                                n_estimators=10, max_depth=3)
        result = tr.run_experiment(params, df, experiment_name="test-fireguard",
                                   holdout_days=10, register=False)
        assert "run_id" in result
        assert "rmse" in result
        assert "mae" in result

    def test_invalid_algorithm_rejected(self, tmp_path):
        import mlflow
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        df = _features_sample(n_provinces=2, n_days=40)
        params = tr.TrainParams(algorithm="unknown")
        with pytest.raises(ValueError, match="algorithm must be"):
            tr.run_experiment(params, df, experiment_name="test-fireguard",
                              holdout_days=10)
