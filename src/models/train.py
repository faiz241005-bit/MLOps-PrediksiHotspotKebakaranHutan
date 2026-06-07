"""
Training pipeline FireGuard — LightGBM regressor + classifier dengan MLflow tracking.

Tugas LK06: latih model + log eksperimen ke MLflow. Output:
    - mlruns/ — local MLflow tracking (dapat diakses via 'mlflow ui')
    - models/{run_id}/ — model artifacts (di-track DVC)

Mendukung:
    - mlflow.log_param: learning_rate, n_estimators, max_depth, num_leaves, dst.
    - mlflow.log_metric: RMSE, MAE, F1 macro, Recall kelas Bahaya, latency
    - mlflow.log_model: artifact LightGBM model
    - Model Registry: register run terbaik sebagai 'fireguard-regressor' / 'fireguard-classifier'

Run example:
    python -m src.models.train --algorithm regressor --learning-rate 0.05 --n-estimators 500
    python -m src.models.train --algorithm classifier --max-depth 6 --learning-rate 0.1
    python -m src.models.train --algorithm regressor --n-estimators 300 --max-depth 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)
from sklearn.model_selection import train_test_split

LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainParams:
    """Hyperparameter set yang akan di-log ke MLflow."""

    algorithm: str  # "regressor" | "classifier"
    learning_rate: float = 0.05
    n_estimators: int = 600
    max_depth: int = 8
    num_leaves: int = 31
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    n_jobs: int = -1


# Fitur input untuk model (semua kolom kecuali ID & targets)
_TARGET_REGRESSOR = "hotspot_count_tomorrow"
_TARGET_CLASSIFIER = "risk_level"
_NON_FEATURE_COLS = {
    "province_id",
    "date",
    _TARGET_REGRESSOR,
    _TARGET_CLASSIFIER,
}


# ---------------------------------------------------------------------------
# Data loading & split
# ---------------------------------------------------------------------------
def _read_latest_features(folder: Path) -> pd.DataFrame:
    """Load parquet terbaru di data/features/."""
    folder = Path(folder)
    files = sorted(folder.glob("training_dataset_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No training dataset found in {folder}. "
            f"Run 'python -m src.features.build_features' first (LK06 Tahap 1)."
        )
    latest = files[-1]
    LOG.info("Loading features from %s", latest.name)
    return pd.read_parquet(latest)


def _time_aware_split(
    df: pd.DataFrame, holdout_days: int = 14
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware train/test split: hari paling lama → train, holdout_days terakhir → test.
    Penting untuk time-series untuk menghindari data leakage.
    """
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].nunique() <= holdout_days:
        # Data terlalu sedikit untuk split waktu; fallback ke 80/20 random (warn)
        LOG.warning(
            "Only %d unique dates; using 80/20 random split instead of time-aware",
            df["date"].nunique(),
        )
        return train_test_split(df, test_size=0.2, random_state=42)

    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    LOG.info(
        "Time-aware split: train=%d rows (≤ %s), test=%d rows (> %s)",
        len(train),
        cutoff.date(),
        len(test),
        cutoff.date(),
    )
    return train, test


def _xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    # Drop fitur non-numeric (kalau ada artefak object dtype)
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df[target]
    return X, y


# ---------------------------------------------------------------------------
# Training: Regressor
# ---------------------------------------------------------------------------
def train_regressor(
    train_df: pd.DataFrame, test_df: pd.DataFrame, params: TrainParams
) -> tuple[object, dict[str, float]]:
    """Train LightGBM regressor untuk hotspot_count_tomorrow. Return (model, metrics)."""
    import lightgbm as lgb  # lazy import (allow features-only test to pass without lightgbm)

    X_train, y_train = _xy(train_df, _TARGET_REGRESSOR)
    X_test, y_test = _xy(test_df, _TARGET_REGRESSOR)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        learning_rate=params.learning_rate,
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
        verbose=-1,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "train_time_seconds": train_time,
        "n_train_rows": float(len(X_train)),
        "n_test_rows": float(len(X_test)),
        "n_features": float(X_train.shape[1]),
    }
    LOG.info("Regressor metrics: %s", metrics)
    return model, metrics


# ---------------------------------------------------------------------------
# Training: Classifier
# ---------------------------------------------------------------------------
def train_classifier(
    train_df: pd.DataFrame, test_df: pd.DataFrame, params: TrainParams
) -> tuple[object, dict[str, float]]:
    """Train LightGBM classifier untuk risk_level (3 kelas)."""
    import lightgbm as lgb

    X_train, y_train = _xy(train_df, _TARGET_CLASSIFIER)
    X_test, y_test = _xy(test_df, _TARGET_CLASSIFIER)

    n_classes = int(y_train.nunique())
    if n_classes < 2:
        raise RuntimeError(
            f"Only {n_classes} class in y_train — need ≥2. "
            f"Tambah data atau perbesar holdout supaya class minoritas ada di train."
        )

    model = lgb.LGBMClassifier(
        objective="multiclass" if n_classes >= 3 else "binary",
        num_class=n_classes if n_classes >= 3 else None,
        learning_rate=params.learning_rate,
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples,
        class_weight="balanced",
        random_state=params.random_state,
        n_jobs=params.n_jobs,
        verbose=-1,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    # Recall kelas Bahaya (label 2) — metrik penting untuk early warning
    recall_bahaya = float(
        recall_score(y_test, y_pred, labels=[2], average="macro", zero_division=0)
    )

    metrics = {
        "f1_macro": f1_macro,
        "recall_bahaya": recall_bahaya,
        "train_time_seconds": train_time,
        "n_train_rows": float(len(X_train)),
        "n_test_rows": float(len(X_test)),
        "n_features": float(X_train.shape[1]),
        "n_classes": float(n_classes),
    }
    LOG.info("Classifier metrics: %s", metrics)
    LOG.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, zero_division=0),
    )
    return model, metrics


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------
def _setup_mlflow(experiment_name: str, tracking_uri: Optional[str] = None) -> None:
    """Konfigurasi MLflow tracking URI + experiment."""
    import mlflow

    uri = (
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or f"file://{_PROJECT_ROOT}/mlruns"
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    LOG.info("MLflow tracking_uri=%s experiment=%s", uri, experiment_name)


def run_experiment(
    params: TrainParams,
    features_df: pd.DataFrame,
    experiment_name: str = "fireguard-ct",
    holdout_days: int = 14,
    register: bool = False,
) -> dict[str, float]:
    """Eksekusi 1 training run end-to-end dengan MLflow logging."""
    import mlflow

    _setup_mlflow(experiment_name)
    train_df, test_df = _time_aware_split(features_df, holdout_days)

    with mlflow.start_run() as run:
        # 1. Log params (semua hyperparameter)
        mlflow.log_params(asdict(params))
        mlflow.log_param("holdout_days", holdout_days)
        mlflow.log_param("dataset_n_rows", len(features_df))
        mlflow.log_param("dataset_n_provinces", features_df["province_id"].nunique())

        # 2. Train + evaluate
        if params.algorithm == "regressor":
            model, metrics = train_regressor(train_df, test_df, params)
            model_name = "fireguard-regressor"
        elif params.algorithm == "classifier":
            model, metrics = train_classifier(train_df, test_df, params)
            model_name = "fireguard-classifier"
        else:
            raise ValueError(
                f"algorithm must be 'regressor' or 'classifier', got {params.algorithm!r}"
            )

        # 3. Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # 4. Log model artifact (with input example untuk schema inference)
        feature_cols = [c for c in train_df.columns if c not in _NON_FEATURE_COLS]
        X_train_numeric = train_df[feature_cols].select_dtypes(include=[np.number])
        input_example = X_train_numeric.head(2)

        if params.algorithm == "regressor":
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=model_name if register else None,
            )
        else:
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=model_name if register else None,
            )

        LOG.info("Run %s logged successfully. Metrics: %s", run.info.run_id, metrics)
        return {"run_id": run.info.run_id, **metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FireGuard training (LK06)")
    p.add_argument("--algorithm", required=True, choices=["regressor", "classifier"])
    p.add_argument(
        "--features-dir", type=Path, default=_PROJECT_ROOT / "data" / "features"
    )
    p.add_argument("--experiment-name", default="fireguard-ct")
    p.add_argument("--holdout-days", type=int, default=14)
    p.add_argument(
        "--register",
        action="store_true",
        help="Register model ke MLflow Model Registry",
    )

    # Hyperparameters yang di-tune untuk variasi run
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Auto-load .env (untuk env vars seperti MLFLOW_TRACKING_URI custom)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    features_df = _read_latest_features(args.features_dir)
    if features_df.empty:
        LOG.error("Empty features dataset")
        return 1

    params = TrainParams(
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
    )

    try:
        result = run_experiment(
            params,
            features_df,
            experiment_name=args.experiment_name,
            holdout_days=args.holdout_days,
            register=args.register,
        )
        LOG.info("Training done: %s", result)
        print(f"Run ID: {result['run_id']}")
        return 0
    except Exception as e:  # noqa: BLE001
        LOG.error("Training failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
