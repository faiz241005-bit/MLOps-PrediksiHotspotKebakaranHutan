"""
MLflow Model Registry helpers — stage transition + load production model.

Tugas LK07: kelola siklus hidup model dari eksperimen ke produksi.
- Transition stage: None → Staging → Production → Archived
- Load model production untuk inferensi
- Promote run terbaik ke registry

CLI examples:
    python -m src.models.registry list --model fireguard-regressor
    python -m src.models.registry transition \
        --model fireguard-regressor --version 1 --stage Staging
    python -m src.models.registry promote-best \
        --model fireguard-regressor --metric rmse --lower-better
    python -m src.models.registry load \
        --model fireguard-regressor --stage Production
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VALID_STAGES = {"None", "Staging", "Production", "Archived"}


def _setup_mlflow(tracking_uri: Optional[str] = None) -> None:
    """Configure MLflow tracking URI."""
    import mlflow
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or f"file://{_PROJECT_ROOT}/mlruns"
    mlflow.set_tracking_uri(uri)
    LOG.info("MLflow tracking_uri=%s", uri)


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------
def list_versions(model_name: str) -> list[dict]:
    """List semua versi dari registered model."""
    import mlflow
    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:  # noqa: BLE001
        LOG.error("Failed to list versions for %s: %s", model_name, e)
        return []

    result = []
    for v in versions:
        result.append({
            "name": v.name,
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "status": v.status,
            "creation_time": v.creation_timestamp,
        })
    return sorted(result, key=lambda x: int(x["version"]))


def transition_stage(model_name: str, version: int, stage: str,
                     archive_existing: bool = True) -> None:
    """
    Transition model version ke stage baru.

    Args:
        model_name: registered model name (e.g., 'fireguard-regressor')
        version: versi yang akan di-transition
        stage: target stage — 'Staging', 'Production', 'Archived', atau 'None'
        archive_existing: jika True, archive versi sebelumnya yang ada di stage yang sama
    """
    if stage not in _VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(_VALID_STAGES)}, got {stage!r}")

    import mlflow
    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing and stage in ("Staging", "Production"),
    )
    LOG.info("Transitioned %s v%s → %s (archive_existing=%s)",
             model_name, version, stage, archive_existing)


def promote_best_run(
    model_name: str, experiment_name: str = "fireguard-ct",
    metric: str = "rmse", lower_better: bool = True,
    target_stage: str = "Staging",
) -> Optional[dict]:
    """
    Cari run terbaik di experiment, register sebagai versi baru,
    transition ke target_stage.

    Returns: dict dengan info versi baru, atau None kalau tidak ada run.
    """
    import mlflow
    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        LOG.error("Experiment %r not found", experiment_name)
        return None

    # Sort order: ASC untuk lower_better, DESC untuk higher_better
    order_by = f"metrics.{metric} {'ASC' if lower_better else 'DESC'}"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[order_by], max_results=1,
    )
    if not runs:
        LOG.error("No runs found in experiment %r", experiment_name)
        return None

    best_run = runs[0]
    best_metric = best_run.data.metrics.get(metric)
    LOG.info("Best run %s: %s=%s",
             best_run.info.run_id[:8], metric, best_metric)

    # Register model dari run terbaik
    model_uri = f"runs:/{best_run.info.run_id}/model"
    try:
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    except Exception as e:  # noqa: BLE001
        LOG.error("Failed to register model: %s", e)
        return None

    # Transition ke target stage
    transition_stage(model_name, int(registered.version), target_stage)

    return {
        "name": registered.name,
        "version": registered.version,
        "stage": target_stage,
        "run_id": best_run.info.run_id,
        f"{metric}": best_metric,
    }


# ---------------------------------------------------------------------------
# Inferensi — load production model
# ---------------------------------------------------------------------------
def load_production_model(model_name: str, stage: str = "Production"):
    """
    Load model dari registry berdasarkan stage.

    Returns: mlflow.pyfunc.PyFuncModel — punya method predict().
    """
    import mlflow
    _setup_mlflow()

    model_uri = f"models:/{model_name}/{stage}"
    LOG.info("Loading model from %s", model_uri)
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        LOG.info("Model loaded successfully: %s", type(model).__name__)
        return model
    except Exception as e:  # noqa: BLE001
        LOG.error("Failed to load model: %s", e)
        raise


def predict_demo(model_name: str = "fireguard-regressor",
                 stage: str = "Production",
                 n_samples: int = 3) -> pd.DataFrame:
    """
    Demo inferensi: load model production, predict pada synthetic input.

    Returns: DataFrame dengan kolom prediksi.
    """
    model = load_production_model(model_name, stage)

    # Generate synthetic input matching feature schema
    # (Untuk demo saja — di produksi pakai data terbaru dari preprocess pipeline)
    rng = np.random.default_rng(42)
    synthetic = pd.DataFrame({
        "hotspot_count": rng.integers(0, 30, n_samples),
        "frp_mean": rng.uniform(5, 30, n_samples),
        "frp_max": rng.uniform(10, 50, n_samples),
        "frp_sum": rng.uniform(50, 500, n_samples),
        "n_daytime": rng.integers(0, 30, n_samples),
        "n_nighttime": rng.integers(0, 30, n_samples),
        "n_confidence_high": rng.integers(0, 30, n_samples),
        "temperature_2m_max": rng.uniform(28, 35, n_samples),
        "temperature_2m_min": rng.uniform(22, 26, n_samples),
        "precipitation_sum": rng.uniform(0, 10, n_samples),
        "windspeed_10m_max": rng.uniform(5, 20, n_samples),
        "winddirection_10m_dominant": rng.uniform(0, 360, n_samples),
        "relative_humidity_2m_mean": rng.uniform(60, 90, n_samples),
        "month": rng.integers(1, 13, n_samples),
        "day_of_year": rng.integers(1, 366, n_samples),
        "month_sin": rng.uniform(-1, 1, n_samples),
        "month_cos": rng.uniform(-1, 1, n_samples),
        "hotspot_count_1d": rng.integers(0, 30, n_samples).astype(float),
        "hotspot_count_3d": rng.integers(0, 90, n_samples).astype(float),
        "hotspot_count_7d": rng.integers(0, 200, n_samples).astype(float),
        "frp_mean_1d": rng.uniform(5, 30, n_samples),
        "frp_mean_3d": rng.uniform(5, 30, n_samples),
        "frp_mean_7d": rng.uniform(5, 30, n_samples),
        "hotspot_count_lag_1d": rng.integers(0, 30, n_samples).astype(float),
        "hotspot_count_lag_3d": rng.integers(0, 30, n_samples).astype(float),
        "hotspot_count_lag_7d": rng.integers(0, 30, n_samples).astype(float),
        "days_since_rain": rng.integers(0, 10, n_samples),
    })

    LOG.info("Predicting on %d synthetic samples", n_samples)
    preds = model.predict(synthetic)
    LOG.info("Predictions: %s", preds)

    result = synthetic.copy()
    result["prediction"] = preds
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FireGuard MLflow Model Registry CLI (LK07)")
    sub = p.add_subparsers(dest="command", required=True)

    # list
    sp = sub.add_parser("list", help="List versions of a registered model")
    sp.add_argument("--model", required=True)

    # transition
    sp = sub.add_parser("transition", help="Transition stage")
    sp.add_argument("--model", required=True)
    sp.add_argument("--version", type=int, required=True)
    sp.add_argument("--stage", required=True, choices=sorted(_VALID_STAGES))
    sp.add_argument("--no-archive", action="store_true",
                    help="Jangan archive existing versi di stage yang sama")

    # promote-best
    sp = sub.add_parser("promote-best", help="Register & promote run terbaik")
    sp.add_argument("--model", required=True)
    sp.add_argument("--experiment", default="fireguard-ct")
    sp.add_argument("--metric", default="rmse")
    sp.add_argument("--higher-better", action="store_true",
                    help="Default: lower is better (RMSE/MAE)")
    sp.add_argument("--target-stage", default="Staging",
                    choices=["Staging", "Production"])

    # load
    sp = sub.add_parser("load", help="Load model + run prediction demo")
    sp.add_argument("--model", required=True)
    sp.add_argument("--stage", default="Production",
                    choices=["Staging", "Production"])
    sp.add_argument("--n-samples", type=int, default=3)

    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
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

    try:
        if args.command == "list":
            for v in list_versions(args.model):
                print(f"v{v['version']:>3} stage={v['stage']:<12} "
                      f"run_id={v['run_id'][:8]}... status={v['status']}")

        elif args.command == "transition":
            transition_stage(args.model, args.version, args.stage,
                             archive_existing=not args.no_archive)

        elif args.command == "promote-best":
            result = promote_best_run(
                args.model, experiment_name=args.experiment,
                metric=args.metric, lower_better=not args.higher_better,
                target_stage=args.target_stage,
            )
            if result is None:
                return 1
            print(f"Promoted {result['name']} v{result['version']} → {result['stage']}")
            print(f"  Run: {result['run_id'][:8]}...  {args.metric}={result.get(args.metric)}")

        elif args.command == "load":
            df = predict_demo(args.model, stage=args.stage, n_samples=args.n_samples)
            print(df[["hotspot_count", "frp_mean", "prediction"]].to_string(index=False))

        return 0
    except Exception as e:  # noqa: BLE001
        LOG.error("Command failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
