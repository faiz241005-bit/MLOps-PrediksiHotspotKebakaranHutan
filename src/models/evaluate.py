"""
Threshold validator — gate untuk LK08 CI/CD pipeline.

Tugas LK08 Tahap 3 & 4: compare metric model baru dengan ambang batas dari LK01.
Kalau lolos validasi, exit 0 (sukses, lanjut auto-register).
Kalau gagal, exit non-zero (block promotion, alert).

Threshold default (dari LK01 metrik target):
    - RMSE ≤ 12 hotspot
    - MAE ≤ 8 hotspot
    - F1 macro ≥ 0.78
    - Recall kelas Bahaya ≥ 0.85

Untuk dataset synthetic / kecil, threshold lebih lenient (CI-friendly).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Thresholds:
    """Ambang batas validasi metric (dari LK01)."""
    rmse_max: float = 12.0
    mae_max: float = 8.0
    f1_macro_min: float = 0.78
    recall_bahaya_min: float = 0.85
    # CI-friendly mode untuk dataset synthetic/kecil
    ci_mode_rmse_max: float = 50.0     # lenient untuk synthetic
    ci_mode_mae_max: float = 30.0
    ci_mode_f1_min: float = 0.30


def evaluate_run(run_id: str, thresholds: Thresholds, ci_mode: bool = False) -> dict:
    """
    Ambil metric dari MLflow run, compare dengan threshold.

    Returns: dict berisi:
        - run_id, metrics (dict), passed (bool), failed_checks (list)
    """
    import mlflow
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"file://{_PROJECT_ROOT}/mlruns"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Run {run_id} not found: {e}") from e

    metrics = dict(run.data.metrics)
    LOG.info("Run %s metrics: %s", run_id[:8], metrics)

    failed = []
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    f1 = metrics.get("f1_macro")
    recall_b = metrics.get("recall_bahaya")

    # Apply threshold (full atau CI-mode)
    rmse_max = thresholds.ci_mode_rmse_max if ci_mode else thresholds.rmse_max
    mae_max = thresholds.ci_mode_mae_max if ci_mode else thresholds.mae_max
    f1_min = thresholds.ci_mode_f1_min if ci_mode else thresholds.f1_macro_min

    # Check regressor metrics (kalau ada)
    if rmse is not None:
        if rmse > rmse_max:
            failed.append(f"rmse={rmse:.2f} > threshold {rmse_max}")
        else:
            LOG.info("✓ rmse=%.2f ≤ %s", rmse, rmse_max)
    if mae is not None:
        if mae > mae_max:
            failed.append(f"mae={mae:.2f} > threshold {mae_max}")
        else:
            LOG.info("✓ mae=%.2f ≤ %s", mae, mae_max)

    # Check classifier metrics (kalau ada)
    if f1 is not None:
        if f1 < f1_min:
            failed.append(f"f1_macro={f1:.3f} < threshold {f1_min}")
        else:
            LOG.info("✓ f1_macro=%.3f ≥ %s", f1, f1_min)
    if recall_b is not None and not ci_mode:
        if recall_b < thresholds.recall_bahaya_min:
            failed.append(
                f"recall_bahaya={recall_b:.3f} < threshold {thresholds.recall_bahaya_min}"
            )
        else:
            LOG.info("✓ recall_bahaya=%.3f ≥ %s", recall_b, thresholds.recall_bahaya_min)

    return {
        "run_id": run_id,
        "metrics": metrics,
        "thresholds": asdict(thresholds),
        "ci_mode": ci_mode,
        "passed": len(failed) == 0,
        "failed_checks": failed,
    }


def evaluate_latest_run(experiment_name: str = "fireguard-ct",
                        thresholds: Optional[Thresholds] = None,
                        ci_mode: bool = False) -> dict:
    """Ambil run paling baru di experiment, evaluasi."""
    import mlflow
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"file://{_PROJECT_ROOT}/mlruns"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {experiment_name!r} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"], max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No runs in experiment {experiment_name!r}")

    return evaluate_run(runs[0].info.run_id, thresholds or Thresholds(), ci_mode)


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate run metrics against thresholds (LK08 gate)"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="MLflow Run ID untuk evaluate")
    g.add_argument("--latest", action="store_true",
                   help="Evaluate run paling baru di experiment")
    p.add_argument("--experiment", default="fireguard-ct")
    p.add_argument("--ci-mode", action="store_true",
                   help="Pakai threshold lebih lenient untuk CI (synthetic data)")
    p.add_argument("--output-json", type=Path,
                   help="Tulis hasil evaluasi ke file JSON")
    p.add_argument("--log-level", default="INFO")
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

    thresholds = Thresholds()

    try:
        if args.latest:
            result = evaluate_latest_run(args.experiment, thresholds, args.ci_mode)
        else:
            result = evaluate_run(args.run_id, thresholds, args.ci_mode)
    except Exception as e:  # noqa: BLE001
        LOG.error("Evaluation failed: %s", e)
        return 2

    LOG.info("Evaluation result: passed=%s", result["passed"])
    if result["failed_checks"]:
        LOG.warning("Failed checks:")
        for check in result["failed_checks"]:
            LOG.warning("  ✗ %s", check)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str))
        LOG.info("Wrote evaluation JSON: %s", args.output_json)

    # Exit code: 0 = passed, 1 = failed (untuk CI gate)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
