"""
FireGuard LK12 — Closed-loop Continuous Training driver.

Tugas:
    1. (Opsional) DVC pull → ambil data feature terbaru
    2. Trigger training via `python -m src.models.train`
    3. Ambil RMSE model baru dari MLflow run terakhir
    4. Ambil RMSE model "Production" saat ini dari MLflow Registry
    5. Bandingkan: kalau new_rmse < prod_rmse * (1 - improvement_threshold),
       promote ke Production. Kalau tidak, biarkan tetap di Staging.
    6. Log semua keputusan ke stdout (juga ke MLflow run sebagai tag).

Usage:
    # Manual trigger
    python -m src.scripts.auto_retrain --reason manual

    # Dari webhook receiver (LK12)
    python -m src.scripts.auto_retrain --reason "grafana_alert:latency_high"

    # Dry-run (skip training, hanya simulasi keputusan)
    python -m src.scripts.auto_retrain --dry-run

    # Skip DVC pull (untuk testing kalau DVC remote belum siap)
    python -m src.scripts.auto_retrain --skip-dvc

Env vars (optional override):
    MLFLOW_TRACKING_URI       default: http://localhost:5000
    FIREGUARD_MODEL_NAME      default: fireguard-regressor
    FIREGUARD_IMPROVEMENT_PCT default: 0.02 (= 2% improvement minimal)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
except ImportError as e:
    print(f"FATAL: mlflow not installed → {e}", file=sys.stderr)
    sys.exit(2)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [auto_retrain] %(message)s",
)
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi default (env-overridable)
# ---------------------------------------------------------------------------
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("FIREGUARD_MODEL_NAME", "fireguard-regressor")
IMPROVEMENT_PCT = float(os.getenv("FIREGUARD_IMPROVEMENT_PCT", "0.02"))
TRAIN_TIMEOUT_S = int(os.getenv("FIREGUARD_TRAIN_TIMEOUT_S", "1800"))  # 30 min
PRIMARY_METRIC = os.getenv("FIREGUARD_METRIC_NAME", "rmse")
METRIC_LOWER_IS_BETTER = True  # RMSE: smaller = better

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name: str
    ok: bool
    detail: str = ""
    elapsed_s: float = 0.0


@dataclass
class CTRunReport:
    triggered_at: str
    reason: str
    stages: list
    new_run_id: Optional[str] = None
    new_metric: Optional[float] = None
    prod_run_id: Optional[str] = None
    prod_metric: Optional[float] = None
    decision: str = "pending"  # "promote" | "keep-staging" | "abort"
    promoted_version: Optional[int] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------
def run_subprocess(
    cmd: list, *, cwd: Path, timeout: int, env: Optional[dict] = None
) -> tuple[int, str]:
    """Subprocess wrapper aman (no shell=True, captured, timeout)."""
    LOG.info("Exec: %s (cwd=%s)", " ".join(cmd), cwd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **(env or {})},
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s: {e}"
    except FileNotFoundError as e:
        return 127, f"Command not found: {e}"


def stage_dvc_pull(skip: bool) -> StageResult:
    started = time.perf_counter()
    if skip:
        return StageResult("dvc_pull", True, "skipped (--skip-dvc)", 0.0)

    rc, out = run_subprocess(
        ["dvc", "pull", "--quiet"],
        cwd=REPO_ROOT,
        timeout=300,
    )
    elapsed = time.perf_counter() - started
    if rc == 0:
        return StageResult("dvc_pull", True, "ok", elapsed)
    # DVC pull bisa gagal kalau remote belum ada data — itu OK untuk CT awal
    LOG.warning("dvc pull rc=%d (lanjut, mungkin remote masih empty):\n%s",
                rc, out[-500:])
    return StageResult("dvc_pull", True, f"warn rc={rc}", elapsed)


def stage_prepare_features(synthetic: bool) -> StageResult:
    """
    Bangun data/features SEBELUM train.

    - synthetic=True  → generate synthetic dataset (CI tanpa real data).
    - synthetic=False → pipeline real: raw → processed (preprocess)
                        → features (build_features).

    Tanpa stage ini, train.py gagal dengan FileNotFoundError karena
    data/features/training_dataset_*.parquet belum ada.

    Catatan: ini terpisah dari --skip-dvc. --skip-dvc hanya melewati
    dvc pull internal (mis. karena CI sudah pull duluan), sedangkan
    synthetic menentukan SUMBER data feature.
    """
    started = time.perf_counter()

    if synthetic:
        rc, out = run_subprocess(
            [sys.executable, "-m", "src.utils.synthetic_data"],
            cwd=REPO_ROOT,
            timeout=300,
        )
        elapsed = time.perf_counter() - started
        if rc != 0:
            return StageResult("prepare_features", False,
                               f"synthetic gen failed rc={rc}\n{out[-800:]}",
                               elapsed)
        return StageResult("prepare_features", True, "synthetic ok", elapsed)

    for mod in ("src.data.preprocess", "src.features.build_features"):
        rc, out = run_subprocess(
            [sys.executable, "-m", mod],
            cwd=REPO_ROOT,
            timeout=600,
        )
        if rc != 0:
            return StageResult("prepare_features", False,
                               f"{mod} failed rc={rc}\n{out[-800:]}",
                               time.perf_counter() - started)
    return StageResult("prepare_features", True,
                       "raw->processed->features ok",
                       time.perf_counter() - started)


def stage_train(dry_run: bool) -> StageResult:
    started = time.perf_counter()
    if dry_run:
        return StageResult("train", True, "dry-run skipped", 0.0)

    # train.py wajib --algorithm. JANGAN pakai --register di sini karena
    # auto_retrain.py akan handle register + transition setelah evaluasi.
    rc, out = run_subprocess(
        [sys.executable, "-m", "src.models.train",
         "--algorithm", "regressor"],
        cwd=REPO_ROOT,
        timeout=TRAIN_TIMEOUT_S,
    )
    elapsed = time.perf_counter() - started
    if rc != 0:
        tail = out[-1500:] if out else "(no output)"
        return StageResult("train", False,
                           f"train failed rc={rc}\n--- tail ---\n{tail}",
                           elapsed)
    return StageResult("train", True, "ok", elapsed)


def _find_latest_run_id(client: MlflowClient, exp_name: str = "Default") -> Optional[str]:
    """Cari MLflow run terakhir di experiment manapun (fallback default)."""
    # Coba experiment "Default" dulu, kalau tidak ada, ambil yang paling baru
    try:
        # Cari di semua experiment, urut by created_at descending
        exps = client.search_experiments()
        for exp in exps:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=1,
            )
            if runs:
                return runs[0].info.run_id
    except MlflowException as e:
        LOG.warning("Could not search runs: %s", e)
    return None


def stage_evaluate_new(
    client: MlflowClient, new_run_id: Optional[str]
) -> tuple[StageResult, Optional[float], Optional[str]]:
    """Ambil metric RMSE dari run terakhir (atau dari run_id eksplisit)."""
    started = time.perf_counter()
    rid = new_run_id or _find_latest_run_id(client)
    if not rid:
        return (StageResult("evaluate_new", False, "no run found", time.perf_counter() - started),
                None, None)
    try:
        run = client.get_run(rid)
        metric = run.data.metrics.get(PRIMARY_METRIC)
        if metric is None:
            return (StageResult("evaluate_new", False,
                                f"metric '{PRIMARY_METRIC}' tidak ada di run {rid}",
                                time.perf_counter() - started),
                    None, rid)
        return (StageResult("evaluate_new", True, f"{PRIMARY_METRIC}={metric:.4f}",
                            time.perf_counter() - started),
                float(metric), rid)
    except MlflowException as e:
        return (StageResult("evaluate_new", False, f"mlflow error: {e}",
                            time.perf_counter() - started),
                None, rid)


def stage_get_production(
    client: MlflowClient,
) -> tuple[StageResult, Optional[float], Optional[str], Optional[int]]:
    """Ambil RMSE & run_id dari current Production model."""
    started = time.perf_counter()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except MlflowException as e:
        return (StageResult("get_production", True,
                            f"no production yet (first-time CT): {e}",
                            time.perf_counter() - started),
                None, None, None)
    if not versions:
        return (StageResult("get_production", True, "no production version",
                            time.perf_counter() - started),
                None, None, None)

    v = versions[0]
    try:
        prod_run = client.get_run(v.run_id)
        prod_metric = prod_run.data.metrics.get(PRIMARY_METRIC)
        return (StageResult("get_production", True,
                            f"v{v.version} {PRIMARY_METRIC}={prod_metric}",
                            time.perf_counter() - started),
                float(prod_metric) if prod_metric is not None else None,
                v.run_id, int(v.version))
    except MlflowException as e:
        return (StageResult("get_production", False, f"mlflow error: {e}",
                            time.perf_counter() - started),
                None, v.run_id, int(v.version))


def stage_decide(
    new_metric: Optional[float],
    prod_metric: Optional[float],
    improvement_pct: float,
) -> tuple[str, str]:
    """Decision logic — return (decision_label, reason_text)."""
    if new_metric is None:
        return "abort", f"new model {PRIMARY_METRIC} unavailable"

    if prod_metric is None:
        # Tidak ada Production sebelumnya → first-time promote
        return "promote", "no existing Production — first promotion"

    if METRIC_LOWER_IS_BETTER:
        threshold = prod_metric * (1.0 - improvement_pct)
        if new_metric < threshold:
            improvement = (prod_metric - new_metric) / prod_metric * 100
            return ("promote",
                    f"new {new_metric:.4f} < threshold {threshold:.4f} "
                    f"(improvement {improvement:.2f}% vs prod {prod_metric:.4f})")
        return ("keep-staging",
                f"new {new_metric:.4f} >= threshold {threshold:.4f} "
                f"(no improvement {improvement_pct*100:.0f}% vs prod {prod_metric:.4f})")
    else:
        # Higher-is-better (e.g. accuracy)
        threshold = prod_metric * (1.0 + improvement_pct)
        if new_metric > threshold:
            return ("promote",
                    f"new {new_metric:.4f} > threshold {threshold:.4f}")
        return ("keep-staging",
                f"new {new_metric:.4f} <= threshold {threshold:.4f}")


def stage_promote(client: MlflowClient, new_run_id: str) -> tuple[StageResult, Optional[int]]:
    """Register new model + transition ke Production, archive existing."""
    started = time.perf_counter()
    try:
        # Daftarkan model dari run baru
        model_uri = f"runs:/{new_run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        new_version = int(mv.version)

        # Transition ke Production + archive yang lama
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )
        return (StageResult("promote", True,
                            f"v{new_version} → Production",
                            time.perf_counter() - started),
                new_version)
    except MlflowException as e:
        return (StageResult("promote", False, f"mlflow error: {e}",
                            time.perf_counter() - started),
                None)


def stage_keep_staging(client: MlflowClient, new_run_id: str) -> tuple[StageResult, Optional[int]]:
    """Register tapi parkir di Staging (tidak promote)."""
    started = time.perf_counter()
    try:
        model_uri = f"runs:/{new_run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        new_version = int(mv.version)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Staging",
            archive_existing_versions=False,
        )
        return (StageResult("keep_staging", True,
                            f"v{new_version} → Staging (tidak promote)",
                            time.perf_counter() - started),
                new_version)
    except MlflowException as e:
        return (StageResult("keep_staging", False, f"mlflow error: {e}",
                            time.perf_counter() - started),
                None)


def tag_run_with_decision(client: MlflowClient, run_id: str, report: CTRunReport) -> None:
    """Set tags pada run untuk audit trail."""
    if not run_id:
        return
    try:
        client.set_tag(run_id, "ct.triggered_at", report.triggered_at)
        client.set_tag(run_id, "ct.reason", report.reason)
        client.set_tag(run_id, "ct.decision", report.decision)
        if report.prod_metric is not None:
            client.set_tag(run_id, "ct.prev_prod_metric", f"{report.prod_metric:.6f}")
    except MlflowException as e:
        LOG.warning("Failed to tag run: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FireGuard CT auto-retrain driver")
    p.add_argument("--reason", default="manual",
                   help="Trigger reason (logged ke tag MLflow run)")
    p.add_argument("--skip-dvc", action="store_true",
                   help="Skip dvc pull internal (mis. CI sudah pull duluan)")
    p.add_argument("--synthetic", action="store_true",
                   help="Pakai synthetic data untuk feature (bukan real "
                        "preprocess->build_features)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip training, hanya simulasi keputusan")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                   help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI})")
    p.add_argument("--improvement-pct", type=float, default=IMPROVEMENT_PCT,
                   help=f"Min improvement required to promote "
                        f"(default: {IMPROVEMENT_PCT})")
    p.add_argument("--report-json", default=None,
                   help="Tulis report JSON ke path ini (untuk CI consume)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    report = CTRunReport(
        triggered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reason=args.reason,
        stages=[],
    )

    LOG.info("=" * 64)
    LOG.info("FireGuard CT — triggered: %s", report.triggered_at)
    LOG.info("Reason : %s", args.reason)
    LOG.info("Tracker: %s", args.tracking_uri)
    LOG.info("Model  : %s", MODEL_NAME)
    LOG.info("Improv : %.2f%%", args.improvement_pct * 100)
    LOG.info("Dry run: %s", args.dry_run)
    LOG.info("=" * 64)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Stage 1: DVC pull
    s = stage_dvc_pull(args.skip_dvc)
    report.stages.append(asdict(s))
    if not s.ok:
        report.error = f"DVC pull failed: {s.detail}"
        report.decision = "abort"
        return _finish(report, args, client)

    # Stage 1.5: Prepare features (raw->processed->features, atau synthetic)
    s = stage_prepare_features(args.synthetic)
    report.stages.append(asdict(s))
    if not s.ok:
        report.error = f"Prepare features failed: {s.detail}"
        report.decision = "abort"
        return _finish(report, args, client)

    # Stage 2: Train
    s = stage_train(args.dry_run)
    report.stages.append(asdict(s))
    if not s.ok:
        report.error = f"Training failed: {s.detail}"
        report.decision = "abort"
        return _finish(report, args, client)

    # Stage 3: Evaluate new
    s, new_metric, new_run_id = stage_evaluate_new(client, None)
    report.stages.append(asdict(s))
    report.new_run_id = new_run_id
    report.new_metric = new_metric
    if not s.ok and not args.dry_run:
        report.error = f"Evaluate failed: {s.detail}"
        report.decision = "abort"
        return _finish(report, args, client)

    # Stage 4: Get production
    s, prod_metric, prod_run_id, _prod_version = stage_get_production(client)
    report.stages.append(asdict(s))
    report.prod_metric = prod_metric
    report.prod_run_id = prod_run_id

    # Stage 5: Decide
    if args.dry_run:
        decision = "dry-run"
        reason_text = "skipped — --dry-run mode"
    else:
        decision, reason_text = stage_decide(new_metric, prod_metric,
                                             args.improvement_pct)
    report.decision = decision
    report.stages.append(asdict(StageResult("decide", True, reason_text, 0.0)))
    LOG.info("DECISION: %s — %s", decision, reason_text)

    # Stage 6: Action
    if decision == "promote" and new_run_id:
        s, ver = stage_promote(client, new_run_id)
        report.stages.append(asdict(s))
        report.promoted_version = ver
    elif decision == "keep-staging" and new_run_id:
        s, ver = stage_keep_staging(client, new_run_id)
        report.stages.append(asdict(s))
        report.promoted_version = ver  # version-nya ada tapi di Staging
    # decision == abort/dry-run → no action

    return _finish(report, args, client)


def _finish(report: CTRunReport, args: argparse.Namespace,
            client: MlflowClient) -> int:
    # Tag run dengan decision
    if report.new_run_id:
        tag_run_with_decision(client, report.new_run_id, report)

    # Tulis JSON report jika diminta
    if args.report_json:
        try:
            Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
            LOG.info("Report written → %s", args.report_json)
        except OSError as e:
            LOG.warning("Could not write report: %s", e)

    # Print summary
    LOG.info("=" * 64)
    LOG.info("CT FINAL: %s", report.decision)
    if report.new_metric is not None:
        LOG.info("  new %s    : %.4f", PRIMARY_METRIC, report.new_metric)
    if report.prod_metric is not None:
        LOG.info("  prod %s   : %.4f", PRIMARY_METRIC, report.prod_metric)
    if report.promoted_version is not None:
        LOG.info("  version   : %d", report.promoted_version)
    if report.error:
        LOG.error("  error     : %s", report.error)
    LOG.info("=" * 64)

    # Exit code: 0 untuk promote/keep-staging/dry-run, 1 untuk abort
    return 0 if report.decision in ("promote", "keep-staging", "dry-run") else 1


if __name__ == "__main__":
    sys.exit(main())
