# LK08 — Panduan CI/CD Automation Pipeline

> Walkthrough lengkap "Code as Trigger" — workflow GitHub Actions yang otomatis
> trigger test → train → evaluate → register saat ada push/PR.

---

## Konsep

```
[Commit / Push]
       │
       ▼
[GitHub Actions Trigger]
       │
   ┌───┴───┐
   ▼       ▼
Test    PR check
   │
   ▼
[Train (synthetic data fallback OR DVC pull)]
   │
   ▼
[Evaluate vs Threshold (LK01)]
   │
   ▼
[Auto-Register kalau lolos → Staging]
   │
   ▼
[Workflow Summary]
```

## Prasyarat

- Branch aktif: `feat/registry-cicd` (atau `main` setelah merge)
- File `.github/workflows/mlops-automation.yaml` sudah di-commit
- Helper scripts ready: `src/utils/synthetic_data.py`, `src/models/evaluate.py`, `src/models/registry.py`

---

## Tahap 1 — Konfigurasi Trigger (sudah dibuat)

File `.github/workflows/mlops-automation.yaml` punya trigger:

```yaml
on:
  push:
    branches: [main]
    paths-ignore:
      - 'docs/**'
      - '*.md'
      - 'LK0*.docx'
  pull_request:
    branches: [main]
  workflow_dispatch:
```

**Path-ignore** menghindari trigger workflow untuk perubahan dokumentasi (cost & noise reduction). `workflow_dispatch` memungkinkan manual trigger via UI untuk testing.

---

## Tahap 2 — Job 1: Automated Testing

Job pertama jalankan `pytest` untuk verifikasi integritas kode:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v --tb=short
```

Kalau ada test gagal, workflow stop di sini — train tidak akan jalan.

---

## Tahap 3 — Job 2: Automated Training

Job ini punya 3 fitur penting:

### A. MinIO sebagai Service Container

```yaml
services:
  minio:
    image: bitnami/minio:2024
    ports: [9000:9000]
    options: >-
      --health-cmd "curl -f http://localhost:9000/minio/health/live"
```

MinIO berjalan sebagai **service container** di dalam workflow — S3-compatible object storage yang **bisa diakses langsung** oleh job. Demonstrate bahwa workflow siap pakai DVC remote production (R2/S3/MinIO Cloud).

### B. Synthetic Data Fallback

```yaml
- name: Generate synthetic features (CI fallback for DVC remote)
  run: |
    python -m src.utils.synthetic_data --n-provinces 5 --n-days 60 --seed 42
```

Generate 300 baris (5 provinsi × 60 hari) synthetic dataset realistic dengan distribusi Poisson untuk hotspot count + seasonal multiplier. Cukup untuk LightGBM training meaningful tanpa external dependency.

### C. Output Run ID untuk Job Berikutnya

```yaml
- id: train_step
  run: |
    OUTPUT=$(python -m src.models.train --algorithm regressor ...)
    RUN_ID=$(echo "$OUTPUT" | grep -oE "Run ID: [a-f0-9]+" | awk '{print $3}')
    echo "run_id=$RUN_ID" >> $GITHUB_OUTPUT
```

Run ID propagated via `needs.train.outputs.run_id` ke Job 3 evaluation.

---

## Tahap 4 — Job 3: Evaluation & Validation

Compare metric dengan threshold dari LK01 (`src/models/evaluate.py`):

```yaml
- run: |
    python -m src.models.evaluate \
      --run-id "${{ needs.train.outputs.run_id }}" \
      --ci-mode \
      --output-json eval_result.json
```

**`--ci-mode`** pakai threshold lebih lenient untuk synthetic data (RMSE ≤ 50, MAE ≤ 30, F1 ≥ 0.30). Mode normal pakai threshold LK01 (RMSE ≤ 12, MAE ≤ 8, F1 ≥ 0.78, Recall Bahaya ≥ 0.85).

**Exit code:**
- `0` → passed, lanjut Job 4
- `1` → failed → workflow gagal di sini, register tidak jalan

---

## Tahap 5 — Job 4: Auto-Registry Update

Kalau evaluasi sukses, auto-register ke MLflow Registry stage `Staging`:

```yaml
register:
  needs: [train, evaluate]
  if: success()    # hanya run kalau semua job sebelumnya OK
  steps:
    - run: |
        python -m src.models.registry promote-best \
          --model fireguard-regressor \
          --experiment fireguard-ct-ci \
          --metric rmse \
          --target-stage Staging
```

Model otomatis muncul di MLflow Registry sebagai `fireguard-regressor` versi baru dengan stage `Staging`. Production deployment butuh approval manual (best practice MLOps).

---

## Tahap 6 — Workflow Summary

Job terakhir tulis ringkasan di **GitHub Actions Summary** (visible di UI):

```yaml
summary:
  if: always()
  steps:
    - run: |
        echo "## FireGuard MLOps Pipeline" >> $GITHUB_STEP_SUMMARY
        echo "| Stage | Status |" >> $GITHUB_STEP_SUMMARY
        ...
```

Setiap workflow run akan punya tab Summary dengan tabel status.

---

## Tahap 7 — Simulasi: Trigger Workflow via Commit

Untuk demonstrasikan **Code as Trigger**, lakukan modifikasi kecil di kode:

### Step 1 — Buat Perubahan Trivial

```bash
# Di laptop atau Codespace
cd ~/Documents/Claude/Projects/MLOps/MLOps-FireGuard   # atau Codespace path

# Edit src/models/train.py: ubah default n_estimators
# Misalnya dari 500 ke 600
sed -i 's/n_estimators: int = 500/n_estimators: int = 600/' src/models/train.py

# Atau edit manual via VS Code

git diff src/models/train.py
```

### Step 2 — Commit & Push

```bash
git add src/models/train.py
git commit -m "chore(train): bump default n_estimators 500 → 600 (trigger CI demo)"
git push origin feat/registry-cicd
```

### Step 3 — Pantau Workflow di GitHub Actions

1. Buka https://github.com/faiz241005-bit/MLOps-PrediksiHotspotKebakaranHutan/actions
2. Akan muncul workflow run baru "MLOps Automation Pipeline (LK08)"
3. Klik untuk masuk detail — terlihat 5 job (Test → Train → Evaluate → Register → Summary)
4. Tunggu ~3-5 menit sampai selesai
5. Status akhir: **success** (semua job hijau)

**Screenshot:**
- Halaman Actions list dengan workflow run terbaru
- Halaman detail run dengan 5 job (semua hijau)
- Tab Summary dengan tabel status

---

## Tahap 8 — Verifikasi Auto-Register Berhasil

Setelah workflow selesai, model baru harusnya terdaftar. Cek dari Codespace:

```bash
python -m src.models.registry list --model fireguard-regressor
```

Output akan menampilkan versi tambahan dari CI run, stage `Staging`.

---

## Tahap 9 — Production Upgrade Path (Opsional / Catatan)

Workflow saat ini pakai **synthetic data** untuk demonstrate konsep. Production upgrade:

### Opsi A — Pakai DVC Remote Cloud (R2/Filebase/S3)

Replace step "Generate synthetic features" dengan:

```yaml
- name: Configure DVC + pull data
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.DVC_REMOTE_ACCESS_KEY }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.DVC_REMOTE_SECRET_KEY }}
  run: |
    dvc remote modify --local production endpointurl https://...
    dvc pull data/features
```

Set secrets di GitHub repo: **Settings → Secrets and variables → Actions**.

### Opsi B — Pakai NASA FIRMS API langsung di CI

```yaml
- name: Fetch real data
  env:
    NASA_FIRMS_API_KEY: ${{ secrets.NASA_FIRMS_API_KEY }}
  run: |
    python -m src.data.ingest_data --provinces all
    python -m src.data.preprocess
    python -m src.features.build_features
```

API key sudah di GitHub Secrets (set saat LK04/LK06).

---

## Troubleshooting

### Workflow gagal di Job "Test"

Test fail biasanya karena code change merusak existing behavior. Run lokal dulu:
```bash
python -m pytest tests/ -v
```

### Workflow gagal di Job "Evaluate"

Synthetic data terlalu noisy → RMSE > threshold lenient CI. Tweak:
- `--seed` di synthetic_data untuk dataset lebih predictable
- Threshold di `src/models/evaluate.py` (class `Thresholds`)

### Workflow gagal di MinIO service container

MinIO image gagal pull atau health check. Cek log workflow → kalau image issue, fallback ke `minio/minio:latest`.

### Self-hosted runner (advanced)

Untuk demonstrate `dvc pull` dari /tmp Codespace, bisa setup self-hosted runner di Codespace (settings → Actions → Runners → New self-hosted runner). Tapi setup ribet dan tidak direkomendasikan untuk demo akademik.

---

## Bukti Submission LK08

| # | Instruksi | Bukti |
|---|---|---|
| 1 | Konfigurasi Trigger | `.github/workflows/mlops-automation.yaml` dengan `on: push/PR/manual` |
| 2 | Automated Testing | Job "test" dengan pytest, screenshot Actions success |
| 3 | Automated Training | Job "train" dengan synthetic data + MinIO service, screenshot Actions success |
| 4 | Evaluation & Validation | Job "evaluate" dengan `--ci-mode` threshold, exit code gate |
| 5 | Auto-Registry update | Job "register" dengan `--target-stage Staging` |
| 6 | Simulasi Perubahan | Commit `n_estimators 500 → 600` trigger workflow, screenshot success run |

---

*LK08 selesai = workflow ter-trigger sukses + 6 bukti + LK08_FireGuard.docx submitted.*
