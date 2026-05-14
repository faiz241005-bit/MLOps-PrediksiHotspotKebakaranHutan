# LK07 — Panduan Model Registry, Versioning, Stage Transition

> Walkthrough lengkap pemenuhan instruksi LK07 di Codespace + MLflow UI.

---

## Prasyarat

- LK06 selesai: model `fireguard-regressor v1` sudah registered di MLflow.
- Branch aktif: `feat/registry-cicd` (atau `main` setelah merge).
- `mlruns/` lokal sudah berisi run dari LK06.

---

## Tahap 1 — Registrasi Model (sudah dilakukan di LK06)

Verifikasi `fireguard-regressor v1` sudah terdaftar:

```bash
cd /workspaces/MLOps-PrediksiHotspotKebakaranHutan

# Pakai helper CLI yang baru
python -m src.models.registry list --model fireguard-regressor
```

Output yang Anda harapkan:
```
v  1 stage=None         run_id=<sha>... status=READY
```

Kalau belum ada (mis. Codespace baru), register manual via MLflow UI atau via CLI:

```bash
python -m src.models.registry promote-best \
    --model fireguard-regressor \
    --experiment fireguard-ct \
    --metric rmse \
    --target-stage None
```

---

## Tahap 2 — Train v2 dengan Params Berbeda

Latih ulang dengan kombinasi hyperparameter baru:

```bash
python -m src.models.train \
    --algorithm regressor \
    --learning-rate 0.03 \
    --n-estimators 700 \
    --max-depth 10 \
    --num-leaves 63 \
    --experiment-name fireguard-ct
```

Catat Run ID dari output. Lalu register sebagai v2:

```bash
# Cara 1: pakai helper (auto pilih best run di experiment)
python -m src.models.registry promote-best \
    --model fireguard-regressor \
    --experiment fireguard-ct \
    --metric rmse \
    --target-stage None

# Verifikasi
python -m src.models.registry list --model fireguard-regressor
```

Output:
```
v  1 stage=None         run_id=<sha-v1>...
v  2 stage=None         run_id=<sha-v2>...
```

**Bukti versioning (#2 instruksi):** dua versi terdaftar, run_id berbeda.

---

## Tahap 3 — Stage Transition

Simulasi alur industri: None → Staging → Production.

### Via CLI (lebih reproducible)

```bash
# Promote v1 ke Staging
python -m src.models.registry transition \
    --model fireguard-regressor \
    --version 1 \
    --stage Staging

# Promote v2 ke Production (auto-archive Staging lama)
python -m src.models.registry transition \
    --model fireguard-regressor \
    --version 2 \
    --stage Production

# Verifikasi state akhir
python -m src.models.registry list --model fireguard-regressor
```

Output:
```
v  1 stage=Staging       run_id=<sha-v1>...
v  2 stage=Production    run_id=<sha-v2>...
```

### Via MLflow UI

1. Buka MLflow UI: `mlflow ui --port 5000 --host 0.0.0.0`
2. Sidebar atas → tab **Models**
3. Klik `fireguard-regressor`
4. Klik versi → dropdown **Stage** → pilih target
5. Centang **Archive existing versions** kalau mau auto-archive lama

**Screenshot Models tab** menampilkan v1=Staging, v2=Production.

---

## Tahap 4 — Sinkronisasi Metadata dengan DVC

Track folder `models/` lewat DVC supaya silsilah data ↔ model terjaga:

```bash
# Buat folder models/ kalau belum ada
mkdir -p models

# Buat metadata file pointer ke versi production
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
client = mlflow.tracking.MlflowClient()
prod = [v for v in client.search_model_versions(\"name='fireguard-regressor'\")
        if v.current_stage == 'Production']
if prod:
    import json
    info = {
        'model_name': prod[0].name,
        'version': prod[0].version,
        'run_id': prod[0].run_id,
        'stage': prod[0].current_stage,
        'source': prod[0].source,
    }
    with open('models/current_production.json', 'w') as f:
        json.dump(info, f, indent=2)
    print('Wrote models/current_production.json')
    print(json.dumps(info, indent=2))
"

# Track via DVC
dvc add models
git add models.dvc .gitignore
dvc push

git add models.dvc
git commit -m "feat(registry): DVC track models/ dengan metadata production pointer (LK07)"
git push origin feat/registry-cicd
```

Catatan: `models.dvc` (metadata kecil) ke-commit Git; binary model artifact ada di `.dvc/cache/` dan remote.

---

## Tahap 5 — Verifikasi Inferensi via Production Model

Test apakah model production dapat dipanggil programatik:

```bash
python -m src.models.registry load \
    --model fireguard-regressor \
    --stage Production \
    --n-samples 5
```

Output:
```
... INFO ... Loading model from models:/fireguard-regressor/Production
... INFO ... Model loaded successfully: <class 'mlflow.lightgbm._LGBModelWrapper'>
... INFO ... Predicting on 5 synthetic samples
... INFO ... Predictions: [12.5, 18.3, 5.1, 8.7, 22.0]

   hotspot_count   frp_mean  prediction
              15      20.43       18.42
              22      14.81       21.07
               5      28.30        7.85
              ...
```

**Bukti inferensi (#5 instruksi):** output prediksi terlihat, model loadable, siap pakai untuk FastAPI endpoint nanti.

---

## Tahap 6 — Screenshot untuk Submission

Siapkan screenshot/output berikut:

1. **`python -m src.models.registry list`** — terlihat v1 + v2 dengan stage berbeda
2. **MLflow UI Models tab** — `fireguard-regressor` dengan multiple versions + stages
3. **Output `mlflow.pyfunc.load_model`** — prediksi terlihat (Tahap 5)
4. **File `models.dvc`** + commit history yang track perubahan registry
5. **Output `models/current_production.json`** — metadata snapshot

---

## Bukti Pemenuhan Instruksi LK07

| # | Instruksi | Bukti |
|---|---|---|
| 1 | Registrasi model | `fireguard-regressor v1` ada di Registry (dari LK06) |
| 2 | Versioning artefak | `v1` + `v2` dengan run_id berbeda |
| 3 | Stage transition | v1=Staging, v2=Production (lihat MLflow UI atau output `registry list`) |
| 4 | DVC sync metadata | `models.dvc` + `models/current_production.json` ter-commit |
| 5 | Verifikasi inferensi | Output `registry load --stage Production` dengan prediksi |

---

*LK07 selesai = 5 bukti di atas + LK07_FireGuard.docx submitted.*
