# LK06 — Panduan Training Pipeline + MLflow

> Step-by-step build features dari output preprocess LK04, train model LightGBM
> dengan MLflow tracking, run 3+ eksperimen, dan register model terbaik.

---

## Prasyarat

- LK05 selesai: DVC sudah init + dataset versioned di `data/raw/`.
- `data/processed/firms_weather_joined_*.parquet` ada (output preprocess.py LK04).
  - Kalau belum, jalankan: `python -m src.data.preprocess`
- `requirements.txt` sudah include `mlflow==2.13.0`, `lightgbm==4.3.0`, `pyarrow==15.0.2`.

---

## Tahap 1 — Feature Engineering

`build_features.py` membaca semua parquet di `data/processed/`, agregasi per
`(province_id, date)`, tambah rolling + lag features + `days_since_rain` +
cyclical encoding + target labels.

```bash
cd /workspaces/MLOps-PrediksiHotspotKebakaranHutan

# Install dependencies kalau ada update
pip install -r requirements.txt

# Build features
python -m src.features.build_features
```

Output yang Anda harapkan:

```
... | INFO | __main__ | Loading processed data from data/processed
... | INFO | __main__ | Loaded NNN rows
... | INFO | src.features.build_features | Drop rows tanpa label: NN → MM
... | INFO | src.features.build_features | Wrote training_dataset_20260512_HHMMSS_UTC.parquet (MM rows, KK cols)
data/features/training_dataset_20260512_HHMMSS_UTC.parquet
```

### Verifikasi output

```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('data/features/training_dataset_*.parquet'))
df = pd.read_parquet(files[-1])
print('Shape:', df.shape)
print()
print('Cols:', list(df.columns))
print()
print('Target distribution (risk_level):')
print(df['risk_level'].value_counts().sort_index())
print()
print('Sample rows:')
print(df[['province_id','date','hotspot_count','hotspot_count_tomorrow','risk_level']].head(5))
"
```

> **Catatan tentang data kecil:** Karena data dari LK04 mungkin hanya 2-3 hari
> per provinsi (real fetch sekali), training dataset akan kecil. Itu **tidak
> masalah** untuk demonstrasi MLflow — fokus LK06 adalah workflow tracking-nya,
> bukan akurasi model. Akurasi akan meningkat seiring akumulasi data dari cron
> berkala (LK07/LK08).

---

## Tahap 2 — Unit Tests (sanity check)

```bash
python -m pytest tests/test_features.py tests/test_train.py -v
```

Output yang Anda harapkan: semua test pass. Test untuk training akan **skip**
otomatis kalau `lightgbm` belum terinstal (pakai `pytest.mark.skipif`).

---

## Tahap 3 — Track Features dengan DVC

Karena features adalah artefak turunan, kita juga track dengan DVC:

```bash
dvc add data/features

# .dvc/data/features.dvc baru dibuat
git status

git add data/features.dvc data/.gitignore
git commit -m "feat(dvc): track features dataset (LK06)"
dvc push
```

---

## Tahap 4 — Run Training Pertama

Jalankan training regressor dengan params default:

```bash
python -m src.models.train \
    --algorithm regressor \
    --learning-rate 0.05 \
    --n-estimators 500 \
    --max-depth 8
```

Output yang Anda harapkan:

```
... | INFO | __main__ | MLflow tracking_uri=file:///workspaces/.../mlruns experiment=fireguard-ct
... | INFO | __main__ | Time-aware split: train=X rows (≤ YYYY-MM-DD), test=Y rows (> YYYY-MM-DD)
[LightGBM] ... training output ...
... | INFO | __main__ | Regressor metrics: {'rmse': ..., 'mae': ..., ...}
... | INFO | __main__ | Run abc123def456... logged successfully.
Run ID: abc123def456...
```

`mlruns/` folder akan dibuat di root repo dengan struktur:

```
mlruns/
├── 0/                       # experiment 0 (Default)
├── 1/                       # experiment 1 (fireguard-ct)
│   ├── <run-id-1>/
│   │   ├── artifacts/
│   │   │   └── model/       # LightGBM model artifact
│   │   ├── metrics/
│   │   ├── params/
│   │   └── tags/
│   └── meta.yaml
└── models/                  # Model Registry
```

---

## Tahap 5 — Run Variasi (Min 3 Run dengan Params Berbeda)

Sesuai instruksi LK06 No. 4: minimal 3 run dengan parameter berbeda.

### Run 2 — learning rate lebih kecil + n_estimators lebih besar

```bash
python -m src.models.train \
    --algorithm regressor \
    --learning-rate 0.01 \
    --n-estimators 1000 \
    --max-depth 6
```

### Run 3 — max_depth lebih dalam + regularisasi lebih kuat

```bash
python -m src.models.train \
    --algorithm regressor \
    --learning-rate 0.1 \
    --n-estimators 300 \
    --max-depth 12 \
    --reg-alpha 0.5 \
    --reg-lambda 0.5
```

### Run 4-5 (opsional) — classifier untuk risk_level

```bash
python -m src.models.train \
    --algorithm classifier \
    --learning-rate 0.05 \
    --n-estimators 500

python -m src.models.train \
    --algorithm classifier \
    --learning-rate 0.1 \
    --n-estimators 300 \
    --max-depth 4
```

Setiap run akan menampilkan **Run ID** unik di akhir output. Catat Run ID-nya
untuk referensi (atau lihat di MLflow UI nanti).

---

## Tahap 6 — Buka MLflow UI

```bash
# Di terminal Codespace
mlflow ui --port 5000 --host 0.0.0.0
```

Output:
```
[2026-05-12 ...] INFO ... Listening at: http://0.0.0.0:5000
```

Codespace akan otomatis **forward port 5000** dan menampilkan toast
notification "**Open in Browser**". Klik untuk buka MLflow UI.

URL akan berupa: `https://<codespace-id>-5000.app.github.dev`

> Kalau port tidak auto-forward: klik tab **Ports** di bawah → cari port 5000 → klik 🌐 ikon
> globe untuk open in browser.

---

## Tahap 7 — Compare Runs di MLflow UI

Di MLflow UI:

1. Klik experiment **fireguard-ct** di sidebar kiri.
2. Anda akan lihat tabel daftar run dengan kolom params + metrics.
3. Centang **checkbox** di kiri 2-3 run yang ingin dibandingkan.
4. Klik tombol **Compare** di atas tabel.
5. Halaman compare menampilkan:
   - **Parameter diff** — beda hyperparameter antar run
   - **Metric chart** — scatter/line chart visual untuk setiap metric
   - **Parallel coordinates plot** — visualisasi multivariat

### Screenshot yang Anda butuhkan untuk submission

1. **Halaman experiment** — daftar 3+ run dengan kolom RMSE / MAE / F1 / params
2. **Halaman compare 3 run** — parallel coordinates plot
3. **Halaman detail 1 run** — section Parameters + Metrics + Artifacts (model)

---

## Tahap 8 — Register Model Terbaik

Pilih run dengan metric paling bagus (mis. RMSE paling kecil untuk regressor).

### Cara 1 — Via MLflow UI

1. Klik run terbaik
2. Klik **Artifacts** tab
3. Pilih folder **model**
4. Klik tombol **Register Model**
5. Nama model: `fireguard-regressor` (atau `fireguard-classifier`)
6. Klik **Register**

Model akan muncul di **Models** tab di sidebar atas.

### Cara 2 — Via CLI (jalankan ulang dengan `--register`)

Untuk run yang sudah ada Run ID, kita bisa **re-register** model:

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
# Ganti <BEST_RUN_ID> dengan Run ID yang terbaik
run_id = '<BEST_RUN_ID>'
result = mlflow.register_model(
    model_uri=f'runs:/{run_id}/model',
    name='fireguard-regressor',
)
print('Registered:', result.name, 'version', result.version)
"
```

Atau jalankan training ulang dengan `--register`:

```bash
python -m src.models.train \
    --algorithm regressor \
    --learning-rate 0.05 \
    --n-estimators 500 \
    --register
```

### Verifikasi Registry

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
client = mlflow.tracking.MlflowClient()
models = client.search_registered_models()
for m in models:
    print(m.name)
    for v in m.latest_versions:
        print(f'  v{v.version} ({v.current_stage}) — run_id={v.run_id}')
"
```

---

## Tahap 9 — Commit Hasil ke Git

```bash
# mlruns/ sudah di-gitignore, jadi tidak ke-commit.
# Kita commit perubahan kode + dokumentasi saja:
git status

# (Optional) commit screenshot/log untuk submission
# Atau biarkan untuk dilampirkan terpisah di docx

git add docs/LK06_TRAINING_GUIDE.md src/features/ src/models/ tests/
git commit -m "feat(training): MLflow-tracked training pipeline (LK06)

- src/features/build_features.py: aggregate per (province,day) + rolling/lag features
- src/models/train.py: LightGBM regressor + classifier dengan MLflow log_param/metric/model
- 3+ run dengan params berbeda; model terbaik di-register ke MLflow Registry
- tests/test_features.py: 13 unit tests
- tests/test_train.py: 8 unit tests (skip kalau lightgbm/mlflow tidak ada)"

git push origin feat/dvc-mlflow
```

---

## Troubleshooting

### `No training dataset found` saat run train.py

`build_features.py` belum jalan atau `data/features/` kosong. Run:
```bash
python -m src.features.build_features
```

### `Only 1 class in y_train` saat classifier training

Data terlalu sedikit / homogen. Solusi:
- Tunggu akumulasi data dari fetch berkala
- Kurangi `--holdout-days` (default 14)
- Atau coba dengan dataset synthetic untuk demonstrasi:
  ```bash
  python -m pytest tests/test_train.py::TestRunExperiment -v
  ```

### MLflow UI tidak load di browser

- Cek port 5000 forwarded di tab **Ports** Codespace
- Pastikan visibility port set ke **Public** (kalau Anda mau share link)
- Coba `mlflow ui --port 5001` kalau 5000 conflict

### `mlflow ui` lambat di Codespace

Default backend MLflow pakai SQLite via filesystem. Untuk dataset run banyak,
boleh upgrade ke local Postgres atau SQLAlchemy:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Bukti Submission LK06

Untuk dosen, siapkan:

1. **Link branch GitHub** `feat/dvc-mlflow` (atau PR-nya nanti)
2. **Screenshot terminal** output training 3+ run (masing-masing menampilkan Run ID + metrics)
3. **Screenshot MLflow UI**:
   - Halaman experiment listing 3+ run
   - Halaman compare 3 run (parallel coordinates plot)
   - Halaman model registry dengan `fireguard-regressor` v1 (atau lebih)
4. **File** `LK06_FireGuard.docx` (akan saya generate setelah Anda kirim screenshot)
5. **Commit history** terkait LK06

---

*Setelah LK06 selesai, kita buka PR `feat/dvc-mlflow` → `main` dan merge sebagai
deliverable gabungan LK05 + LK06.*
