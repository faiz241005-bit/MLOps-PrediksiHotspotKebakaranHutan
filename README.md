## 1. Tujuan Proyek

**FireGuard** adalah platform berbasis AI yang memprediksi jumlah dan tingkat risiko titik panas (hotspot) kebakaran hutan/lahan untuk **satu hari ke depan**, per provinsi di Indonesia.

Proyek ini diinisiasi sebagai tugas mata kuliah **MLOps** dengan fokus utama:

1. Membangun pipeline **production-oriented**, bukan eksperimen sekali pakai.
2. Menggunakan **data streaming nyata** (NASA FIRMS update 2× sehari + cuaca per jam).
3. Menerapkan **Continual Training (CT)** dengan empat trigger berbeda untuk menjaga relevansi model terhadap *seasonal concept drift* (musim kemarau ↔ musim hujan).
4. Menggunakan tooling industri standar: **GitHub Codespaces, GitHub Actions, DVC, MLflow, Docker, FastAPI**.

### Target Pengguna

| Segmen | Use case |
|---|---|
| Perusahaan sawit / HTI | Compliance KLHK, deteksi risiko di lahan konsesi |
| BNPB / KLHK | Early warning nasional, dashboard pemerintah |
| BPBD Provinsi | Monitoring per kabupaten, alokasi sumber daya pemadaman |
| Asuransi perkebunan | Risk-based pricing premi |

---

## 2. Struktur Direktori

Mengikuti konvensi *Cookiecutter Data Science* dengan adaptasi MLOps:

```
MLOps-FireGuard/
├── .devcontainer/
│   └── devcontainer.json       # Config GitHub Codespaces (Python 3.11)
├── .github/
│   └── workflows/              # CI/CD: data_fetch, ct_train, ci_test, cd_deploy
├── config/                     # params.yaml, paths.yaml (file *.example.yaml saja yang di-commit)
├── data/
│   ├── raw/                    # Data mentah dari API (DVC-tracked, gitignored)
│   ├── processed/              # Data setelah cleaning (DVC-tracked)
│   └── features/               # Feature-engineered data (DVC-tracked)
├── docs/
│   └── BRANCHING.md            # Strategi GitHub Flow + code-quality guidelines
├── models/                     # Model artifacts (DVC-tracked, gitignored)
├── notebooks/                  # EDA & eksperimen (01_initial_eda.ipynb, dst.)
├── src/
│   ├── data/                   # fetch_firms.py, fetch_weather.py, preprocess.py
│   ├── features/               # build_features.py
│   ├── models/                 # train.py, evaluate.py, predict.py
│   └── monitoring/             # drift_detector.py, performance_monitor.py
├── tests/                      # Unit & integration tests (pytest)
├── .gitignore                  # Lengkap untuk Python + DVC + MLflow + secrets
├── LICENSE                     # MIT
├── README.md                   # File ini
└── requirements.txt            # Pinned dependencies
```

> **Catatan keamanan:** folder `data/`, `models/`, `mlruns/`, dan file `.env` **tidak masuk** Git — masing-masing di-track via DVC atau di-store di GitHub Secrets / runner environment.

---

## 3. Cara Menjalankan dengan GitHub Codespaces

### 3.1 Prasyarat

* Akun GitHub (disarankan daftar dulu di [GitHub Education](https://github.com/education/students) untuk mendapat akses Codespaces gratis lebih besar).
* API key gratis dari [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/map_key/).

### 3.2 Buka di Codespaces

1. Klik tombol hijau **`<> Code`** di halaman repo GitHub → tab **Codespaces** → **Create codespace on main**.
2. Tunggu 2–4 menit. Codespace akan:
   * Pull image `mcr.microsoft.com/devcontainers/python:3.11-bullseye`.
   * Install dependencies dari `requirements.txt` via `postCreateCommand`.
   * Install ekstensi VS Code (Python, Jupyter, DVC, GitHub Actions, dst).
3. Setelah siap, terminal terbuka otomatis di dalam container.

### 3.3 Set API Keys (Secrets)

```bash
# Di Codespace (sekali saja)
cp .env.example .env
# Edit .env, isi NASA_FIRMS_API_KEY, dst.
# .env sudah ada di .gitignore sehingga TIDAK akan ter-commit.
```

Untuk GitHub Actions, simpan kunci di **Repo Settings → Secrets and variables → Actions**:

* `NASA_FIRMS_API_KEY`
* `DVC_REMOTE_URL` (jika pakai S3/GDrive)

### 3.4 Verifikasi Setup

```bash
python --version          # harus Python 3.11.x
pip list | grep mlflow    # harus terinstal
dvc --version             # harus terinstal
```

### 3.5 Jalankan EDA

```bash
jupyter lab               # buka notebooks/01_initial_eda.ipynb (akan ditambahkan di branch feat/initial-eda)
```

### 3.6 Jalankan MLflow UI (lokal di Codespace)

```bash
mlflow ui --port 5000
# Codespace akan auto-forward ke browser
```

---

## 3a. Menjalankan Skrip Pengumpul Data

Pipeline data fetch ada di `src/data/`. Tiga skrip utama berjalan berurutan
untuk membawa data mentah ke bentuk siap-feature.

### 3a.1 Ingestion — NASA FIRMS + Open-Meteo

```bash
# Pastikan NASA_FIRMS_API_KEY sudah di .env
python -m src.data.ingest_data --provinces all
```

Argumen yang berguna:

| Flag | Default | Fungsi |
|---|---|---|
| `--provinces` | wajib | List provinsi (`riau kalteng kalbar sumsel jambi`) atau `all` |
| `--sources` | `firms,weather` | Pilih sumber data yang di-fetch |
| `--raw-dir` | `data/raw/` | Folder output (di-DVC-track) |
| `--log-level` | `INFO` | `DEBUG` untuk troubleshoot |

Output: `data/raw/firms/{province}_*.csv` + `data/raw/weather/{province}_*.csv`,
masing-masing dengan suffix timestamp UTC.

### 3a.2 Preprocess — Cleaning + Join

```bash
python -m src.data.preprocess
```

Menggabungkan hotspot detection dengan cuaca per provinsi & hari (WIB, UTC+7).
Output: `data/processed/firms_weather_joined_*.parquet`.

### 3a.3 Feature Engineering

```bash
python -m src.features.build_features
```

Bangun 27 fitur (rolling, lag, cyclical, days_since_rain) + 2 target
(`hotspot_count_tomorrow`, `risk_level`). Output: `data/features/training_dataset_*.parquet`.

> **One-liner end-to-end:**
> ```bash
> python -m src.data.ingest_data --provinces all && \
> python -m src.data.preprocess && \
> python -m src.features.build_features
> ```

Detail flag & troubleshooting di [`docs/LK04_RUN_GUIDE.md`](docs/LK04_RUN_GUIDE.md).

---

## 3b. Versioning Data dengan DVC

Folder `data/` (`raw/`, `processed/`, `features/`) **tidak masuk Git** — dilacak
oleh **DVC** (Data Version Control). Alur penambahan versi data:

### 3b.1 Tambah Versi Baru

Setelah fetch data baru di section 3a:

```bash
dvc add data/raw/firms data/raw/weather
```

Output: file `.dvc` (kecil, hash + pointer) yang **masuk Git**, sementara data
asli masuk `.dvc/cache`. Pesan akhir akan kasih perintah `git add` yang siap
di-paste.

```bash
git add data/raw/firms.dvc data/raw/weather.dvc data/raw/.gitignore
git commit -m "data: bump FIRMS+weather snapshot $(date +%Y-%m-%d)"
```

### 3b.2 Push ke Remote DVC

```bash
dvc push
```

DVC akan upload content yang berubah ke remote (lokal `/tmp/fireguard-dvc-remote`
saat ini; production upgrade ke S3/R2/Filebase). Daftar remote tersedia:

```bash
dvc remote list
```

### 3b.3 Pull di Mesin Lain

Setelah `git pull` ambil file `.dvc`, jalankan:

```bash
dvc pull
```

DVC akan rekonstruksi `data/raw/`, `data/processed/`, `data/features/`
berdasarkan hash. Tidak perlu fetch ulang dari API.

### 3b.4 Lihat Riwayat Versi

```bash
git log --oneline -- data/raw/firms.dvc      # commit yang ubah snapshot FIRMS
dvc data status                              # state lokal vs remote
```

Setiap commit Git yang mengubah file `.dvc` = satu versi dataset. Bisa
`git checkout <hash>` lalu `dvc checkout` untuk kembali ke state lama.

Detail (cara setup remote cloud + recovery) di [`docs/LK05_DVC_GUIDE.md`](docs/LK05_DVC_GUIDE.md).

---

## 3c. Versi Model Aktif untuk Inferensi

Status model produksi di-track di file [`models/current_production.json`](models/current_production.json):

```json
{
  "model_name": "fireguard-regressor",
  "version": 1,
  "stage": "Production",
  "run_id": "11cc4affd84247dab2397f13b704712d"
}
```

### Versi Aktif Saat Ini

| Field | Nilai |
|---|---|
| **Registered model** | `fireguard-regressor` |
| **Version** | `1` |
| **Stage** | `Production` |
| **Algoritma** | LightGBM Regressor |
| **Target** | `hotspot_count_tomorrow` (prediksi jumlah hotspot besok) |

### Alasan Versi Ini Aktif

1. **Lolos threshold LK01** — RMSE ≤ 12 hotspot, MAE ≤ 8 hotspot saat di-evaluate di test set out-of-time (April–Mei 2026 vs train Jan–Mar 2026).
2. **Lolos CI/CD gate (LK08)** — workflow `mlops-automation.yaml` menjalankan `src/models/evaluate.py` dengan `--ci-mode` lenient untuk synthetic, dan threshold full untuk run dengan real data. Versi 1 lolos keduanya.
3. **Promoted via registry CLI** — di-transition dari `Staging` → `Production` lewat `python -m src.models.registry transition --model fireguard-regressor --version 1 --stage Production`, dengan flag `--archive-existing` untuk auto-archive versi lama.
4. **Reproducible** — `run_id` `11cc4af…` merujuk ke MLflow run yang menyimpan: hyperparameter, training data hash (DVC), feature importances, metric, dan artifact model. Bisa di-load lagi kapan saja dengan `mlflow.pyfunc.load_model("models:/fireguard-regressor/Production")`.

### Cara Cek dari Codespace

```bash
# Listing semua versi + stage-nya
python -m src.models.registry list --model fireguard-regressor

# Test load model production
python -c "import mlflow; m=mlflow.pyfunc.load_model('models:/fireguard-regressor/Production'); print('OK:', m.metadata)"
```

### Kapan Akan Ada Versi Baru?

Trigger retrain otomatis (lihat section 5 — Continual Training Strategy). Setiap retrain yang lolos threshold akan auto-promote ke `Staging`. Promosi `Staging → Production` saat ini **manual** (best practice MLOps — butuh approval) lewat CLI `registry transition` atau workflow `workflow_dispatch`.

Detail siklus hidup model di [`docs/LK07_REGISTRY_GUIDE.md`](docs/LK07_REGISTRY_GUIDE.md).

---

## 3d. Cara Menjalankan dengan Docker Compose

Stack penuh dapat dijalankan dengan **satu perintah** menggunakan Docker
Compose. Saat semua service aktif, ada **7 container** (LK09 + LK10 + LK11):

| Service | Port | Replicas | Peran |
|---|---|---|---|
| `mlflow-server` | 5000 | 1 | Tracking + Model Registry (sqlite backend, filesystem artifact store) |
| `mlflow-model-server` | 8010-8012 | **3** | MLflow native serving via `/invocations` (horizontal scaling LK10) |
| `streamlit-dashboard` | 8501 | 1 | Folium UI — call API via metrics-proxy |
| `metrics-proxy` | 9000 | 1 | FastAPI sidecar → `/metrics` (LK11) |
| `prometheus` | 9090 | 1 | TSDB scraper (LK11) |
| `grafana` | 3000 | 1 | Dashboard UI (LK11) |
| `cadvisor` | 8085 | 1 | Container resource exporter (LK11) |

### 3d.1 Prasyarat

- Docker Engine ≥ 24 + Docker Compose v2 (sudah include di Codespaces)
- Image `fireguard/mlflow-model-server:1.0` sudah di-build (lihat 3d.2 di bawah)

### 3d.2 Build MLflow Model Image (Sekali Saja)

Image untuk model serving dibuat otomatis dari MLflow Registry — tidak butuh
Dockerfile manual:

```bash
# Pastikan mlflow-server jalan dulu (untuk source model)
docker compose up -d mlflow-server
sleep 30

# Build image otomatis dari model di Registry stage Production
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models build-docker \
  --model-uri "models:/fireguard-regressor/Production" \
  --name fireguard/mlflow-model-server:1.0 \
  --env-manager local
```

⏱️ **~5-15 menit** (download deps + build layers).

### 3d.3 Run Full Stack

```bash
docker compose up -d
sleep 60   # tunggu semua replica healthy
docker compose ps
```

Output diharapkan (4 container total — 1 mlflow-server + 3 replicas):
```
NAME                                            IMAGE                                  STATUS
fireguard-mlflow                                fireguard/mlflow-server:1.0            Up (healthy)
mlops-fireguard-mlflow-model-server-1           fireguard/mlflow-model-server:1.0      Up (healthy)
mlops-fireguard-mlflow-model-server-2           fireguard/mlflow-model-server:1.0      Up (healthy)
mlops-fireguard-mlflow-model-server-3           fireguard/mlflow-model-server:1.0      Up (healthy)
```

### 3d.4 Akses Endpoint API

**Cara MLflow native:** kirim DataFrame format `dataframe_split`:

```bash
# Ke replica 1 (port 8010)
curl -X POST http://localhost:8010/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["hotspot_count", "frp_mean", ...],
      "data": [[1323, 50.0, ...]]
    }
  }'

# Response: {"predictions": [323.531]}
```

Untuk demo balanced antar replicas, ganti port: `8010`, `8011`, atau `8012`.

Health endpoint: `curl http://localhost:8010/ping` (return string kosong + status 200).

Buka MLflow UI di browser: <http://localhost:5000>.

### 3d.5 Dynamic Scaling (LK10)

Tambah replicas dinamis tanpa modify YAML:

```bash
# Scale up ke 5 replicas
docker compose up -d --scale mlflow-model-server=5

# Scale down ke 1 replica
docker compose up -d --scale mlflow-model-server=1

# Verify
docker compose ps
```

> **Catatan port:** Default port range di YAML `8010-8012` cuma cover 3 replicas. Untuk 5 replicas, edit YAML jadi `8010-8014:8080`, atau gunakan reverse proxy (nginx) di depan untuk load balancing dengan 1 port external.

### 3d.6 End-to-End Demo Script

```bash
# Predict tomorrow per provinsi via /invocations (skip fetch untuk cepat)
python -m src.scripts.predict_tomorrow --skip-fetch

# Atau dengan fresh data dari NASA FIRMS
python -m src.scripts.predict_tomorrow
```

### 3d.7 Lifecycle

```bash
docker compose logs -f mlflow-model-server   # follow logs (semua replicas)
docker compose restart mlflow-model-server   # restart semua replicas
docker compose down                          # stop (volumes tetap)
docker compose down -v                       # stop + hapus volumes (data hilang)
```

Detail lengkap LK09 (network, volume, healthcheck) di [`docs/LK09_DOCKER_GUIDE.md`](docs/LK09_DOCKER_GUIDE.md).
Detail LK10 (MLflow serving, replicas, scaling) di [`docs/LK10_SERVING_GUIDE.md`](docs/LK10_SERVING_GUIDE.md).

---

## 3e. Observability — Prometheus + Grafana (LK11)

Tiga container baru di stack untuk monitoring production-grade:

- **metrics-proxy** (FastAPI sidecar) — intercept `/invocations`, instrumentasi
  latency, throughput, dan **prediction value distribution** (untuk drift
  detection). Expose `/metrics` dalam format Prometheus.
- **prometheus** — scrape metrics-proxy + cAdvisor tiap 15 detik. TSDB
  retention 15 hari.
- **grafana** — dashboard `FireGuard MLOps Observability (LK11)` auto-loaded
  via provisioning. 10 panel: throughput, latency p50/p95/p99, error rate,
  CPU/RAM per container, dan **heatmap distribusi prediksi** untuk drift.
- **cadvisor** — exporter container resource metrics dari cgroups.

### 3e.1 Akses UI

| Service | URL | Credentials |
|---|---|---|
| Grafana | <http://localhost:3000> | `admin` / `admin` |
| Prometheus | <http://localhost:9090> | — |
| cAdvisor | <http://localhost:8085> | — |
| metrics-proxy `/metrics` | <http://localhost:9000/metrics> | — |

> ⚠️ **Security:** default Grafana password `admin/admin` hanya untuk demo.
> Untuk production, set `GF_SECURITY_ADMIN_PASSWORD` dari secret manager.

### 3e.2 Cek Prometheus Scraping

```bash
# Open: http://localhost:9090/targets
# Semua 3 target harus UP:
#   - mlflow-metrics-proxy
#   - cadvisor
#   - prometheus (self)
```

### 3e.3 Generate Beban Kerja (Demo)

```bash
# Default: 50 request, concurrency 5
python -m src.scripts.load_test_lk11

# Heavy: 500 request, 20 concurrent
python -m src.scripts.load_test_lk11 --requests 500 --concurrency 20

# Demo drift detection: 2 fase berbeda
python -m src.scripts.load_test_lk11 --requests 100 --scenario low
sleep 60
python -m src.scripts.load_test_lk11 --requests 100 --scenario high
# → Heatmap di Grafana akan shift ke nilai prediksi yang lebih besar
```

### 3e.4 Dashboard Panel — Apa yang Dimonitor

| Panel | Metric | Insight |
|---|---|---|
| Throughput RPS | `rate(mlflow_requests_total)` | Beban request masuk |
| Latency p50/p95/p99 | `histogram_quantile(...)` | SLA inference time |
| Error Rate | rate 4xx+5xx ÷ rate total | Kesehatan service |
| Request per Status | grouped by status code | Pola error/sukses |
| CPU per Container | cAdvisor `container_cpu_usage_seconds_total` | Resource pressure |
| RAM per Container | cAdvisor `container_memory_working_set_bytes` | Memory growth/leak |
| **Prediction Heatmap** | `mlflow_prediction_value_bucket` | **Data drift detection** |
| Prediction Quantiles | p50/p95 dari prediction value | Trend shift |

### 3e.5 Deteksi Model Decay

Dashboard ini memvisualisasikan **4 sinyal model decay** sekaligus:

1. **Data drift** — heatmap prediksi shift dari baseline → input distribution
   berubah (musim, demografi, dll).
2. **Latency degradation** — p99 naik bertahap → model semakin lambat (ukuran
   model bertambah, atau memory pressure).
3. **Error rate creep** — 4xx muncul → schema mismatch dengan training data.
4. **Memory leak** — RAM monoton naik tanpa decay → restart replica diperlukan.

Detail lengkap setup + screenshot guide + analisis decay di
[`docs/LK11_OBSERVABILITY_GUIDE.md`](docs/LK11_OBSERVABILITY_GUIDE.md).

---

## 4. Branching Strategy — GitHub Flow

Detail lengkap di [`docs/BRANCHING.md`](docs/BRANCHING.md).

Ringkasan:

* `main` — **always deployable**. Tidak boleh push langsung.
* `feat/*` — fitur baru (mis. `feat/initial-eda`, `feat/data-fetcher`).
* `fix/*` — perbaikan bug.
* `exp/*` — eksperimen model (boleh long-lived, tidak harus di-merge).

Setiap perubahan masuk lewat **Pull Request** dengan minimal 1 self-review pada catatan PR (proyek individual).

---

## 5. Continual Training Strategy

Empat trigger retrain (lihat detail di LK01_FireGuard.docx):

| Tipe | Mekanisme |
|---|---|
| **Seasonal CT** | Cron 1 April & 1 November setiap tahun |
| **Performance CT** | RMSE rolling 7 hari naik > 20% dari baseline |
| **Event-based CT** | > 500 hotspot/hari → retrain immediate |
| **Manual CT** | `workflow_dispatch` di GitHub Actions |

---

## 6. Lisensi

[MIT](LICENSE) — bebas dipakai dengan atribusi.

---

## 7. Kontributor

* **Izu** — pengembang utama (mahasiswa MLOps, individual project)

---

*Dokumen ini akan diperbarui setiap LK selesai. Pertanyaan/issue: buka GitHub Issue di repo ini.*
