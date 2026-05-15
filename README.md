# MLOps-FireGuard

> Forest Fire Hotspot Detection System — Continual-Learning MLOps Project
> Wilayah: Kalimantan Tengah, Kalimantan Barat, Riau, Sumatera Selatan, Jambi
> Status: 🚧 Under construction (LK01–LK09)

---

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
* `BMKG_API_KEY` (opsional, BMKG saat ini publik tanpa key)
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

## 3a. Cara Menjalankan dengan Docker Compose (LK09)

Stack penuh dapat dijalankan dengan **satu perintah** menggunakan Docker
Compose. Compose akan men-orchestrate dua container:

| Service | Port | Peran |
|---|---|---|
| `mlflow-server` | 5000 | Tracking + Model Registry (sqlite backend, filesystem artifact store) |
| `api-service`   | 8000 | FastAPI inference (`/predict`, `/health`, `/model-info`) |

### 3a.1 Prasyarat

- Docker Engine ≥ 24 + Docker Compose v2 (sudah include di Codespaces)

### 3a.2 Build & Run

```bash
docker compose up -d --build
```

Tunggu ~30–60 detik sampai kedua container `healthy`. Cek status:

```bash
docker compose ps
```

Output diharapkan:
```
NAME                IMAGE                          STATUS              PORTS
fireguard-mlflow    fireguard/mlflow-server:1.0    Up (healthy)        0.0.0.0:5000->5000/tcp
fireguard-api       fireguard/api-service:1.0      Up (healthy)        0.0.0.0:8000->8000/tcp
```

### 3a.3 Verifikasi

```bash
curl http://localhost:8000/health
# {"status":"ok","ready":true,"uptime_sec":...}

curl http://localhost:8000/model-info
# {"name":"fireguard-regressor","version":"...","stage":"Production",...}
```

Buka MLflow UI di browser: <http://localhost:5000>.

### 3a.4 Lifecycle

```bash
docker compose logs -f api-service    # follow logs satu service
docker compose restart api-service    # restart kalau perlu reload model
docker compose down                   # stop (volumes tetap)
docker compose down -v                # stop + hapus volumes (data hilang)
```

Detail lengkap (network, named volume, healthcheck, troubleshooting, seeding
registry) ada di [`docs/LK09_DOCKER_GUIDE.md`](docs/LK09_DOCKER_GUIDE.md).

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

## 6. Roadmap LK

| LK | Topik | Status |
|---|---|---|
| LK01 | Inisiasi proyek + Continual Learning strategy | ✅ Done |
| LK02 | Repo + Codespaces + GitHub Flow | ✅ Done |
| LK03 | Pipeline arsitektur (ETL diagram) | ✅ Done |
| LK04 | Data fetcher + preprocessing | ✅ Done |
| LK05 | DVC versioning | ✅ Done |
| LK06 | Training pipeline + MLflow tracking | ✅ Done |
| LK07 | Model Registry + stage transition | ✅ Done |
| LK08 | CI/CD GitHub Actions (automation pipeline) | ✅ Done |
| LK09 | Docker Compose orchestration (api + mlflow) | ✅ Done |
| LK10 | Monitoring & drift detection | ⏳ Pending |

---

## 7. Lisensi

[MIT](LICENSE) — bebas dipakai dengan atribusi.

---

## 8. Kontributor

* **Izu** — pengembang utama (mahasiswa MLOps, individual project)

---

*Dokumen ini akan diperbarui setiap LK selesai. Pertanyaan/issue: buka GitHub Issue di repo ini.*
