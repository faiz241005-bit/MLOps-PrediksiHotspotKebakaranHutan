# LK09 — Panduan Docker Compose Orchestration

> Walkthrough lengkap "Multi-Service Container Orchestration" — mengangkat
> sistem FireGuard ke `docker compose` dengan dua layanan yang saling
> berkomunikasi via custom bridge network.

---

## Konsep

```
┌────────────────────── HOST ──────────────────────┐
│  http://localhost:5000     http://localhost:8000  │
│         │                          │              │
└─────────┼──────────────────────────┼──────────────┘
          ▼                          ▼
   ┌──────────────────────────────────────────────┐
   │  fireguard-net (custom bridge network)        │
   │                                                │
   │   ┌──────────────────┐   ┌──────────────────┐│
   │   │  mlflow-server   │◄──┤  api-service     ││
   │   │  port 5000       │HTTP│  FastAPI :8000  ││
   │   │  HEALTHCHECK ✓   │    │  HEALTHCHECK ✓  ││
   │   └──┬───────────┬───┘    └──────────────────┘│
   │      │           │                            │
   │   ┌──▼──┐    ┌───▼─────┐                      │
   │   │db   │    │artifacts│   (named volumes)    │
   │   │vol  │    │  vol    │                      │
   │   └─────┘    └─────────┘                      │
   └────────────────────────────────────────────────┘
```

## Prasyarat

- Docker Engine ≥ 24 + Docker Compose v2 (sudah include di Codespaces & Docker Desktop)
- File `Dockerfile.api`, `Dockerfile.mlflow`, `docker-compose.yaml` di root (sudah di-commit di branch `feat/docker-compose`)
- (Opsional) Model artefact `fireguard-regressor` sudah di-register di MLflow Registry — kalau belum, lihat **Tahap 5**

---

## Tahap 1 — Identifikasi Layanan

| Service | Image | Port (host:container) | Peran |
|---|---|---|---|
| `mlflow-server` | `fireguard/mlflow-server:1.0` | `5000:5000` | Tracking server + Model Registry (sqlite backend, filesystem artifact store, `--serve-artifacts`) |
| `api-service` | `fireguard/api-service:1.0` | `8000:8000` | FastAPI inference (`/predict`, `/health`, `/model-info`) yang load model dari registry via HTTP |

**Kenapa dua service ini?** Ini adalah pattern paling minimal untuk MLOps serving:
*tracking metadata* terpisah dari *inference runtime*. Mengikuti prinsip
*single responsibility per container*.

---

## Tahap 2 — Penulisan Manifest YAML

File `docker-compose.yaml` di root proyek (sudah dibuat). Struktur top-level:

```yaml
services:
  mlflow-server: { ... }
  api-service:   { ... }
networks:
  fireguard-net: { driver: bridge }
volumes:
  mlflow-db: {}
  mlflow-artifacts: {}
```

Versi key `version: "3.x"` sengaja dihilangkan — Compose v2 sudah deprecate
field tersebut.

---

## Tahap 3 — Konfigurasi Jaringan (Custom Bridge Network)

```yaml
networks:
  fireguard-net:
    name: fireguard-net
    driver: bridge
```

Setiap service tergabung di network ini, sehingga:

- `api-service` bisa pakai URL `http://mlflow-server:5000` (DNS resolution otomatis lewat nama service)
- **TIDAK** boleh pakai `localhost` antar container — `localhost` di container = container itu sendiri
- Default bridge network Docker tidak menyediakan DNS otomatis; **custom bridge** memberikan service discovery built-in

---

## Tahap 4 — Manajemen Data (Named Volumes)

```yaml
volumes:
  mlflow-db:
    name: fireguard-mlflow-db
  mlflow-artifacts:
    name: fireguard-mlflow-artifacts
```

Di-mount di `mlflow-server`:

```yaml
volumes:
  - mlflow-db:/mlflow/db             # sqlite metadata
  - mlflow-artifacts:/mlflow/artifacts  # model artifacts (.pkl, model.yaml)
```

**Perilaku:**
| Perintah | Efek pada volume |
|---|---|
| `docker compose down` | Volumes **tetap** — data aman |
| `docker compose down -v` | Volumes **dihapus** — semua run + model **hilang** |
| `docker volume ls` | List semua volume (cari prefix `fireguard-`) |

**Kenapa named volumes, bukan bind mount?**
- Portable: tidak tergantung path absolut host (penting saat pindah dari Windows ke Linux)
- Permission lebih clean (Docker yang manage owner, bukan host filesystem)
- Tidak mencemari working directory (`./mlruns` tidak tercipta di repo)

---

## Tahap 5 — Pengaturan Ketergantungan (`depends_on`)

```yaml
api-service:
  depends_on:
    mlflow-server:
      condition: service_healthy
```

Tanpa `condition: service_healthy`, api-service akan start saat mlflow-server
sekedar **berjalan** (process up) — bisa jadi MLflow belum siap menerima HTTP
request. Dengan `service_healthy`, Compose menunggu sampai HEALTHCHECK MLflow
berstatus `healthy` dulu baru start api-service.

HEALTHCHECK di `Dockerfile.mlflow`:
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --silent --fail http://localhost:5000/ || exit 1
```

---

## Tahap 6 — Eksekusi & Verifikasi

### Step 1 — Build images

```bash
cd ~/Documents/Claude/Projects/MLOps/MLOps-FireGuard  # atau Codespace path
docker compose build
```

Output: dua image baru di `docker images`:
```
fireguard/mlflow-server   1.0   ...   ~450 MB
fireguard/api-service     1.0   ...   ~750 MB
```

### Step 2 — Up (detached)

```bash
docker compose up -d
```

Output kira-kira:
```
[+] Running 4/4
 ✔ Network fireguard-net                  Created
 ✔ Volume "fireguard-mlflow-db"           Created
 ✔ Volume "fireguard-mlflow-artifacts"    Created
 ✔ Container fireguard-mlflow             Started
 ⠿ Container fireguard-api                Waiting for healthcheck
 ✔ Container fireguard-api                Started
```

### Step 3 — Cek status

```bash
docker compose ps
```

Expected (kedua service status **healthy**):
```
NAME                IMAGE                          STATUS                   PORTS
fireguard-mlflow    fireguard/mlflow-server:1.0    Up 1 min (healthy)       0.0.0.0:5000->5000/tcp
fireguard-api       fireguard/api-service:1.0      Up 30 sec (healthy)      0.0.0.0:8000->8000/tcp
```

**Screenshot wajib untuk LK09:**
1. `docker compose ps` dengan status `healthy` di kedua container

### Step 4 — Verifikasi komunikasi antar container

#### A. Cek mlflow-server (dari host)

```bash
curl http://localhost:5000/
# Output: <html>...MLflow...</html>  (HTTP 200)
```

#### B. Cek api-service health

```bash
curl http://localhost:8000/health
# {"status":"ok","ready":true,"uptime_sec":45.12}
```

`ready: true` artinya api-service **berhasil load model dari mlflow-server** via
network `fireguard-net`. **Inilah bukti komunikasi antar container.**

#### C. Cek model-info

```bash
curl http://localhost:8000/model-info
```
```json
{
  "name": "fireguard-regressor",
  "version": "3",
  "stage": "Production",
  "loaded_at": 1747350123.45,
  "ready": true,
  "tracking_uri": "http://mlflow-server:5000",
  "feature_count": 27
}
```

`tracking_uri: "http://mlflow-server:5000"` — pakai **nama service**, bukan localhost.

**Screenshot wajib untuk LK09:**
2. Output `curl /health` + `curl /model-info` — api berhasil menarik model dari mlflow-server

#### D. (Opsional) Test inference

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hotspot_count": 12, "frp_mean": 18.5, "frp_max": 27.0, "frp_sum": 222,
    "n_daytime": 5, "n_nighttime": 7, "n_confidence_high": 4,
    "temperature_2m_max": 33.2, "temperature_2m_min": 24.1,
    "precipitation_sum": 0.3, "windspeed_10m_max": 11.0,
    "winddirection_10m_dominant": 145.0, "relative_humidity_2m_mean": 62,
    "month": 8, "day_of_year": 220, "month_sin": 0.866, "month_cos": -0.5,
    "hotspot_count_1d": 10, "hotspot_count_3d": 28, "hotspot_count_7d": 60,
    "frp_mean_1d": 16, "frp_mean_3d": 17, "frp_mean_7d": 15,
    "hotspot_count_lag_1d": 8, "hotspot_count_lag_3d": 22, "hotspot_count_lag_7d": 50,
    "days_since_rain": 5
  }'
```

Response:
```json
{
  "hotspot_count_tomorrow": 14.23,
  "risk_level": 2,
  "risk_label": "Bahaya",
  "served_by": {"name": "fireguard-regressor", "version": "3", "stage": "Production"}
}
```

---

## Tahap 7 — Seed Model ke Registry (kalau Registry kosong)

Kalau `/model-info` return `ready: false`, artinya registry kosong. Buat model
"smoke" dengan synthetic data + auto-register:

```bash
# 1. Pakai docker exec ke api-service untuk akses env yang sudah set MLFLOW_TRACKING_URI
docker compose exec api-service bash

# Di dalam container:
python -m src.utils.synthetic_data --n-provinces 5 --n-days 60 --seed 42
python -m src.models.train --algorithm regressor --register-as fireguard-regressor
python -m src.models.registry transition --model fireguard-regressor --version 1 --stage Production
exit

# 2. Restart api-service supaya load ulang model
docker compose restart api-service
sleep 5
curl http://localhost:8000/health   # ready: true
```

> **Alternatif (lebih mudah):** lakukan train + register dari host Codespace
> dengan `MLFLOW_TRACKING_URI=http://localhost:5000`, lalu restart api-service.

---

## Tahap 8 — Lifecycle Commands

| Tujuan | Perintah |
|---|---|
| Build ulang image setelah ganti code | `docker compose build` |
| Start semua service (detached) | `docker compose up -d` |
| Cek status & port | `docker compose ps` |
| Follow logs satu service | `docker compose logs -f api-service` |
| Restart satu service | `docker compose restart api-service` |
| Stop (volumes tetap) | `docker compose down` |
| Stop + hapus volumes (data hilang) | `docker compose down -v` |
| Eksekusi shell di container | `docker compose exec api-service bash` |
| Cek resource usage | `docker stats fireguard-mlflow fireguard-api` |
| Cek network detail | `docker network inspect fireguard-net` |

---

## Troubleshooting

### `api-service` exit dengan "Connection refused"
Mlflow belum healthy. Cek `docker compose logs mlflow-server`. Kalau healthcheck
gagal terus, naikkan `start_period` di compose. Atau cek apakah port 5000
bentrok dengan service lain di host.

### `/health` return `ready: false` tapi mlflow-server healthy
Registry kosong — tidak ada model di stage `Production` maupun `Staging`. Lihat
**Tahap 7** untuk seeding.

### Build fail dengan "no space left on device"
```bash
docker system prune -af --volumes  # hapus image+volume+cache tidak terpakai
```

### Port 5000 atau 8000 sudah dipakai
Edit `ports:` di compose, misal `5001:5000` dan `8001:8000`. Ingat: kalau
ubah port mlflow, **JANGAN** ubah `MLFLOW_TRACKING_URI` env karena itu
internal network — port internal container tetap 5000.

### Container exit code 137 / OOMKilled
Hit memory limit di `deploy.resources.limits`. Naikkan `memory:` di compose,
atau audit kebocoran memori di code.

### Permission denied saat tulis ke volume
Image jalan sebagai non-root user (uid 10001/10002). Volume managed oleh
Docker, jadi ownership otomatis benar. Kalau error muncul, biasanya karena
bind mount manual ke folder host yang dimiliki root.

---

## Security Checklist LK09

- [x] Non-root user di kedua image (uid 10001 untuk API, 10002 untuk MLflow)
- [x] Multi-stage Dockerfile.api → image runtime tanpa gcc/build tools
- [x] `.dockerignore` block `.env`, `.git`, `data/`, `mlruns/` (cegah leak)
- [x] `security_opt: no-new-privileges:true` di compose
- [x] `deploy.resources.limits` (memory hard-cap)
- [x] HEALTHCHECK + `depends_on.condition: service_healthy`
- [x] Tidak ada credential hardcoded — semua via env var
- [x] Pydantic v2 strict validation di `/predict` (bounds di setiap field)
- [x] Error handler mask internal exception → tidak leak stack trace ke client

## Memory Hygiene Checklist

- [x] Model di-load **sekali** di FastAPI lifespan startup (bukan per-request)
- [x] `_STATE["model"] = None` di shutdown → bantu GC release memori
- [x] `--workers 1` di uvicorn → tidak duplikasi model di RAM per worker
- [x] MLflow client tidak cache run history yang membesar terus
- [x] Hard memory limit di compose → mencegah balloon ke seluruh host RAM

---

## Bukti Submission LK09

| # | Instruksi | Bukti |
|---|---|---|
| 1 | Identifikasi Layanan | `mlflow-server` + `api-service` di `docker-compose.yaml` |
| 2 | Manifest YAML | File `docker-compose.yaml` di root |
| 3 | Custom Bridge Network | `networks.fireguard-net.driver: bridge` |
| 4 | Volumes Persisten | `volumes.mlflow-db`, `mlflow-artifacts` |
| 5 | Pengaturan Ketergantungan | `depends_on.mlflow-server.condition: service_healthy` |
| 6 | Eksekusi & Verifikasi | Screenshot `docker compose ps` (healthy) + `curl /health` (ready:true) |

---

*LK09 selesai = `docker compose up -d` sukses, 2 service `healthy`, api ↔ mlflow communication terbukti, LK09_FireGuard.docx submitted.*
