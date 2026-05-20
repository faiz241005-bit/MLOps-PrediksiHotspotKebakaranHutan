# LK10 — Panduan Model Serving & Horizontal Scaling

> Walkthrough lengkap mengoperasionalkan model dari MLflow Registry menjadi
> layanan REST API via `mlflow models serve` + `mlflow models build-docker`,
> dengan simulasi horizontal scaling melalui `deploy.replicas: 3` di Docker
> Compose.

---

## Konsep

```
┌────────────────────────────────────────────────────────────────┐
│  MLflow Model Registry (fireguard-regressor v2 Production)     │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             │ mlflow models build-docker
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  Docker Image: fireguard/mlflow-model-server:1.0              │
│  (auto-generated dari MLflow — tidak butuh Dockerfile manual)  │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             │ deploy: replicas: 3
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  Docker Compose Stack                                           │
│                                                                 │
│   Replica 1 :8010  ────┐                                       │
│   Replica 2 :8011  ────┼───── POST /invocations                │
│   Replica 3 :8012  ────┘      Body: {"dataframe_split": {...}} │
│                                                                 │
│   3 instances of same image, untuk simulasi high-load serving  │
└────────────────────────────────────────────────────────────────┘
```

---

## Prasyarat

- Model `fireguard-regressor` versi terbaru sudah di stage **Production** di MLflow Registry (sudah dilakukan di LK07-LK09)
- Docker Compose stack dari LK09 sudah bisa jalan (`mlflow-server` minimum)
- MLflow CLI ter-install (`pip install mlflow==2.13.0` — sudah di `requirements.txt`)

---

## Tahap 1 — Verifikasi Model di Production

```bash
# Pastikan mlflow-server jalan
docker compose up -d mlflow-server
sleep 30

# Cek model versi yang Production
python -m src.models.registry list --model fireguard-regressor
```

Expected output:
```
v1 | stage=Archived
v2 | stage=Production    ← model yang akan di-serve
```

---

## Tahap 2 — Model Serving Lokal (Eksplorasi)

Sebelum containerize, test dulu `mlflow models serve` di Codespace host:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000

mlflow models serve \
  -m "models:/fireguard-regressor/Production" \
  --port 9999 \
  --host 0.0.0.0 \
  --no-conda
```

Process akan running di foreground. Buka terminal baru, test:

```bash
# Health check
curl http://localhost:9999/ping
# Response: kosong (status 200)

# Predict
curl -X POST http://localhost:9999/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["hotspot_count", "frp_mean", "frp_max", "frp_sum",
                  "n_daytime", "n_nighttime", "n_confidence_high",
                  "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                  "windspeed_10m_max", "winddirection_10m_dominant",
                  "relative_humidity_2m_mean",
                  "month", "day_of_year", "month_sin", "month_cos",
                  "hotspot_count_1d", "hotspot_count_3d", "hotspot_count_7d",
                  "frp_mean_1d", "frp_mean_3d", "frp_mean_7d",
                  "hotspot_count_lag_1d", "hotspot_count_lag_3d", "hotspot_count_lag_7d",
                  "days_since_rain"],
      "data": [[1323, 50.0, 100.0, 5000, 500, 800, 300,
                33.0, 24.0, 0.5, 12.0, 180.0, 65,
                9, 265, -0.5, -0.866,
                1323.0, 2500.0, 4000.0,
                50.0, 45.0, 40.0,
                1000.0, 2200.0, 3500.0, 15]]
    }
  }'
```

Response: `{"predictions": [323.531]}`

Stop server lokal (Ctrl+C) sebelum lanjut ke containerize.

---

## Tahap 3 — Containerize Model via `build-docker`

```bash
mlflow models build-docker \
  --model-uri "models:/fireguard-regressor/Production" \
  --name fireguard/mlflow-model-server:1.0 \
  --env-manager local
```

⏱️ **5-15 menit**.

**Apa yang dilakukan:**

1. MLflow tarik model artifact (`model.pkl`, `MLmodel`, `conda.yaml`, `requirements.txt`) dari Registry
2. Build base Python image dengan semua dependency model
3. Inject MLflow serving script di entry point
4. Tag image sebagai `fireguard/mlflow-model-server:1.0`

Verify image ada:

```bash
docker images | grep fireguard/mlflow-model-server
```

Expected:
```
fireguard/mlflow-model-server   1.0   <hash>   ...   ~2.5 GB
```

Image MLflow native lebih besar dari custom FastAPI (~700 MB) karena include full Python + scikit + lightgbm + dependencies model, plus generic MLflow wrapper.

---

## Tahap 4 — Modify `docker-compose.yaml` dengan `replicas: 3`

Service `mlflow-model-server` di docker-compose.yaml:

```yaml
mlflow-model-server:
  image: fireguard/mlflow-model-server:1.0
  # JANGAN set container_name (konflik dengan replicas)
  restart: unless-stopped
  networks:
    - fireguard-net
  ports:
    - "8010-8012:8080"          # 3 host ports → 1 port di tiap container
  depends_on:
    mlflow-server:
      condition: service_healthy
  healthcheck:
    test: ["CMD", "curl", "--silent", "--fail", "http://localhost:8080/ping"]
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 30s
  deploy:
    replicas: 3                  # ← LK10 KEY REQUIREMENT
    resources:
      limits:
        memory: 800m
        cpus: "0.5"
  security_opt:
    - no-new-privileges:true
```

**Key konfigurasi:**

| Konfigurasi | Penjelasan |
|---|---|
| `image: ...` | Pakai image yang sudah di-build (bukan `build:` dari Dockerfile) |
| Tidak ada `container_name` | **WAJIB** dihapus karena container name harus unique, replicas membuat banyak container |
| `ports: "8010-8012:8080"` | Range 3 port host masing-masing forward ke port 8080 (default MLflow) di container |
| `deploy.replicas: 3` | 3 instance container running paralel |
| `deploy.resources.limits` | Setiap replica dibatasi 800 MB RAM + 0.5 CPU core |

---

## Tahap 5 — Run Stack & Verify 3 Replicas

```bash
docker compose up -d
sleep 60
docker compose ps
```

Expected output (4 container — 1 mlflow-server + 3 replicas):

```
NAME                                          IMAGE                                STATUS              PORTS
fireguard-mlflow                              fireguard/mlflow-server:1.0          Up 1 min (healthy)  0.0.0.0:5000->5000/tcp
mlops-fireguard-mlflow-model-server-1         fireguard/mlflow-model-server:1.0    Up 1 min (healthy)  0.0.0.0:8010->8080/tcp
mlops-fireguard-mlflow-model-server-2         fireguard/mlflow-model-server:1.0    Up 1 min (healthy)  0.0.0.0:8011->8080/tcp
mlops-fireguard-mlflow-model-server-3         fireguard/mlflow-model-server:1.0    Up 1 min (healthy)  0.0.0.0:8012->8080/tcp
```

**Screenshot wajib untuk LK10:**
1. `docker compose ps` dengan 3 replicas `(healthy)`

---

## Tahap 6 — Uji Coba Endpoint Antar Replicas

Test ke setiap replica untuk demonstrate semua respond:

```bash
for port in 8010 8011 8012; do
  echo "=== Replica @ port $port ==="
  curl -s http://localhost:$port/ping | head -5
  curl -s -X POST http://localhost:$port/invocations \
    -H "Content-Type: application/json" \
    -d '{
      "dataframe_split": {
        "columns": ["hotspot_count", "frp_mean", "frp_max", "frp_sum",
                    "n_daytime", "n_nighttime", "n_confidence_high",
                    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                    "windspeed_10m_max", "winddirection_10m_dominant",
                    "relative_humidity_2m_mean",
                    "month", "day_of_year", "month_sin", "month_cos",
                    "hotspot_count_1d", "hotspot_count_3d", "hotspot_count_7d",
                    "frp_mean_1d", "frp_mean_3d", "frp_mean_7d",
                    "hotspot_count_lag_1d", "hotspot_count_lag_3d", "hotspot_count_lag_7d",
                    "days_since_rain"],
        "data": [[1323, 50.0, 100.0, 5000, 500, 800, 300, 33.0, 24.0, 0.5, 12.0, 180.0, 65,
                  9, 265, -0.5, -0.866, 1323.0, 2500.0, 4000.0, 50.0, 45.0, 40.0,
                  1000.0, 2200.0, 3500.0, 15]]
      }
    }' | python3 -m json.tool
done
```

Expected: ketiga replicas return prediksi sama (model identik).

**Screenshot wajib untuk LK10:**
2. Output predict dari salah satu replica (input + output)

---

## Tahap 7 — Dynamic Scaling

Tambah replicas dinamis tanpa modify YAML:

```bash
# Scale UP ke 5 replicas
docker compose up -d --scale mlflow-model-server=5
sleep 30
docker compose ps

# Scale DOWN ke 1 replica
docker compose up -d --scale mlflow-model-server=1
docker compose ps
```

⚠️ **Catatan port range:** Default YAML cuma cover port 8010-8012 (3 replicas). Untuk lebih dari 3:
- **Opsi A:** Edit `ports: "8010-8014:8080"` untuk 5 replicas
- **Opsi B:** Pakai single port + reverse proxy nginx untuk load balancing

---

## Tahap 8 — End-to-End Demo (predict_tomorrow.py)

Saya update `src/scripts/predict_tomorrow.py` untuk pakai endpoint `/invocations`:

```bash
# Skip fetch — instant
python -m src.scripts.predict_tomorrow --skip-fetch

# Full pipeline (fetch + predict)
python -m src.scripts.predict_tomorrow
```

Expected output:
```
======================================================================
🔥 FireGuard — Prediksi Hotspot Besok per Provinsi
======================================================================
  Endpoint         : http://localhost:8010/invocations
  Server           : MLflow native (LK10, replicas:3)
  ...
======================================================================

Provinsi                    Today     Tomorrow Risk
----------------------------------------------------------------------
Kalimantan Tengah            1323        323.5 🔴 Bahaya
Kalimantan Barat              215        180.2 🔴 Bahaya
...
```

---

## Format Input MLflow `/invocations`

3 format yang didukung MLflow native:

### Format 1: `dataframe_split` (REKOMENDASI)

```json
{
  "dataframe_split": {
    "columns": ["col1", "col2", ...],
    "data": [[v1, v2, ...]]
  }
}
```

### Format 2: `dataframe_records`

```json
{
  "dataframe_records": [
    {"col1": v1, "col2": v2, ...}
  ]
}
```

### Format 3: `instances` (deprecated tapi masih support)

```json
{
  "instances": [[v1, v2, ...]]
}
```

Kami pakai `dataframe_split` karena schema-explicit dan match dengan MLflow model signature.

---

## Perbandingan LK09 vs LK10

| Aspek | LK09 (custom FastAPI) | LK10 (MLflow native) |
|---|---|---|
| **Source code** | Manual `src/api/main.py` | Auto-generated |
| **Build** | Manual `Dockerfile.api` | `mlflow models build-docker` |
| **Endpoint** | `/predict` (custom) + `/health` + `/model-info` | `/invocations` + `/ping` |
| **Input format** | Pydantic schema (named fields) | DataFrame format (columns + data) |
| **Output** | Rich JSON (count + risk_level + label + served_by) | Raw predictions array |
| **Replicas** | 1 (single container) | **3** (horizontal scaling) |
| **Use case** | Production with custom logic | Quick deployment + scaling |

---

## Security & Memory Checklist LK10

- [x] **Image dari MLflow signature** — model loaded dari registry, bukan path lokal
- [x] **No `container_name`** — agar replicas tidak konflik
- [x] **Port range** — 1 port per replica, tidak overlapping
- [x] **`security_opt: no-new-privileges`** — anti privilege escalation
- [x] **`deploy.resources.limits`** — hard memory cap per replica (800 MB)
- [x] **`depends_on: condition: service_healthy`** — wait MLflow-server siap sebelum start
- [x] **Healthcheck** — Docker tahu replica sehat/tidak (auto-restart kalau crash)

---

## Bukti Submission LK10

| # | Instruksi | Bukti |
|---|---|---|
| 1 | Model di Registry Production | `registry list` output: v2 stage=Production |
| 2 | `mlflow models serve` lokal | Tahap 2 — curl /invocations berhasil |
| 3 | Uji endpoint | Tahap 6 — output prediksi 3 replicas |
| 4 | `mlflow models build-docker` | Image `fireguard/mlflow-model-server:1.0` ada di `docker images` |
| 5 | `replicas: 3` di compose | Tahap 4 — block YAML lengkap |
| 6 | Verifikasi 3 replicas running | Tahap 5 — `docker compose ps` (3 healthy) |
| 7 | README.md instruksi | Section 3d di README.md dengan akses endpoint + dynamic scaling |

---

## Troubleshooting

### `mlflow models build-docker` gagal "model not found"

```bash
# Pastikan tracking URI benar
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models list      # cek model exist

# Pastikan stage Production ada
python -m src.models.registry list --model fireguard-regressor
```

### Replicas tidak start (hanya 1 yang up)

```bash
# Cek apakah port range cukup
docker compose ps
# Kalau bentrok: edit ports range di YAML, atau scale down dulu
docker compose down
docker compose up -d
```

### Predict return 400 / schema error

Format input harus match dengan model signature. Common mistake:
- Kolom missing → tambahkan dengan default value
- Order kolom salah → pakai `_FEATURE_COLUMNS` di `predict_tomorrow.py` untuk reference

### Image size besar (~2.5 GB)

Normal untuk MLflow native — include full Python runtime + libs + model. Untuk production yang care size, custom FastAPI (LK09) lebih hemat (~700 MB).

---

*LK10 selesai = 3 replicas healthy, predict endpoint working, screenshot bukti, dan LK10_FireGuard.docx submitted.*
