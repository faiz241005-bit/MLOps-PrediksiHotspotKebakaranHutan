# LK11 — Observability & Dashboard Guide

Panduan setup Prometheus + Grafana untuk monitoring FireGuard MLOps stack.

---

## Arsitektur

```
                              ┌──────────────────────┐
                              │   Grafana (3000)     │
                              │   - Dashboards       │
                              └──────────┬───────────┘
                                         │ PromQL
                              ┌──────────▼───────────┐
                              │  Prometheus (9090)   │
                              │  - 15s scrape        │
                              │  - 15d retention     │
                              └────┬─────────────┬───┘
                                   │             │
                  ┌────────────────┘             └──────────────┐
                  │ /metrics                                    │ /metrics
                  ▼                                             ▼
        ┌──────────────────┐                          ┌──────────────────┐
        │ metrics-proxy    │                          │ cAdvisor (8085)  │
        │ (FastAPI, 9000)  │                          │ - CPU per cont.  │
        │ - latency hist   │                          │ - RAM per cont.  │
        │ - request count  │                          │ - net I/O        │
        │ - prediction val │                          └──────────────────┘
        └────────┬─────────┘
                 │ POST /invocations
                 ▼ (Docker DNS round-robin)
        ┌──────────────────────────────────┐
        │  mlflow-model-server (x3)        │
        │  fireguard/mlflow-model-server   │
        │  ports 8010-8012 → 8080          │
        └──────────────────────────────────┘
                 ▲
                 │ FIREGUARD_API_URL=http://metrics-proxy:9000
                 │
        ┌────────┴──────────┐
        │ Streamlit (8501)  │
        └───────────────────┘
```

---

## Quickstart

### Prerequisite

- Sudah selesai LK10 (image `fireguard/mlflow-model-server:1.0` ada di local registry)
- Docker Compose v2+

### Jalankan stack

```bash
# Build & start semua service (7 container saat replicas=3)
docker compose up -d --build

# Cek status — semua harus healthy
docker compose ps
```

Ekspektasi output:
```
NAME                            STATUS                   PORTS
fireguard-cadvisor              Up X minutes (healthy)   0.0.0.0:8085->8080/tcp
fireguard-dashboard             Up X minutes (healthy)   0.0.0.0:8501->8501/tcp
fireguard-grafana               Up X minutes (healthy)   0.0.0.0:3000->3000/tcp
fireguard-metrics-proxy         Up X minutes (healthy)   0.0.0.0:9000->9000/tcp
fireguard-mlflow                Up X minutes (healthy)   0.0.0.0:5000->5000/tcp
fireguard-prometheus            Up X minutes (healthy)   0.0.0.0:9090->9090/tcp
mlops_mlflow-model-server_1     Up X minutes (healthy)   0.0.0.0:8010->8080/tcp
mlops_mlflow-model-server_2     Up X minutes (healthy)   0.0.0.0:8011->8080/tcp
mlops_mlflow-model-server_3     Up X minutes (healthy)   0.0.0.0:8012->8080/tcp
```

### Akses UI

| Service | URL | Credentials |
|---|---|---|
| MLflow Tracking | http://localhost:5000 | — |
| Streamlit Dashboard | http://localhost:8501 | — |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |
| Prometheus | http://localhost:9090 | — |
| cAdvisor (debug) | http://localhost:8085 | — |
| metrics-proxy `/metrics` | http://localhost:9000/metrics | — |

---

## Verifikasi Setup

### 1. Cek metrics-proxy expose metrics

```bash
curl http://localhost:9000/metrics | head -40
```

Ekspektasi: ada baris `mlflow_request_duration_seconds_bucket{...}`, `mlflow_requests_total{...}`, dll.

### 2. Cek Prometheus berhasil scrape

Buka http://localhost:9090/targets

Semua 3 target harus `UP`:
- `mlflow-metrics-proxy (1/1 up)`
- `cadvisor (1/1 up)`
- `prometheus (1/1 up)`

**📸 Screenshot ini untuk bukti scraping (laporan LK11).**

### 3. Cek Grafana datasource auto-provisioned

1. Login Grafana → **Connections → Data sources** → `Prometheus` sudah terdaftar
2. Klik **Test** → harus muncul "Successfully queried the Prometheus API"

### 4. Cek dashboard auto-loaded

**Dashboards → FireGuard → FireGuard MLOps Observability (LK11)**

Dashboard punya 10 panel. Awalnya kosong karena belum ada trafik.

---

## Simulasi Beban Kerja (LK11 poin 5)

### Cara 1: Streamlit "Predict besok"

1. Buka http://localhost:8501
2. Klik tombol **🔮 Predict besok (5 provinsi)** di sidebar
3. Tiap klik = 5 request ke metrics-proxy

### Cara 2: Load test script (RECOMMENDED untuk demo)

```bash
# Skenario mixed (default)
python -m src.scripts.load_test_lk11 --requests 200 --concurrency 10

# Untuk demonstrasi drift detection: 2 fase berbeda
# Fase 1: low-fire scenario (10 menit awal)
python -m src.scripts.load_test_lk11 --requests 200 --concurrency 5 --scenario low

# Tunggu 1-2 menit (biar terlihat di heatmap)

# Fase 2: high-fire scenario — drift heatmap akan SHIFT ke kanan
python -m src.scripts.load_test_lk11 --requests 200 --concurrency 5 --scenario high
```

### Cara 3: Manual curl (loop)

```bash
for i in {1..100}; do
  curl -s -X POST http://localhost:9000/invocations \
    -H "Content-Type: application/json" \
    -d '{"dataframe_split": {"columns": ["hotspot_count", ...], "data": [[...]]}}' \
    > /dev/null
  sleep 0.5
done
```

---

## Screenshot untuk Laporan LK11

Setelah load test berjalan, ambil screenshot **selama trafik masih live** (panel aktif):

### Wajib (sesuai instruksi)

1. **Grafana Dashboard penuh** — semua 10 panel terisi data
   - URL: http://localhost:3000/d/fireguard-lk11
   - Time range: Last 15 minutes (atau 30 menit)
   - Refresh: 15s

2. **Prometheus Targets** — bukti scraping
   - URL: http://localhost:9090/targets
   - Tampilkan: 3 target semua UP

### Tambahan (memperkuat laporan)

3. **Panel detail — Latensi Inferensi p50/p95/p99**
4. **Panel detail — Throughput per Status**
5. **Panel detail — CPU/RAM per Container** (terlihat 3 replicas mlflow-model-server)
6. **Panel detail — Prediction Distribution Heatmap** (sebelum & sesudah scenario shift)

---

## Analisis Monitoring untuk Model Decay (LK11 poin 3)

### Bagaimana dashboard mendeteksi penurunan performa model?

**1. Distribusi Nilai Prediksi (panel heatmap + quantile)**

Model decay paling sering terlihat dari **pergeseran distribusi prediksi**. Misal:

- Baseline: 80% prediksi <10 hotspots, 20% antara 10-50
- Setelah 1 bulan: 60% prediksi <10, 40% prediksi >50
- → Sinyal: input data drift (musim berganti) atau model semakin "panik" (prediksi over)

Heatmap memvisualisasikan ini dengan **bucket prediction value (sumbu Y) × waktu (sumbu X)**. Warna gelap = density tinggi.

**2. Latensi p99 yang naik perlahan**

Kalau p99 latensi naik bertahap (dari 200ms → 500ms → 1s dalam beberapa minggu), kemungkinan:
- Model file size membesar (re-training tambah parameter)
- Memory pressure di container (panel RAM akan korelasi)
- → Trigger investigasi & retraining

**3. Error rate creeping up**

Panel "Error Rate" menunjukkan rasio 4xx/5xx. Kalau muncul 4xx baru (misal 422 Unprocessable):
- Skema input berubah / data baru tidak sesuai signature model
- → Mismatch antara data production vs training data

**4. CPU/RAM container yang tidak stabil**

Kalau salah satu replica memori naik terus tanpa turun (saw-tooth pattern hilang) → kemungkinan memory leak di model wrapper. Bisa restart untuk recovery sementara, tapi perlu investigation.

**5. Throughput drop tanpa perubahan trafik**

Kalau RPS turun tapi error rate tetap rendah → kemungkinan throttling atau bottleneck baru. Korelasi dengan panel CPU.

### Action Plan Operasional

| Sinyal | Indikator dashboard | Action |
|---|---|---|
| Data drift | Heatmap prediksi shift signifikan | Schedule retraining dengan data terbaru |
| Latency degradation | p99 > SLA (mis. 1s) selama 10 min | Scale replicas naik, investigate model |
| Error spike | Error rate > 5% selama 5 min | Cek logs, mungkin rollback model version |
| Memory leak | RAM monoton naik selama jam-jam-an | Restart replica, jadwalkan fix |
| Resource pressure | CPU > 80% sustained | Auto-scale (`docker compose up --scale=N`) |

---

## Troubleshooting

### "Target DOWN" di Prometheus

```bash
# Cek apakah metrics-proxy hidup
docker compose logs metrics-proxy --tail 50

# Cek dari dalam network
docker compose exec prometheus wget -O- http://metrics-proxy:9000/metrics | head
```

### Grafana dashboard kosong walaupun ada trafik

1. Cek datasource: **Connections → Data sources → Prometheus → Test**
2. Buka **Explore**, jalankan query: `mlflow_requests_total`
3. Kalau data ada di Explore tapi panel kosong → cek time range panel (klik kanan atas)

### "503 upstream unreachable" dari proxy

```bash
# Cek mlflow-model-server replicas
docker compose ps mlflow-model-server

# Cek healthcheck
docker compose logs mlflow-model-server --tail 20
```

### cAdvisor target DOWN

cAdvisor butuh akses ke `/sys`, `/var/lib/docker`. Kalau di Codespace ada permission issue, alternatif:
- Skip cAdvisor (CPU/RAM panel akan kosong) — masih bisa demo latensi & drift
- Atau jalankan di host Docker biasa (bukan dalam Codespace dev container)

---

## Reset & Cleanup

```bash
# Stop semua, keep volumes (data Prometheus & Grafana persist)
docker compose down

# Stop & hapus volumes (FRESH START)
docker compose down -v
```

---

## Security Notes (production checklist)

- [ ] Ganti `GF_SECURITY_ADMIN_PASSWORD` dari `admin` ke nilai dari secret manager
- [ ] Tutup port 8085 (cAdvisor) dari public — hanya untuk internal monitoring
- [ ] Tutup port 9090 (Prometheus) dari public — hanya untuk Grafana
- [ ] Aktifkan HTTPS di Grafana via reverse proxy (Caddy/nginx)
- [ ] Set retention Prometheus sesuai compliance kamu (`--storage.tsdb.retention.time`)
- [ ] Backup `grafana-data` & `prometheus-data` volumes secara berkala
