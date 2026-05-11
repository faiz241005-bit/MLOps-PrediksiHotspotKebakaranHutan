# LK04 — Panduan Run Ingestion & Preprocessing

> Panduan teknis untuk menjalankan pipeline ingestion FireGuard di Codespace,
> uji real fetch ke NASA FIRMS dan Open-Meteo, lalu push ke branch `feat/data-pipeline`.

---

## Prasyarat

- Codespace MLOps-PrediksiHotspotKebakaranHutan sudah aktif (sesuai LK02).
- `.env` sudah berisi `NASA_FIRMS_API_KEY` valid.
- Sudah `pull` perubahan terbaru dari `main` (atau pull branch baru ini).

---

## Tahap 1 — Pull / Setup Branch

```bash
cd /workspaces/MLOps-PrediksiHotspotKebakaranHutan
git fetch origin
git checkout main && git pull

# Buat branch eksperimen untuk pekerjaan LK03+LK04
git checkout -b feat/data-pipeline
```

Verifikasi file baru sudah ada:

```bash
ls src/data/
# Harus ada: __init__.py  fetch_firms.py  fetch_weather.py  ingest_data.py  preprocess.py

ls tests/
# Harus ada: __init__.py  conftest.py  test_fetch_firms.py  test_preprocess.py
```

---

## Tahap 2 — Install Dependency Tambahan

`requirements.txt` sudah diupdate dengan `pyarrow` (untuk parquet). Install:

```bash
pip install -r requirements.txt
```

Atau install yang baru saja:

```bash
pip install pyarrow==16.1.0
```

---

## Tahap 3 — Jalankan Unit Test (no real network)

```bash
python -m pytest tests/ -v
```

Output yang benar: **31 passed**. Kalau ada test gagal, kabarkan saya.

---

## Tahap 4 — Setup Config untuk Production

```bash
cp config/params.example.yaml config/params.yaml
# Tidak perlu edit kalau default sudah sesuai (5 provinsi).
# File config/params.yaml ini sengaja NOT committed (private setting).
```

> Catatan: `params.yaml` belum di-gitignore. Kalau Anda merubahnya dan tidak mau commit, tambahkan ke `.gitignore`. Default-nya saya biarkan commitable supaya CI tetap punya config.

---

## Tahap 5 — Test Real Fetch (1 provinsi dulu)

### 5.1 — Fetch NASA FIRMS untuk Riau

```bash
python -m src.data.fetch_firms \
    --province riau \
    --bbox 0.0 100.0 4.5 106.5 \
    --day-range 2 \
    --log-level INFO
```

Output yang benar:

```
2026-05-11 ... | INFO | src.data.fetch_firms | Fetching FIRMS province=riau ...
2026-05-11 ... | INFO | src.data.fetch_firms | Wrote riau_20260511_HHMMSS_UTC.csv (N lines, M bytes)
/workspaces/MLOps-PrediksiHotspotKebakaranHutan/data/raw/firms/riau_20260511_HHMMSS_UTC.csv
```

Periksa isinya (head 3 baris):

```bash
head -3 data/raw/firms/riau_*.csv
```

Harus ada header `latitude,longitude,bright_ti4,...` dan beberapa baris data (musim kemarau biasanya banyak; musim hujan bisa kosong/sedikit).

### 5.2 — Fetch cuaca Open-Meteo untuk Riau

```bash
python -m src.data.fetch_weather \
    --province riau \
    --bbox 0.0 100.0 4.5 106.5 \
    --past-days 7 \
    --forecast-days 1
```

Output:

```
... INFO ... Wrote riau_20260511_HHMMSS_UTC.csv (8 rows, ~10 cols)
data/raw/weather/riau_20260511_HHMMSS_UTC.csv
```

### 5.3 — Orchestrator: fetch semua provinsi sekaligus

```bash
python -m src.data.ingest_data --provinces all --sources firms,weather
```

Atau hanya 2 provinsi:

```bash
python -m src.data.ingest_data --provinces riau kalteng --sources firms,weather
```

---

## Tahap 6 — Test Preprocessing (Cleaning + Join)

```bash
python -m src.data.preprocess
```

Output:

```
... INFO ... Loading FIRMS from data/raw/firms
... INFO ... Loaded FIRMS: NNN rows
... INFO ... Loading weather from data/raw/weather
... INFO ... Loaded weather: NN rows
... INFO ... FIRMS dropna critical: ... → ... rows
... INFO ... FIRMS dedup: ... → ... rows
... INFO ... Weather cleaned: ... rows
... INFO ... Wrote firms_weather_joined_20260511_HHMMSS_UTC.parquet (X rows, Y cols)
data/processed/firms_weather_joined_20260511_HHMMSS_UTC.parquet
```

Verifikasi output parquet:

```bash
python -c "
import pandas as pd
import glob
file = sorted(glob.glob('data/processed/firms_weather_joined_*.parquet'))[-1]
df = pd.read_parquet(file)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print(df.head(3))
"
```

---

## Tahap 7 — Test Non-Destructive Append

Jalankan fetch ulang sekarang — file lama harus **tetap ada**, file baru ditambahkan:

```bash
ls -la data/raw/firms/  # catat berapa file ada
python -m src.data.fetch_firms --province riau --bbox 0.0 100.0 4.5 106.5
ls -la data/raw/firms/  # harus tambah 1 file baru
```

Ini mendemonstrasikan **simulasi periodik** (instruksi LK04 No. 3): "skrip dapat dijalankan ulang untuk mengambil data terbaru tanpa menimpa data lama".

---

## Tahap 8 — Commit & Push

### 8.1 — Security check sebelum commit (WAJIB)

```bash
# Pastikan tidak ada API key di file yang akan di-commit
git status
git diff --cached
grep -rE "(api_key|API_KEY)\s*=\s*[\"'][a-zA-Z0-9]{8,}" src/ tests/ docs/ \
    && echo "STOP: ada potensi key hardcoded!" || echo "OK: tidak ada key di kode"

# Pastikan .env tidak akan ke-commit
git check-ignore -v .env
```

### 8.2 — Sample data ke repo (bukti ingestion bekerja)

Untuk submission, kita commit **1 sample CSV** ke repo (per sumber) sebagai bukti pipeline jalan. Sample harus **kecil** dan **tidak sensitive**:

```bash
# Pilih file FIRMS yang ada datanya (cek jumlah baris)
wc -l data/raw/firms/*.csv

# Ambil 1 yang terkecil/representatif untuk sample (override .gitignore data/raw/)
ls -t data/raw/firms/*.csv | head -1 > /tmp/firms_sample.txt
ls -t data/raw/weather/*.csv | head -1 > /tmp/weather_sample.txt

# Force-add (bypass .gitignore) sample files saja
git add -f $(cat /tmp/firms_sample.txt)
git add -f $(cat /tmp/weather_sample.txt)
```

> **Catatan:** Cara lain (lebih bersih) adalah memakai DVC untuk versioning sample. Untuk LK04, force-add ke Git sudah cukup karena ukuran file kecil (< 100KB biasanya).

### 8.3 — Commit semua perubahan

```bash
git add src/data/ tests/ docs/LK04_RUN_GUIDE.md pytest.ini requirements.txt config/

git status   # review apa yang akan di-commit

git commit -m "feat(data): implement ingestion + preprocessing pipeline (LK03+LK04)

- src/data/fetch_firms.py: NASA FIRMS fetcher with retry, allow-list URL,
  timestamp filename (non-destructive), schema validation
- src/data/fetch_weather.py: Open-Meteo daily weather fetcher
- src/data/ingest_data.py: orchestrator CLI for multi-province + multi-source
- src/data/preprocess.py: cleaning, dedup, timezone (WIB), join FIRMS+weather,
  write to data/processed/ as parquet
- tests/: 31 unit tests covering security (URL allow-list, path traversal,
  bbox validation), schema validation, dedup, join behavior; all using mock
  HTTP (no real network)
- docs/LK04_RUN_GUIDE.md: end-to-end run instructions
- requirements.txt: pin pyarrow 16.1.0
- Sample raw data committed as proof of ingestion (data/raw/*/riau_*.csv)"
```

### 8.4 — Push & buka PR

```bash
git push -u origin feat/data-pipeline
```

Lalu di GitHub UI:

1. Buka https://github.com/faiz241005-bit/MLOps-PrediksiHotspotKebakaranHutan/pulls
2. Klik **Compare & pull request** untuk `feat/data-pipeline`
3. PR title: `feat(data): pipeline ingestion + preprocessing (LK03+LK04)`
4. PR body — copy-paste template ini:

```markdown
## Ringkasan
Implementasi pipeline ETL FireGuard untuk LK03 (perencanaan) + LK04 (implementasi).

## Yang Ditambahkan
- 4 skrip Python production-ready (fetch_firms, fetch_weather, ingest_data, preprocess)
- 31 unit test (semua pakai mock HTTP, no real network)
- Run guide di docs/LK04_RUN_GUIDE.md
- Sample raw data (proof of real fetch)

## Security Checklist
- [x] API key dari env, tidak hard-coded
- [x] URL allow-list (mencegah SSRF)
- [x] Path traversal guard di output writer
- [x] Timeout 30s + retry exponential backoff pada semua HTTP call
- [x] HTTP redirect dimatikan (allow_redirects=False)
- [x] Tidak ada `print(api_key)` atau log API key

## Memory/Resource Hygiene
- [x] requests.Session pakai context manager (auto-close)
- [x] File I/O pakai context manager
- [x] Tidak ada global session yang persisten

## Tests
`python -m pytest tests/` — 31 passed
```

5. Klik **Create pull request** → tunggu CI hijau (kalau workflow `.github/workflows/ci_test.yml` aktif) → **Squash & merge**.

### 8.5 — Sync lokal setelah merge

```bash
git checkout main
git pull origin main
git branch -d feat/data-pipeline
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Pastikan Anda jalankan dari root repo: `cd /workspaces/MLOps-PrediksiHotspotKebakaranHutan`. Dan pakai `python -m src.data.fetch_firms`, bukan `python src/data/fetch_firms.py`.

### `Empty CSV returned for ...`

Kemungkinan musim hujan / area tidak ada hotspot dalam day_range terakhir. Coba `--day-range 10` (maksimum FIRMS) atau ganti bbox ke area yang sedang banyak titik api.

### `Disallowed host: ...`

Anda mengubah URL di luar allow-list. Ini security feature — jangan di-bypass. Tambahkan host baru ke `_ALLOWED_HOSTS` di kode kalau memang perlu.

### Open-Meteo `400 Bad Request`

Coba kurangi `past_days` (max 92) atau `forecast_days` (max 16) sesuai docs Open-Meteo.

---

## Bukti Submission LK04

Untuk dosen, siapkan:

1. **Link PR `feat/data-pipeline`** di GitHub (closed/merged).
2. **Screenshot terminal** menampilkan `python -m src.data.ingest_data --provinces riau` dengan output sukses + filename ber-timestamp.
3. **Screenshot file** di `data/raw/firms/` dan `data/raw/weather/` (terlihat beberapa file dengan timestamp berbeda → demonstrasi non-destructive).
4. **Screenshot pytest** `31 passed`.
5. **File output** `data/processed/firms_weather_joined_*.parquet` (boleh attach atau tunjukkan via `pandas.read_parquet`).

---

*LK04 selesai = PR merged + 5 bukti di atas. Lanjut ke LK05 (feature engineering + training).*
