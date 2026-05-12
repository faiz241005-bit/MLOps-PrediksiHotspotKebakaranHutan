# LK05 — Panduan Run DVC Integration

> Step-by-step setup DVC, tracking dataset LK04, simulasi continual learning,
> dan demonstrasi `dvc diff` antara dua versi data.

---

## Prasyarat

- `feat/data-pipeline` sudah merged ke `main` (LK04 done).
- Codespace aktif di branch baru `feat/dvc-mlflow`.
- Data hasil real fetch LK04 sudah ada di `data/raw/firms/` dan `data/raw/weather/`.
- API key NASA FIRMS tetap di `.env` Codespace (untuk simulasi continual learning).

---

## Tahap 1 — Setup Branch & Install DVC

### 1.1 — Buat branch baru dari main

```bash
cd /workspaces/MLOps-PrediksiHotspotKebakaranHutan
git checkout main
git pull origin main
git checkout -b feat/dvc-mlflow
```

### 1.2 — Install DVC + plugin Google Drive

```bash
# Pakai requirements yang sudah saya update (dvc 3.50.1 + dvc-gdrive 3.0.1)
pip install -r requirements.txt

# Verifikasi
dvc --version       # harus 3.50.1
dvc remote --help   # harus list 'add', 'modify', dst.
```

---

## Tahap 2 — Inisialisasi DVC

```bash
# Init di root repo
dvc init

# Verifikasi struktur baru
ls -la .dvc/
# Harus ada: .dvcignore, config, .gitignore (file)
```

`dvc init` membuat:
- `.dvc/config` — config repo (kosong awalnya)
- `.dvc/.gitignore` — block cache dari Git
- `.dvcignore` — pattern untuk skip file

```bash
# Commit metadata DVC ke Git (BUKAN data binary-nya)
git add .dvc/.gitignore .dvc/config .dvcignore
git status
git commit -m "feat(dvc): initialize DVC for dataset versioning (LK05)"
```

---

## Tahap 3 — Setup Google Drive Remote

### 3.1 — Buat folder di Google Drive

1. Buka https://drive.google.com
2. New → Folder → nama: **FireGuard-DVC-Storage**
3. Buka folder yang baru dibuat
4. **Copy folder ID** dari URL — bagian setelah `/folders/`:
   ```
   https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz12345
                                          └─ ini folder ID ─┘
   ```
   Simpan folder ID-nya.

### 3.2 — Konfigurasi remote DVC

Ganti `<FOLDER_ID>` dengan ID dari langkah 3.1:

```bash
dvc remote add -d gdrive gdrive://<FOLDER_ID>
dvc remote modify gdrive gdrive_acknowledge_abuse true

# Verifikasi config
cat .dvc/config
```

Output `cat` harusnya:
```
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://1AbCdEfGhIjKlMnOpQrStUvWxYz12345
    gdrive_acknowledge_abuse = true
```

### 3.3 — Commit config remote

```bash
git add .dvc/config
git commit -m "feat(dvc): add Google Drive remote storage (LK05 #6 optional)"
```

> **Catatan keamanan:** folder ID Google Drive **tidak rahasia** seperti API key — boleh di-commit. Yang harus dirahasiakan adalah token OAuth (yang akan dibuat di Tahap 4) — token disimpan di `.dvc/tmp/gdrive-user-credentials.json` yang sudah otomatis di-gitignore oleh DVC.

---

## Tahap 4 — Track Dataset Awal (Hasil LK04)

### 4.1 — DVC add folder data

```bash
# Track folder firms (semua CSV di dalamnya)
dvc add data/raw/firms

# Track folder weather
dvc add data/raw/weather
```

DVC akan:
1. Hitung hash MD5 setiap file di folder
2. Pindahkan file ke `.dvc/cache/`
3. Buat **symlink/reflink/copy** di lokasi asal supaya kode tetap baca dari path yang sama
4. Buat file metadata:
   - `data/raw/firms.dvc`
   - `data/raw/weather.dvc`

```bash
# Cek isi .dvc file (small JSON-like metadata)
cat data/raw/firms.dvc
```

Output kira-kira:
```yaml
outs:
- md5: a1b2c3d4e5f6... .dir
  size: 12345
  nfiles: 1
  hash: md5
  path: firms
```

### 4.2 — Commit metadata DVC ke Git

```bash
git status
# Harus muncul:
#   new file:   data/raw/firms.dvc
#   new file:   data/raw/weather.dvc
#   modified:   data/raw/.gitignore (DVC auto-add pattern untuk block raw data)

git add data/raw/firms.dvc data/raw/weather.dvc data/raw/.gitignore
git commit -m "feat(dvc): track initial dataset from LK04 ingestion"
```

### 4.3 — Push binary ke Google Drive

```bash
# Pertama kali: OAuth browser flow
dvc push
```

**Pengalaman OAuth:**
1. DVC akan print URL ke terminal: `Go to the following link: https://accounts.google.com/o/oauth2/auth?...`
2. Buka URL di browser
3. Login dengan akun Google Anda (yang punya akses ke folder DVC-Storage)
4. Klik **Allow** untuk grant DVC akses
5. Browser akan tampilkan **kode authorization**
6. Copy kode, paste balik ke terminal yang nunggu input
7. DVC simpan token ke `.dvc/tmp/gdrive-user-credentials.json` (gitignored)

Setelah token didapat, DVC upload file ke Google Drive. Output:
```
2 files pushed
```

Buka https://drive.google.com → folder FireGuard-DVC-Storage → harusnya ada subfolder hash-named (mis. `a1/b2c3d4...`).

---

## Tahap 5 — Simulasi Continual Learning

Ini bagian penting untuk demonstrasi: tambah data baru, lihat hash berubah.

### 5.1 — Fetch data baru (tunggu interval atau langsung)

```bash
# Fetch lagi (timestamp baru, file CSV baru ditambahkan)
python -m src.data.fetch_firms \
    --province riau \
    --bbox 0.0 100.0 4.5 106.5 \
    --day-range 2

python -m src.data.fetch_firms \
    --province kalteng \
    --bbox -3.5 110.5 1.5 116.5 \
    --day-range 2

python -m src.data.fetch_weather \
    --province riau \
    --bbox 0.0 100.0 4.5 106.5

# Lihat folder data
ls -la data/raw/firms/
ls -la data/raw/weather/
```

Harus ada file CSV baru dengan timestamp lebih baru. Karena nama file ber-timestamp, file lama tetap dipertahankan (non-destructive).

### 5.2 — DVC add ulang (track perubahan)

```bash
dvc add data/raw/firms
dvc add data/raw/weather
```

Output kali ini:
```
✓ Added 'data/raw/firms.dvc'
✓ Added 'data/raw/weather.dvc'
```

Sekarang `.dvc` files punya **hash MD5 berbeda** (karena isi folder berubah).

```bash
# Cek perubahan
git diff data/raw/firms.dvc
```

Anda akan lihat diff seperti:
```diff
-- md5: a1b2c3d4e5f6... .dir   # hash lama
-  nfiles: 1
+- md5: f7e8d9c0a1b2... .dir   # hash baru
+  nfiles: 2
```

### 5.3 — Push versi baru ke Google Drive

```bash
dvc push
```

DVC hanya upload **chunk baru** (deduplication). File lama yang masih sama tidak di-upload ulang — efisien!

### 5.4 — Commit versi baru ke Git

```bash
git add data/raw/firms.dvc data/raw/weather.dvc
git commit -m "feat(dvc): version 2 of dataset (continual learning sim)

- Added new FIRMS snapshots for Riau & Kalteng
- Added new Open-Meteo snapshot for Riau
- DVC hash updated; data binary pushed to Google Drive remote"
```

---

## Tahap 6 — Audit & Diff (Bukti Versioning)

### 6.1 — `dvc diff` antara dua commit

```bash
# Bandingkan dengan commit sebelumnya
dvc diff HEAD^ HEAD
```

Output kira-kira:
```
Modified:
    data/raw/firms
    data/raw/weather

files summary: 0 added, 0 removed, 2 modified
```

Untuk detail per-file:
```bash
dvc diff HEAD^ HEAD --json
```

### 6.2 — Audit lineage (silsilah perubahan)

```bash
# Lihat history DVC tracked files via Git
git log --all --oneline data/raw/firms.dvc data/raw/weather.dvc
```

Output:
```
<sha-v2> feat(dvc): version 2 of dataset (continual learning sim)
<sha-v1> feat(dvc): track initial dataset from LK04 ingestion
```

### 6.3 — Reproduce state lama (kalau perlu)

Untuk roll back data ke versi v1 (mis. untuk replicate training experiment lama):

```bash
git checkout <sha-v1> -- data/raw/firms.dvc data/raw/weather.dvc
dvc checkout
# data/raw/firms dan data/raw/weather sekarang berisi snapshot v1
```

Kembali ke versi terbaru:
```bash
git checkout main -- data/raw/firms.dvc data/raw/weather.dvc
dvc checkout
```

Ini adalah **superpower DVC**: Anda bisa "time travel" antara versi dataset tanpa duplikasi storage.

---

## Tahap 7 — Verifikasi & Screenshot untuk Submission

Untuk LK05.docx, siapkan screenshot/output dari:

1. **Output `dvc --version`** (bukti DVC terinstal)
2. **Output `dvc remote list`** (bukti remote Google Drive ter-config)
3. **Output `dvc status`** (clean state)
4. **Isi `data/raw/firms.dvc`** dan `data/raw/weather.dvc` (small metadata files)
5. **Output `dvc diff HEAD^ HEAD`** (bukti perbandingan dua versi)
6. **Screenshot Google Drive folder** menampilkan subfolder hash (bukti remote push sukses)
7. **Output `git log --oneline -5`** menampilkan dua commit DVC

---

## Tahap 8 — Push Branch

```bash
git push -u origin feat/dvc-mlflow
```

> Belum buka PR sekarang — kita akan lanjutkan dengan LK06 (training pipeline) di branch yang sama. PR dibuka setelah LK05 + LK06 dua-duanya selesai.

---

## Troubleshooting

### `dvc remote modify gdrive ... → Error: gdrive_acknowledge_abuse must be set`

```bash
dvc remote modify gdrive gdrive_acknowledge_abuse true
```
(Sudah ada di guide, kalau lupa run-nya.)

### `dvc push → Authentication failed`

Token OAuth expired atau corrupted. Hapus dan re-auth:
```bash
rm -rf .dvc/tmp/gdrive-user-credentials.json
dvc push   # akan trigger OAuth flow ulang
```

### `dvc add → Cache directory full`

```bash
# Pindahkan cache ke folder dengan space lebih besar (jarang terjadi di Codespace)
dvc cache dir /path/to/larger/disk
```

### `git push → too large objects`

Berarti ada file binary masuk Git tanpa via DVC. Cek:
```bash
git diff --cached --stat | sort -k2 -n -r | head -5
```
Kalau ada file > 100 MB, unstage dan track via DVC dulu.

---

## Bukti Submission LK05

Untuk dosen, siapkan:

1. **Link branch GitHub** `feat/dvc-mlflow` (atau PR-nya nanti)
2. **Screenshot terminal** menampilkan output 7 command di Tahap 7
3. **Screenshot Google Drive folder** dengan beberapa subfolder hash-named
4. **2 commit message** terkait DVC di history Git
5. **(Opsional)** file `LK05_FireGuard.docx` yang akan saya generate setelah Anda kirim output verifikasi

---

*Setelah LK05 selesai, saya tulis `train.py` untuk LK06 di branch yang sama.*
