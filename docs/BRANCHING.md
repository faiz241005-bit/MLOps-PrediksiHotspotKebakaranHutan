# Branching Strategy & Code Quality Guidelines

> **Strategy:** GitHub Flow (lightweight, sesuai untuk proyek individual production-oriented)

---

## 1. Branching Rules

### 1.1 `main`
* Selalu **deployable**.
* **Tidak boleh** `git push` langsung ke `main`.
* Hanya menerima merge via Pull Request yang sudah lolos CI.

### 1.2 Naming Convention

| Prefix | Tujuan | Contoh |
|---|---|---|
| `feat/` | Fitur baru (LK01–LK08, modul baru, integrasi API) | `feat/initial-eda`, `feat/dvc-setup`, `feat/firms-fetcher` |
| `fix/` | Bug fix | `fix/timezone-handling`, `fix/empty-frp-rows` |
| `exp/` | Eksperimen model — boleh long-lived | `exp/lightgbm-tuning`, `exp/lstm-baseline` |
| `chore/` | Maintenance (deps, formatting) | `chore/bump-mlflow-2.13` |
| `docs/` | Hanya dokumentasi | `docs/update-readme-codespace` |

### 1.3 Workflow Per LK

```bash
# 1. Pastikan main up-to-date
git checkout main
git pull origin main

# 2. Buat branch untuk LK / fitur
git checkout -b feat/initial-eda

# 3. Kerjakan, commit kecil-kecil
git add notebooks/01_initial_eda.ipynb
git commit -m "feat(eda): initial NASA FIRMS exploration on Riau Aug 2024"

# 4. Push
git push -u origin feat/initial-eda

# 5. Buka PR di GitHub UI:
#    base: main  ←  compare: feat/initial-eda
#    Self-review: pastikan diff bersih, tidak ada secret leak
#    Tunggu CI pass, lalu Merge (Squash & merge direkomendasikan untuk linear history)

# 6. Setelah merge, hapus branch lokal
git checkout main
git pull
git branch -d feat/initial-eda
```

### 1.4 Commit Message Convention

Mengikuti [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body opsional>

<footer opsional, untuk BREAKING CHANGE atau referensi issue>
```

| Type | Kapan dipakai |
|---|---|
| `feat` | Fitur baru |
| `fix` | Bug fix |
| `docs` | Dokumentasi saja |
| `style` | Formatting, tidak ubah logika |
| `refactor` | Refactor tanpa ubah behavior |
| `test` | Tambah/ubah test |
| `chore` | Build, deps, config |
| `perf` | Performance improvement |
| `ci` | GitHub Actions config |

Contoh: `feat(data): add NASA FIRMS fetcher with retry`

---

## 2. Pull Request Checklist

Setiap PR (termasuk self-review) harus mencentang:

- [ ] **Tidak ada secret/API key** ter-commit (cek `git diff --staged | grep -iE "key|token|password|secret"`).
- [ ] `.env`, `mlruns/`, `data/raw/*` masih ada di `.gitignore`.
- [ ] Dependency baru ditambah ke `requirements.txt` dengan **versi pinned**.
- [ ] Lulus linting lokal (`black .`, `flake8 src/ tests/`, `isort src/ tests/`).
- [ ] Lulus unit test (`pytest -q`).
- [ ] Notebook output sudah di-strip (`nbstripout` aktif via pre-commit).
- [ ] Untuk fitur API call: ada **timeout** dan **retry/backoff**.
- [ ] Untuk fitur baca file/network: pakai **context manager** (`with`) — tidak boleh leak file descriptor.

---

## 3. Code Quality Guidelines

### 3.1 Security Checklist (wajib)

| Risiko | Mitigasi |
|---|---|
| API key leak ke Git | `.env` di `.gitignore`; gunakan GitHub Actions Secrets di CI |
| API key di log | Jangan `print()` env vars; gunakan structured logging dengan field allow-list |
| SSRF dari URL eksternal | Validasi URL ke domain allow-list (`firms.modaps.eosdis.nasa.gov`, `api.open-meteo.com`, `api.bmkg.go.id`, `api.openaq.org`) |
| Path traversal saat read/write | Gunakan `pathlib.Path` + `.resolve()` + cek prefix; jangan concat string mentah |
| Pickle deserialization | Hindari `pickle.load` dari source tidak terpercaya; pakai `joblib` dengan integrity check (mis. SHA256) |
| Dependency CVE | Audit periodik: `pip-audit` di CI |

### 3.2 Memory/Resource Leak Prevention

```python
# ❌ JANGAN
f = open("data.csv")
df = pd.read_csv(f)   # f tidak pernah di-close kalau exception

# ✅ LAKUKAN
with open("data.csv") as f:
    df = pd.read_csv(f)
```

```python
# ❌ JANGAN — session global yang tidak pernah di-close
SESSION = requests.Session()

def fetch(url):
    return SESSION.get(url).json()  # leak di long-running process

# ✅ LAKUKAN — context manager per fetch batch
def fetch_batch(urls: list[str]) -> list[dict]:
    with requests.Session() as session:
        session.headers.update({"User-Agent": "FireGuard/1.0"})
        return [session.get(u, timeout=30).json() for u in urls]
```

```python
# ❌ JANGAN — leak matplotlib figures di loop
for province in provinces:
    fig = plt.figure()
    # ... plot ...
    plt.savefig(f"{province}.png")
    # fig tidak di-close → memory tumbuh!

# ✅ LAKUKAN
for province in provinces:
    fig, ax = plt.subplots()
    # ... plot ...
    fig.savefig(f"{province}.png")
    plt.close(fig)            # explicit close
```

### 3.3 General Code Standards

* **Type hints wajib** di semua function publik di `src/`.
* **Docstring** format Google style untuk module dan function publik.
* **Logging > print** — gunakan `logging.getLogger(__name__)`.
* **Konfigurasi tidak hard-coded** — selalu lewat `config/*.yaml` atau env var.
* **Test coverage target**: ≥ 70% untuk modul `src/data/` dan `src/models/`.

---

## 4. Validasi Sebelum Merge ke `main`

1. CI hijau (linting + tests).
2. Diff sudah di-review (untuk individual project: self-review minimal 1 cycle, beri komentar di PR).
3. Tidak ada `TODO:` atau `FIXME:` tanpa issue link.
4. Untuk LK: pastikan dokumen LK terkait di-update di branch yang sama.

---

*File ini hidup — update saat ada konvensi baru. Last updated: Mei 2026.*
