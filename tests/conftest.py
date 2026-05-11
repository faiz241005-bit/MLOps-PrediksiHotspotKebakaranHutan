"""Pytest fixtures global untuk FireGuard tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Tambah project root ke sys.path supaya `from src.data ... import ...` jalan
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
