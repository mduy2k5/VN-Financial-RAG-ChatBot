"""Resolve ignored, machine-local evaluation inputs without tracking them in Git."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_DATA_ROOT = PROJECT_ROOT / "local_data" / "archive"


def local_data_path(*parts: str) -> Path:
    """Return a path under ``FINANCIAL_LOCAL_DATA_ROOT`` or the ignored archive."""
    root = Path(os.environ.get("FINANCIAL_LOCAL_DATA_ROOT", DEFAULT_LOCAL_DATA_ROOT))
    return root.joinpath(*parts)
