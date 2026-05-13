"""Resolve path to Airline_Delay_Cause.csv (user filename variants)."""

from __future__ import annotations

import os
from pathlib import Path


def default_csv_path() -> Path:
    """CSV lives next to package modules (same directory as app.py), or sibling filenames."""
    pkg_dir = Path(__file__).resolve().parent
    for name in ("Airline_Delay_Cause.csv", "Airline_Delay_cause.csv"):
        p = pkg_dir / name
        if p.is_file():
            return p
    return pkg_dir / "Airline_Delay_Cause.csv"


def resolved_csv_path() -> Path:
    override = os.environ.get("AIRLINE_DELAY_CSV_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return default_csv_path()
