"""Persistent storage for analysis windows (slider positions).

Saves to `saved_windows.json` in the project root, keyed by CSV filename.
Works for both NordBord and ForceFrame pages.
"""
import json
import os
from datetime import datetime
from pathlib import Path

# Always resolve relative to this file's directory (project root)
_STORE = Path(__file__).parent / "saved_windows.json"


def _load_all() -> dict:
    if _STORE.exists():
        try:
            return json.loads(_STORE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_all(data: dict) -> None:
    _STORE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_windows(filename: str) -> dict | None:
    """Return saved windows dict for filename, or None if not found."""
    return _load_all().get(filename)


def save_windows(filename: str, windows: dict) -> None:
    """Persist windows dict for filename (overwrites existing entry)."""
    all_data = _load_all()
    all_data[filename] = {
        **windows,
        "_saved_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    _save_all(all_data)


def delete_windows(filename: str) -> None:
    """Remove saved windows for filename."""
    all_data = _load_all()
    all_data.pop(filename, None)
    _save_all(all_data)


def has_windows(filename: str) -> bool:
    return filename in _load_all()
