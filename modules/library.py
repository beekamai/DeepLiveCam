"""Saved source sets ("favourites"): a name for one or more photos of a person.

Stored in ``library.json`` next to the app (kept out of git): each entry
carries the photo paths and, for photos with several faces, which face was
picked.  Applying an entry restores the whole source set in one click.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import modules.globals

LIBRARY_FILE = os.path.join(os.path.dirname(modules.globals.ROOT_DIR), "library.json")


def _read() -> Dict[str, dict]:
    try:
        with open(LIBRARY_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(entries: Dict[str, dict]) -> None:
    with open(LIBRARY_FILE, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)


def names() -> List[str]:
    return sorted(_read())


def save(name: str, paths: List[str], picks: Optional[Dict[str, list]] = None) -> None:
    """Remember ``paths`` (and their face picks) under ``name``."""
    name = name.strip()
    if not name or not paths:
        return
    entries = _read()
    entries[name] = {
        "paths": list(paths),
        "picks": {p: list(pt) for p, pt in (picks or {}).items() if p in paths},
    }
    _write(entries)


def load(name: str) -> Optional[dict]:
    entry = _read().get(name)
    if not entry:
        return None
    paths = [p for p in entry.get("paths", []) if os.path.exists(p)]
    if not paths:
        return None
    return {"paths": paths, "picks": {p: pt for p, pt in entry.get("picks", {}).items() if p in paths}}


def delete(name: str) -> None:
    entries = _read()
    if entries.pop(name, None) is not None:
        _write(entries)
