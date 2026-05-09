"""Filesystem locations for AI Counsel runtime artifacts."""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """User-writable directory for the decision graph DB and the server log.

    Resolution order:
      1. ``$AI_COUNSEL_DATA_HOME`` (used as-is)
      2. ``$XDG_DATA_HOME/ai-counsel``
      3. ``~/.local/share/ai-counsel``

    The directory is created on first call.
    """
    override = os.environ.get("AI_COUNSEL_DATA_HOME")
    if override:
        base = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = (Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share") / "ai-counsel"
    base.mkdir(parents=True, exist_ok=True)
    return base
