"""Figure saving utilities.

Centralizes matplotlib figure saving across analysis scripts:
- creates parent folders
- enforces file extension to match format
- sanitizes filenames for Windows

Keep this module lightweight and dependency-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

_WINDOWS_INVALID_CHARS = '<>:"/\\|?*'


def stringify_for_filename(value: object) -> str:
    """Convert common objects (lists/arrays/etc.) into filename-friendly strings."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return "-".join(str(v) for v in value)
    return str(value)


def sanitize_filename_component(name: object) -> str:
    """Sanitize a filename component for Windows filesystems."""
    text = str(name)
    text = "".join("_" if ch in _WINDOWS_INVALID_CHARS else ch for ch in text)
    # Avoid trailing spaces/dots which Windows strips/blocks.
    text = text.strip().rstrip(".")
    # Normalize repeated whitespace.
    text = " ".join(text.split())
    return text if text else "figure"


def ensure_suffix(path: Path, frmt: str) -> Path:
    """Ensure `path` ends with the suffix implied by `frmt` (without leading dot)."""
    safe_frmt = str(frmt).lstrip(".")
    if not safe_frmt:
        return path
    desired = f".{safe_frmt}"
    if path.suffix.lower() != desired.lower():
        return path.with_suffix(desired)
    return path


def save_figure(
    fig,
    path_out: str | Path,
    frmt: str,
    savefig_kw: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Path:
    """Save matplotlib figure to `path_out`.

    - Creates parent folder.
    - Sanitizes filename (stem) for Windows.
    - Enforces suffix to match `frmt`.

    Returns the final Path saved.
    """
    path = Path(path_out)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_frmt = str(frmt).lstrip(".")
    path = ensure_suffix(path, safe_frmt)
    path = path.with_name(f"{sanitize_filename_component(path.stem)}{path.suffix}")

    save_kwargs: dict[str, Any] = dict(savefig_kw or {})
    save_kwargs.update(overrides)

    fig.savefig(str(path), format=safe_frmt, **save_kwargs)
    return path
