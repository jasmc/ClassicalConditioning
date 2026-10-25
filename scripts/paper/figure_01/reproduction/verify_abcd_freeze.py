"""Verify only the existing historical Figure 1 A-D freeze without changing its artifacts."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import json
from pathlib import Path
import sys

sys.path.insert(0, str(_COMMAND_REPOSITORY/"src/classical_conditioning"))
from figure_freeze import digest

REPO = _COMMAND_REPOSITORY


def main():
    layout_path = REPO / "configs/paper-figures/figure1-assembly.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    root = Path(layout["storage_root"])
    target = root / "frozen/2026-10-05-ABCD"
    manifest_path = target / "freeze.json"
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in frozen["panels"]:
            if digest(root / item["source"]) != item["sha256"]:
                raise ValueError("Frozen artwork changed")
        print("Existing freeze verified:", manifest_path)
        return
    raise RuntimeError(
        "This historical script only verifies the existing A-D freeze. "
        "New panel/figure freezes must use scripts/paper/shared/freeze.py or "
        "classical-conditioning freeze-figure with a verified candidate manifest."
    )


if __name__ == "__main__":
    main()
