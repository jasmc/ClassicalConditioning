"""Verify the historical Figure 1 A-D freeze without changing its artifacts."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src/classical_conditioning"))
from figure_freeze import digest

REPO = Path(__file__).resolve().parents[1]


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
        "New panel/figure freezes must use scripts/freeze_figure.py or "
        "classical-conditioning freeze-figure with a verified candidate manifest."
    )


if __name__ == "__main__":
    main()
