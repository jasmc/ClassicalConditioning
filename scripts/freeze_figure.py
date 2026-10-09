"""Standalone freeze gate, usable without importing the scientific libraries."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src/classical_conditioning"))
from figure_freeze import run_cli

if __name__ == "__main__":
    run_cli()
