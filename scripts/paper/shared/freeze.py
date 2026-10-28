"""Standalone freeze gate, usable without importing the scientific libraries."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

from pathlib import Path
import sys

sys.path.insert(0, str(_COMMAND_REPOSITORY/"src/classical_conditioning"))
from figure_freeze import run_cli

if __name__ == "__main__":
    run_cli()
