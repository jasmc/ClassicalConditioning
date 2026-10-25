"""Compatibility CLI for the bout-only D/E review; frozen original is archived."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

from pathlib import Path
import argparse
import sys
ROOT = _COMMAND_REPOSITORY
sys.path.insert(0, str(ROOT / "src"))
from classical_conditioning.analysis.review.bout_block_review import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true', help='Retained compatibility option')
    args = parser.parse_args()
    from scripts.paper.figure_02.panels_d_e.reproduction.review_bout_only import main as run
    return run(args.output)


if __name__ == '__main__':
    main()
