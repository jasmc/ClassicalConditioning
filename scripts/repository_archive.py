"""Stable compatibility path; maintained implementation is scoped under scripts/."""
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'src')]
if __name__ == '__main__':
    import runpy
    runpy.run_module('scripts.maintenance.archive', run_name='__main__')
else:
    from scripts.maintenance.archive import *
