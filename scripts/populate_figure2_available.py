"""Compatibility path retained for the immutable element specification."""
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'src')]
if __name__ == '__main__':
    import runpy
    runpy.run_module('scripts.paper.figure_02.reproduction.recover_saved_panels', run_name='__main__')
else:
    from scripts.paper.figure_02.reproduction.recover_saved_panels import *
