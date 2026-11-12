"""Compatibility import for the maintained numerical calculation."""
import sys
from classical_conditioning.analysis import signed_bout_heatmap as _implementation
sys.modules[__name__] = _implementation
