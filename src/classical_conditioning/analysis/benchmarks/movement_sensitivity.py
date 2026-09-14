"""Movement-state sensitivity benchmark entry points.

The detector implementation remains in :mod:`analysis.movement_state`, where
the normal pipeline uses it.  This module makes the optional parameter sweep
explicitly discoverable as a benchmark rather than a pipeline stage.
"""

from classical_conditioning.analysis.movement_state import (
    MovementSensitivityResult,
    build_movement_sensitivity_report,
    evaluate_smoothing_sensitivity,
)

__all__ = [
    "MovementSensitivityResult",
    "build_movement_sensitivity_report",
    "evaluate_smoothing_sensitivity",
]
