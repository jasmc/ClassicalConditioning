"""Movement-state sensitivity benchmark entry points.

The detector implementation remains in :mod:`analysis.movement_state`, where
the normal pipeline uses it.  This module makes the optional parameter sweep
explicitly discoverable as a benchmark rather than a pipeline stage.

Review note: this is an import façade only. It does not change detector
behaviour or cause sensitivity analysis to run during normal pipeline work.
"""

# Re-export benchmark interfaces from their sole implementation module.
from classical_conditioning.analysis.movement_state import (
    MovementSensitivityResult,
    build_movement_sensitivity_report,
    evaluate_smoothing_sensitivity,
)

# Keep benchmark-only names explicit for source/API review.
__all__ = [
    "MovementSensitivityResult",
    "build_movement_sensitivity_report",
    "evaluate_smoothing_sensitivity",
]
