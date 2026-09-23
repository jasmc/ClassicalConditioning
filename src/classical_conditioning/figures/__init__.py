"""Public figure builders and export modes.

Review note: this module is an import-only façade. It contains no figure logic;
each exported name is owned by the focused module named in its import block.
"""

# Core renderer types and file-export helper.
from classical_conditioning.figures.export import (
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
# Cohort comparison visualization.
from classical_conditioning.figures.metric_comparison import (
    build_metric_comparison_figure,
)
# Learning-onset and diagnostic visualizations.
from classical_conditioning.figures.learning_onset import (
    build_learning_onset_figure,
)
# Additional diagnostics used to review onset modelling behaviour.
from classical_conditioning.figures.learning_diagnostics import (
    build_learning_diagnostics_figure,
)
# Ratio-style cohort response figures.
from classical_conditioning.figures.cohort_response import (
    build_block_profile_figure,
    build_catch_profile_figure,
    build_event_aligned_ratio_figure,
    build_selected_block_ratio_figure,
    build_trial_ratio_figure,
)
# Per-recording temporal-profile figure builder.
from classical_conditioning.figures.temporal_profiles import (
    build_candidate_profile_figure,
)
from classical_conditioning.figures.example_traces import build_example_trace_figure

# The supported rendering API; private plotting helpers are intentionally absent.
__all__ = [
    "FigureMode",
    "FigureProvenance",
    "build_candidate_profile_figure",
    "build_example_trace_figure",
    "build_block_profile_figure",
    "build_catch_profile_figure",
    "build_event_aligned_ratio_figure",
    "build_metric_comparison_figure",
    "build_learning_onset_figure",
    "build_learning_diagnostics_figure",
    "build_selected_block_ratio_figure",
    "build_trial_ratio_figure",
    "export_matplotlib_figure",
]
