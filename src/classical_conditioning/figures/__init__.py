"""Versioned figure builders and export modes."""

from classical_conditioning.figures.export import (
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.metric_comparison import (
    build_metric_comparison_figure,
)
from classical_conditioning.figures.learning_onset import (
    build_learning_onset_figure,
)
from classical_conditioning.figures.learning_diagnostics import (
    build_learning_diagnostics_figure,
)
from classical_conditioning.figures.cohort_response import (
    build_event_aligned_ratio_figure,
    build_selected_block_ratio_figure,
    build_trial_ratio_figure,
)
from classical_conditioning.figures.temporal_profiles import (
    build_candidate_profile_figure,
)

__all__ = [
    "FigureMode",
    "FigureProvenance",
    "build_candidate_profile_figure",
    "build_event_aligned_ratio_figure",
    "build_metric_comparison_figure",
    "build_learning_onset_figure",
    "build_learning_diagnostics_figure",
    "build_selected_block_ratio_figure",
    "build_trial_ratio_figure",
    "export_matplotlib_figure",
]
