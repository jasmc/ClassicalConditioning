"""Versioned figure builders and export modes."""

from classical_conditioning.figures.export import (
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.legacy_review import (
    LegacyReviewFigureResult,
    build_legacy_preprocessing_review_figure,
)
from classical_conditioning.figures.metric_comparison import (
    build_metric_comparison_figure,
)
from classical_conditioning.figures.temporal_profiles import (
    build_candidate_profile_figure,
)

__all__ = [
    "FigureMode",
    "FigureProvenance",
    "LegacyReviewFigureResult",
    "build_candidate_profile_figure",
    "build_legacy_preprocessing_review_figure",
    "build_metric_comparison_figure",
    "export_matplotlib_figure",
]
