"""Versioned preprocessing implementations.

Import concrete recipes lazily so optional analysis dependencies are loaded
only by the stages that need them.
"""

from typing import Any

__all__ = [
    "CandidateMetricConfig",
    "CandidateMetricResult",
    "CorrectedPreprocessConfig",
    "CorrectedPreprocessResult",
    "build_candidate_activity_metrics",
    "build_candidate_activity_metrics_from_corrected",
    "build_corrected_preprocessing",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CandidateMetricConfig",
        "CandidateMetricResult",
        "build_candidate_activity_metrics",
    }:
        from classical_conditioning.preprocessing import candidates_v1

        return getattr(candidates_v1, name)
    if name == "build_candidate_activity_metrics_from_corrected":
        from classical_conditioning.preprocessing import candidates_corrected_v1

        return getattr(candidates_corrected_v1, name)
    if name in {
        "CorrectedPreprocessConfig",
        "CorrectedPreprocessResult",
        "build_corrected_preprocessing",
    }:
        from classical_conditioning.preprocessing import corrected_v1

        return getattr(corrected_v1, name)
    raise AttributeError(name)
