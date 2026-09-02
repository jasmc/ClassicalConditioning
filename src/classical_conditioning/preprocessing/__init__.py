"""Versioned preprocessing implementations.

Import concrete recipes from their modules so optional legacy dependencies do
not load when only corrected/candidate stages are used.
"""

from typing import Any

__all__ = [
    "CandidateMetricConfig",
    "CandidateMetricResult",
    "CorrectedPreprocessConfig",
    "CorrectedPreprocessResult",
    "LegacyPreprocessingConfig",
    "LegacyPreprocessingResult",
    "build_candidate_activity_metrics",
    "build_candidate_activity_metrics_from_corrected",
    "build_corrected_preprocessing",
    "preprocess_legacy_recording",
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
    if name in {
        "LegacyPreprocessingConfig",
        "LegacyPreprocessingResult",
        "preprocess_legacy_recording",
    }:
        from classical_conditioning.preprocessing import legacy_v1

        return getattr(legacy_v1, name)
    raise AttributeError(name)
