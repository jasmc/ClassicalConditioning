"""Versioned preprocessing implementations.

Import concrete recipes lazily so optional analysis dependencies are loaded
only by the stages that need them.
"""

# Lazy imports isolate optional/heavy processing dependencies from basic package
# import and make the public names below the supported preprocessing surface.
from typing import Any

# Public recipe types and stage entry points, without exposing internal helpers.
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
    # Route each requested public name to its owning implementation only when
    # needed; the direct-intake metric writer is an explicit benchmark route.
    if name in {
        "CandidateMetricConfig",
        "CandidateMetricResult",
    }:
        from classical_conditioning.preprocessing import candidate_metric_kernel

        return getattr(candidate_metric_kernel, name)
    if name == "build_candidate_activity_metrics":
        from classical_conditioning.preprocessing.benchmarks import candidate_metrics_from_intake

        return candidate_metrics_from_intake.build_candidate_activity_metrics
    if name == "build_candidate_activity_metrics_from_corrected":
        from classical_conditioning.preprocessing import candidate_metrics_from_corrected_frames

        return getattr(candidate_metrics_from_corrected_frames, name)
    if name in {
        "CorrectedPreprocessConfig",
        "CorrectedPreprocessResult",
        "build_corrected_preprocessing",
    }:
        from classical_conditioning.preprocessing import corrected_frame_preprocessing

        return getattr(corrected_frame_preprocessing, name)
    raise AttributeError(name)
