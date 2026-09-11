"""Versioned analytical transformations."""

from classical_conditioning.analysis.temporal_profiles import (
    TemporalProfileConfig,
    TemporalProfileResult,
    build_candidate_temporal_profiles,
)

__all__ = [
    "MovementCalibrationConfig",
    "MovementStateResult",
    "MovementSensitivityResult",
    "MetricComparisonConfig",
    "MetricComparisonResult",
    "CandidateRunnerResult",
    "TrialOutcomeConfig",
    "TrialOutcomeResult",
    "TemporalProfileConfig",
    "TemporalProfileResult",
    "TraceReviewResult",
    "build_candidate_movement_state",
    "build_candidate_metric_comparison",
    "run_candidate_development_pipeline",
    "build_candidate_trial_outcomes",
    "build_movement_sensitivity_report",
    "build_candidate_temporal_profiles",
    "build_trace_review",
]


def __getattr__(name: str):
    if name in {
        "MovementCalibrationConfig",
        "MovementStateResult",
        "MovementSensitivityResult",
        "build_candidate_movement_state",
        "build_movement_sensitivity_report",
    }:
        from classical_conditioning.analysis import movement_state

        return getattr(movement_state, name)
    if name in {"TraceReviewResult", "build_trace_review"}:
        from classical_conditioning.analysis import trace_review

        return getattr(trace_review, name)
    if name in {
        "MetricComparisonConfig",
        "MetricComparisonResult",
        "build_candidate_metric_comparison",
    }:
        from classical_conditioning.analysis import metric_comparison

        return getattr(metric_comparison, name)
    if name in {
        "CandidateRunnerResult",
        "run_candidate_development_pipeline",
    }:
        from classical_conditioning.analysis import candidate_runner

        return getattr(candidate_runner, name)
    if name in {
        "TrialOutcomeConfig",
        "TrialOutcomeResult",
        "build_candidate_trial_outcomes",
    }:
        from classical_conditioning.analysis import trial_outcomes

        return getattr(trial_outcomes, name)
    raise AttributeError(name)
