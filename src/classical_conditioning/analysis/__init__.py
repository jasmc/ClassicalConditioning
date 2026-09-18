"""Versioned analytical transformations with a lazy public API.

Review note: temporal profiles are a lightweight common export; the remaining
stages load lazily so importing ``analysis`` does not initialize every route.
"""

# Re-export the temporal-profile stage directly because it is a common endpoint.
from classical_conditioning.analysis.temporal_profiles import (
    TemporalProfileConfig,
    TemporalProfileResult,
    build_candidate_temporal_profiles,
)

# Names intentionally exposed to callers; helper functions remain module-local.
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
    "CohortTrialOutcomesResult",
    "AnalysisEligibilityResult",
    "build_candidate_movement_state",
    "build_candidate_metric_comparison",
    "run_candidate_development_pipeline",
    "build_candidate_trial_outcomes",
    "build_movement_sensitivity_report",
    "build_candidate_temporal_profiles",
    "build_trace_review",
    "build_cohort_trial_outcomes",
    "build_analysis_eligibility_artifact",
]


def __getattr__(name: str):
    # Each branch maps a public symbol group to its sole implementation module.
    # This avoids eager imports and makes ownership visible during source review.
    if name in {
        "CohortTrialOutcomesResult",
        "AnalysisEligibilityResult",
        "build_cohort_trial_outcomes",
        "build_analysis_eligibility_artifact",
    }:
        from classical_conditioning.analysis import cohort_outcomes

        return getattr(cohort_outcomes, name)
    if name in {
        "MovementCalibrationConfig",
        "MovementStateResult",
        "build_candidate_movement_state",
    }:
        from classical_conditioning.analysis import movement_state

        return getattr(movement_state, name)
    if name in {"TraceReviewResult", "build_trace_review"}:
        from classical_conditioning.analysis.review import trace_review

        return getattr(trace_review, name)
    if name in {"MovementSensitivityResult", "build_movement_sensitivity_report"}:
        from classical_conditioning.analysis.benchmarks import movement_sensitivity

        return getattr(movement_sensitivity, name)
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
    # Match normal Python module behaviour for names outside the public surface.
    raise AttributeError(name)
