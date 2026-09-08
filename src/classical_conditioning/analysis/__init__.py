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
    "LegacyLogMedianConfig",
    "LegacyLogMedianResult",
    "LegacyStandardMainConfig",
    "LegacyStandardMainResult",
    "LegacyScaledVigorConfig",
    "LegacyScaledVigorResult",
    "LegacyScaledVigorCohortResult",
    "LegacyNormalizedVigorConfig",
    "LegacyNormalizedVigorResult",
    "LegacyStatisticsConfig",
    "LegacyStatisticsResult",
    "LegacyRunnerResult",
    "MetricComparisonConfig",
    "MetricComparisonResult",
    "CandidateRunnerResult",
    "TrialOutcomeConfig",
    "TrialOutcomeResult",
    "OutcomeComparisonResult",
    "TemporalProfileConfig",
    "TemporalProfileResult",
    "TraceReviewResult",
    "build_candidate_movement_state",
    "build_legacy_logmedian",
    "build_legacy_standard_main",
    "build_legacy_scaled_vigor",
    "build_legacy_scaled_vigor_cohort",
    "build_legacy_normalized_vigor",
    "build_legacy_statistics",
    "run_legacy_analysis_pipeline",
    "build_candidate_metric_comparison",
    "run_candidate_development_pipeline",
    "build_candidate_trial_outcomes",
    "build_legacy_candidate_outcome_comparison",
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
        "LegacyLogMedianConfig",
        "LegacyLogMedianResult",
        "build_legacy_logmedian",
    }:
        from classical_conditioning.analysis import legacy_logmedian

        return getattr(legacy_logmedian, name)
    if name in {
        "LegacyStandardMainConfig",
        "LegacyStandardMainResult",
        "build_legacy_standard_main",
    }:
        from classical_conditioning.analysis import legacy_standard_main

        return getattr(legacy_standard_main, name)
    if name in {
        "LegacyScaledVigorConfig",
        "LegacyScaledVigorResult",
        "LegacyScaledVigorCohortResult",
        "build_legacy_scaled_vigor",
        "build_legacy_scaled_vigor_cohort",
    }:
        from classical_conditioning.analysis import legacy_scaled_vigor

        return getattr(legacy_scaled_vigor, name)
    if name in {
        "LegacyNormalizedVigorConfig",
        "LegacyNormalizedVigorResult",
        "build_legacy_normalized_vigor",
    }:
        from classical_conditioning.analysis import legacy_normalized_vigor

        return getattr(legacy_normalized_vigor, name)
    if name in {
        "LegacyStatisticsConfig",
        "LegacyStatisticsResult",
        "build_legacy_statistics",
    }:
        from classical_conditioning.analysis import legacy_statistics

        return getattr(legacy_statistics, name)
    if name in {"LegacyRunnerResult", "run_legacy_analysis_pipeline"}:
        from classical_conditioning.analysis import legacy_runner

        return getattr(legacy_runner, name)
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
    if name in {
        "OutcomeComparisonResult",
        "build_legacy_candidate_outcome_comparison",
    }:
        from classical_conditioning.analysis import outcome_comparison

        return getattr(outcome_comparison, name)
    raise AttributeError(name)
