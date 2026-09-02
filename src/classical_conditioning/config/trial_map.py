"""Canonical experiment trial-map construction and export."""

from __future__ import annotations

from typing import Any

from classical_conditioning.config.domain import Alignment, ExperimentSpec, TrialSpec
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ConfigurationError


def _block_5_ids(trials: tuple[TrialSpec, ...]) -> dict[tuple[str, int], int]:
    """Assign consecutive five-trial blocks within each alignment."""
    mapping: dict[tuple[str, int], int] = {}
    for alignment in Alignment:
        ordered = sorted(
            (trial for trial in trials if trial.alignment is alignment),
            key=lambda trial: trial.trial_number,
        )
        for index, trial in enumerate(ordered):
            mapping[(trial.alignment.value, trial.trial_number)] = (index // 5) + 1
    return mapping


def build_trial_map(experiment: ExperimentSpec) -> dict[str, Any]:
    """Build the immutable expected trial map for one experiment."""
    block_5 = _block_5_ids(experiment.analysis_trials)
    rows: list[dict[str, Any]] = []
    for trial in experiment.analysis_trials:
        trial_id = f"{trial.alignment.value}-{trial.trial_number:04d}"
        rows.append(
            {
                "trial_id": trial_id,
                "trial_number": trial.trial_number,
                "alignment": trial.alignment.value,
                "phase": trial.phase.value,
                "block_5_id": block_5[(trial.alignment.value, trial.trial_number)],
                "block_10_id": trial.block_10_id,
                "block_10_name": trial.block_10_name,
                "catch": trial.catch,
                "event_id": None,
                "expected_us_time_s": None,
            }
        )

    cs_count = sum(1 for row in rows if row["alignment"] == Alignment.CS.value)
    us_count = sum(1 for row in rows if row["alignment"] == Alignment.US.value)
    if not rows:
        raise ConfigurationError("Trial map cannot be empty.")
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise ConfigurationError("Trial map trial_id values must be unique.")

    return {
        "artifact_kind": "experiment-trial-map-v1",
        "experiment_id": experiment.experiment_id,
        "paradigm": experiment.paradigm.value,
        "cs_duration_s": experiment.cs_duration_s,
        "conditioned_response_window_s": {
            "start_s": experiment.conditioned_response_window.start_s,
            "end_s": experiment.conditioned_response_window.end_s,
        },
        "minimum_cs_trials": experiment.minimum_cs_trials,
        "minimum_us_trials": experiment.minimum_us_trials,
        "row_count": len(rows),
        "cs_trial_count": cs_count,
        "us_trial_count": us_count,
        "rows": rows,
    }


def get_experiment_trial_map(experiment_name: str) -> dict[str, Any]:
    return build_trial_map(get_experiment_spec(experiment_name))
