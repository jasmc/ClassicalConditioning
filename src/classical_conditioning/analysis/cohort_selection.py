from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from classical_conditioning.analysis.cohort_outcomes import load_cohort_trial_outcomes
from classical_conditioning.exceptions import ConfigurationError

@dataclass(frozen=True)
class FrozenCohortSelection:
    """Authenticated primary population used by a cohort figure."""

    cohort_id: str
    cohort_hash: str
    experiment_id: str
    metric_recipe: str
    recording_ids: tuple[str, ...]
    condition_by_recording: dict[str, str]
    fish_by_recording: dict[str, str]
    input_artifacts: tuple[dict[str, str], ...]


def load_primary_cohort(project_dir: Path, cohort_id: str) -> FrozenCohortSelection:
    # Load the reviewed cohort once and freeze the identity lookup used by plots.
    outcomes, summary = load_cohort_trial_outcomes(project_dir, cohort_id)
    included = outcomes.loc[
        :, ["experiment_id", "recording_id", "fish_id", "condition_id"]
    ].drop_duplicates()
    if included.empty:
        raise ConfigurationError(f"Cohort {cohort_id!r} has no primary fish.")
    experiments = included["experiment_id"].astype(str).unique()
    if len(experiments) != 1:
        raise ConfigurationError(
            "A response figure must contain one experiment; "
            f"cohort {cohort_id!r} contains {sorted(experiments)}."
        )
    recording_ids = tuple(included["recording_id"].astype(str))
    condition_by_recording = dict(
        zip(
            included["recording_id"].astype(str),
            included["condition_id"].astype(str),
            strict=True,
        )
    )
    fish_by_recording = dict(
        zip(
            included["recording_id"].astype(str),
            included["fish_id"].astype(str),
            strict=True,
        )
    )
    artifacts = tuple(
        {
            "recipe": "cohort-trial-outcomes",
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
        }
        for record in summary["artifacts"].values()
    )
    return FrozenCohortSelection(
        cohort_id=cohort_id,
        cohort_hash=str(summary["cohort_hash"]),
        experiment_id=str(experiments[0]),
        metric_recipe=str(summary["metric_recipe"]),
        recording_ids=recording_ids,
        condition_by_recording=condition_by_recording,
        fish_by_recording=fish_by_recording,
        input_artifacts=artifacts,
    )

