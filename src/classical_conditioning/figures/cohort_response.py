"""Fish-weighted descriptive cohort figures for response/baseline ratios.

These figures deliberately separate a directly interpretable plotted ratio from
population inference.  A response/baseline ratio is useful for displaying the
within-trial change, but low baselines can make ratios unstable.  Consequently
these builders show fish-level distributions and do not derive p-values from
the ratio.  Condition-aware LME contrasts belong to the learning-onset
analysis once its model and contrast configuration have been approved.

Review note: these are fish-weighted descriptive views over a frozen primary
cohort. Ratio eligibility and zero/missing denominators remain explicit rather
than being imputed or silently interpreted as no response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_conditioning.analysis.cohort_outcomes import (
    load_cohort_trial_outcomes,
)
from classical_conditioning.analysis.metric_comparison import (
    _verify_temporal_profiles,
)
from classical_conditioning.analysis.movement_state import (
    CandidateMetricSource,
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config import Alignment, get_experiment_spec
from classical_conditioning.exceptions import ConfigurationError, SchemaValidationError
from classical_conditioning.figures.export import (
    FigureExportResult,
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.metric_comparison import (
    CONDITION_DISPLAY,
    _condition_colors,
)
from classical_conditioning.figures.theme import (
    DEFAULT_THEME,
    DOUBLE_COLUMN_MM,
    apply_theme,
    mm_to_in,
    style_axes,
)


RATIO_OUTCOMES = {
    "total-activity": {
        "trial_response": "response_total_activity",
        "trial_baseline": "baseline_total_activity",
        "profile_value": "Total activity mean",
        "label": "Total activity",
    },
    "conditional-intensity": {
        "trial_response": "conditional_intensity",
        "trial_baseline": "baseline_conditional_intensity",
        "profile_value": "Conditional intensity mean",
        "label": "Conditional intensity",
    },
}


@dataclass(frozen=True)
class SelectedBlock:
    """One prespecified five-CS-trial block for the descriptive figure."""

    label: str
    start_trial: int
    end_trial: int


# Manuscript block selection is explicit: the final five Pre-Train trials,
# then the first and final five Test trials. It is independent of ten-trial
# acquisition block labels elsewhere in the experiment definition.
DEFAULT_SELECTED_BLOCKS = (
    SelectedBlock("Final Pre-train", 10, 14),
    SelectedBlock("Early Test", 65, 69),
    SelectedBlock("Late Test", 90, 94),
)


@dataclass(frozen=True)
class TemporalTrialGroup:
    """One experiment-resolved set of trials pooled within fish before plotting."""

    label: str
    trial_numbers: tuple[int, ...]
    order: int = 0


def configured_catch_group(experiment_name: str) -> tuple[TemporalTrialGroup, ...]:
    """Resolve the single pooled CS-catch group from the experiment contract."""
    trials = get_experiment_spec(experiment_name).catch_trial_numbers(Alignment.CS)
    if not trials:
        raise ConfigurationError(
            f"Experiment {experiment_name!r} declares no CS catch trials."
        )
    return (TemporalTrialGroup("Catch trials", trials, 0),)


def configured_cs_block_groups(experiment_name: str) -> tuple[TemporalTrialGroup, ...]:
    """Resolve the declared CS ten-trial groups from the experiment contract."""
    return tuple(
        TemporalTrialGroup(label, trials, order)
        for order, (label, trials) in enumerate(
            get_experiment_spec(experiment_name).trial_blocks(Alignment.CS)
        )
    )


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


def _require_ratio_outcome(outcome_id: str) -> dict[str, str]:
    # Restrict ratio plots to outcomes with semantically matched baseline values.
    try:
        return RATIO_OUTCOMES[outcome_id]
    except KeyError as error:
        choices = ", ".join(sorted(RATIO_OUTCOMES))
        raise ConfigurationError(
            f"Response/baseline ratio figures support only: {choices}. "
            "The other candidate outcomes do not yet have a matched "
            "baseline estimand."
        ) from error


def _metric_slug(metric_id: str) -> str:
    # Turn a metric identifier into a safe, stable filename component.
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", metric_id).strip("-.").lower()
    if not slug:
        raise ConfigurationError("Metric ID must contain a filename-safe character.")
    return slug


def _validate_analysis_id(analysis_id: str) -> None:
    # Reject non-portable IDs before they become output-directory components.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def _load_primary_cohort(project_dir: Path, cohort_id: str) -> FrozenCohortSelection:
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


def _validate_outcome_cohort_identity(
    outcomes: pd.DataFrame,
    cohort: FrozenCohortSelection,
) -> None:
    # Verify all table identities still match the cohort selected for the figure.
    required = {"experiment_id", "recording_id", "fish_id", "condition_id"}
    missing = required.difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(
            f"Trial outcomes are missing cohort identity columns: {sorted(missing)}"
        )
    identities = outcomes.loc[:, sorted(required)].drop_duplicates()
    if identities["recording_id"].astype(str).duplicated().any():
        raise SchemaValidationError(
            "A recording has inconsistent experiment, fish, or condition identity."
        )
    observed = set(identities["recording_id"].astype(str))
    expected = set(cohort.recording_ids)
    if observed != expected:
        raise SchemaValidationError(
            "Trial-outcome recordings do not match the primary cohort; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}."
        )
    for row in identities.itertuples(index=False):
        recording_id = str(row.recording_id)
        if (
            str(row.experiment_id) != cohort.experiment_id
            or str(row.fish_id) != cohort.fish_by_recording[recording_id]
            or str(row.condition_id) != cohort.condition_by_recording[recording_id]
        ):
            raise SchemaValidationError(
                f"Trial-outcome identity disagrees with cohort for {recording_id!r}."
            )


def _safe_positive_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    # Divide only finite values with a positive baseline, marking others missing.
    values = numerator.to_numpy(dtype=float)
    baseline = denominator.to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(baseline) & (baseline > 0)
    ratio = np.full(values.shape, np.nan, dtype=float)
    np.divide(values, baseline, out=ratio, where=valid)
    return ratio


def _trial_ratio_rows(
    outcomes: pd.DataFrame,
    *,
    metric_id: str,
    outcome_id: str,
    alignment: str,
) -> pd.DataFrame:
    # Convert eligible trial outcomes into one response/baseline row per trial.
    fields = _require_ratio_outcome(outcome_id)
    if alignment not in {"CS", "US"}:
        raise ConfigurationError("Ratio figure alignment must be CS or US.")
    required = {
        "fish_id",
        "condition_id",
        "alignment",
        "trial_number",
        "metric_id",
        fields["trial_response"],
        fields["trial_baseline"],
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(
            f"Trial outcomes are missing columns: {sorted(missing)}"
        )
    selected = outcomes.loc[
        (outcomes["alignment"].astype(str) == alignment)
        & (outcomes["metric_id"].astype(str) == metric_id)
    ].copy()
    if selected.empty:
        raise ConfigurationError(
            f"No {alignment} trial outcomes for metric {metric_id!r}."
        )
    selected["Response / baseline"] = _safe_positive_ratio(
        selected[fields["trial_response"]], selected[fields["trial_baseline"]]
    )
    return selected


def summarize_selected_block_ratios(
    outcomes: pd.DataFrame,
    *,
    metric_id: str,
    outcome_id: str = "total-activity",
    alignment: str = "CS",
    selected_blocks: tuple[SelectedBlock, ...] = DEFAULT_SELECTED_BLOCKS,
    min_trials_per_fish_block: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize trial ratios as fish medians, then cohort medians and IQRs.

    Each fish has equal weight in the condition-level result.  The returned
    frames are, respectively, one row per fish/block and one row per
    condition/block.  This is descriptive aggregation, not a statistical test.
    """
    if min_trials_per_fish_block < 1:
        raise ConfigurationError("Minimum trials per fish/block must be positive.")
    if not selected_blocks:
        raise ConfigurationError("At least one selected block is required.")
    labels = [block.label for block in selected_blocks]
    if len(labels) != len(set(labels)):
        raise ConfigurationError("Selected-block labels must be unique.")
    selected = _trial_ratio_rows(
        outcomes,
        metric_id=metric_id,
        outcome_id=outcome_id,
        alignment=alignment,
    )
    frames: list[pd.DataFrame] = []
    for block in selected_blocks:
        if block.start_trial > block.end_trial:
            raise ConfigurationError(f"Invalid selected block {block.label!r}.")
        block_rows = selected.loc[
            selected["trial_number"].between(block.start_trial, block.end_trial)
        ].copy()
        block_rows["Selected block"] = block.label
        block_rows["Selected block order"] = len(frames)
        frames.append(block_rows)
    trial_ratios = pd.concat(frames, ignore_index=True)
    observed = (
        trial_ratios.groupby(
            ["fish_id", "condition_id", "Selected block", "Selected block order"],
            observed=True,
            sort=False,
        )["Response / baseline"]
        .agg(
            [
                ("Fish median response / baseline", "median"),
                ("Contributing trials", "count"),
            ]
        )
        .reset_index()
    )
    fish_identities = selected.loc[:, ["fish_id", "condition_id"]].drop_duplicates()
    block_identities = pd.DataFrame(
        {
            "Selected block": labels,
            "Selected block order": range(len(selected_blocks)),
        }
    )
    fish = fish_identities.merge(block_identities, how="cross").merge(
        observed,
        on=["fish_id", "condition_id", "Selected block", "Selected block order"],
        how="left",
        validate="one_to_one",
    )
    fish["Contributing trials"] = fish["Contributing trials"].fillna(0).astype(int)
    fish["Eligible"] = fish["Contributing trials"] >= min_trials_per_fish_block
    fish["Ineligible reason"] = np.where(
        fish["Eligible"], "", "fewer_than_minimum_finite_trial_ratios"
    )
    eligible_fish = fish.loc[fish["Eligible"]].copy()
    group = (
        eligible_fish.groupby(
            ["condition_id", "Selected block", "Selected block order"],
            observed=True,
            sort=False,
        )["Fish median response / baseline"]
        .agg(
            [
                ("Cohort median response / baseline", "median"),
                (
                    "Cohort Q25 response / baseline",
                    lambda values: values.quantile(0.25),
                ),
                (
                    "Cohort Q75 response / baseline",
                    lambda values: values.quantile(0.75),
                ),
                ("Fish count", "size"),
            ]
        )
        .reset_index()
    )
    return fish, group


def summarize_trial_ratios(
    outcomes: pd.DataFrame,
    *,
    metric_id: str,
    outcome_id: str = "total-activity",
    alignment: str = "CS",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize a trial-number trajectory with equal fish weight."""
    selected = _trial_ratio_rows(
        outcomes,
        metric_id=metric_id,
        outcome_id=outcome_id,
        alignment=alignment,
    )
    fish = (
        selected.dropna(subset=["Response / baseline"])
        .groupby(
            ["fish_id", "condition_id", "trial_number"],
            observed=True,
            sort=False,
        )["Response / baseline"]
        .agg(
            [
                ("Fish median response / baseline", "median"),
                ("Contributing rows", "size"),
            ]
        )
        .reset_index()
    )
    if fish.empty:
        raise ConfigurationError("No finite trial response/baseline ratios.")
    group = (
        fish.groupby(["condition_id", "trial_number"], observed=True, sort=False)[
            "Fish median response / baseline"
        ]
        .agg(
            [
                ("Cohort median response / baseline", "median"),
                (
                    "Cohort Q25 response / baseline",
                    lambda values: values.quantile(0.25),
                ),
                (
                    "Cohort Q75 response / baseline",
                    lambda values: values.quantile(0.75),
                ),
                ("Fish count", "size"),
            ]
        )
        .reset_index()
    )
    return fish, group


def summarize_event_aligned_ratios(
    profiles: pd.DataFrame,
    *,
    metric_id: str,
    outcome_id: str = "total-activity",
    alignment: str = "CS",
    condition_by_recording: dict[str, str],
    fish_by_recording: dict[str, str] | None = None,
    baseline_window_s: tuple[float, float] = (-15.0, 0.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build equal-fish event-aligned ratio trajectories and their IQRs."""
    fields = _require_ratio_outcome(outcome_id)
    if alignment not in {"CS", "US"}:
        raise ConfigurationError("Ratio figure alignment must be CS or US.")
    if (
        len(baseline_window_s) != 2
        or not np.isfinite(baseline_window_s).all()
        or baseline_window_s[0] >= baseline_window_s[1]
    ):
        raise ConfigurationError("Baseline window must be a finite increasing pair.")
    required = {
        "Recording ID",
        "Trial type",
        "Trial number",
        "Time bin center (s)",
        "Metric ID",
        fields["profile_value"],
    }
    missing = required.difference(profiles.columns)
    if missing:
        raise SchemaValidationError(
            f"Temporal profiles are missing columns: {sorted(missing)}"
        )
    selected = profiles.loc[
        (profiles["Trial type"].astype(str) == alignment)
        & (profiles["Metric ID"].astype(str) == metric_id)
    ].copy()
    if selected.empty:
        raise ConfigurationError(
            f"No {alignment} temporal profiles for metric {metric_id!r}."
        )
    selected["condition_id"] = selected["Recording ID"].astype(str).map(
        condition_by_recording
    )
    if selected["condition_id"].isna().any():
        unknown = sorted(
            selected.loc[selected["condition_id"].isna(), "Recording ID"]
            .astype(str)
            .unique()
        )
        raise ConfigurationError(f"No condition identity for recordings: {unknown}")
    fish_by_recording = fish_by_recording or {
        recording_id: recording_id for recording_id in condition_by_recording
    }
    selected["fish_id"] = selected["Recording ID"].astype(str).map(
        fish_by_recording
    )
    if selected["fish_id"].isna().any():
        unknown = sorted(
            selected.loc[selected["fish_id"].isna(), "Recording ID"]
            .astype(str)
            .unique()
        )
        raise ConfigurationError(f"No fish identity for recordings: {unknown}")
    time = selected["Time bin center (s)"].to_numpy(dtype=float)
    baseline = selected.loc[
        (time >= baseline_window_s[0]) & (time < baseline_window_s[1])
    ]
    baseline_means = (
        baseline.groupby(["Recording ID", "Trial number"], observed=True)[
            fields["profile_value"]
        ]
        .mean()
        .rename("Trial baseline")
        .reset_index()
    )
    selected = selected.merge(
        baseline_means,
        on=["Recording ID", "Trial number"],
        how="left",
        validate="many_to_one",
    )
    selected["Response / baseline"] = _safe_positive_ratio(
        selected[fields["profile_value"]], selected["Trial baseline"]
    )
    fish = (
        selected.dropna(subset=["Response / baseline"])
        .groupby(
            ["Recording ID", "fish_id", "condition_id", "Time bin center (s)"],
            observed=True,
            sort=False,
        )["Response / baseline"]
        .agg(
            [
                ("Fish median response / baseline", "median"),
                ("Contributing trials", "size"),
            ]
        )
        .reset_index()
    )
    group = (
        fish.groupby(
            ["condition_id", "Time bin center (s)"],
            observed=True,
            sort=False,
        )["Fish median response / baseline"]
        .agg(
            [
                ("Cohort median response / baseline", "median"),
                (
                    "Cohort Q25 response / baseline",
                    lambda values: values.quantile(0.25),
                ),
                (
                    "Cohort Q75 response / baseline",
                    lambda values: values.quantile(0.75),
                ),
                ("Fish count", "size"),
            ]
        )
        .reset_index()
    )
    return fish, group


def summarize_scaled_activity_trial_groups(
    profiles: pd.DataFrame,
    *,
    metric_id: str,
    trial_groups: tuple[TemporalTrialGroup, ...],
    condition_by_recording: dict[str, str],
    fish_by_recording: dict[str, str],
    minimum_coverage: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool configured trials within fish, then summarize fish equally.

    Values below the existing temporal-profile coverage threshold are masked,
    never converted to zero. The returned tables retain contributing trial and
    fish counts so catch/block plots expose their changing support.
    """
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ConfigurationError("Minimum temporal-profile coverage must be 0-1.")
    if not trial_groups:
        raise ConfigurationError("At least one temporal trial group is required.")
    labels = [group.label for group in trial_groups]
    if len(labels) != len(set(labels)):
        raise ConfigurationError("Temporal trial-group labels must be unique.")
    required = {
        "Recording ID",
        "Trial type",
        "Trial number",
        "Time bin center (s)",
        "Metric ID",
        "Scaled total activity",
        "Valid expected fraction",
    }
    missing = required.difference(profiles.columns)
    if missing:
        raise SchemaValidationError(
            f"Temporal profiles are missing columns: {sorted(missing)}"
        )
    selected = profiles.loc[
        (profiles["Trial type"].astype(str) == "CS")
        & (profiles["Metric ID"].astype(str) == metric_id)
    ].copy()
    if selected.empty:
        raise ConfigurationError(f"No CS temporal profiles for metric {metric_id!r}.")
    selected["condition_id"] = selected["Recording ID"].astype(str).map(
        condition_by_recording
    )
    selected["fish_id"] = selected["Recording ID"].astype(str).map(
        fish_by_recording
    )
    if selected[["condition_id", "fish_id"]].isna().any().any():
        unknown = sorted(
            selected.loc[
                selected[["condition_id", "fish_id"]].isna().any(axis=1),
                "Recording ID",
            ]
            .astype(str)
            .unique()
        )
        raise ConfigurationError(
            f"No frozen cohort identity for temporal-profile recordings: {unknown}"
        )

    trial_to_group: dict[int, TemporalTrialGroup] = {}
    for trial_group in trial_groups:
        if not trial_group.trial_numbers:
            raise ConfigurationError(
                f"Temporal trial group {trial_group.label!r} is empty."
            )
        for trial_number in trial_group.trial_numbers:
            if trial_number in trial_to_group:
                raise ConfigurationError(
                    f"CS trial {trial_number} belongs to more than one profile group."
                )
            trial_to_group[int(trial_number)] = trial_group
    selected["Profile group"] = selected["Trial number"].map(
        {trial: group.label for trial, group in trial_to_group.items()}
    )
    selected["Profile group order"] = selected["Trial number"].map(
        {trial: group.order for trial, group in trial_to_group.items()}
    )
    selected = selected.loc[selected["Profile group"].notna()].copy()
    if selected.empty:
        raise ConfigurationError("Configured profile groups have no temporal rows.")
    covered = (
        pd.to_numeric(selected["Valid expected fraction"], errors="coerce")
        >= minimum_coverage
    )
    finite = np.isfinite(
        pd.to_numeric(selected["Scaled total activity"], errors="coerce")
    )
    selected["Covered scaled total activity"] = selected[
        "Scaled total activity"
    ].where(covered & finite)

    valid = selected.dropna(subset=["Covered scaled total activity"])
    if valid.empty:
        raise ConfigurationError(
            "No scaled temporal-profile values pass the coverage threshold."
        )
    fish = (
        valid.groupby(
            [
                "Recording ID",
                "fish_id",
                "condition_id",
                "Profile group",
                "Profile group order",
                "Time bin center (s)",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            **{
                "Fish median scaled total activity": (
                    "Covered scaled total activity",
                    "median",
                ),
                "Contributing trials": ("Trial number", "nunique"),
            }
        )
        .reset_index()
    )
    group = (
        fish.groupby(
            [
                "condition_id",
                "Profile group",
                "Profile group order",
                "Time bin center (s)",
            ],
            observed=True,
            sort=False,
        )["Fish median scaled total activity"]
        .agg(
            **{
                "Cohort median scaled total activity": "median",
                "Cohort Q25 scaled total activity": lambda values: values.quantile(0.25),
                "Cohort Q75 scaled total activity": lambda values: values.quantile(0.75),
                "Fish count": "size",
            }
        )
        .reset_index()
    )
    return fish, group


def _plot_selected_blocks(
    fish: pd.DataFrame,
    group: pd.DataFrame,
    *,
    experiment_name: str | None,
) -> tuple[Any, list[str], dict[str, dict[str, str]]]:
    # Plot prespecified block summaries with fish points and cohort IQR evidence.
    eligible_fish = fish.loc[fish["Eligible"]].copy()
    if eligible_fish.empty:
        raise ConfigurationError("No fish meet the selected-block ratio coverage rule.")
    colors = _condition_colors(experiment_name)
    apply_theme(DEFAULT_THEME)
    width, _ = mm_to_in(DOUBLE_COLUMN_MM, 90.0)
    figure, axis = plt.subplots(figsize=(width, 90.0 / 25.4), layout="constrained")
    axis.axhline(1.0, color="0.6", linewidth=0.6, zorder=1)
    conditions = list(dict.fromkeys(eligible_fish["condition_id"].astype(str)))
    blocks = (
        eligible_fish.loc[:, ["Selected block", "Selected block order"]]
        .drop_duplicates()
        .sort_values("Selected block order")
    )
    offsets = (
        np.linspace(-0.18, 0.18, len(conditions))
        if len(conditions) > 1
        else [0.0]
    )
    rng = np.random.default_rng(0)
    mappings: dict[str, dict[str, str]] = {}
    for offset, condition in zip(offsets, conditions):
        condition_fish = eligible_fish.loc[
            eligible_fish["condition_id"].astype(str) == condition
        ]
        condition_group = group.loc[group["condition_id"].astype(str) == condition]
        xs = condition_group["Selected block order"].to_numpy(dtype=float) + offset
        medians = condition_group["Cohort median response / baseline"].to_numpy(
            dtype=float
        )
        lower = medians - condition_group[
            "Cohort Q25 response / baseline"
        ].to_numpy(dtype=float)
        upper = (
            condition_group["Cohort Q75 response / baseline"].to_numpy(dtype=float)
            - medians
        )
        error = axis.errorbar(
            xs,
            medians,
            yerr=np.vstack([lower, upper]),
            fmt="o-",
            color=colors.get(condition, "0.2"),
            markersize=4,
            capsize=2,
            zorder=4,
            label=_condition_label_with_count(condition, condition_group),
        )
        line_id = f"cohort-median__{condition}"
        error.lines[0].set_gid(line_id)
        mappings[line_id] = {
            "condition": condition,
            "value_field": "Cohort median response / baseline",
            "spread": "fish_IQR",
            "aggregation": "equal_fish_median",
        }
        for _, block in blocks.iterrows():
            values = condition_fish.loc[
                condition_fish["Selected block order"] == block["Selected block order"],
                "Fish median response / baseline",
            ].to_numpy(dtype=float)
            if values.size:
                points = axis.scatter(
                    np.full(values.size, float(block["Selected block order"]) + offset)
                    + rng.uniform(-0.035, 0.035, values.size),
                    values,
                    s=16,
                    color=colors.get(condition, "0.2"),
                    edgecolors="0.15",
                    linewidths=0.25,
                    alpha=0.72,
                    zorder=3,
                )
                point_id = (
                    f"fish-points__{condition}__"
                    f"{int(block['Selected block order'])}"
                )
                points.set_gid(point_id)
                mappings[point_id] = {
                    "condition": condition,
                    "block": str(block["Selected block"]),
                    "value_field": "Fish median response / baseline",
                }
    axis.set_xticks(blocks["Selected block order"])
    axis.set_xticklabels(blocks["Selected block"])
    style_axes(
        axis,
        theme=DEFAULT_THEME,
        xlabel="Prespecified five-trial block",
        ylabel="Activity response / pre-CS baseline",
    )
    axis.set_title(
        "Selected blocks — fish medians and cohort median [IQR]\n"
        "Lower ratio = stronger motor suppression"
    )
    axis.legend(loc="best")
    return figure, ["A"], mappings


def _condition_label_with_count(condition: str, summary: pd.DataFrame) -> str:
    # Display a condition name with its available fish count or count range.
    counts = summary["Fish count"].dropna().astype(int)
    display = CONDITION_DISPLAY.get(condition, condition)
    if counts.empty:
        return f"{display} (n=0)"
    low = int(counts.min())
    high = int(counts.max())
    return f"{display} (n={low})" if low == high else f"{display} (n={low}–{high})"


def _plot_trial_ratios(
    fish: pd.DataFrame,
    group: pd.DataFrame,
    *,
    experiment_name: str | None,
) -> tuple[Any, list[str], dict[str, dict[str, str]]]:
    # Plot every fish trajectory beneath an equal-fish cohort median and IQR.
    colors = _condition_colors(experiment_name)
    apply_theme(DEFAULT_THEME)
    width, _ = mm_to_in(DOUBLE_COLUMN_MM, 90.0)
    figure, axis = plt.subplots(figsize=(width, 90.0 / 25.4), layout="constrained")
    axis.axhline(1.0, color="0.6", linewidth=0.6, zorder=1)
    mappings: dict[str, dict[str, str]] = {}
    for condition in dict.fromkeys(fish["condition_id"].astype(str)):
        color = colors.get(condition, "0.2")
        condition_fish = fish.loc[fish["condition_id"].astype(str) == condition]
        for fish_id, frame in condition_fish.groupby("fish_id", observed=True):
            frame = frame.sort_values("trial_number")
            trace = axis.plot(
                frame["trial_number"],
                frame["Fish median response / baseline"],
                color=color,
                alpha=0.12,
                linewidth=0.45,
                zorder=2,
            )[0]
            trace_id = f"fish-trajectory__{condition}__{fish_id}"
            trace.set_gid(trace_id)
            mappings[trace_id] = {
                "condition": condition,
                "fish_id": str(fish_id),
                "value_field": "Fish median response / baseline",
            }
        summary = group.loc[group["condition_id"].astype(str) == condition].sort_values(
            "trial_number"
        )
        x = summary["trial_number"].to_numpy(dtype=float)
        median = summary["Cohort median response / baseline"].to_numpy(dtype=float)
        q25 = summary["Cohort Q25 response / baseline"].to_numpy(dtype=float)
        q75 = summary["Cohort Q75 response / baseline"].to_numpy(dtype=float)
        ribbon = axis.fill_between(x, q25, q75, color=color, alpha=0.18, zorder=3)
        ribbon_id = f"cohort-iqr__{condition}"
        ribbon.set_gid(ribbon_id)
        mappings[ribbon_id] = {
            "condition": condition,
            "value_field": "Cohort response / baseline IQR",
            "aggregation": "equal_fish_median",
        }
        line = axis.plot(
            x,
            median,
            color=color,
            linewidth=1.2,
            zorder=4,
            label=_condition_label_with_count(condition, summary),
        )[0]
        line_id = f"cohort-median__{condition}"
        line.set_gid(line_id)
        mappings[line_id] = {
            "condition": condition,
            "value_field": "Cohort median response / baseline",
            "aggregation": "equal_fish_median",
        }
    style_axes(
        axis,
        theme=DEFAULT_THEME,
        xlabel="CS trial number",
        ylabel="Activity response / pre-CS baseline",
    )
    axis.set_title(
        "Trial trajectory — fish traces and cohort median [IQR]\n"
        "Lower ratio = stronger motor suppression"
    )
    axis.legend(loc="best")
    return figure, ["A"], mappings


def _plot_event_aligned(
    fish: pd.DataFrame,
    group: pd.DataFrame,
    *,
    experiment_name: str | None,
) -> tuple[Any, list[str], dict[str, dict[str, str]]]:
    # Plot event-relative fish and cohort ratios, retaining the onset reference.
    if fish.empty:
        raise ConfigurationError("No finite event-aligned response/baseline ratios.")
    colors = _condition_colors(experiment_name)
    apply_theme(DEFAULT_THEME)
    width, _ = mm_to_in(DOUBLE_COLUMN_MM, 90.0)
    figure, axis = plt.subplots(figsize=(width, 90.0 / 25.4), layout="constrained")
    axis.axhline(1.0, color="0.6", linewidth=0.6, zorder=1)
    axis.axvline(0.0, color=DEFAULT_THEME.cs_color, linewidth=0.6, zorder=1)
    mappings: dict[str, dict[str, str]] = {}
    for condition in dict.fromkeys(fish["condition_id"].astype(str)):
        color = colors.get(condition, "0.2")
        condition_fish = fish.loc[fish["condition_id"].astype(str) == condition]
        for fish_id, frame in condition_fish.groupby("fish_id", observed=True):
            trace = axis.plot(
                frame["Time bin center (s)"],
                frame["Fish median response / baseline"],
                color=color,
                alpha=0.16,
                linewidth=0.55,
                zorder=2,
            )[0]
            trace_id = f"fish-trajectory__{condition}__{fish_id}"
            trace.set_gid(trace_id)
            mappings[trace_id] = {
                "condition": condition,
                "fish_id": str(fish_id),
                "value_field": "Fish median response / baseline",
            }
        summary = group.loc[group["condition_id"].astype(str) == condition].sort_values(
            "Time bin center (s)"
        )
        x = summary["Time bin center (s)"].to_numpy(dtype=float)
        median = summary["Cohort median response / baseline"].to_numpy(dtype=float)
        q25 = summary["Cohort Q25 response / baseline"].to_numpy(dtype=float)
        q75 = summary["Cohort Q75 response / baseline"].to_numpy(dtype=float)
        ribbon = axis.fill_between(x, q25, q75, color=color, alpha=0.18, zorder=3)
        ribbon_id = f"cohort-iqr__{condition}"
        ribbon.set_gid(ribbon_id)
        mappings[ribbon_id] = {
            "condition": condition,
            "value_field": "Cohort response / baseline IQR",
            "aggregation": "equal_fish_median",
        }
        line = axis.plot(
            x,
            median,
            color=color,
            linewidth=1.2,
            zorder=4,
            label=_condition_label_with_count(condition, summary),
        )[0]
        line_id = f"cohort-median__{condition}"
        line.set_gid(line_id)
        mappings[line_id] = {
            "condition": condition,
            "value_field": "Cohort median response / baseline",
            "aggregation": "equal_fish_median",
        }
    style_axes(
        axis,
        theme=DEFAULT_THEME,
        xlabel="Time from CS onset (s)",
        ylabel="Activity response / pre-CS baseline",
    )
    axis.set_title(
        "Event-aligned ratio — fish medians and cohort median [IQR]\n"
        "Lower ratio = stronger motor suppression"
    )
    axis.legend(loc="best")
    return figure, ["A"], mappings


def _plot_scaled_activity_groups(
    fish: pd.DataFrame,
    group: pd.DataFrame,
    *,
    experiment_name: str,
    title: str,
) -> tuple[Any, list[str], dict[str, dict[str, str]]]:
    """Plot one panel per trial group using cohort medians and fish IQRs."""
    groups = (
        group.loc[:, ["Profile group", "Profile group order"]]
        .drop_duplicates()
        .sort_values("Profile group order")
    )
    panel_count = len(groups)
    columns = min(3, panel_count)
    rows = int(np.ceil(panel_count / columns))
    apply_theme(DEFAULT_THEME)
    width, _ = mm_to_in(DOUBLE_COLUMN_MM, max(58.0, 50.0 * rows))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, max(58.0, 50.0 * rows) / 25.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    colors = _condition_colors(experiment_name)
    experiment = get_experiment_spec(experiment_name)
    mappings: dict[str, dict[str, str]] = {}
    panel_ids: list[str] = []
    for panel_index, (_, profile_group) in enumerate(groups.iterrows()):
        axis = axes.flat[panel_index]
        panel_id = chr(ord("A") + panel_index)
        panel_ids.append(panel_id)
        axis.axvspan(
            0.0,
            experiment.cs_duration_s,
            color=DEFAULT_THEME.cs_color,
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        group_name = str(profile_group["Profile group"])
        panel_summary = group.loc[group["Profile group"] == group_name]
        panel_fish = fish.loc[fish["Profile group"] == group_name]
        for condition in dict.fromkeys(panel_summary["condition_id"].astype(str)):
            summary = panel_summary.loc[
                panel_summary["condition_id"].astype(str) == condition
            ].sort_values("Time bin center (s)")
            color = colors.get(condition, "0.2")
            x = summary["Time bin center (s)"].to_numpy(dtype=float)
            median = summary[
                "Cohort median scaled total activity"
            ].to_numpy(dtype=float)
            q25 = summary["Cohort Q25 scaled total activity"].to_numpy(dtype=float)
            q75 = summary["Cohort Q75 scaled total activity"].to_numpy(dtype=float)
            ribbon = axis.fill_between(x, q25, q75, color=color, alpha=0.18, zorder=2)
            ribbon_id = f"{panel_id}__cohort-iqr__{condition}"
            ribbon.set_gid(ribbon_id)
            mappings[ribbon_id] = {
                "condition": condition,
                "profile_group": group_name,
                "value_field": "fish_IQR_scaled_total_activity",
            }
            line = axis.plot(
                x,
                median,
                color=color,
                linewidth=1.2,
                zorder=3,
                label=_condition_label_with_count(condition, summary),
            )[0]
            line_id = f"{panel_id}__cohort-median__{condition}"
            line.set_gid(line_id)
            mappings[line_id] = {
                "condition": condition,
                "profile_group": group_name,
                "value_field": "Cohort median scaled total activity",
                "aggregation": "trials_within_fish_then_equal_fish_median",
            }
            contributing = panel_fish.loc[
                panel_fish["condition_id"].astype(str) == condition,
                "Contributing trials",
            ]
            mappings[f"{panel_id}__coverage__{condition}"] = {
                "condition": condition,
                "profile_group": group_name,
                "fish_count_min": str(int(summary["Fish count"].min())),
                "fish_count_max": str(int(summary["Fish count"].max())),
                "contributing_trials_median": (
                    f"{float(contributing.median()):g}" if not contributing.empty else "0"
                ),
            }
        style_axes(
            axis,
            theme=DEFAULT_THEME,
            xlabel="Time from CS onset (s)",
            ylabel="Scaled total activity",
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_title(group_name)
        if panel_index == 0:
            axis.legend(loc="best")
    for axis in axes.flat[panel_count:]:
        axis.set_visible(False)
    figure.suptitle(title)
    return figure, panel_ids, mappings


def _load_verified_cohort_profiles(
    project_dir: Path,
    cohort: FrozenCohortSelection,
    *,
    metric_recipe: str,
) -> tuple[pd.DataFrame, tuple[Any, ...]]:
    """Authenticate and concatenate one temporal-profile artifact per cohort fish."""
    if metric_recipe != cohort.metric_recipe:
        raise ConfigurationError(
            "Requested metric recipe differs from the cohort outcome artifact."
        )
    route: CandidateMetricSource = resolve_candidate_metric_source(
        metric_recipe=metric_recipe
    )
    verified = tuple(
        _verify_temporal_profiles(project_dir, recording_id, route)
        for recording_id in cohort.recording_ids
    )
    profiles = pd.concat(
        [pd.read_parquet(item.path) for item in verified],
        ignore_index=True,
    )
    observed = set(profiles["Recording ID"].astype(str))
    expected = set(cohort.recording_ids)
    if observed != expected:
        raise SchemaValidationError(
            "Temporal-profile recordings do not match the primary cohort; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}."
        )
    return profiles, verified


def _output_base(
    project_dir: Path,
    analysis_id: str,
    *,
    mode: FigureMode,
    name: str,
) -> Path:
    # Construct the canonical output base for PNG or publication-vector exports.
    return (
        project_dir
        / "Figures"
        / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
        / "Analyses"
        / analysis_id
        / name
    )


def build_selected_block_ratio_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    outcome_id: str = "total-activity",
    metric_recipe: str = "tail-candidate-corrected",
    mode: FigureMode,
    selected_blocks: tuple[SelectedBlock, ...] = DEFAULT_SELECTED_BLOCKS,
    min_trials_per_fish_block: int = 3,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render the selected-block response/baseline descriptive cohort figure."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Cohort ratio figures are Matplotlib-only.")
    _validate_analysis_id(analysis_id)
    project_dir = project_dir.resolve()
    cohort = _load_primary_cohort(project_dir, cohort_id)
    if metric_recipe != cohort.metric_recipe:
        raise ConfigurationError(
            "Requested metric recipe differs from the cohort outcome artifact."
        )
    outcomes, _ = load_cohort_trial_outcomes(project_dir, cohort_id)
    _validate_outcome_cohort_identity(outcomes, cohort)
    fish, group = summarize_selected_block_ratios(
        outcomes,
        metric_id=metric_id,
        outcome_id=outcome_id,
        selected_blocks=selected_blocks,
        min_trials_per_fish_block=min_trials_per_fish_block,
    )
    figure, panel_ids, mappings = _plot_selected_blocks(
        fish, group, experiment_name=cohort.experiment_id
    )
    name = f"cohort-selected-block-ratio_{_metric_slug(metric_id)}_{outcome_id}"
    source_path = Path(__file__).resolve()
    provenance = FigureProvenance(
        figure_id=(
            f"cohort-selected-block-ratio-{_metric_slug(metric_id)}-"
            f"{outcome_id}"
        ),
        analysis_recipe="cohort-response-ratio-figures",
        source_file=str(source_path),
        source_symbol="build_selected_block_ratio_figure",
        source_hash=sha256_file(source_path),
        reproduction_snippet=(
            "python -m classical_conditioning figure-cohort-selected-block-ratio "
            f"--project-dir \"{project_dir}\" --analysis-id {analysis_id} "
            f"--cohort-id {cohort_id} --metric {metric_id} --outcome {outcome_id} "
            f"--metric-recipe {metric_recipe} --mode {mode.value}"
        ),
        input_artifacts=cohort.input_artifacts,
        cohort_hash=cohort.cohort_hash,
        artist_mappings=mappings,
    )
    try:
        return export_matplotlib_figure(
            figure,
            _output_base(project_dir, analysis_id, mode=mode, name=name),
            provenance,
            mode=mode,
            panel_ids=panel_ids,
            overwrite=overwrite,
        )
    finally:
        plt.close(figure)


def build_trial_ratio_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    outcome_id: str = "total-activity",
    metric_recipe: str = "tail-candidate-corrected",
    mode: FigureMode,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render the descriptive trial-number response/baseline trajectory."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Cohort ratio figures are Matplotlib-only.")
    _validate_analysis_id(analysis_id)
    project_dir = project_dir.resolve()
    cohort = _load_primary_cohort(project_dir, cohort_id)
    if metric_recipe != cohort.metric_recipe:
        raise ConfigurationError(
            "Requested metric recipe differs from the cohort outcome artifact."
        )
    outcomes, _ = load_cohort_trial_outcomes(project_dir, cohort_id)
    _validate_outcome_cohort_identity(outcomes, cohort)
    fish, group = summarize_trial_ratios(
        outcomes,
        metric_id=metric_id,
        outcome_id=outcome_id,
    )
    figure, panel_ids, mappings = _plot_trial_ratios(
        fish, group, experiment_name=cohort.experiment_id
    )
    name = f"cohort-trial-ratio_{_metric_slug(metric_id)}_{outcome_id}"
    source_path = Path(__file__).resolve()
    provenance = FigureProvenance(
        figure_id=f"cohort-trial-ratio-{_metric_slug(metric_id)}-{outcome_id}",
        analysis_recipe="cohort-response-ratio-figures",
        source_file=str(source_path),
        source_symbol="build_trial_ratio_figure",
        source_hash=sha256_file(source_path),
        reproduction_snippet=(
            "python -m classical_conditioning figure-cohort-trial-ratio "
            f"--project-dir \"{project_dir}\" --analysis-id {analysis_id} "
            f"--cohort-id {cohort_id} --metric {metric_id} --outcome {outcome_id} "
            f"--metric-recipe {metric_recipe} --mode {mode.value}"
        ),
        input_artifacts=cohort.input_artifacts,
        cohort_hash=cohort.cohort_hash,
        artist_mappings=mappings,
    )
    try:
        return export_matplotlib_figure(
            figure,
            _output_base(project_dir, analysis_id, mode=mode, name=name),
            provenance,
            mode=mode,
            panel_ids=panel_ids,
            overwrite=overwrite,
        )
    finally:
        plt.close(figure)


def _build_scaled_activity_profile_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    metric_recipe: str,
    mode: FigureMode,
    trial_groups: tuple[TemporalTrialGroup, ...],
    figure_name: str,
    figure_title: str,
    command_name: str,
    source_symbol: str,
    minimum_coverage: float,
    overwrite: bool,
) -> FigureExportResult:
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Cohort temporal-profile figures are Matplotlib-only.")
    _validate_analysis_id(analysis_id)
    project_dir = project_dir.resolve()
    cohort = _load_primary_cohort(project_dir, cohort_id)
    profiles, verified = _load_verified_cohort_profiles(
        project_dir,
        cohort,
        metric_recipe=metric_recipe,
    )
    fish, group = summarize_scaled_activity_trial_groups(
        profiles,
        metric_id=metric_id,
        trial_groups=trial_groups,
        condition_by_recording=cohort.condition_by_recording,
        fish_by_recording=cohort.fish_by_recording,
        minimum_coverage=minimum_coverage,
    )
    figure, panel_ids, mappings = _plot_scaled_activity_groups(
        fish,
        group,
        experiment_name=cohort.experiment_id,
        title=figure_title,
    )
    source_path = Path(__file__).resolve()
    provenance = FigureProvenance(
        figure_id=f"{figure_name.replace('_', '-')}-{_metric_slug(metric_id)}",
        analysis_recipe="cohort-scaled-activity-profile-figures",
        source_file=str(source_path),
        source_symbol=source_symbol,
        source_hash=sha256_file(source_path),
        reproduction_snippet=(
            f"python -m classical_conditioning {command_name} "
            f"--project-dir \"{project_dir}\" --analysis-id {analysis_id} "
            f"--cohort-id {cohort_id} --metric {metric_id} "
            f"--metric-recipe {metric_recipe} --mode {mode.value} "
            f"--minimum-coverage {minimum_coverage:g}"
        ),
        input_artifacts=(
            *cohort.input_artifacts,
            *(
                {"path": str(item.path), "sha256": item.digest}
                for item in verified
            ),
        ),
        cohort_hash=cohort.cohort_hash,
        artist_mappings=mappings,
    )
    try:
        return export_matplotlib_figure(
            figure,
            _output_base(
                project_dir,
                analysis_id,
                mode=mode,
                name=f"{figure_name}_{_metric_slug(metric_id)}",
            ),
            provenance,
            mode=mode,
            panel_ids=panel_ids,
            overwrite=overwrite,
        )
    finally:
        plt.close(figure)


def build_catch_profile_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    metric_recipe: str = "tail-candidate-corrected",
    mode: FigureMode,
    minimum_coverage: float = 0.9,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render the pooled configured-catch scaled-activity cohort profile."""
    cohort = _load_primary_cohort(project_dir.resolve(), cohort_id)
    groups = configured_catch_group(cohort.experiment_id)
    trial_text = ", ".join(str(value) for value in groups[0].trial_numbers)
    return _build_scaled_activity_profile_figure(
        project_dir,
        cohort_id=cohort_id,
        analysis_id=analysis_id,
        metric_id=metric_id,
        metric_recipe=metric_recipe,
        mode=mode,
        trial_groups=groups,
        figure_name="cohort-catch-profile",
        figure_title=f"Configured catch trials ({trial_text})",
        command_name="figure-cohort-catch-profile",
        source_symbol="build_catch_profile_figure",
        minimum_coverage=minimum_coverage,
        overwrite=overwrite,
    )


def build_block_profile_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    metric_recipe: str = "tail-candidate-corrected",
    mode: FigureMode,
    minimum_coverage: float = 0.9,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render all declared CS ten-trial scaled-activity cohort profiles."""
    cohort = _load_primary_cohort(project_dir.resolve(), cohort_id)
    return _build_scaled_activity_profile_figure(
        project_dir,
        cohort_id=cohort_id,
        analysis_id=analysis_id,
        metric_id=metric_id,
        metric_recipe=metric_recipe,
        mode=mode,
        trial_groups=configured_cs_block_groups(cohort.experiment_id),
        figure_name="cohort-block-profile",
        figure_title="Declared ten-trial CS blocks",
        command_name="figure-cohort-block-profile",
        source_symbol="build_block_profile_figure",
        minimum_coverage=minimum_coverage,
        overwrite=overwrite,
    )


def build_event_aligned_ratio_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    outcome_id: str = "total-activity",
    metric_recipe: str = "tail-candidate-corrected",
    mode: FigureMode,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render the event-aligned response/baseline descriptive cohort figure."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Cohort ratio figures are Matplotlib-only.")
    _validate_analysis_id(analysis_id)
    project_dir = project_dir.resolve()
    cohort = _load_primary_cohort(project_dir, cohort_id)
    profiles, verified = _load_verified_cohort_profiles(
        project_dir,
        cohort,
        metric_recipe=metric_recipe,
    )
    fish, group = summarize_event_aligned_ratios(
        profiles,
        metric_id=metric_id,
        outcome_id=outcome_id,
        condition_by_recording=cohort.condition_by_recording,
        fish_by_recording=cohort.fish_by_recording,
    )
    figure, panel_ids, mappings = _plot_event_aligned(
        fish, group, experiment_name=cohort.experiment_id
    )
    name = f"cohort-event-aligned-ratio_{_metric_slug(metric_id)}_{outcome_id}"
    source_path = Path(__file__).resolve()
    provenance = FigureProvenance(
        figure_id=(
            f"cohort-event-aligned-ratio-{_metric_slug(metric_id)}-"
            f"{outcome_id}"
        ),
        analysis_recipe="cohort-response-ratio-figures",
        source_file=str(source_path),
        source_symbol="build_event_aligned_ratio_figure",
        source_hash=sha256_file(source_path),
        reproduction_snippet=(
            "python -m classical_conditioning figure-cohort-event-aligned-ratio "
            f"--project-dir \"{project_dir}\" --analysis-id {analysis_id} "
            f"--cohort-id {cohort_id} --metric {metric_id} --outcome {outcome_id} "
            f"--metric-recipe {metric_recipe} --mode {mode.value}"
        ),
        input_artifacts=(
            *cohort.input_artifacts,
            *(
                {"path": str(item.path), "sha256": item.digest}
                for item in verified
            ),
        ),
        cohort_hash=cohort.cohort_hash,
        artist_mappings=mappings,
    )
    try:
        return export_matplotlib_figure(
            figure,
            _output_base(project_dir, analysis_id, mode=mode, name=name),
            provenance,
            mode=mode,
            panel_ids=panel_ids,
            overwrite=overwrite,
        )
    finally:
        plt.close(figure)
