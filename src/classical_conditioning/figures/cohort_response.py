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


# These match the historical selected blocks while declaring their numeric
# membership directly.  Candidate trial outcomes do not otherwise carry a
# five-trial block name.
DEFAULT_SELECTED_BLOCKS = (
    SelectedBlock("Early Pre-train", 5, 9),
    SelectedBlock("Early Test", 65, 69),
    SelectedBlock("Late Test", 90, 94),
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
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", metric_id).strip("-.").lower()
    if not slug:
        raise ConfigurationError("Metric ID must contain a filename-safe character.")
    return slug


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def _load_primary_cohort(project_dir: Path, cohort_id: str) -> FrozenCohortSelection:
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


def _plot_selected_blocks(
    fish: pd.DataFrame,
    group: pd.DataFrame,
    *,
    experiment_name: str | None,
) -> tuple[Any, list[str], dict[str, dict[str, str]]]:
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


def _output_base(
    project_dir: Path,
    analysis_id: str,
    *,
    mode: FigureMode,
    name: str,
) -> Path:
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
    metric_recipe: str = "tail-candidate-corrected-v1",
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
    metric_recipe: str = "tail-candidate-corrected-v1",
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


def build_event_aligned_ratio_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    outcome_id: str = "total-activity",
    metric_recipe: str = "tail-candidate-corrected-v1",
    mode: FigureMode,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render the event-aligned response/baseline descriptive cohort figure."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Cohort ratio figures are Matplotlib-only.")
    _validate_analysis_id(analysis_id)
    project_dir = project_dir.resolve()
    cohort = _load_primary_cohort(project_dir, cohort_id)
    if metric_recipe != cohort.metric_recipe:
        raise ConfigurationError(
            "Requested metric recipe differs from the cohort outcome artifact."
        )
    route: CandidateMetricSource = resolve_candidate_metric_source(
        metric_recipe=metric_recipe
    )
    verified = [
        _verify_temporal_profiles(project_dir, recording_id, route)
        for recording_id in cohort.recording_ids
    ]
    profiles = pd.concat(
        [pd.read_parquet(item.path) for item in verified],
        ignore_index=True,
    )
    observed_recordings = set(profiles["Recording ID"].astype(str))
    expected_recordings = set(cohort.recording_ids)
    if observed_recordings != expected_recordings:
        raise SchemaValidationError(
            "Temporal-profile recordings do not match the primary cohort; "
            f"missing={sorted(expected_recordings - observed_recordings)}, "
            f"unexpected={sorted(observed_recordings - expected_recordings)}."
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
