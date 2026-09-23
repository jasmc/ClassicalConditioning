"""Command-line interface for local analysis operations.

Review note: this module only translates command-line arguments into validated
stage calls. Scientific calculations and artifact authentication remain owned
by the focused modules imported inside each command-dispatch branch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from classical_conditioning.environment import ensure_supported_runtime
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.intake import intake_recording, intake_recordings


def _preprocess_recipe(value: str) -> str:
    """Resolve a user-facing preprocessing mode to its internal recipe ID."""
    # Keep the user-facing mode short while resolving one frozen recipe identity.
    aliases = {
        "corrected": "corrected-preprocess",
        "corrected-preprocess": "corrected-preprocess",
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "preprocessing mode must be 'corrected'."
        ) from error


def _activity_metric_recipe(value: str) -> str:
    """Resolve a user-facing metric source to its internal recipe ID."""
    # Map friendly development/corrected labels to compatible metric recipe IDs.
    aliases = {
        "development": "tail-candidate-development",
        "corrected": "tail-candidate-corrected",
        "tail-candidate-development": "tail-candidate-development",
        "tail-candidate-corrected": "tail-candidate-corrected",
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "metric source must be 'development' or 'corrected'."
        ) from error


def _warn_deprecated_exploratory_inference(command: str) -> None:
    """Make retained historical inference commands visibly non-routine."""
    print(
        f"WARNING: {command} is a deprecated exploratory comparison and does not "
        "estimate the condition-aware learning effect. Use learning-onset for "
        "condition-aware inference.",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    # Declare every supported command and its input contract without executing work.
    parser = argparse.ArgumentParser(prog="classical-conditioning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    intake = subparsers.add_parser(
        "intake",
        help="Convert one immutable acquisition triplet to lossless Parquet.",
    )
    intake.add_argument("--input-dir", type=Path, required=True)
    intake.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Local Paper data directory containing Processed data and Quality checks.",
    )
    intake.add_argument("--chunk-rows", type=int, default=250_000)
    intake.add_argument("--preview-rows", type=int, default=25)
    intake.add_argument(
        "--recording-id",
        help="Select one recording when the input directory contains several triplets.",
    )
    intake.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived intake artifacts; raw files remain read-only.",
    )

    intake_batch = subparsers.add_parser(
        "intake-batch",
        help="Intake every complete triplet, optionally filtered by condition.",
    )
    intake_batch.add_argument("--input-dir", type=Path, required=True)
    intake_batch.add_argument("--project-dir", type=Path, required=True)
    intake_batch.add_argument("--recording-id", action="append")
    intake_batch.add_argument(
        "--keep-condition",
        action="append",
        dest="keep_conditions",
        help="Filename condition token to keep; repeat to keep several (e.g. control).",
    )
    intake_batch.add_argument("--chunk-rows", type=int, default=250_000)
    intake_batch.add_argument("--preview-rows", type=int, default=25)
    intake_batch.add_argument("--overwrite", action="store_true")

    retry_intake = subparsers.add_parser(
        "retry-intake",
        help="Retry one previously failed recording after correcting its source or settings.",
    )
    retry_intake.add_argument("--input-dir", type=Path, required=True)
    retry_intake.add_argument("--project-dir", type=Path, required=True)
    retry_intake.add_argument("--recording-id", required=True)

    inventory = subparsers.add_parser(
        "inventory",
        help="Discover and hash local raw recording triplets without modifying them.",
    )
    inventory.add_argument("--input-dir", type=Path, required=True)
    inventory.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path outside the raw-data tree; omit to print the inventory.",
    )
    inventory.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing inventory output; raw files remain read-only.",
    )
    inventory.add_argument(
        "--skip-hashes",
        action="store_true",
        help="List completeness quickly without calculating source SHA-256 hashes.",
    )
    inventory.add_argument(
        "--inspect-tracking-headers",
        action="store_true",
        help="Read tracking headers only to summarize point-count stability.",
    )

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Prepare corrected frames from intake Parquet.",
    )
    preprocess.add_argument("--project-dir", type=Path, required=True)
    preprocess.add_argument("--recording-id", required=True)
    preprocess.add_argument(
        "--recipe",
        type=_preprocess_recipe,
        default="corrected-preprocess",
        metavar="corrected",
        help="Corrected frame preparation (the only supported preprocessing mode).",
    )
    preprocess.add_argument("--experiment", default="allDelay")
    preprocess.add_argument("--batch-size", type=int, default=250_000)
    preprocess.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser(
        "compare",
        help="Compare two row-aligned versioned Parquet artifacts.",
    )
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--absolute-tolerance", type=float, default=0.0)
    compare.add_argument("--relative-tolerance", type=float, default=0.0)

    activity = subparsers.add_parser(
        "activity-metrics",
        help="Build exploratory whole-tail activity metrics.",
    )
    activity.add_argument("--project-dir", type=Path, required=True)
    activity.add_argument("--recording-id", required=True)
    activity.add_argument(
        "--recipe",
        type=_activity_metric_recipe,
        default="tail-candidate-development",
        metavar="{development,corrected}",
        help=(
            "Metric input lineage: corrected is the normal gap-aware route; "
            "development reads intake artifacts directly for benchmarking."
        ),
    )
    activity.add_argument("--batch-size", type=int, default=250_000)
    activity.add_argument("--overwrite", action="store_true")

    profiles = subparsers.add_parser(
        "temporal-profiles",
        help="Align candidate metrics to protocol events and build time bins.",
    )
    profiles.add_argument("--project-dir", type=Path, required=True)
    profiles.add_argument("--recording-id", required=True)
    profiles.add_argument(
        "--recipe",
        choices=(
            "candidate-temporal-outcomes",
            "candidate-temporal-outcomes-corrected",
        ),
        default="candidate-temporal-outcomes",
    )
    profiles.add_argument("--experiment", default="allDelay")
    profiles.add_argument("--overwrite", action="store_true")

    figure_profiles = subparsers.add_parser(
        "figure-candidate-profiles",
        help="Render candidate temporal profiles from saved panel data.",
    )
    figure_profiles.add_argument("--project-dir", type=Path, required=True)
    figure_profiles.add_argument("--recording-id", required=True)
    figure_profiles.add_argument("--trial-type", choices=("CS", "US"), default="CS")
    figure_profiles.add_argument(
        "--figure",
        choices=(
            "total-activity-raw",
            "total-activity-scaled",
            "signed-log-vigor",
            "conditional-intensity-raw",
            "bout-outcomes",
        ),
        default="total-activity-raw",
        help=(
            "Which figure to render. The activity and signed-vigor figures have one row "
            "per metric; bout-outcomes has one row per detector-dependent "
            "outcome and is metric-free."
        ),
    )
    figure_profiles.add_argument(
        "--mode",
        choices=("publication", "static", "interactive"),
        required=True,
    )
    figure_profiles.add_argument(
        "--recipe",
        choices=(
            "candidate-temporal-outcomes",
            "candidate-temporal-outcomes-corrected",
        ),
        default="candidate-temporal-outcomes",
    )
    figure_profiles.add_argument("--overwrite", action="store_true")

    example_traces = subparsers.add_parser(
        "figure-example-traces",
        help="Plot matched tail-angle and selected-vigor traces for one fish and CS trials.",
    )
    example_traces.add_argument("--project-dir", type=Path, required=True)
    example_traces.add_argument("--recording-id", required=True)
    example_traces.add_argument("--experiment", choices=("allDelay", "all3sTrace", "all10sTrace"), required=True)
    example_traces.add_argument("--trial", type=int, action="append", required=True,
                                help="Global CS trial number; repeat to choose multiple rows.")
    example_traces.add_argument("--metric", choices=(
        "tail_length_weighted_angular_l1",
        "whole_tail_xy_mean_speed_normalized",
        "legacy_distal_angular_speed",
    ), required=True)
    example_traces.add_argument("--tail-point", type=int, default=15)
    example_traces.add_argument("--window-start", type=float, default=-20.0)
    example_traces.add_argument("--window-end", type=float, default=20.0)
    example_traces.add_argument("--mode", choices=("publication", "static"), default="static")
    example_traces.add_argument("--overwrite", action="store_true")

    figure_metric_comparison = subparsers.add_parser(
        "figure-metric-comparison",
        help="Render a cohort candidate metric comparison from saved summaries.",
    )
    figure_metric_comparison.add_argument("--project-dir", type=Path, required=True)
    figure_metric_comparison.add_argument("--analysis-id", required=True)
    figure_metric_comparison.add_argument("--trial-type", choices=("CS", "US"), default="CS")
    figure_metric_comparison.add_argument(
        "--outcome",
        choices=(
            "total-activity",
            "movement-probability",
            "fraction-time-moving",
            "conditional-intensity",
            "bout-rate",
        ),
        default="movement-probability",
    )
    figure_metric_comparison.add_argument(
        "--recipe",
        choices=(
            "candidate-metric-comparison",
            "candidate-metric-comparison-corrected",
        ),
        default="candidate-metric-comparison-corrected",
    )
    figure_metric_comparison.add_argument(
        "--mode",
        choices=("publication", "static"),
        default="static",
    )
    figure_metric_comparison.add_argument("--overwrite", action="store_true")

    figure_selected_block_ratio = subparsers.add_parser(
        "figure-cohort-selected-block-ratio",
        help=(
            "Render fish-weighted response/baseline summaries for Early "
            "Pre-train, Early Test, and Late Test."
        ),
    )
    figure_selected_block_ratio.add_argument("--project-dir", type=Path, required=True)
    figure_selected_block_ratio.add_argument("--analysis-id", required=True)
    figure_selected_block_ratio.add_argument("--cohort-id", required=True)
    figure_selected_block_ratio.add_argument("--metric", required=True)
    figure_selected_block_ratio.add_argument(
        "--outcome",
        choices=("total-activity", "conditional-intensity"),
        default="total-activity",
    )
    figure_selected_block_ratio.add_argument(
        "--metric-recipe",
        choices=("tail-candidate-corrected", "tail-candidate-development"),
        default="tail-candidate-corrected",
    )
    figure_selected_block_ratio.add_argument(
        "--mode", choices=("publication", "static"), default="static"
    )
    figure_selected_block_ratio.add_argument("--overwrite", action="store_true")

    figure_trial_ratio = subparsers.add_parser(
        "figure-cohort-trial-ratio",
        help="Render the fish-weighted response/baseline trajectory by CS trial.",
    )
    figure_trial_ratio.add_argument("--project-dir", type=Path, required=True)
    figure_trial_ratio.add_argument("--analysis-id", required=True)
    figure_trial_ratio.add_argument("--cohort-id", required=True)
    figure_trial_ratio.add_argument("--metric", required=True)
    figure_trial_ratio.add_argument(
        "--outcome",
        choices=("total-activity", "conditional-intensity"),
        default="total-activity",
    )
    figure_trial_ratio.add_argument(
        "--metric-recipe",
        choices=("tail-candidate-corrected", "tail-candidate-development"),
        default="tail-candidate-corrected",
    )
    figure_trial_ratio.add_argument(
        "--mode", choices=("publication", "static"), default="static"
    )
    figure_trial_ratio.add_argument("--overwrite", action="store_true")

    figure_event_aligned_ratio = subparsers.add_parser(
        "figure-cohort-event-aligned-ratio",
        help=(
            "Render fish-weighted event-aligned response/baseline ratio "
            "trajectories."
        ),
    )
    figure_event_aligned_ratio.add_argument("--project-dir", type=Path, required=True)
    figure_event_aligned_ratio.add_argument("--analysis-id", required=True)
    figure_event_aligned_ratio.add_argument("--cohort-id", required=True)
    figure_event_aligned_ratio.add_argument("--metric", required=True)
    figure_event_aligned_ratio.add_argument(
        "--outcome",
        choices=("total-activity", "conditional-intensity"),
        default="total-activity",
    )
    figure_event_aligned_ratio.add_argument(
        "--metric-recipe",
        choices=("tail-candidate-corrected", "tail-candidate-development"),
        default="tail-candidate-corrected",
    )
    figure_event_aligned_ratio.add_argument(
        "--mode", choices=("publication", "static"), default="static"
    )
    figure_event_aligned_ratio.add_argument("--overwrite", action="store_true")

    for command, help_text in (
        (
            "figure-cohort-catch-profile",
            "Render the pooled configured-catch scaled-total-activity profile.",
        ),
        (
            "figure-cohort-block-profile",
            "Render scaled-total-activity profiles for every declared CS block.",
        ),
    ):
        profile_parser = subparsers.add_parser(command, help=help_text)
        profile_parser.add_argument("--project-dir", type=Path, required=True)
        profile_parser.add_argument("--analysis-id", required=True)
        profile_parser.add_argument("--cohort-id", required=True)
        profile_parser.add_argument("--metric", required=True)
        profile_parser.add_argument(
            "--metric-recipe",
            choices=("tail-candidate-corrected", "tail-candidate-development"),
            default="tail-candidate-corrected",
        )
        profile_parser.add_argument(
            "--mode", choices=("publication", "static"), default="static"
        )
        profile_parser.add_argument(
            "--minimum-coverage", type=float, default=0.9
        )
        profile_parser.add_argument("--overwrite", action="store_true")

    movement = subparsers.add_parser(
        "movement-state",
        help="Calibrate exploratory movement state and bouts for candidate metrics.",
    )
    movement.add_argument("--project-dir", type=Path, required=True)
    movement.add_argument("--recording-id", required=True)
    movement.add_argument(
        "--recipe",
        choices=("movement-candidate", "movement-candidate-corrected"),
        default="movement-candidate",
    )
    movement.add_argument("--overwrite", action="store_true")

    sensitivity = subparsers.add_parser(
        "movement-sensitivity",
        help="Compare candidate detector smoothing variants without full duplicate outputs.",
    )
    sensitivity.add_argument("--project-dir", type=Path, required=True)
    sensitivity.add_argument("--recording-id", required=True)
    sensitivity.add_argument("--overwrite", action="store_true")

    trace_review = subparsers.add_parser(
        "trace-review",
        help="Build balanced local trace windows for human detector review.",
    )
    trace_review.add_argument("--project-dir", type=Path, required=True)
    trace_review.add_argument("--recording-id", required=True)
    trace_review.add_argument("--overwrite", action="store_true")

    metric_comparison = subparsers.add_parser(
        "compare-candidate-metrics",
        help="Apply identical descriptive outcomes to all candidate metrics.",
    )
    metric_comparison.add_argument("--project-dir", type=Path, required=True)
    metric_comparison.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Authenticated candidate recording ID; repeat for every recording.",
    )
    metric_comparison.add_argument("--analysis-id", required=True)
    metric_comparison.add_argument(
        "--recipe",
        choices=(
            "candidate-metric-comparison",
            "candidate-metric-comparison-corrected",
        ),
        default="candidate-metric-comparison",
    )
    metric_comparison.add_argument("--experiment", default="allDelay")
    metric_comparison.add_argument("--overwrite", action="store_true")

    candidate_runner = subparsers.add_parser(
        "candidate-runner",
        help="Run the corrected candidate pipeline or an explicitly selected direct-intake benchmark.",
    )
    candidate_runner.add_argument("--project-dir", type=Path, required=True)
    candidate_runner.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Authenticated recording ID; repeat for every development recording.",
    )
    candidate_runner.add_argument("--analysis-id", required=True)
    candidate_runner.add_argument("--experiment", default="allDelay")
    candidate_runner.add_argument("--batch-size", type=int, default=250_000)
    candidate_runner.add_argument(
        "--recipe",
        choices=(
            "candidate-development-runner",
            "candidate-corrected-runner",
        ),
        default="candidate-corrected-runner",
    )
    candidate_runner.add_argument("--overwrite", action="store_true")
    candidate_runner.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stage banners, step flags, and progress bars.",
    )

    trial_outcomes = subparsers.add_parser(
        "candidate-trial-outcomes",
        help="Build exact measured-time trial outcomes for all candidate metrics.",
    )
    trial_outcomes.add_argument("--project-dir", type=Path, required=True)
    trial_outcomes.add_argument("--recording-id", required=True)
    trial_outcomes.add_argument("--experiment", default="allDelay")
    trial_outcomes.add_argument(
        "--recipe",
        choices=(
            "candidate-trial-outcomes",
            "candidate-trial-outcomes-corrected",
        ),
        default="candidate-trial-outcomes",
    )
    trial_outcomes.add_argument("--overwrite", action="store_true")

    freeze_cohort = subparsers.add_parser(
        "freeze-cohort",
        help="Validate and freeze an explicitly reviewed cohort manifest.",
    )
    freeze_cohort.add_argument("--project-dir", type=Path, required=True)
    freeze_cohort.add_argument("--input", type=Path, required=True)
    freeze_cohort.add_argument("--cohort-id", required=True)
    freeze_cohort.add_argument("--policy-id", required=True)
    freeze_cohort.add_argument(
        "--recipe",
        choices=("cohort-manifest",),
        default="cohort-manifest",
    )

    apply_cohort = subparsers.add_parser(
        "apply-cohort",
        help="Filter a table to fish included in a frozen cohort manifest.",
    )
    apply_cohort.add_argument("--project-dir", type=Path, required=True)
    apply_cohort.add_argument(
        "--cohort-id",
        required=True,
        help="Frozen cohort identity under Processed data/Cohorts.",
    )
    apply_cohort.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Parquet or CSV containing experiment_id and fish_id columns.",
    )
    apply_cohort.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination Parquet path for the filtered table.",
    )
    apply_cohort.add_argument(
        "--include-column",
        default="primary_included",
        help="Boolean cohort include column (default: primary_included).",
    )
    apply_cohort.add_argument("--overwrite", action="store_true")

    cohort_outcomes = subparsers.add_parser(
        "build-cohort-trial-outcomes",
        help="Apply one frozen cohort and publish the population trial table.",
    )
    cohort_outcomes.add_argument("--project-dir", type=Path, required=True)
    cohort_outcomes.add_argument("--cohort-id", required=True)
    cohort_outcomes.add_argument(
        "--metric-recipe",
        choices=("tail-candidate-corrected", "tail-candidate-development"),
        default="tail-candidate-corrected",
    )
    cohort_outcomes.add_argument("--overwrite", action="store_true")

    eligibility = subparsers.add_parser(
        "build-analysis-eligibility",
        help="Publish outcome eligibility without changing cohort membership.",
    )
    eligibility.add_argument("--project-dir", type=Path, required=True)
    eligibility.add_argument("--cohort-id", required=True)
    eligibility.add_argument("--analysis-id", required=True)
    eligibility.add_argument("--metric", required=True)
    eligibility.add_argument(
        "--outcome",
        choices=("total-activity", "conditional-intensity"),
        default="total-activity",
    )
    eligibility.add_argument("--alignment", choices=("CS", "US"), default="CS")
    eligibility.add_argument("--min-baseline-samples", type=int, default=1)
    eligibility.add_argument("--min-response-samples", type=int, default=1)
    eligibility.add_argument("--overwrite", action="store_true")

    learning_onset = subparsers.add_parser(
        "learning-onset",
        help="Fit condition-aware block and longitudinal learning models.",
    )
    learning_onset.add_argument("--project-dir", type=Path, required=True)
    learning_onset.add_argument("--cohort-id", required=True)
    learning_onset.add_argument("--analysis-id", required=True)
    learning_onset.add_argument(
        "--metric",
        required=True,
        help=(
            "Metric ID to select from cohort trial outcomes; only rows for this "
            "metric enter eligibility, models, resampling, and figures."
        ),
    )
    learning_onset.add_argument(
        "--outcome",
        choices=("total-activity", "conditional-intensity"),
        default="total-activity",
        help="Response/baseline outcome derived from the selected metric.",
    )
    learning_onset.add_argument(
        "--alignment",
        choices=("CS", "US"),
        default="CS",
        help="Trial alignment to analyze (default: CS).",
    )
    learning_onset.add_argument(
        "--control-condition",
        default="control",
        help="Reference condition for model encoding and learning contrasts.",
    )
    learning_onset.add_argument(
        "--test-condition",
        required=True,
        help="Condition compared with control as the learned/test group.",
    )
    learning_onset.add_argument(
        "--pretraining-block",
        choices=(
            "Pre-train",
            "Train 1",
            "Train 2",
            "Train 3",
            "Train 4",
            "Train 5",
            "Test 1",
            "Test 2",
            "Test 3",
        ),
        default="Pre-train",
        help="Reference block used to define change (default: Pre-train).",
    )
    learning_onset.add_argument(
        "--delta-min",
        type=float,
        required=True,
        help="Minimum positive learning contrast required for support.",
    )
    learning_onset.add_argument(
        "--persistence-trials",
        type=int,
        default=3,
        help="Consecutive supported scheduled trials required for onset.",
    )
    learning_onset.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals and simultaneous bands.",
    )
    learning_onset.add_argument(
        "--spline-df",
        type=int,
        default=5,
        help="Degrees of freedom for the primary cubic trial spline.",
    )
    learning_onset.add_argument(
        "--skip-categorical-sensitivity",
        action="store_true",
        help="Skip the optional categorical-trial sensitivity fit.",
    )
    learning_onset.add_argument(
        "--sensitivity-optimizer",
        default="powell",
        help="Alternate optimizer for diagnostic refits; use 'none' to disable.",
    )
    learning_onset.add_argument(
        "--skip-random-intercept-sensitivity",
        action="store_true",
        help="Skip the random-intercept-only diagnostic refit.",
    )
    learning_onset.add_argument(
        "--late-block",
        action="append",
        dest="late_blocks",
        help=(
            "Late block for fish-level robustness; repeat to pool blocks. "
            "The last value is the primary leave-one-fish-out block."
        ),
    )
    learning_onset.add_argument(
        "--min-baseline-samples",
        type=int,
        default=1,
        help="Minimum valid samples required in a trial baseline window.",
    )
    learning_onset.add_argument(
        "--min-response-samples",
        type=int,
        default=1,
        help="Minimum valid samples required in a trial response window.",
    )
    learning_onset.add_argument(
        "--activity-offset",
        type=float,
        default=1e-6,
        help=(
            "Positive offset before logging total activity; ignored for "
            "conditional intensity."
        ),
    )
    learning_onset.add_argument(
        "--random-effects-formula",
        default="1 + trial_scaled",
        help="statsmodels random-effects formula grouped by fish.",
    )
    learning_onset.add_argument(
        "--disable-random-intercept-fallback",
        action="store_true",
        help="Do not retry a failed requested random structure as intercept-only.",
    )
    learning_onset.add_argument(
        "--optimizer",
        default="lbfgs",
        help="Primary statsmodels MixedLM optimizer (default: lbfgs).",
    )
    learning_onset.add_argument(
        "--bootstrap",
        type=int,
        default=499,
        help="Requested whole-fish longitudinal bootstrap refits.",
    )
    learning_onset.add_argument(
        "--min-successful-bootstrap",
        type=int,
        default=100,
        help="Minimum successful refits required to accept the simultaneous band.",
    )
    learning_onset.add_argument(
        "--min-bootstrap-success-fraction",
        type=float,
        default=0.8,
        help="Minimum successful-refit fraction required for the band.",
    )
    learning_onset.add_argument(
        "--permutations",
        type=int,
        default=9999,
        help="Condition-label permutations for fish-level robustness.",
    )
    learning_onset.add_argument(
        "--seed",
        type=int,
        default=20260917,
        help="Random seed shared by bootstrap and permutation procedures.",
    )
    learning_onset.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the complete existing artifact set for this analysis ID.",
    )

    figure_learning_onset = subparsers.add_parser(
        "figure-learning-onset",
        help="Render trajectories, trial contrasts, and planned block contrasts.",
    )
    figure_learning_onset.add_argument("--project-dir", type=Path, required=True)
    figure_learning_onset.add_argument("--analysis-id", required=True)
    figure_learning_onset.add_argument(
        "--mode", choices=("publication", "static"), default="static"
    )
    figure_learning_onset.add_argument("--overwrite", action="store_true")

    figure_learning_diagnostics = subparsers.add_parser(
        "figure-learning-diagnostics",
        help="Render residual diagnostics for the learning mixed models.",
    )
    figure_learning_diagnostics.add_argument(
        "--project-dir", type=Path, required=True
    )
    figure_learning_diagnostics.add_argument("--analysis-id", required=True)
    figure_learning_diagnostics.add_argument(
        "--mode", choices=("publication", "static"), default="static"
    )
    figure_learning_diagnostics.add_argument("--overwrite", action="store_true")

    plan_batch = subparsers.add_parser(
        "plan-batch",
        help="Write a deterministic per-recording stage work manifest for coverage/resume.",
    )
    plan_batch.add_argument("--project-dir", type=Path, required=True)
    plan_batch.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Recording ID; repeat for every batch member.",
    )
    plan_batch.add_argument("--batch-id", required=True)
    plan_batch.add_argument(
        "--metric-recipe",
        choices=(
            "tail-candidate-corrected",
            "tail-candidate-development",
        ),
        default="tail-candidate-corrected",
    )
    plan_batch.add_argument(
        "--selection",
        choices=("all", "pending", "failed"),
        default="all",
    )
    plan_batch.add_argument("--overwrite", action="store_true")

    execute_batch = subparsers.add_parser(
        "execute-batch",
        help="Run pending/failed/all recordings for a candidate route and refresh the work manifest.",
    )
    execute_batch.add_argument("--project-dir", type=Path, required=True)
    execute_batch.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Recording ID; repeat for every batch member.",
    )
    execute_batch.add_argument("--batch-id", required=True)
    execute_batch.add_argument(
        "--analysis-id",
        help="Optional runner analysis ID; defaults to <batch-id>-run.",
    )
    execute_batch.add_argument(
        "--metric-recipe",
        choices=(
            "tail-candidate-corrected",
            "tail-candidate-development",
        ),
        default="tail-candidate-corrected",
    )
    execute_batch.add_argument(
        "--selection",
        choices=("pending", "failed", "all"),
        default="pending",
    )
    execute_batch.add_argument("--experiment", default="allDelay")
    execute_batch.add_argument("--batch-size", type=int, default=250_000)
    execute_batch.add_argument(
        "--no-overwrite-failed",
        action="store_true",
        help="When selecting failed stages, verify only and do not rebuild.",
    )
    execute_batch.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stage banners, step flags, and progress bars.",
    )

    mixed_effects = subparsers.add_parser(
        "candidate-mixed-effects",
        help="DEPRECATED exploratory mixed-effects comparison; use learning-onset.",
    )
    mixed_effects.add_argument("--project-dir", type=Path, required=True)
    mixed_effects.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Recording ID; repeat for every included fish.",
    )
    mixed_effects.add_argument("--analysis-id", required=True)
    mixed_effects.add_argument(
        "--metric-recipe",
        choices=(
            "tail-candidate-corrected",
            "tail-candidate-development",
        ),
        default="tail-candidate-corrected",
    )
    mixed_effects.add_argument("--overwrite", action="store_true")

    fish_permutation = subparsers.add_parser(
        "candidate-fish-permutation",
        help="DEPRECATED exploratory early-vs-late permutation; use learning-onset.",
    )
    fish_permutation.add_argument("--project-dir", type=Path, required=True)
    fish_permutation.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Recording ID; repeat for every included fish.",
    )
    fish_permutation.add_argument("--analysis-id", required=True)
    fish_permutation.add_argument(
        "--metric-recipe",
        choices=(
            "tail-candidate-corrected",
            "tail-candidate-development",
        ),
        default="tail-candidate-corrected",
    )
    fish_permutation.add_argument("--overwrite", action="store_true")

    fish_bootstrap = subparsers.add_parser(
        "candidate-fish-bootstrap",
        help="DEPRECATED exploratory early-vs-late bootstrap; use learning-onset.",
    )
    fish_bootstrap.add_argument("--project-dir", type=Path, required=True)
    fish_bootstrap.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Recording ID; repeat for every included fish.",
    )
    fish_bootstrap.add_argument("--analysis-id", required=True)
    fish_bootstrap.add_argument(
        "--metric-recipe",
        choices=(
            "tail-candidate-corrected",
            "tail-candidate-development",
        ),
        default="tail-candidate-corrected",
    )
    fish_bootstrap.add_argument("--overwrite", action="store_true")

    model_input = subparsers.add_parser(
        "candidate-model-input",
        help="DEPRECATED exploratory model-input export; use learning-onset.",
    )
    model_input.add_argument("--project-dir", type=Path, required=True)
    model_input.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Recording ID; repeat for every included fish.",
    )
    model_input.add_argument("--analysis-id", required=True)
    model_input.add_argument(
        "--metric-recipe",
        choices=(
            "tail-candidate-corrected",
            "tail-candidate-development",
        ),
        default="tail-candidate-corrected",
    )
    model_input.add_argument("--overwrite", action="store_true")

    resolve_config = subparsers.add_parser(
        "resolve-config",
        help="Write resolved recipe JSON, trial map, and source-trace report.",
    )
    resolve_config.add_argument("--project-dir", type=Path, required=True)
    resolve_config.add_argument(
        "--runner-recipe",
        choices=("candidate-corrected-runner", "candidate-development-runner"),
        default="candidate-corrected-runner",
    )
    resolve_config.add_argument("--experiment", default="allDelay")
    resolve_config.add_argument("--overwrite", action="store_true")

    audit_tracking = subparsers.add_parser(
        "audit-tracking",
        help="Inventory tracking columns without assuming they are all angles.",
    )
    audit_tracking.add_argument("--input", type=Path, required=True)
    audit_tracking.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path; omit to print the audit.",
    )
    audit_tracking.add_argument("--sample-rows", type=int, default=2_000)
    audit_tracking.add_argument("--overwrite", action="store_true")

    validate_raw = subparsers.add_parser(
        "validate-raw",
        help="Validate one local camera/tracking/protocol triplet without writing Parquet.",
    )
    validate_raw.add_argument("--input-dir", type=Path, required=True)
    validate_raw.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; omit to print the summary.",
    )
    validate_raw.add_argument("--overwrite", action="store_true")

    environment = subparsers.add_parser(
        "environment-report",
        help="Report pinned package, numerical-library, and figure settings.",
    )
    environment.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; omit to print the report.",
    )

    run_pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Run inventory, verified intake, corrected analysis, and available figures.",
    )
    run_pipeline.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Strict JSON file with raw_dir, save_dir, experiment, and analysis_id.",
    )
    run_pipeline.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stage banners, step flags, and progress bars.",
    )

    discarding = subparsers.add_parser(
        "assess-discarding",
        help="Audit technical readiness and exploratory legacy behavior without excluding fish.",
    )
    discarding.add_argument("--raw-dir", type=Path, required=True)
    discarding.add_argument("--project-dir", type=Path, required=True)
    discarding.add_argument("--analysis-id", required=True)
    discarding.add_argument("--experiment", required=True)
    discarding.add_argument("--metric", required=True)
    discarding.add_argument(
        "--metric-recipe", type=_activity_metric_recipe, default="tail-candidate-corrected"
    )
    discarding.add_argument("--technical-policy", type=Path)
    discarding.add_argument("--disable-check", action="append", default=[])
    discarding.add_argument("--recording-id", action="append")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    # Fail early on unsupported runtime, parse arguments, then dispatch one command.
    ensure_supported_runtime()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "assess-discarding":
        from classical_conditioning.analysis.discarding import assess_discarding

        result = assess_discarding(
            args.raw_dir, args.project_dir,
            analysis_id=args.analysis_id, experiment=args.experiment,
            metric_id=args.metric, metric_recipe=args.metric_recipe,
            technical_policy_path=args.technical_policy,
            disabled_rules=args.disable_check, recording_ids=args.recording_id,
        )
        print(f"Technical assessment: {result.technical_path}")
        print(f"Exploratory assessment: {result.exploratory_path}")
        print(f"Rule-by-rule flow: {result.flow_path}")
        print(f"Assessment hash: {result.assessment_hash}")
        return

    if args.command == "environment-report":
        from classical_conditioning.environment import (
            format_environment_report,
            write_environment_report,
        )

        if args.output is None:
            print(format_environment_report())
        else:
            print(f"Environment report: {write_environment_report(args.output)}")
        return

    if args.command == "run-pipeline":
        from classical_conditioning.pipeline import run_pipeline
        from classical_conditioning.progress import default_progress
        from classical_conditioning.run_config import load_pipeline_run_config

        config = load_pipeline_run_config(args.config.resolve())
        progress = default_progress(enabled=not args.quiet and config.show_progress)
        result = run_pipeline(config, progress=progress)
        print(f"Raw: {config.raw_dir}")
        print(f"Save: {config.save_dir}")
        print(f"Experiment: {config.experiment}")
        print(f"Recordings: {len(result.recording_ids)}")
        print(f"Intake completed: {len(result.intake_completed)}")
        print(f"Intake skipped: {len(result.intake_skipped)}")
        print(f"Intake failed: {len(result.intake_failed)}")
        if result.candidate_runner_status:
            print(f"Candidate analysis: {config.resolved_candidate_analysis_id()}")
        if result.figure_paths:
            print("Figures:")
            for path in result.figure_paths:
                print(f"  {path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "resolve-config":
        from classical_conditioning.config import export_resolved_config

        result = export_resolved_config(
            args.project_dir,
            experiment_name=args.experiment,
            runner_recipe=args.runner_recipe,
            overwrite=args.overwrite,
        )
        print(f"Recipe: {result.recipe_id}")
        print(f"Experiment: {result.experiment_id}")
        print(f"Config hash: {result.config_hash}")
        print(f"Resolved config: {result.config_path}")
        print(f"Trial map: {result.trial_map_path}")
        print(f"Source report: {result.source_report_path}")
        return

    if args.command == "audit-tracking":
        import json

        from classical_conditioning.ingestion.tracking_audit import (
            audit_tracking_file,
            write_tracking_audit,
        )

        if args.output is None:
            print(
                json.dumps(
                    audit_tracking_file(
                        args.input,
                        sample_rows=args.sample_rows,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            output = write_tracking_audit(
                args.input,
                args.output,
                sample_rows=args.sample_rows,
                overwrite=args.overwrite,
            )
            print(f"Tracking audit: {output}")
        return

    if args.command == "validate-raw":
        import json

        from classical_conditioning.ingestion.validate_raw import validate_raw_triplet

        result = validate_raw_triplet(
            args.input_dir,
            output=args.output,
            overwrite=args.overwrite,
        )
        if args.output is None:
            print(json.dumps(result.summary, indent=2, sort_keys=True))
        else:
            print(f"Recording: {result.recording_id}")
            print(f"Status: {result.summary['status']}")
            print(f"Raw validation: {result.report_path}")
        return

    if args.command == "inventory":
        import json

        from classical_conditioning.inventory import (
            build_recording_inventory,
            write_recording_inventory,
        )

        if args.output is None:
            print(
                json.dumps(
                    build_recording_inventory(
                        args.input_dir,
                        hash_files=not args.skip_hashes,
                        inspect_tracking_headers=args.inspect_tracking_headers,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            output = write_recording_inventory(
                args.input_dir,
                args.output,
                hash_files=not args.skip_hashes,
                inspect_tracking_headers=args.inspect_tracking_headers,
                overwrite=args.overwrite,
            )
            print(f"Recording inventory: {output}")
        return

    if args.command == "intake":
        result = intake_recording(
            input_dir=args.input_dir,
            project_dir=args.project_dir,
            recording_id=args.recording_id,
            chunk_rows=args.chunk_rows,
            preview_rows=args.preview_rows,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Status: {result.status}")
        print(f"Processed data: {result.processed_dir}")
        print(f"Integrity report: {result.report_path}")
        return

    if args.command == "intake-batch":
        result = intake_recordings(
            input_dir=args.input_dir,
            project_dir=args.project_dir,
            keep_conditions=args.keep_conditions,
            recording_ids=args.recording_id,
            chunk_rows=args.chunk_rows,
            preview_rows=args.preview_rows,
            overwrite=args.overwrite,
        )
        print(f"Selected: {len(result.recording_ids)}")
        print(f"Completed: {len(result.completed)}")
        print(f"Skipped: {len(result.skipped)}")
        print(f"Failed: {len(result.failed)}")
        for recording_id, reason in result.failed:
            print(f"FAILED {recording_id}: {reason}")
        return

    if args.command == "retry-intake":
        result = intake_recordings(
            input_dir=args.input_dir,
            project_dir=args.project_dir,
            recording_ids=(args.recording_id,),
            retry_failed=True,
        )
        if result.failed or result.incomplete:
            reasons = result.failed + result.incomplete
            raise ConfigurationError(
                f"Intake retry did not produce a ready recording: {reasons}"
            )
        print(f"Ready: {args.recording_id}")
        print(f"Ledger: {result.ledger_path}")
        return

    if args.command == "preprocess":
        from classical_conditioning.preprocessing.corrected_frame_preprocessing import (
            build_corrected_preprocessing,
        )

        result = build_corrected_preprocessing(
            args.project_dir,
            args.recording_id,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Rows: {result.row_count:,}")
        print(f"Derivative-valid rows: {result.derivative_valid_count:,}")
        print(f"Frames: {result.frames_path}")
        print(f"Summary: {result.summary_path}")
        return
    if args.command == "trace-review":
        from classical_conditioning.analysis import build_trace_review

        result = build_trace_review(
            args.project_dir,
            args.recording_id,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Review windows: {result.window_count}")
        print(f"Interactive review: {result.interactive_figure_path}")
        print(f"Annotation template: {result.annotation_path}")
        return

    if args.command == "compare-candidate-metrics":
        from classical_conditioning.analysis import (
            build_candidate_metric_comparison,
        )
        from classical_conditioning.analysis.movement_state import (
            COMPARISON_RECIPE_TO_METRIC_SOURCE,
        )

        metric_recipe = COMPARISON_RECIPE_TO_METRIC_SOURCE[args.recipe]
        result = build_candidate_metric_comparison(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            experiment_name=args.experiment,
            metric_recipe=metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Metric source: {metric_recipe}")
        for name, path in result.artifact_paths.items():
            print(f"{name}: {path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "candidate-runner":
        from classical_conditioning.analysis import (
            run_candidate_development_pipeline,
        )
        from classical_conditioning.analysis.movement_state import (
            RUNNER_RECIPE_TO_METRIC_SOURCE,
        )
        from classical_conditioning.progress import default_progress

        metric_recipe = RUNNER_RECIPE_TO_METRIC_SOURCE[args.recipe]
        progress = default_progress(enabled=not args.quiet)
        result = run_candidate_development_pipeline(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            experiment_name=args.experiment,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            metric_recipe=metric_recipe,
            runner_recipe=args.recipe,
            progress=progress,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Metric source: {metric_recipe}")
        print(f"Manifest: {result.manifest_path}")
        for recording_id, entries in result.step_status.items():
            for step_name, state in entries.items():
                print(f"{recording_id}:{step_name}={state}")
        return

    if args.command == "candidate-trial-outcomes":
        from classical_conditioning.analysis.movement_state import (
            TRIAL_RECIPE_TO_METRIC_SOURCE,
        )
        from classical_conditioning.analysis import (
            build_candidate_trial_outcomes,
        )

        metric_recipe = TRIAL_RECIPE_TO_METRIC_SOURCE[args.recipe]
        result = build_candidate_trial_outcomes(
            args.project_dir,
            args.recording_id,
            experiment_name=args.experiment,
            metric_recipe=metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Metric source: {metric_recipe}")
        print(f"Rows: {result.row_count}")
        print(f"Outcomes: {result.outcomes_path}")
        print(f"Coverage: {result.coverage_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "freeze-cohort":
        import pandas as pd
        import pyarrow.parquet as pq

        from classical_conditioning.cohort import freeze_cohort_manifest

        suffix = args.input.suffix.lower()
        if suffix == ".parquet":
            reviewed = pq.read_table(args.input).to_pandas()
        elif suffix == ".csv":
            reviewed = pd.read_csv(args.input)
        else:
            parser.error("freeze-cohort --input must be Parquet or CSV.")
        result = freeze_cohort_manifest(
            args.project_dir,
            reviewed,
            cohort_id=args.cohort_id,
            policy_id=args.policy_id,
        )
        print(f"Cohort: {result.cohort_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Rows: {result.row_count}")
        print(f"Primary included: {result.primary_count}")
        print(f"Logical hash: {result.logical_content_sha256}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "apply-cohort":
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        from classical_conditioning.artifacts import publish_transaction, sha256_file
        from classical_conditioning.cohort import apply_cohort, RECIPE_ID

        manifest_path = (
            args.project_dir
            / "Processed data"
            / "Cohorts"
            / args.cohort_id
            / f"{RECIPE_ID}.parquet"
        )
        if not manifest_path.is_file():
            parser.error(f"Missing frozen cohort manifest: {manifest_path}")
        suffix = args.input.suffix.lower()
        if suffix == ".parquet":
            data = pq.read_table(args.input).to_pandas()
        elif suffix == ".csv":
            data = pd.read_csv(args.input)
        else:
            parser.error("apply-cohort --input must be Parquet or CSV.")
        filtered = apply_cohort(
            data,
            pq.read_table(manifest_path).to_pandas(),
            include_column=args.include_column,
        )
        output = args.output
        if output.exists() and not args.overwrite:
            parser.error(f"Output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = args.project_dir.resolve()
        from classical_conditioning.artifacts import artifact_staging

        with artifact_staging(
            staging_parent,
            prefix=f".{args.cohort_id}-apply-cohort-",
        ) as staging_root:
            staged = staging_root / output.name
            table = pa.Table.from_pandas(filtered, preserve_index=False, safe=True)
            pq.write_table(table, staged, compression="zstd", write_statistics=True)
            publish_transaction(
                ((staged, output),),
                staging_root,
                overwrite=args.overwrite,
            )
        print(f"Cohort: {args.cohort_id}")
        print(f"Include column: {args.include_column}")
        print(f"Input rows: {len(data)}")
        print(f"Output rows: {len(filtered)}")
        print(f"Output: {output}")
        print(f"SHA256: {sha256_file(output)}")
        return

    if args.command == "build-cohort-trial-outcomes":
        from classical_conditioning.analysis import build_cohort_trial_outcomes

        result = build_cohort_trial_outcomes(
            args.project_dir,
            cohort_id=args.cohort_id,
            metric_recipe=args.metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Cohort: {result.cohort_id}")
        print(f"Cohort hash: {result.cohort_hash}")
        print(f"Fish: {result.fish_count}")
        print(f"Rows: {result.row_count}")
        print(f"Outcomes: {result.outcomes_path}")
        print(f"Sample flow: {result.sample_flow_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "build-analysis-eligibility":
        from classical_conditioning.analysis import (
            build_analysis_eligibility_artifact,
        )

        result = build_analysis_eligibility_artifact(
            args.project_dir,
            cohort_id=args.cohort_id,
            analysis_id=args.analysis_id,
            metric_id=args.metric,
            outcome_id=args.outcome,
            alignment=args.alignment,
            min_baseline_samples=args.min_baseline_samples,
            min_response_samples=args.min_response_samples,
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Cohort: {result.cohort_id}")
        print(f"Rows: {result.row_count}")
        print(f"Eligible: {result.eligible_count}")
        print(f"Eligibility: {result.eligibility_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "learning-onset":
        from classical_conditioning.analysis.inference import (
            LearningOnsetConfig,
            build_learning_onset_analysis,
        )

        result = build_learning_onset_analysis(
            args.project_dir,
            cohort_id=args.cohort_id,
            analysis_id=args.analysis_id,
            config=LearningOnsetConfig(
                metric_id=args.metric,
                outcome_id=args.outcome,
                alignment=args.alignment,
                control_condition=args.control_condition,
                test_condition=args.test_condition,
                pretraining_block=args.pretraining_block,
                delta_min=args.delta_min,
                persistence_trials=args.persistence_trials,
                confidence_level=args.confidence_level,
                spline_df=args.spline_df,
                run_categorical_sensitivity=(
                    not args.skip_categorical_sensitivity
                ),
                sensitivity_optimizer=(
                    None
                    if args.sensitivity_optimizer.lower() == "none"
                    else args.sensitivity_optimizer
                ),
                run_random_intercept_sensitivity=(
                    not args.skip_random_intercept_sensitivity
                ),
                late_blocks=tuple(args.late_blocks or ("Test 2", "Test 3")),
                min_baseline_samples=args.min_baseline_samples,
                min_response_samples=args.min_response_samples,
                activity_offset=args.activity_offset,
                random_effects_formula=args.random_effects_formula,
                allow_random_intercept_fallback=(
                    not args.disable_random_intercept_fallback
                ),
                optimizer=args.optimizer,
                n_bootstrap=args.bootstrap,
                min_successful_bootstrap=args.min_successful_bootstrap,
                min_bootstrap_success_fraction=(
                    args.min_bootstrap_success_fraction
                ),
                n_permutations=args.permutations,
                seed=args.seed,
            ),
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Cohort: {result.cohort_id}")
        print(f"Cohort hash: {result.cohort_hash}")
        print(f"Block contrasts: {result.block_contrasts_path}")
        print(f"Trial contrasts: {result.trial_contrasts_path}")
        print(f"Onset: {result.onset_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "figure-learning-onset":
        from classical_conditioning.figures import (
            FigureMode,
            build_learning_onset_figure,
        )

        result = build_learning_onset_figure(
            args.project_dir,
            args.analysis_id,
            mode=FigureMode(args.mode),
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command == "figure-learning-diagnostics":
        from classical_conditioning.figures import (
            FigureMode,
            build_learning_diagnostics_figure,
        )

        result = build_learning_diagnostics_figure(
            args.project_dir,
            args.analysis_id,
            mode=FigureMode(args.mode),
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command == "plan-batch":
        from classical_conditioning.operations.batch_work import (
            write_batch_work_manifest,
        )

        result = write_batch_work_manifest(
            args.project_dir,
            args.recording_id,
            batch_id=args.batch_id,
            metric_recipe=args.metric_recipe,
            selection=args.selection,
            overwrite=args.overwrite,
        )
        print(f"Batch: {result.batch_id}")
        print(f"Recipe: batch-work-manifest")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Rows: {result.row_count}")
        print(f"Complete: {result.complete_count}")
        print(f"Pending: {result.pending_count}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "execute-batch":
        from classical_conditioning.operations.batch_work import (
            execute_batch_work,
        )
        from classical_conditioning.progress import default_progress

        progress = default_progress(enabled=not args.quiet)
        result = execute_batch_work(
            args.project_dir,
            args.recording_id,
            batch_id=args.batch_id,
            metric_recipe=args.metric_recipe,
            selection=args.selection,
            analysis_id=args.analysis_id,
            experiment_name=args.experiment,
            batch_size=args.batch_size,
            overwrite_failed=not args.no_overwrite_failed,
            progress=progress,
        )
        print(f"Batch: {result.batch_id}")
        print(f"Selection: {result.selection}")
        print(f"Metric source: {result.metric_recipe}")
        print(f"Before pending/failed: {result.before_pending_count}/{result.before_failed_count}")
        print(f"After complete/pending/failed: {result.after_complete_count}/{result.after_pending_count}/{result.after_failed_count}")
        if result.runner_manifest_path is not None:
            print(f"Runner manifest: {result.runner_manifest_path}")
        print(f"Work manifest: {result.refreshed_manifest_path}")
        return

    if args.command == "candidate-mixed-effects":
        _warn_deprecated_exploratory_inference(args.command)
        from classical_conditioning.analysis.inference.mixed_effects import (
            build_candidate_mixed_effects,
        )

        result = build_candidate_mixed_effects(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            metric_recipe=args.metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: candidate-mixed-effects")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Model input: {result.model_input_path}")
        print(f"Coefficients: {result.coefficients_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "candidate-fish-permutation":
        _warn_deprecated_exploratory_inference(args.command)
        from classical_conditioning.analysis.inference.fish_permutation import (
            build_candidate_fish_permutation,
        )

        result = build_candidate_fish_permutation(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            metric_recipe=args.metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: candidate-fish-permutation")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Fish effects: {result.fish_effects_path}")
        print(f"Population: {result.population_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "candidate-fish-bootstrap":
        _warn_deprecated_exploratory_inference(args.command)
        from classical_conditioning.analysis.inference.fish_bootstrap import (
            build_candidate_fish_bootstrap,
        )

        result = build_candidate_fish_bootstrap(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            metric_recipe=args.metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: candidate-fish-bootstrap")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Fish effects: {result.fish_effects_path}")
        print(f"Population: {result.population_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "candidate-model-input":
        _warn_deprecated_exploratory_inference(args.command)
        from classical_conditioning.analysis.inference.model_input import (
            build_candidate_model_input_artifact,
        )

        result = build_candidate_model_input_artifact(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            metric_recipe=args.metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: candidate-model-input")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Rows: {result.row_count}")
        print(f"Fish: {result.fish_count}")
        print(f"Model input: {result.model_input_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "movement-sensitivity":
        from classical_conditioning.analysis import (
            build_movement_sensitivity_report,
        )

        result = build_movement_sensitivity_report(
            args.project_dir,
            args.recording_id,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "figure-example-traces":
        from classical_conditioning.figures import FigureMode, build_example_trace_figure

        result = build_example_trace_figure(
            args.project_dir,
            args.recording_id,
            trial_numbers=args.trial,
            metric_id=args.metric,
            experiment=args.experiment,
            mode=FigureMode(args.mode),
            tail_point=args.tail_point,
            window_s=(args.window_start, args.window_end),
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command == "figure-candidate-profiles":
        from classical_conditioning.figures import (
            FigureMode,
            build_candidate_profile_figure,
        )

        result = build_candidate_profile_figure(
            args.project_dir,
            args.recording_id,
            mode=FigureMode(args.mode),
            trial_type=args.trial_type,
            figure_id=args.figure,
            temporal_recipe=args.recipe,
            overwrite=args.overwrite,
        )
        if isinstance(result, Path):
            print(f"Interactive figure: {result}")
        else:
            for output in result.outputs:
                print(f"Figure: {output}")
            print(f"Provenance: {result.sidecar}")
        return

    if args.command == "figure-metric-comparison":
        from classical_conditioning.figures import (
            FigureMode,
            build_metric_comparison_figure,
        )

        result = build_metric_comparison_figure(
            args.project_dir,
            args.analysis_id,
            mode=FigureMode(args.mode),
            trial_type=args.trial_type,
            outcome_id=args.outcome,
            comparison_recipe=args.recipe,
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command == "figure-cohort-selected-block-ratio":
        from classical_conditioning.figures import (
            FigureMode,
            build_selected_block_ratio_figure,
        )

        result = build_selected_block_ratio_figure(
            args.project_dir,
            cohort_id=args.cohort_id,
            analysis_id=args.analysis_id,
            metric_id=args.metric,
            outcome_id=args.outcome,
            metric_recipe=args.metric_recipe,
            mode=FigureMode(args.mode),
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command == "figure-cohort-trial-ratio":
        from classical_conditioning.figures import FigureMode, build_trial_ratio_figure

        result = build_trial_ratio_figure(
            args.project_dir,
            cohort_id=args.cohort_id,
            analysis_id=args.analysis_id,
            metric_id=args.metric,
            outcome_id=args.outcome,
            metric_recipe=args.metric_recipe,
            mode=FigureMode(args.mode),
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command == "figure-cohort-event-aligned-ratio":
        from classical_conditioning.figures import (
            FigureMode,
            build_event_aligned_ratio_figure,
        )

        result = build_event_aligned_ratio_figure(
            args.project_dir,
            cohort_id=args.cohort_id,
            analysis_id=args.analysis_id,
            metric_id=args.metric,
            outcome_id=args.outcome,
            metric_recipe=args.metric_recipe,
            mode=FigureMode(args.mode),
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command in {
        "figure-cohort-catch-profile",
        "figure-cohort-block-profile",
    }:
        from classical_conditioning.figures import (
            FigureMode,
            build_block_profile_figure,
            build_catch_profile_figure,
        )

        builder = (
            build_catch_profile_figure
            if args.command == "figure-cohort-catch-profile"
            else build_block_profile_figure
        )
        result = builder(
            args.project_dir,
            cohort_id=args.cohort_id,
            analysis_id=args.analysis_id,
            metric_id=args.metric,
            metric_recipe=args.metric_recipe,
            mode=FigureMode(args.mode),
            minimum_coverage=args.minimum_coverage,
            overwrite=args.overwrite,
        )
        for output in result.outputs:
            print(f"Figure: {output}")
        print(f"Provenance: {result.sidecar}")
        return

    if args.command == "movement-state":
        from classical_conditioning.analysis.movement_state import (
            MOVEMENT_RECIPE_TO_METRIC_SOURCE,
            build_candidate_movement_state,
        )

        metric_recipe = MOVEMENT_RECIPE_TO_METRIC_SOURCE[args.recipe]
        result = build_candidate_movement_state(
            args.project_dir,
            args.recording_id,
            metric_recipe=metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Metric source: {metric_recipe}")
        print(f"Rows: {result.row_count:,}")
        print(f"Movement state: {result.movement_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "temporal-profiles":
        from classical_conditioning.analysis.movement_state import (
            TEMPORAL_RECIPE_TO_METRIC_SOURCE,
        )
        from classical_conditioning.analysis import build_candidate_temporal_profiles

        metric_recipe = TEMPORAL_RECIPE_TO_METRIC_SOURCE[args.recipe]
        result = build_candidate_temporal_profiles(
            args.project_dir,
            args.recording_id,
            experiment_name=args.experiment,
            metric_recipe=metric_recipe,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Metric source: {metric_recipe}")
        print(f"Rows: {result.row_count:,}")
        print(f"Trials: CS={result.cs_trial_count}, US={result.us_trial_count}")
        print(f"Profiles: {result.profiles_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "compare":
        from classical_conditioning.comparison import compare_parquet_artifacts

        result = compare_parquet_artifacts(
            args.left,
            args.right,
            args.output,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        print(f"Row counts equal: {result.row_counts_equal}")
        print(f"Common columns: {result.common_column_count}")
        print(f"Report: {result.output}")
        return

    if args.command == "activity-metrics":
        if args.recipe == "tail-candidate-development":
            from classical_conditioning.preprocessing.benchmarks.candidate_metrics_from_intake import (
                build_candidate_activity_metrics,
            )

            result = build_candidate_activity_metrics(
                args.project_dir,
                args.recording_id,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
        elif args.recipe == "tail-candidate-corrected":
            from classical_conditioning.preprocessing.candidate_metrics_from_corrected_frames import (
                build_candidate_activity_metrics_from_corrected,
            )

            result = build_candidate_activity_metrics_from_corrected(
                args.project_dir,
                args.recording_id,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
        else:
            raise ValueError(f"Unsupported activity-metrics recipe: {args.recipe}")
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Rows: {result.row_count:,}")
        print(f"Valid derivatives: {result.valid_derivative_count:,}")
        print(f"Metrics: {result.metrics_path}")
        print(f"Summary: {result.summary_path}")
        return
