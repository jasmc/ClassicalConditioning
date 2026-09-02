"""Command-line interface for local analysis operations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from classical_conditioning.intake import intake_recording, intake_recordings


def build_parser() -> argparse.ArgumentParser:
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
        help="Run one versioned preprocessing recipe from intake Parquet.",
    )
    preprocess.add_argument("--project-dir", type=Path, required=True)
    preprocess.add_argument("--recording-id", required=True)
    preprocess.add_argument(
        "--recipe",
        choices=("legacy-paper-v1", "corrected-preprocess-v1"),
        default="legacy-paper-v1",
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

    compare_pickle = subparsers.add_parser(
        "compare-legacy-pickle",
        help="Compare a local historical per-fish pickle with legacy Parquet.",
    )
    compare_pickle.add_argument("--pickle", type=Path, required=True)
    compare_pickle.add_argument("--parquet", type=Path, required=True)
    compare_pickle.add_argument("--output", type=Path, required=True)
    compare_pickle.add_argument("--batch-size", type=int, default=250_000)
    compare_pickle.add_argument("--overwrite", action="store_true")

    activity = subparsers.add_parser(
        "activity-metrics",
        help="Build exploratory whole-tail activity metrics.",
    )
    activity.add_argument("--project-dir", type=Path, required=True)
    activity.add_argument("--recording-id", required=True)
    activity.add_argument(
        "--recipe",
        choices=(
            "tail-candidate-development-v1",
            "tail-candidate-corrected-v1",
        ),
        default="tail-candidate-development-v1",
    )
    activity.add_argument("--batch-size", type=int, default=250_000)
    activity.add_argument("--overwrite", action="store_true")

    compare_routes = subparsers.add_parser(
        "compare-routes",
        help="Summarize legacy versus candidate preprocessing behavior.",
    )
    compare_routes.add_argument("--project-dir", type=Path, required=True)
    compare_routes.add_argument("--recording-id", required=True)
    compare_routes.add_argument("--overwrite", action="store_true")

    profiles = subparsers.add_parser(
        "temporal-profiles",
        help="Align candidate metrics to protocol events and build time bins.",
    )
    profiles.add_argument("--project-dir", type=Path, required=True)
    profiles.add_argument("--recording-id", required=True)
    profiles.add_argument(
        "--recipe",
        choices=(
            "candidate-temporal-outcomes-v2",
            "candidate-temporal-outcomes-corrected-v2",
        ),
        default="candidate-temporal-outcomes-v2",
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
        "--outcome",
        choices=(
            "total-activity",
            "movement-probability",
            "fraction-time-moving",
            "conditional-intensity",
            "bout-rate",
        ),
        default="total-activity",
    )
    figure_profiles.add_argument(
        "--mode",
        choices=("publication", "static", "interactive"),
        required=True,
    )
    figure_profiles.add_argument(
        "--recipe",
        choices=(
            "candidate-temporal-outcomes-v2",
            "candidate-temporal-outcomes-corrected-v2",
        ),
        default="candidate-temporal-outcomes-v2",
    )
    figure_profiles.add_argument("--overwrite", action="store_true")

    figure_metric_comparison = subparsers.add_parser(
        "figure-metric-comparison",
        help="Render a cohort five-metric comparison from saved summaries.",
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
            "candidate-metric-comparison-v1",
            "candidate-metric-comparison-corrected-v1",
        ),
        default="candidate-metric-comparison-corrected-v1",
    )
    figure_metric_comparison.add_argument(
        "--mode",
        choices=("publication", "static"),
        default="static",
    )
    figure_metric_comparison.add_argument("--overwrite", action="store_true")

    figure_legacy = subparsers.add_parser(
        "figure-legacy-review",
        help="Render a compact QC figure from frozen legacy stage-1 samples.",
    )
    figure_legacy.add_argument("--project-dir", type=Path, required=True)
    figure_legacy.add_argument("--recording-id", required=True)
    figure_legacy.add_argument(
        "--mode",
        choices=("publication", "static"),
        default="static",
    )
    figure_legacy.add_argument("--overwrite", action="store_true")

    movement = subparsers.add_parser(
        "movement-state",
        help="Calibrate exploratory movement state and bouts for candidate metrics.",
    )
    movement.add_argument("--project-dir", type=Path, required=True)
    movement.add_argument("--recording-id", required=True)
    movement.add_argument(
        "--recipe",
        choices=("movement-candidate-v1", "movement-candidate-corrected-v1"),
        default="movement-candidate-v1",
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

    logmedian = subparsers.add_parser(
        "legacy-logmedian",
        help="Reproduce the historical LogMedian stage-3 transform.",
    )
    logmedian.add_argument("--project-dir", type=Path, required=True)
    logmedian.add_argument("--recording-id", required=True)
    logmedian.add_argument(
        "--recipe",
        choices=("historical-logmedian-v1",),
        default="historical-logmedian-v1",
    )
    logmedian.add_argument("--overwrite", action="store_true")

    standard_main = subparsers.add_parser(
        "legacy-standard-main",
        help="Reproduce the frozen standard-main stage-3 grouping transform.",
    )
    standard_main.add_argument("--project-dir", type=Path, required=True)
    standard_main.add_argument("--recording-id", required=True)
    standard_main.add_argument(
        "--recipe",
        choices=("legacy-standard-main-v1",),
        default="legacy-standard-main-v1",
    )
    standard_main.add_argument("--experiment", default="allDelay")
    standard_main.add_argument("--read-batch-rows", type=int, default=250_000)
    standard_main.add_argument("--overwrite", action="store_true")

    scaled_vigor = subparsers.add_parser(
        "legacy-scaled-vigor",
        help="Reproduce frozen standard-main stage-4 pooled aggregation.",
    )
    scaled_vigor.add_argument("--project-dir", type=Path, required=True)
    scaled_vigor.add_argument("--recording-id", required=True)
    scaled_vigor.add_argument(
        "--recipe",
        choices=("legacy-scaled-vigor-v1",),
        default="legacy-scaled-vigor-v1",
    )
    scaled_vigor.add_argument("--overwrite", action="store_true")

    normalized_vigor = subparsers.add_parser(
        "legacy-normalized-vigor",
        help="Reproduce frozen standard-main stage-5 per-trial windows.",
    )
    normalized_vigor.add_argument("--project-dir", type=Path, required=True)
    normalized_vigor.add_argument("--recording-id", required=True)
    normalized_vigor.add_argument(
        "--recipe",
        choices=("legacy-normalized-vigor-v1",),
        default="legacy-normalized-vigor-v1",
    )
    normalized_vigor.add_argument("--experiment", default="allDelay")
    normalized_vigor.add_argument("--overwrite", action="store_true")

    legacy_statistics = subparsers.add_parser(
        "legacy-statistics",
        help="Run frozen stage-5 inference over explicit normalized-vigor inputs.",
    )
    legacy_statistics.add_argument("--project-dir", type=Path, required=True)
    legacy_statistics.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Authenticated recording ID; repeat for every cohort member.",
    )
    legacy_statistics.add_argument("--analysis-id", required=True)
    legacy_statistics.add_argument("--alignment", choices=("CS", "US"), default="CS")
    legacy_statistics.add_argument(
        "--recipe",
        choices=("legacy-statistics-v1",),
        default="legacy-statistics-v1",
    )
    legacy_statistics.add_argument("--experiment", default="allDelay")
    legacy_statistics.add_argument("--overwrite", action="store_true")

    legacy_runner = subparsers.add_parser(
        "legacy-runner",
        help="Run the frozen legacy stage-3/4/5 pipeline for a local cohort.",
    )
    legacy_runner.add_argument("--project-dir", type=Path, required=True)
    legacy_runner.add_argument(
        "--recording-id",
        action="append",
        required=True,
        help="Authenticated recording ID; repeat for every cohort member.",
    )
    legacy_runner.add_argument("--analysis-id", required=True)
    legacy_runner.add_argument("--alignment", choices=("CS", "US"), default="CS")
    legacy_runner.add_argument(
        "--recipe",
        choices=("legacy-runner-v1",),
        default="legacy-runner-v1",
    )
    legacy_runner.add_argument("--experiment", default="allDelay")
    legacy_runner.add_argument("--read-batch-rows", type=int, default=250_000)
    legacy_runner.add_argument("--overwrite", action="store_true")
    legacy_runner.add_argument(
        "--skip-statistics",
        action="store_true",
        help="Only run the legacy stage-3/4/5 recording transforms without cohort inference.",
    )

    metric_comparison = subparsers.add_parser(
        "compare-candidate-metrics",
        help="Apply identical descriptive outcomes to all five candidate metrics.",
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
            "candidate-metric-comparison-v1",
            "candidate-metric-comparison-corrected-v1",
        ),
        default="candidate-metric-comparison-v1",
    )
    metric_comparison.add_argument("--experiment", default="allDelay")
    metric_comparison.add_argument("--overwrite", action="store_true")

    candidate_runner = subparsers.add_parser(
        "candidate-runner",
        help="Run the non-approved five-metric candidate-development pipeline.",
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
            "candidate-development-runner-v1",
            "candidate-corrected-runner-v1",
        ),
        default="candidate-development-runner-v1",
    )
    candidate_runner.add_argument("--overwrite", action="store_true")

    trial_outcomes = subparsers.add_parser(
        "candidate-trial-outcomes",
        help="Build exact measured-time trial outcomes for all five candidates.",
    )
    trial_outcomes.add_argument("--project-dir", type=Path, required=True)
    trial_outcomes.add_argument("--recording-id", required=True)
    trial_outcomes.add_argument("--experiment", default="allDelay")
    trial_outcomes.add_argument(
        "--recipe",
        choices=(
            "candidate-trial-outcomes-v1",
            "candidate-trial-outcomes-corrected-v1",
        ),
        default="candidate-trial-outcomes-v1",
    )
    trial_outcomes.add_argument("--overwrite", action="store_true")

    outcome_comparison = subparsers.add_parser(
        "legacy-candidate-outcome-comparison",
        help="Compare frozen legacy and candidate trial outcomes descriptively.",
    )
    outcome_comparison.add_argument("--project-dir", type=Path, required=True)
    outcome_comparison.add_argument("--recording-id", required=True)
    outcome_comparison.add_argument(
        "--recipe",
        choices=("legacy-candidate-outcome-comparison-v1",),
        default="legacy-candidate-outcome-comparison-v1",
    )
    outcome_comparison.add_argument("--overwrite", action="store_true")

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
        choices=("cohort-manifest-v1",),
        default="cohort-manifest-v1",
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
            "tail-candidate-corrected-v1",
            "tail-candidate-development-v1",
        ),
        default="tail-candidate-corrected-v1",
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
            "tail-candidate-corrected-v1",
            "tail-candidate-development-v1",
        ),
        default="tail-candidate-corrected-v1",
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

    mixed_effects = subparsers.add_parser(
        "candidate-mixed-effects",
        help="Fit exploratory fish-grouped mixed-effects models on candidate trial outcomes.",
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
            "tail-candidate-corrected-v1",
            "tail-candidate-development-v1",
        ),
        default="tail-candidate-corrected-v1",
    )
    mixed_effects.add_argument("--overwrite", action="store_true")

    fish_permutation = subparsers.add_parser(
        "candidate-fish-permutation",
        help="Fish-level early-vs-late effect with sign-flip permutation (not mixed-effects).",
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
            "tail-candidate-corrected-v1",
            "tail-candidate-development-v1",
        ),
        default="tail-candidate-corrected-v1",
    )
    fish_permutation.add_argument("--overwrite", action="store_true")

    fish_bootstrap = subparsers.add_parser(
        "candidate-fish-bootstrap",
        help="Fish-level percentile bootstrap CI for early-vs-late effects (resamples fish only).",
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
            "tail-candidate-corrected-v1",
            "tail-candidate-development-v1",
        ),
        default="tail-candidate-corrected-v1",
    )
    fish_bootstrap.add_argument("--overwrite", action="store_true")

    model_input = subparsers.add_parser(
        "candidate-model-input",
        help="Freeze the shared candidate model-input table used by LME and fish-permutation.",
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
            "tail-candidate-corrected-v1",
            "tail-candidate-development-v1",
        ),
        default="tail-candidate-corrected-v1",
    )
    model_input.add_argument("--overwrite", action="store_true")

    resolve_config = subparsers.add_parser(
        "resolve-config",
        help="Write resolved recipe JSON, trial map, and source-trace report.",
    )
    resolve_config.add_argument("--project-dir", type=Path, required=True)
    resolve_config.add_argument(
        "--recipe",
        choices=("legacy-paper-v1",),
        default="legacy-paper-v1",
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
        help="Run intake and selected analysis routes from a JSON config file.",
    )
    run_pipeline.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON file with raw_dir, save_dir, experiment, analysis_id, and routes.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

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
        from classical_conditioning.run_config import load_pipeline_run_config

        config = load_pipeline_run_config(args.config.resolve())
        result = run_pipeline(config)
        print(f"Raw: {config.raw_dir}")
        print(f"Save: {config.save_dir}")
        print(f"Experiment: {config.experiment}")
        print(f"Recordings: {len(result.recording_ids)}")
        print(f"Intake completed: {len(result.intake_completed)}")
        print(f"Intake skipped: {len(result.intake_skipped)}")
        print(f"Intake failed: {len(result.intake_failed)}")
        if result.legacy_runner_status:
            print(f"Legacy analysis: {config.resolved_legacy_analysis_id()}")
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
            recipe_id=args.recipe,
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

    if args.command == "preprocess":
        if args.recipe == "legacy-paper-v1":
            from classical_conditioning.preprocessing.legacy_v1 import (
                preprocess_legacy_recording,
            )

            result = preprocess_legacy_recording(
                project_dir=args.project_dir,
                recording_id=args.recording_id,
                experiment_name=args.experiment,
                overwrite=args.overwrite,
            )
            print(f"Recording: {result.recording_id}")
            print(f"Recipe: {args.recipe}")
            print(f"Rows: {result.row_count:,}")
            print(f"Trials: CS={result.cs_trial_count}, US={result.us_trial_count}")
            print(f"Samples: {result.samples_path}")
            print(f"Summary: {result.summary_path}")
            return

        if args.recipe == "corrected-preprocess-v1":
            from classical_conditioning.preprocessing.corrected_v1 import (
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

        raise ValueError(f"Unsupported preprocessing recipe: {args.recipe}")
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

    if args.command == "legacy-logmedian":
        from classical_conditioning.analysis import build_legacy_logmedian

        result = build_legacy_logmedian(
            args.project_dir,
            args.recording_id,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Rows: {result.row_count:,}")
        print(f"Samples: {result.samples_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "legacy-standard-main":
        from classical_conditioning.analysis import build_legacy_standard_main

        result = build_legacy_standard_main(
            args.project_dir,
            args.recording_id,
            experiment_name=args.experiment,
            read_batch_rows=args.read_batch_rows,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        for alignment, path in result.samples_paths.items():
            print(
                f"{alignment}: rows={result.row_counts[alignment]:,}, "
                f"trials={result.trial_counts[alignment]}, samples={path}"
            )
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "legacy-scaled-vigor":
        from classical_conditioning.analysis import build_legacy_scaled_vigor

        result = build_legacy_scaled_vigor(
            args.project_dir,
            args.recording_id,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        for key, path in result.artifact_paths.items():
            print(f"{key}: rows={result.row_counts[key]:,}, artifact={path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "legacy-normalized-vigor":
        from classical_conditioning.analysis import build_legacy_normalized_vigor

        result = build_legacy_normalized_vigor(
            args.project_dir,
            args.recording_id,
            experiment_name=args.experiment,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        for alignment, path in result.artifact_paths.items():
            print(
                f"{alignment}: rows={result.row_counts[alignment]:,}, "
                f"artifact={path}"
            )
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "legacy-statistics":
        from classical_conditioning.analysis import build_legacy_statistics

        result = build_legacy_statistics(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            alignment=args.alignment,
            experiment_name=args.experiment,
            overwrite=args.overwrite,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: {args.recipe}")
        for name, path in result.artifact_paths.items():
            print(f"{name}: {path}")
        print(f"Model errors: {len(result.model_errors)}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "legacy-runner":
        from classical_conditioning.analysis import run_legacy_analysis_pipeline

        result = run_legacy_analysis_pipeline(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            experiment_name=args.experiment,
            alignment=args.alignment,
            read_batch_rows=args.read_batch_rows,
            overwrite=args.overwrite,
            run_statistics=not args.skip_statistics,
        )
        print(f"Analysis: {result.analysis_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Status: {result.status}")
        print(f"Manifest: {result.manifest_path}")
        for recording_id, entries in result.step_status.items():
            if isinstance(entries, dict):
                for step_name, state in entries.items():
                    print(f"{recording_id}:{step_name}={state}")
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

        metric_recipe = RUNNER_RECIPE_TO_METRIC_SOURCE[args.recipe]
        result = run_candidate_development_pipeline(
            args.project_dir,
            args.recording_id,
            analysis_id=args.analysis_id,
            experiment_name=args.experiment,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            metric_recipe=metric_recipe,
            runner_recipe=args.recipe,
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

    if args.command == "legacy-candidate-outcome-comparison":
        from classical_conditioning.analysis import (
            build_legacy_candidate_outcome_comparison,
        )

        result = build_legacy_candidate_outcome_comparison(
            args.project_dir,
            args.recording_id,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Recipe: {args.recipe}")
        print(f"Matched rows: {result.matched_row_count}")
        print(f"Matched outcomes: {result.matched_path}")
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

    if args.command == "plan-batch":
        from classical_conditioning.analysis.batch_work import (
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
        print(f"Recipe: batch-work-manifest-v1")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Rows: {result.row_count}")
        print(f"Complete: {result.complete_count}")
        print(f"Pending: {result.pending_count}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "execute-batch":
        from classical_conditioning.analysis.batch_work import (
            execute_batch_work,
        )

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
        from classical_conditioning.analysis.mixed_effects import (
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
        print(f"Recipe: candidate-mixed-effects-v1")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Model input: {result.model_input_path}")
        print(f"Coefficients: {result.coefficients_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "candidate-fish-permutation":
        from classical_conditioning.analysis.fish_permutation import (
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
        print(f"Recipe: candidate-fish-permutation-v1")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Fish effects: {result.fish_effects_path}")
        print(f"Population: {result.population_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "candidate-fish-bootstrap":
        from classical_conditioning.analysis.fish_bootstrap import (
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
        print(f"Recipe: candidate-fish-bootstrap-v1")
        print(f"Metric source: {args.metric_recipe}")
        print(f"Fish effects: {result.fish_effects_path}")
        print(f"Population: {result.population_path}")
        print(f"Summary: {result.summary_path}")
        return

    if args.command == "candidate-model-input":
        from classical_conditioning.analysis.model_input import (
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
        print(f"Recipe: candidate-model-input-v1")
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
            outcome_id=args.outcome,
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
        if isinstance(result, Path):
            print(f"Interactive figure: {result}")
        else:
            for output in result.outputs:
                print(f"Figure: {output}")
            print(f"Provenance: {result.sidecar}")
        return

    if args.command == "figure-legacy-review":
        from classical_conditioning.figures import (
            FigureMode,
            build_legacy_preprocessing_review_figure,
        )

        result = build_legacy_preprocessing_review_figure(
            args.project_dir,
            args.recording_id,
            mode=FigureMode(args.mode),
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Trial counts: {result.trial_counts}")
        for path in result.outputs:
            print(f"Figure: {path}")
        print(f"Summary: {result.summary_path}")
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

    if args.command == "compare-routes":
        from classical_conditioning.comparison import (
            write_preprocessing_route_comparison,
        )

        result = write_preprocessing_route_comparison(
            args.project_dir,
            args.recording_id,
            overwrite=args.overwrite,
        )
        print(f"Recording: {result.recording_id}")
        print(f"Legacy trial samples: {result.legacy_final_rows:,}")
        print(f"Candidate frame rows: {result.candidate_frame_rows:,}")
        print(f"Report: {result.output}")
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

    if args.command == "compare-legacy-pickle":
        from classical_conditioning.preprocessing.legacy_pickle_compare import (
            compare_legacy_pickle_to_parquet,
        )

        result = compare_legacy_pickle_to_parquet(
            args.pickle,
            args.parquet,
            args.output,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(f"Row counts equal: {result.row_counts_equal}")
        print(f"Scientific columns equal: {result.all_scientific_columns_equal}")
        if result.first_divergence is not None:
            print(f"First divergence: {result.first_divergence}")
        if result.original_frame_classification is not None:
            print(f"Original-frame classification: {result.original_frame_classification}")
            if (
                result.onset_vigor_correlation is not None
                and result.full_trial_vigor_correlation is not None
            ):
                print(
                    "Onset vigor corr: "
                    f"{result.onset_vigor_correlation:.4f}; "
                    "full-trial vigor corr: "
                    f"{result.full_trial_vigor_correlation:.4f}"
                )
            if result.vigor_correlation_after_rate_warp is not None:
                print(
                    "Vigor corr after rate-warp: "
                    f"{result.vigor_correlation_after_rate_warp:.4f}"
                )
            if result.lag_versus_predicted_rate_warp_correlation is not None:
                print(
                    "Lag vs predicted rate-warp corr: "
                    f"{result.lag_versus_predicted_rate_warp_correlation:.4f}"
                )
        print(f"Report: {result.report_path}")
        return

    if args.command == "activity-metrics":
        if args.recipe == "tail-candidate-development-v1":
            from classical_conditioning.preprocessing.candidates_v1 import (
                build_candidate_activity_metrics,
            )

            result = build_candidate_activity_metrics(
                args.project_dir,
                args.recording_id,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
        elif args.recipe == "tail-candidate-corrected-v1":
            from classical_conditioning.preprocessing.candidates_corrected_v1 import (
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
