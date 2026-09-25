"""Check that the complete 3sTrace review is authenticated before J: removal."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from classical_conditioning.analysis.cohort_outcomes import load_cohort_trial_outcomes
from classical_conditioning.analysis.figure4 import (
    load_classification_manifest, load_figure4_analysis,
)
from classical_conditioning.analysis.inference.learning_onset import load_learning_onset_analysis
from classical_conditioning.analysis.trial_outcomes import trial_outcome_settings_for_experiment
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cohort import load_cohort_manifest, logical_cohort_hash


METRIC = "tail_length_weighted_angular_l1"
COHORT = "all3sTrace-full-exploratory"


def sha256_with_retry(path: Path) -> str:
    for attempt in range(3):
        try:
            return sha256_file(path)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError("Unreachable SHA-256 retry state")


def verify(project: Path, source: Path, report_path: Path) -> dict:
    project, source, report_path = project.resolve(), source.resolve(), report_path.resolve()
    expected_project = Path(r"F:\Digested Data\all3sTrace-full-v1").resolve()
    expected_source = Path(r"J:\Digested Data\all3sTrace-full-v1").resolve()
    if project != expected_project or source != expected_source:
        raise ValueError("Move readiness is restricted to the exact 3sTrace J: and F: projects")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (Path(report["source_project"]).resolve() != source
            or Path(report["destination_project"]).resolve() != project
            or report["completed_file_count"] != len(report["files"])):
        raise ValueError("Verified-copy report does not match the exact project paths")
    source_files = {str(item["relative_path"]): item for item in report["files"]}
    if len(source_files) != report["completed_file_count"]:
        raise ValueError("Verified-copy report has duplicate file paths")
    for relative, record in source_files.items():
        path = (source / relative).resolve()
        if source not in path.parents or not path.is_file():
            raise ValueError(f"J: source member is outside the project or missing: {relative}")
        if path.stat().st_size != record["size_bytes"] or sha256_with_retry(path).upper() != record["sha256"].upper():
            raise ValueError(f"J: source member changed since verified transfer: {relative}")

    staging_report_path = report_path.with_name("verified-staging-copy.json")
    staging_report = json.loads(staging_report_path.read_text(encoding="utf-8"))
    if (Path(staging_report["source_project"]).resolve() != source
            or Path(staging_report["destination_project"]).resolve() != project
            or staging_report["folder_count"] != 2
            or staging_report["file_count"] != len(staging_report["files"])):
        raise ValueError("Verified staging-copy report does not match the exact project paths")
    staging_files = {str(item["relative_path"]): item for item in staging_report["files"]}
    if len(staging_files) != staging_report["file_count"] or len(staging_files) != 8:
        raise ValueError("Verified staging-copy report has an unexpected file set")
    expected_staging_folders = {
        ".20230307_12-intake-be77eadf21eb4411b2c33512fd892950",
        ".20230309_11-intake-2fb3ba2619a945f692660c78b09d5466",
    }
    for relative, record in staging_files.items():
        if Path(relative).parts[0] not in expected_staging_folders:
            raise ValueError(f"Unexpected staging folder: {relative}")
        for root in (source, project):
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"Staging member escaped or is missing: {path}")
            if path.stat().st_size != record["size_bytes"] or sha256_with_retry(path).upper() != record["sha256"].upper():
                raise ValueError(f"Staging member changed since verified transfer: {path}")
    actual_source_files = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    if actual_source_files != set(source_files) | set(staging_files):
        raise ValueError("J: project contains files outside the two verified transfer reports")

    pipeline = json.loads((project / "Metadata" / "all3sTrace-full_pipeline_run.json").read_text(encoding="utf-8"))
    selected = set(pipeline["recording_ids"])
    if (pipeline["status"] != "complete" or len(selected) != 59
            or selected != set(pipeline["active_recording_ids"])
            or pipeline.get("stage_errors") or not pipeline.get("candidate_runner_status")):
        raise ValueError("F: candidate pipeline is not complete for all 59 fish")
    inventory = json.loads((project / "Metadata" / "recording_inventory.json").read_text(encoding="utf-8"))
    complete = {row["recording_id"]: row for row in inventory["records"] if row["status"] == "COMPLETE"}
    if selected != set(complete):
        raise ValueError("F: candidate fish and complete raw inventory differ")
    condition_counts = {condition: sum(row["condition_id"] == condition for row in complete.values())
                        for condition in ("trace", "control")}
    if condition_counts != {"trace": 40, "control": 19}:
        raise ValueError(f"F: cohort condition counts differ: {condition_counts}")
    expected_trial_settings = trial_outcome_settings_for_experiment("all3sTrace")
    for recording_id in selected:
        summary_path = project / "Quality checks" / recording_id / "candidate-trial-outcomes-corrected_summary.json"
        trial_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (trial_summary.get("experiment") != "all3sTrace"
                or trial_summary.get("config") != expected_trial_settings):
            raise ValueError(f"{recording_id}: trial outcomes do not use the 0–13 s response window")

    cohort = load_cohort_manifest(project, COHORT)
    primary = cohort.loc[cohort["primary_included"].astype(bool)]
    if len(primary) != 59 or set(primary["recording_id"]) != selected:
        raise ValueError("Frozen exploratory cohort does not contain all 59 complete fish")
    cohort_hash = logical_cohort_hash(cohort)
    outcomes, _ = load_cohort_trial_outcomes(project, COHORT)
    if set(outcomes["recording_id"].astype(str)) != selected:
        raise ValueError("Frozen cohort outcomes do not cover all complete fish")
    comparison_dir = project / "Processed data" / "Analyses" / "figure3-3strace-window13" / "legacy-tail-l1"
    comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    wip = next((item for item in comparison["variants"] if item["variant"] == "legacy-wip"), None)
    if (comparison.get("cohort_hash") != cohort_hash or comparison.get("metric_id") != METRIC
            or wip is None or wip.get("status") != "completed"):
        raise ValueError("3sTrace legacy-wip comparison is incomplete or mismatched")
    labels_path = project / "Processed data" / "Analyses" / "figure4-3strace-window13" / "learner-labels.csv"
    labels, classifier = load_classification_manifest(labels_path, METRIC, ("all3sTrace",))
    if (len(labels) != 59 or classifier["cohort_hashes"] != {"all3sTrace": cohort_hash}
            or classifier["classifier_execution_id"] != "legacy-wip-3strace-tail-l1-window13-exploratory"):
        raise ValueError("3sTrace provisional classifier manifest is incomplete or mismatched")
    figure4_path = project / "Processed data" / "Analyses" / "figure4-3strace-window13" / "figure4" / "analysis.json"
    figure4, tables = load_figure4_analysis(figure4_path)
    if (figure4["analysis_scope"] != "partial_assay_review"
            or figure4["scientific_status"] != "descriptive_provisional_legacy_rule"
            or figure4["cohort_hashes"] != {"all3sTrace": cohort_hash}
            or len(tables["sample-flow"]) != 59):
        raise ValueError("Figure 4B analysis is incomplete or mismatched")
    learning_id = "all3sTrace-full-learning-onset-window13"
    _, learning = load_learning_onset_analysis(project, learning_id)
    if learning.get("cohort_hash") != cohort_hash:
        raise ValueError("Learning-onset analysis uses a different cohort")

    figure_paths = [
        project / "Figures" / "PNG" / "20230307_12" / f"figure-1-F-3strace_{METRIC}.png",
        *(project / "Figures" / "PNG" / "Analyses" / "figure2-3strace-review"
          / f"figure-2B-3strace_{kind}_{METRIC}.png" for kind in ("signed", "coverage")),
        *(project / "Figures" / "PNG" / "Analyses" / "figure2-3strace-window13"
          / f"cohort-{kind}-ratio_{METRIC.replace('_', '-')}_total-activity.png"
          for kind in ("selected-block", "trial")),
        project / "Figures" / "PNG" / "Analyses" / "figure3-3strace-window13"
        / f"figure-3-3strace-review_{METRIC}.png",
        *(project / "Figures" / "PNG" / "Analyses" / "figure4-3strace-window13"
          / "all3sTrace" / f"{kind}_{METRIC}.png" for kind in (
              "figure-4", "supplement-single-catches", "supplement-movement",
              "supplement-coverage", "supplement-single-catch-movement",
              "supplement-single-catch-coverage")),
        *(project / "Figures" / "PNG" / "Analyses" / learning_id / f"{kind}.png"
          for kind in ("learning-diagnostics", "learning-onset")),
    ]
    missing = [str(path) for path in figure_paths if not path.is_file()]
    if missing:
        raise ValueError(f"3sTrace review figures are missing: {missing}")
    return {
        "status": "ready_to_remove_j_3strace_project",
        "source_project": str(source), "destination_project": str(project),
        "source_completed_files_reverified": len(source_files),
        "staging_files_reverified_on_both_drives": len(staging_files),
        "fish": condition_counts, "cohort_hash": cohort_hash,
        "classifier_version": "legacy-wip", "metric_id": METRIC,
        "conditioned_learners": int((labels["condition_id"].eq("trace")
                                        & labels["classifier_label"].eq("Learner")).sum()),
        "control_flagged": int((labels["condition_id"].eq("control")
                                 & labels["classifier_label"].eq("Learner")).sum()),
        "figure_count": len(figure_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(r"F:\Digested Data\all3sTrace-full-v1"))
    parser.add_argument("--source-dir", type=Path, default=Path(r"J:\Digested Data\all3sTrace-full-v1"))
    parser.add_argument("--transfer-report", type=Path, default=Path(__file__).resolve().parents[1]
                        / "outputs" / "trace-transfer-review" / "verified-copy.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.project_dir, args.source_dir, args.transfer_report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
