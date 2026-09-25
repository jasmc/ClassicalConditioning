"""Freeze the complete 3sTrace cohort and package provisional WIP labels.

The cohort policy is an explicit exploratory all-complete decision authorized
for this run. The archived WIP classifier remains a same-data description.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cohort import freeze_cohort_manifest, load_cohort_manifest, logical_cohort_hash


EXPERIMENT = "all3sTrace"
ANALYSIS_ID = "all3sTrace-full"
COHORT_ID = "all3sTrace-full-exploratory"
METRIC = "tail_length_weighted_angular_l1"
METRICS = (METRIC, "whole_tail_xy_mean_speed_normalized", "legacy_distal_angular_speed")
VARIANT = "legacy-wip"


def freeze(project: Path, *, cohort_id: str = COHORT_ID) -> Path:
    project = project.resolve()
    pipeline_path = project / "Metadata" / f"{ANALYSIS_ID}_pipeline_run.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    selected = tuple(pipeline["recording_ids"])
    if (pipeline.get("status") != "complete" or not selected
            or set(selected) != set(pipeline["active_recording_ids"])
            or pipeline.get("stage_errors") or not pipeline.get("candidate_runner_status")):
        raise ValueError("The 3sTrace candidate pipeline is not complete for every selected fish")
    inventory = json.loads((project / "Metadata" / "recording_inventory.json").read_text(encoding="utf-8"))
    if not inventory.get("source_hashes_included"):
        raise ValueError("The source inventory lacks file hashes")
    records = {str(row["recording_id"]): row for row in inventory["records"] if row.get("recording_id")}
    if set(selected) != {name for name, row in records.items()
                         if row["status"] == "COMPLETE" and row["condition_id"] in {"trace", "control"}}:
        raise ValueError("Pipeline fish differ from the complete fixed trace/control inventory")
    if {records[name]["condition_id"] for name in selected} != {"trace", "control"}:
        raise ValueError("Both fixed trace and matched controls are required")
    assessment_path = Path(pipeline["selection_assessment"])
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if (assessment.get("selected_metric") != METRIC
            or assessment.get("input_identity", {}).get("experiment") != EXPERIMENT):
        raise ValueError("The selected-metric assessment is missing or mismatched")
    reviewed_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for name in selected:
        source_qc = project / "Metadata" / f"{name}_source_manifest.json"
        if not source_qc.is_file():
            raise ValueError(f"Missing source manifest for {name}")
        rows.append({
            "experiment_id": EXPERIMENT, "recording_id": name, "fish_id": name,
            "condition_id": records[name]["condition_id"], "technical_valid": True,
            "technical_exclusion_reason": "", "behavioral_engagement": None,
            "behavioral_engagement_reason": "", "primary_included": True,
            "sensitivity_population_ids": [], "review_status": "approved",
            "reviewer": "Codex-exploratory-all-complete", "reviewed_at": reviewed_at,
            "source_qc_artifact_id": f"Metadata/{name}_source_manifest.json",
        })
    reviewed = pd.DataFrame(rows)
    result = freeze_cohort_manifest(
        project, reviewed, cohort_id=cohort_id,
        policy_id="exploratory-all-complete-triplets",
    )
    return result.manifest_path


def classify(project: Path, comparison_dir: Path, *, cohort_id: str = COHORT_ID,
             analysis_id: str = "figure4-3strace-exploratory",
             assessment_summary: Path | None = None,
             classifier_execution_id: str = "legacy-wip-3strace-tail-l1-exploratory",
             metric_id: str = METRIC) -> Path:
    project, comparison_dir = project.resolve(), comparison_dir.resolve()
    if metric_id not in METRICS:
        raise ValueError(f"Unsupported 3sTrace classifier metric: {metric_id}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ValueError("Analysis ID contains unsafe path characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", classifier_execution_id):
        raise ValueError("Classifier execution ID contains unsafe path characters")
    cohort_path = project / "Processed data" / "Cohorts" / cohort_id / "cohort-manifest.parquet"
    comparison_path = comparison_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    cohort = load_cohort_manifest(project, cohort_id)
    cohort_hash = logical_cohort_hash(cohort)
    if (comparison.get("experiment_id") != EXPERIMENT or comparison.get("metric_id") != metric_id
            or comparison.get("cohort_hash") != cohort_hash
            or comparison.get("cohort_sha256") != sha256_file(cohort_path)):
        raise ValueError("Legacy comparison does not match the frozen 3sTrace cohort and metric")
    variant = next((item for item in comparison["variants"] if item["variant"] == VARIANT), None)
    if variant is None or variant["status"] != "completed":
        raise ValueError("The archived WIP variant did not complete")
    if sha256_file(comparison_dir / f"{VARIANT}.parquet") != variant["result_sha256"]:
        raise ValueError("The archived WIP result hash differs from its comparison")
    compared = pd.read_csv(comparison_dir / "fish-comparison.csv")
    selected = cohort.loc[cohort["primary_included"].astype(bool),
                          ["experiment_id", "recording_id", "fish_id", "condition_id"]]
    key = ["experiment_id", "condition_id", "fish_id"]
    joined = selected.merge(compared, on=key, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Legacy fish comparison and frozen primary cohort differ")
    eligible = joined[f"{VARIANT}_status"].eq("classified")
    flag = joined[f"{VARIANT}_learner"]
    if flag[eligible].isna().any():
        raise ValueError("Classified fish have missing WIP learner flags")
    label = pd.Series("Unclassified", index=joined.index)
    label.loc[eligible & flag.eq(True)] = "Learner"
    label.loc[eligible & flag.eq(False)] = "Non-learner"
    execution_id = classifier_execution_id
    validation_mode = "descriptive_same_data"
    output = joined[key + ["recording_id"]].copy()
    output["classifier_label"] = label
    output["classification_eligible"] = eligible.astype(bool)
    output["ineligible_reason"] = eligible.map({True: "", False: "legacy-wip did not return a classification"})
    output["input_metric_id"] = metric_id
    output["cohort_hash"] = cohort_hash
    output["classifier_execution_id"] = execution_id
    output["validation_mode"] = validation_mode
    output = output.sort_values(key).reset_index(drop=True)
    if assessment_summary is None:
        pipeline = json.loads((project / "Metadata" / f"{ANALYSIS_ID}_pipeline_run.json").read_text(encoding="utf-8"))
        assessment_path = Path(pipeline["selection_assessment"]).resolve()
    else:
        assessment_path = assessment_summary.resolve()
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if (assessment.get("selected_metric") != metric_id
            or assessment.get("input_identity", {}).get("experiment") != EXPERIMENT
            or set(assessment.get("input_identity", {}).get("selected_recording_ids", ()))
            != set(output["recording_id"])):
        raise ValueError("Learner labels require an assessment for the same metric and fish")
    destination = project / "Processed data" / "Analyses" / analysis_id / "learner-labels.csv"
    metadata_path = destination.with_suffix(".manifest.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Exploratory learner manifest already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    metadata = {
        "schema": "figure4-classifier-manifest/1.0",
        "scientific_status": "descriptive_provisional_legacy_rule",
        "table_sha256": sha256_file(destination), "input_metric_id": metric_id,
        "classifier_execution_id": execution_id, "validation_mode": validation_mode,
        "cohort_hashes": {EXPERIMENT: cohort_hash},
        "selection_assessments": {EXPERIMENT: {
            "path": str(assessment_path), "sha256": sha256_file(assessment_path),
            "assessment_hash": assessment["assessment_hash"],
        }},
        "source_comparison": str(comparison_path),
        "source_comparison_sha256": sha256_file(comparison_path),
        "algorithm_sha256": variant["algorithm_sha256"],
        "algorithm_config_sha256": variant["config_sha256"],
        "conditioned_learner_count": int((output["condition_id"].eq("trace")
                                           & output["classifier_label"].eq("Learner")).sum()),
        "control_flag_count": int((output["condition_id"].eq("control")
                                   & output["classifier_label"].eq("Learner")).sum()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "classify"))
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--cohort-id", default=COHORT_ID)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--metric", choices=METRICS, default=METRIC)
    parser.add_argument("--analysis-id", default="figure4-3strace-exploratory")
    parser.add_argument("--assessment-summary", type=Path)
    parser.add_argument("--classifier-execution-id",
                        default="legacy-wip-3strace-tail-l1-exploratory")
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze(args.project_dir, cohort_id=args.cohort_id)
    else:
        if args.comparison_dir is None:
            parser.error("classify requires --comparison-dir")
        result = classify(args.project_dir, args.comparison_dir, cohort_id=args.cohort_id,
                          analysis_id=args.analysis_id,
                          assessment_summary=args.assessment_summary,
                          classifier_execution_id=args.classifier_execution_id,
                          metric_id=args.metric)
    print(result)


if __name__ == "__main__":
    main()
