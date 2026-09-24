"""Authenticated, fish-weighted learner-stratified Figure 4 panel data."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.metric_comparison import _verify_temporal_profiles
from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source
from classical_conditioning.analysis.trial_outcomes import _verify_inputs
from classical_conditioning.artifacts import artifact_staging, publish_transaction, sha256_file
from classical_conditioning.config import Alignment, get_experiment_spec
from classical_conditioning.config.domain import ConditionRole, Phase
from classical_conditioning.exceptions import ConfigurationError, SchemaValidationError
from classical_conditioning.figures.cohort_response import _load_primary_cohort
from classical_conditioning.figures.example_traces import METRIC_COLUMNS
from classical_conditioning.figures.signed_bout_heatmap import (
    BIN_WIDTH_S, SIGNAL, WINDOW_S, calculate_fish_heatmaps,
)


EXPERIMENTS = ("allDelay", "all3sTrace", "all10sTrace")
STRATA = (
    "Conditioned learner", "Conditioned nonlearner",
    "Control learner-flagged", "Control nonlearner",
)
TABLES = ("trial-bins", "fish-bins", "group-bins", "sample-flow")
RECIPE = "figure4-learner-signed-profiles/1.0"
MINIMUM_COVERAGE = 0.9


def _require_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ConfigurationError("Analysis ID must contain only letters, numbers, dot, underscore or hyphen.")


def load_classification_manifest(path: Path, metric_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a frozen table plus its hash-bound Gate L metadata."""
    path = path.resolve()
    metadata_path = path.with_suffix(".manifest.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if (metadata.get("table_sha256") != digest
            or metadata.get("input_metric_id") != metric_id
            or not metadata.get("classifier_execution_id")
            or not metadata.get("validation_mode")
            or set(metadata.get("cohort_hashes", {})) != set(EXPERIMENTS)
            or set(metadata.get("selection_assessments", {})) != set(EXPERIMENTS)):
        raise ConfigurationError("Classification manifest hash, metric, classifier, validation mode or cohort map is invalid.")
    for experiment_id, assessment in metadata["selection_assessments"].items():
        assessment_path = Path(assessment["path"]).resolve()
        if sha256_file(assessment_path) != assessment.get("sha256"):
            raise ConfigurationError(f"{experiment_id}: selection assessment summary hash is invalid.")
        assessment_summary = json.loads(assessment_path.read_text(encoding="utf-8"))
        if (assessment_summary.get("assessment_hash") != assessment.get("assessment_hash")
                or assessment_summary.get("selected_metric") != metric_id
                or assessment_summary.get("input_identity", {}).get("experiment") != experiment_id
                or assessment_summary.get("input_identity", {}).get("metric_id") != metric_id
                or assessment_summary.get("input_identity", {}).get("metric_recipe") != "tail-candidate-corrected"):
            raise ConfigurationError(f"{experiment_id}: selection assessment identity differs from the classifier.")
        filenames = {"technical": "technical-assessment.parquet",
                     "exploratory": "exploratory-assessment.parquet",
                     "rules": "legacy-rule-results.parquet",
                     "flow": "discarding-flow.parquet",
                     "details": "rule-details.parquet"}
        artifacts = assessment_summary.get("artifacts", {})
        if set(artifacts) != set(filenames):
            raise ConfigurationError(f"{experiment_id}: selection assessment artifact map is incomplete.")
        for name, filename in filenames.items():
            if sha256_file(assessment_path.parent / filename) != artifacts[name]:
                raise ConfigurationError(f"{experiment_id}: selection assessment {name} artifact has changed.")
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ConfigurationError("Classification table must be Parquet or CSV.")
    required = {"experiment_id", "condition_id", "fish_id", "classifier_label", "classification_eligible", "ineligible_reason"}
    if required - set(frame):
        raise SchemaValidationError(f"Classification table is missing {sorted(required - set(frame))}.")
    key = ["experiment_id", "condition_id", "fish_id"]
    if frame.duplicated(key).any() or frame[key].isna().any().any():
        raise SchemaValidationError("Classification fish keys must be complete and unique.")
    labels = frame["classifier_label"].astype(str)
    if not labels.isin(("Learner", "Non-learner", "Unclassified")).all():
        raise SchemaValidationError("classifier_label must be Learner, Non-learner or Unclassified.")
    eligibility = frame["classification_eligible"]
    if eligibility.isna().any() or not eligibility.isin((True, False)).all():
        raise SchemaValidationError("classification_eligible must be Boolean for every fish.")
    if ((labels.eq("Unclassified") == eligibility.astype(bool))).any():
        raise SchemaValidationError("Unclassified labels must be ineligible; classified labels must be eligible.")
    if (labels.eq("Unclassified") & frame["ineligible_reason"].fillna("").astype(str).str.strip().eq("")).any():
        raise SchemaValidationError("Unclassified fish require an ineligible reason.")
    if "input_metric_id" in frame and not frame["input_metric_id"].astype(str).eq(metric_id).all():
        raise SchemaValidationError("Fish rows contain a different classifier input metric.")
    for column in ("classifier_execution_id", "validation_mode"):
        if column in frame and not frame[column].astype(str).eq(str(metadata[column])).all():
            raise SchemaValidationError(f"Fish rows contain a different {column}.")
    if "cohort_hash" in frame:
        expected_hash = frame["experiment_id"].map(metadata["cohort_hashes"])
        if expected_hash.isna().any() or not frame["cohort_hash"].astype(str).eq(expected_hash.astype(str)).all():
            raise SchemaValidationError("Fish rows contain a different cohort hash.")
    metadata["table_path"] = str(path)
    metadata["metadata_path"] = str(metadata_path)
    metadata["metadata_sha256"] = sha256_file(metadata_path)
    return frame, metadata


def assign_plot_strata(fish: pd.DataFrame) -> pd.DataFrame:
    """Keep biological reference role while exposing control classifier flags."""
    result = fish.copy()
    role = result["cohort_role"].astype(str)
    label = result["classifier_label"].astype(str)
    result["plot_stratum"] = "Unclassified"
    result.loc[role.eq("conditioned") & label.eq("Learner"), "plot_stratum"] = STRATA[0]
    result.loc[role.eq("conditioned") & label.eq("Non-learner"), "plot_stratum"] = STRATA[1]
    result.loc[role.eq("reference") & label.eq("Learner"), "plot_stratum"] = STRATA[2]
    result.loc[role.eq("reference") & label.eq("Non-learner"), "plot_stratum"] = STRATA[3]
    return result


def verify_expected_us(protocol: pd.DataFrame, experiment_id: str) -> tuple[float, int]:
    """Resolve the paired-training US latency from recorded events and spec."""
    spec = get_experiment_spec(experiment_id)
    cycles = protocol.loc[protocol["Type"].astype(str).eq("Cycle")].sort_values("Beg", kind="stable")
    reinforcers = protocol.loc[protocol["Type"].astype(str).eq("Reinforcer"), "Beg"].to_numpy(dtype=float)
    if len(cycles) < 94:
        raise ConfigurationError(f"{experiment_id}: fewer than 94 recorded CS events.")
    observed: list[float] = []
    catch = set(spec.catch_trial_numbers(Alignment.CS))
    for trial in spec.analysis_trials:
        if trial.alignment is not Alignment.CS:
            continue
        onset = float(cycles.iloc[trial.trial_number - 1]["Beg"])
        latencies = (reinforcers - onset) / 1000.0
        in_window = latencies[(latencies >= 0) & (latencies <= 20.1)]
        if trial.phase is not Phase.TRAIN or trial.trial_number in catch:
            if len(in_window):
                raise ConfigurationError(f"{experiment_id}: non-US CS {trial.trial_number} contains a US event.")
        elif len(in_window) != 1:
            raise ConfigurationError(f"{experiment_id}: paired CS {trial.trial_number} has {len(in_window)} US events.")
        else:
            observed.append(float(in_window[0]))
    expected = next(condition for condition in spec.conditions if condition.role is ConditionRole.CONDITIONED)
    declared = np.asarray(expected.us_latency_s, dtype=float)
    measured = np.asarray(observed, dtype=float)
    if not len(measured) or len(measured) != len(declared) or np.max(np.abs(measured - declared)) > 0.1:
        raise ConfigurationError(
            f"{experiment_id}: recorded paired-US latency {np.median(measured) if len(measured) else 'missing'} s "
            f"disagrees with ExperimentSpec {np.median(declared) if len(declared) else 'missing'} s; "
            "resolve the protocol definition before Figure 4 rendering."
        )
    if np.ptp(measured) > 0.1:
        raise ConfigurationError(f"{experiment_id}: paired-US timing is not consistent across trials.")
    return float(np.median(measured)), len(measured)


def profile_groups(experiment_id: str) -> list[dict[str, Any]]:
    spec = get_experiment_spec(experiment_id)
    groups = [
        {"group_type": "block", "group_name": label, "group_order": index,
         "trials": tuple(trials)}
        for index, (label, trials) in enumerate(spec.trial_blocks(Alignment.CS))
    ]
    catches = spec.catch_trial_numbers(Alignment.CS)
    groups.append({"group_type": "pooled_catch", "group_name": "All catch trials",
                   "group_order": len(groups), "trials": catches})
    groups.extend({"group_type": "individual_catch", "group_name": f"Catch {trial}",
                   "group_order": len(groups) + index, "trials": (trial,)}
                  for index, trial in enumerate(catches))
    return groups


def summarize_trial_bins(trial_bins: pd.DataFrame, experiment_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median over trials within fish, then median/IQR over equally weighted fish."""
    assignments = pd.DataFrame(
        {"trial_number": trial, "group_type": group["group_type"],
         "group_name": group["group_name"], "group_order": group["group_order"]}
        for group in profile_groups(experiment_id) for trial in group["trials"]
    )
    expanded = trial_bins.merge(assignments, on="trial_number", how="inner", validate="many_to_many")
    keys = ["experiment_id", "recording_id", "fish_id", "condition_id", "cohort_role",
            "classifier_label", "plot_stratum", "group_type", "group_name", "group_order", "time_s"]
    fish_bins = (
        expanded.groupby(keys, dropna=False, observed=True, sort=False)
        .agg(signed=("signed_log_vigor", "median"),
             signed_trials=("signed_log_vigor", "count"),
             movement=("movement_probability", "median"),
             movement_trials=("movement_probability", "count"))
        .reset_index()
    )
    displayed = fish_bins.loc[fish_bins["plot_stratum"].isin(STRATA)]
    group_keys = ["experiment_id", "plot_stratum", "group_type", "group_name", "group_order", "time_s"]
    summary_rows = []
    for key, part in displayed.groupby(group_keys, observed=True, sort=False):
        row = dict(zip(group_keys, key, strict=True))
        row["total_group_fish"] = int(part["fish_id"].nunique())
        for signal in ("signed", "movement"):
            valid = part.loc[np.isfinite(part[signal]), signal]
            row[f"{signal}_median"] = float(valid.median()) if len(valid) else np.nan
            row[f"{signal}_q25"] = float(valid.quantile(.25)) if len(valid) else np.nan
            row[f"{signal}_q75"] = float(valid.quantile(.75)) if len(valid) else np.nan
            row[f"{signal}_fish"] = int(len(valid))
            row[f"{signal}_trials"] = int(part.loc[valid.index, f"{signal}_trials"].sum())
        summary_rows.append(row)
    # Retain empty classifier groups so all panels render, even when every fish
    # in one experiment is unclassified or a group has no valid signed bins.
    grid = pd.DataFrame(
        {"experiment_id": experiment_id, "plot_stratum": stratum,
         "group_type": group["group_type"], "group_name": group["group_name"],
         "group_order": group["group_order"], "time_s": time}
        for group in profile_groups(experiment_id)
        for stratum in STRATA
        for time in sorted(trial_bins["time_s"].unique())
    )
    value_columns = ["total_group_fish", *(f"{signal}_{field}"
                     for signal in ("signed", "movement")
                     for field in ("median", "q25", "q75", "fish", "trials"))]
    group_bins = grid.merge(pd.DataFrame(summary_rows, columns=group_keys + value_columns), on=group_keys, how="left",
                            validate="one_to_one")
    for column in ("total_group_fish", "signed_fish", "signed_trials",
                   "movement_fish", "movement_trials"):
        group_bins[column] = pd.to_numeric(group_bins[column], errors="coerce").fillna(0).astype(int)
    return fish_bins, group_bins


def _read_recording(project_dir: Path, recording_id: str, metric_id: str, metric_recipe: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    verified = _verify_temporal_profiles(project_dir, recording_id, source)
    _, metric_path, movement_path, protocol_path, _, _ = _verify_inputs(
        project_dir, recording_id, metric_recipe=metric_recipe
    )
    metrics = pq.read_table(metric_path, columns=["FrameID", "AbsoluteTime", METRIC_COLUMNS[metric_id]]).to_pandas()
    movement = pq.read_table(movement_path, columns=["FrameID", "AbsoluteTime", "valid", "moving", "bout_id"]).to_pandas()
    protocol = pq.read_table(protocol_path).to_pandas()
    cycles = protocol.loc[protocol["Type"].astype(str).eq("Cycle")].sort_values(
        "Beg", kind="stable"
    ).reset_index(drop=True)
    if len(cycles) < 94:
        raise SchemaValidationError(f"{recording_id}: Figure 4 requires 94 recorded CS cycles.")
    profiles = pq.read_table(verified.path, columns=["Recording ID", "Trial type", "Trial number", "Time bin center (s)", "Metric ID", "Valid expected fraction", "Movement probability"]).to_pandas()
    profiles = profiles.loc[(profiles["Trial type"] == "CS") & (profiles["Metric ID"] == metric_id)
                            & profiles["Time bin center (s)"].between(-20, 20, inclusive="neither")].copy()
    signed = calculate_fish_heatmaps(metrics, movement, cycles, recording_id=recording_id,
                                     metric_ids=(metric_id,))
    keys = ["Recording ID", "Trial number", "Time bin center (s)", "Metric ID"]
    if profiles.duplicated(keys).any() or signed.duplicated(keys).any():
        raise SchemaValidationError("Duplicate Figure 4 trial/time bins.")
    joined = signed.merge(profiles[keys + ["Valid expected fraction", "Movement probability"]],
                          on=keys, how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise SchemaValidationError(f"Missing authenticated temporal coverage for {recording_id}.")
    covered = pd.to_numeric(joined["Valid expected fraction"], errors="coerce") >= MINIMUM_COVERAGE
    joined["signed_log_vigor"] = pd.to_numeric(joined["Signed log vigor"], errors="coerce").where(covered)
    joined["movement_probability"] = pd.to_numeric(joined["Movement probability"], errors="coerce").where(covered)
    joined["time_s"] = joined["Time bin center (s)"].astype(float)
    joined["trial_number"] = joined["Trial number"].astype(int)
    joined["recording_id"] = recording_id
    joined["coverage"] = joined["Valid expected fraction"].astype(float)
    inputs = [{"path": str(path), "sha256": sha256_file(path)}
              for path in (metric_path, movement_path, protocol_path, verified.path)]
    return joined[["recording_id", "trial_number", "time_s", "signed_log_vigor", "movement_probability", "coverage"]], protocol, profiles, inputs


def analyze_figure4(
    project_dir: Path, *, analysis_id: str, metric_id: str,
    cohort_ids: dict[str, str], learner_manifest: Path,
    experiment_dirs: dict[str, Path] | None = None, overwrite: bool = False,
) -> Path:
    """Publish signed trial, fish and group bins before any figure is rendered."""
    _require_id(analysis_id)
    if metric_id not in METRIC_COLUMNS or set(cohort_ids) != set(EXPERIMENTS):
        raise ConfigurationError("Figure 4 requires one supported metric and three experiment cohort IDs.")
    manifest, classifier = load_classification_manifest(learner_manifest, metric_id)
    experiment_dirs = experiment_dirs or {}
    cohorts = {}
    fish_frames = []
    for experiment_id in EXPERIMENTS:
        root = experiment_dirs.get(experiment_id, project_dir).resolve()
        cohort = _load_primary_cohort(root, cohort_ids[experiment_id])
        if cohort.experiment_id != experiment_id or classifier["cohort_hashes"][experiment_id] != cohort.cohort_hash:
            raise ConfigurationError(f"{experiment_id}: learner manifest and authenticated cohort disagree.")
        if cohort.metric_recipe != "tail-candidate-corrected":
            raise ConfigurationError("Figure 4 requires corrected candidate metrics.")
        cohorts[experiment_id] = (root, cohort)
        spec = get_experiment_spec(experiment_id)
        role_by_condition = {item.condition_id: (
            "reference" if item.role is ConditionRole.CONTROL else "conditioned"
        ) for item in spec.conditions}
        fish_frames.append(pd.DataFrame({
            "experiment_id": experiment_id, "recording_id": list(cohort.recording_ids),
            "fish_id": [cohort.fish_by_recording[item] for item in cohort.recording_ids],
            "condition_id": [cohort.condition_by_recording[item] for item in cohort.recording_ids],
            "cohort_role": [role_by_condition[cohort.condition_by_recording[item]] for item in cohort.recording_ids],
        }))
    fish = pd.concat(fish_frames, ignore_index=True)
    key = ["experiment_id", "condition_id", "fish_id"]
    if fish.duplicated(key).any() or fish.duplicated(["experiment_id", "recording_id"]).any():
        raise SchemaValidationError("Cohort fish or recording identity is not unique.")
    joined = fish.merge(manifest, on=key, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise SchemaValidationError("Classification manifest and primary cohort fish do not match exactly.")
    if "recording_id_y" in joined:
        if not joined["recording_id_x"].astype(str).eq(joined["recording_id_y"].astype(str)).all():
            raise SchemaValidationError("Classifier changed a fish's recording identity.")
        joined = joined.rename(columns={"recording_id_x": "recording_id"}).drop(columns="recording_id_y")
    if "cohort_role_y" in joined:
        if not joined["cohort_role_x"].eq(joined["cohort_role_y"]).all():
            raise SchemaValidationError("Classifier changed a fish's cohort role.")
        joined = joined.rename(columns={"cohort_role_x": "cohort_role"}).drop(columns="cohort_role_y")
    joined = assign_plot_strata(joined)
    flow = joined[["experiment_id", "recording_id", "fish_id", "condition_id", "cohort_role",
                   "classifier_label", "classification_eligible", "ineligible_reason", "plot_stratum"]].copy()
    trial_parts: list[pd.DataFrame] = []
    inputs = [{"path": classifier["table_path"], "sha256": classifier["table_sha256"]},
              {"path": classifier["metadata_path"], "sha256": classifier["metadata_sha256"]}]
    inputs.extend({"path": item["path"], "sha256": item["sha256"]}
                  for item in classifier["selection_assessments"].values())
    inputs.extend({"path": item["path"], "sha256": item["sha256"]}
                  for _, cohort in cohorts.values()
                  for item in getattr(cohort, "input_artifacts", ()))
    timing = {}
    for experiment_id, (root, cohort) in cohorts.items():
        times = []
        for recording_id in cohort.recording_ids:
            bins, protocol, _, record_inputs = _read_recording(root, recording_id, metric_id, cohort.metric_recipe)
            inputs.extend(record_inputs)
            identity = joined.loc[(joined["experiment_id"] == experiment_id)
                                  & (joined["recording_id"] == recording_id)].iloc[0]
            for column in ("experiment_id", "fish_id", "condition_id", "cohort_role", "classifier_label", "plot_stratum"):
                bins[column] = identity[column]
            trial_parts.append(bins)
            if identity["cohort_role"] == "conditioned":
                time, count = verify_expected_us(protocol, experiment_id)
                times.append(time)
        if not times or max(times) - min(times) > .1:
            raise ConfigurationError(f"{experiment_id}: expected-US time cannot be verified across conditioned fish.")
        timing[experiment_id] = {"expected_us_s": float(np.median(times)),
                                 "conditioned_recordings": len(times), "source": "authenticated paired-training Reinforcer events"}
    trial_bins = pd.concat(trial_parts, ignore_index=True)
    fish_parts, group_parts = [], []
    for experiment_id in EXPERIMENTS:
        subset = trial_bins.loc[trial_bins["experiment_id"] == experiment_id]
        fish_part, group_part = summarize_trial_bins(subset, experiment_id)
        fish_parts.append(fish_part)
        group_parts.append(group_part)
    tables = {"trial-bins": trial_bins, "fish-bins": pd.concat(fish_parts, ignore_index=True),
              "group-bins": pd.concat(group_parts, ignore_index=True), "sample-flow": flow}
    for frame in tables.values():
        frame["metric_id"] = metric_id
        frame["cohort_hash"] = frame["experiment_id"].map(classifier["cohort_hashes"])
        frame["selection_assessment_hash"] = frame["experiment_id"].map({
            experiment: record["assessment_hash"]
            for experiment, record in classifier["selection_assessments"].items()
        })
        frame["classifier_execution_id"] = classifier["classifier_execution_id"]
        frame["validation_mode"] = classifier["validation_mode"]
    root = project_dir.resolve() / "Processed data" / "Analyses" / analysis_id / "figure4"
    destinations = {name: root / f"{name}.parquet" for name in TABLES}
    summary_path = root / "analysis.json"
    marker_path = root / "complete.json"
    all_paths = [*destinations.values(), summary_path, marker_path]
    if not overwrite and any(path.exists() for path in all_paths):
        raise FileExistsError("Figure 4 analysis outputs already exist; use --overwrite to replace them.")
    with artifact_staging(root.parent, prefix=".figure4-") as stage:
        staged = {name: stage / path.name for name, path in destinations.items()}
        for name, frame in tables.items():
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), staged[name], compression="zstd")
        table_hashes = {name: sha256_file(path) for name, path in staged.items()}
        summary = {
            "recipe": RECIPE, "analysis_id": analysis_id, "metric_id": metric_id,
            "signal": SIGNAL, "window_s": list(WINDOW_S), "bin_width_s": BIN_WIDTH_S,
            "baseline_s": [-20.0, 0.0], "minimum_coverage": MINIMUM_COVERAGE,
            "aggregation": "trial_median_within_fish_then_equal_fish_median_and_iqr",
            "classifier_execution_id": classifier["classifier_execution_id"],
            "validation_mode": classifier["validation_mode"], "classifier_manifest_sha256": classifier["table_sha256"],
            "cohort_ids": cohort_ids, "cohort_hashes": classifier["cohort_hashes"],
            "selection_assessments": classifier["selection_assessments"],
            "expected_us": timing, "inputs": inputs,
            "tables": {name: {"path": str(destinations[name]), "sha256": table_hashes[name],
                              "rows": len(tables[name])} for name in TABLES},
            "scientific_status": "descriptive_same_data", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (stage / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (stage / "complete.json").write_text(json.dumps({"status": "complete", "recipe": RECIPE,
                "summary_sha256": sha256_file(stage / "analysis.json"), "tables_sha256": table_hashes}, indent=2) + "\n", encoding="utf-8")
        publish_transaction(tuple((staged[name], destinations[name]) for name in TABLES)
                            + ((stage / "analysis.json", summary_path), (stage / "complete.json", marker_path)),
                            stage, overwrite=overwrite)
    return summary_path


def load_figure4_analysis(summary_path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Authenticate persisted analysis data; rendering never redoes analysis."""
    summary_path = summary_path.resolve()
    root = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    marker = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if (summary.get("recipe") != RECIPE or marker.get("recipe") != RECIPE
            or marker.get("status") != "complete"
            or marker.get("summary_sha256") != sha256_file(summary_path)):
        raise ConfigurationError("Figure 4 analysis marker or summary is invalid.")
    tables = {}
    for name in TABLES:
        entry = summary["tables"][name]
        path = (root / f"{name}.parquet").resolve()
        if (Path(entry["path"]).resolve() != path or sha256_file(path) != entry["sha256"]
                or marker["tables_sha256"].get(name) != entry["sha256"]):
            raise ConfigurationError(f"Figure 4 {name} table has changed.")
        tables[name] = pd.read_parquet(path)
        if len(tables[name]) != entry["rows"]:
            raise ConfigurationError(f"Figure 4 {name} row count has changed.")
    return summary, tables
