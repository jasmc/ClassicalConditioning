"""Technical and exploratory legacy-rule assessment without deleting fish.

The exploratory result is a projection of historical rules onto the selected
corrected metric.  It is never a reviewed cohort or a classifier result.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.analysis.candidate_runner import (
    _verify_corrected_preprocess,
    _verify_temporal,
)
from classical_conditioning.analysis.movement_state import (
    METRIC_IDS,
    resolve_candidate_metric_source,
)
from classical_conditioning.analysis.trial_outcomes import verify_candidate_trial_outcomes
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ClassicalConditioningError, ConfigurationError
from classical_conditioning.inventory import build_recording_inventory
from classical_conditioning.paths import assert_project_dir_allowed


# Source references are data as well as nearby comments: every published rule
# row identifies the historical operation whose behavior is being projected.
LEGACY_SOURCES: dict[str, str] = {
    "readable_fish": "Archive/historical-scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py::run_discard (lines 48, 1656-1680)",
    "last_us": "Archive/historical-scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py::check_viability (lines 1467-1485)",
    "train_us": "Archive/historical-scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py::check_train (lines 1486-1497)",
    "retrain_us": "Archive/historical-scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py::check_retrain (lines 1498-1505)",
    "baseline_bouts": "Archive/historical-scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py::check_baseline (lines 1507-1520, guard 1707-1711)",
    "cr_bouts": "Archive/historical-scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py::check_cr (lines 1521-1533, guard 1721-1724)",
    "discard_propagation": "Archive/historical-scripts/3_FishGrouping.py::main (lines 679-803); Archive/historical-scripts/4_ScaledVigorPlotting.py::filter_discarded_fish_ids (lines 155-188); Archive/historical-scripts/5_NormalizedVigorPlotting.py::APPLY_FISH_DISCARD (line 67)",
    "learner_inputs": "Archive/historical-scripts/6_LearnersQuantification.py::prepare_data/_filter_fish_by_trials (lines 776-934); Archive/historical-scripts/6_LearnersQuantification_new.py (lines 1024-1182); Archive/historical-scripts/6_LearnersQuantification_improved.py (lines 433-535); Archive/historical-scripts/6_LearnersQuantification_WIP.py (lines 511-624)",
}
RULE_ORDER = tuple(LEGACY_SOURCES)
DEFAULT_METRIC = "legacy_distal_angular_speed"
FIVE_TRIAL_BLOCKS = {
    name: frozenset(range(start, start + 5))
    for name, start in (
        ("Early Pre-Train", 5), ("Late Pre-Train", 10),
        ("Early Train", 15), ("Train 2", 20), ("Train 3", 25),
        ("Train 4", 30), ("Train 5", 35), ("Train 6", 40),
        ("Train 7", 45), ("Train 8", 50), ("Train 9", 55),
        ("Late Train", 60), ("Early Test", 65), ("Test 2", 70),
        ("Test 3", 75), ("Test 4", 80), ("Test 5", 85),
        ("Late Test", 90),
    )
}
QC_BLOCKS = {
    "Late Pre-Train": FIVE_TRIAL_BLOCKS["Late Pre-Train"],
    "Early Test": FIVE_TRIAL_BLOCKS["Early Test"],
    "Late Test": FIVE_TRIAL_BLOCKS["Late Test"],
}
LEARNER_EPOCHS = {
    "pretrain": ("Early Pre-Train", "Late Pre-Train"),
    "late_train_early_test": (
        "Train 6", "Train 7", "Train 8", "Train 9",
        "Late Train", "Early Test",
    ),
    "late_test": ("Test 5", "Late Test"),
}


@dataclass(frozen=True)
class DiscardAssessmentResult:
    output_dir: Path
    technical_path: Path
    exploratory_path: Path
    rules_path: Path
    flow_path: Path
    summary_path: Path
    assessment_hash: str


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_technical_policy(path: Path | None) -> dict[str, Any]:
    """An absent policy is a draft evidence audit, never an inclusion decision."""
    policy = {"approval_status": "draft", "min_matched_frames": 1,
              "min_valid_frame_fraction": 0.0}
    if path is not None:
        incoming = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(incoming, dict) or set(incoming) - {
            "approval_status", "min_matched_frames", "min_valid_frame_fraction",
            "approved_by", "approved_at",
        }:
            raise ConfigurationError("Technical policy has unknown or invalid fields.")
        policy.update(incoming)
    if policy["approval_status"] not in {"draft", "approved"}:
        raise ConfigurationError("Technical policy approval_status must be draft or approved.")
    if (not isinstance(policy["min_matched_frames"], int)
            or isinstance(policy["min_matched_frames"], bool)
            or policy["min_matched_frames"] < 1):
        raise ConfigurationError("min_matched_frames must be a positive integer.")
    fraction = policy["min_valid_frame_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ConfigurationError("min_valid_frame_fraction must be between zero and one.")
    if policy["approval_status"] == "approved" and not (
        policy.get("approved_by") and policy.get("approved_at")
    ):
        raise ConfigurationError("An approved technical policy needs reviewer and date.")
    return policy


def _bout_in_window(
    times_ms: np.ndarray, moving: np.ndarray, event_ms: int,
    start_s: float, end_s: float,
) -> bool:
    # Historical window_mask uses inclusive ge/le; test Bout on frames, not
    # bout starts. Source: 1_Preprocessing...Discarding.py::window_mask.
    relative = (times_ms - event_ms) / 1000.0
    return bool(np.any(moving & (relative >= start_s) & (relative <= end_s)))


def evaluate_legacy_bouts(
    movement: pd.DataFrame, protocol: pd.DataFrame, *, experiment: str,
) -> tuple[dict[str, tuple[str, str]], list[dict[str, Any]]]:
    """Project executable preprocessing checks onto authenticated bouts."""
    times = movement["AbsoluteTime"].to_numpy(dtype=np.int64)
    moving = movement["moving"].to_numpy(dtype=bool) & movement["valid"].to_numpy(dtype=bool)
    events = protocol.loc[protocol["Type"].astype(str).isin(["Cycle", "Reinforcer"])].sort_values("Beg", kind="stable")
    us = events.loc[events["Type"].astype(str).eq("Reinforcer")].reset_index(drop=True)
    cs = events.loc[events["Type"].astype(str).eq("Cycle")].reset_index(drop=True)
    result: dict[str, tuple[str, str]] = {}
    details: list[dict[str, Any]] = []

    # Source: 1_Preprocessing...Discarding.py::check_viability. The final
    # observed US event is used, not a fabricated scheduled event.
    if us.empty:
        result["last_us"] = ("fail", "no_us_trials")
    else:
        last = us.iloc[-1]
        end_value = pd.to_numeric(last["End"], errors="coerce")
        end_s = (float(end_value) - float(last["Beg"])) / 1000.0
        has_bout = _bout_in_window(times, moving, int(last["Beg"]), 0, 5)
        result["last_us"] = (
            ("pass", "") if np.isfinite(end_s) and end_s >= 0.4 and has_bout
            else ("fail", "last_us_end_missing_or_short" if not np.isfinite(end_s) or end_s < 0.4 else "last_us_no_bout")
        )
        details.append({"rule_id": "last_us", "trial_number": len(us), "observed_end_s": end_s,
                        "bout_present": has_bout})

    # Sources: check_train/check_retrain in the same file. All migrated assays
    # currently declare Train US; an absent Train group is a failure.
    spec = get_experiment_spec_for_events(experiment)
    train_numbers = spec["train"]
    retrain_numbers = spec["retrain"]
    for rule, numbers in (("train_us", train_numbers), ("retrain_us", retrain_numbers)):
        observed = [(number, us.iloc[number - 1]) for number in sorted(numbers) if number <= len(us)]
        if not observed:
            result[rule] = ("pass", "") if rule == "retrain_us" else ("fail", "no_train_us")
            continue
        missing_bout = [number for number, event in observed
                        if not _bout_in_window(times, moving, int(event["Beg"]), 0, 5)]
        result[rule] = ("fail", f"no_bout_trials:{','.join(map(str, missing_bout))}") if missing_bout else ("pass", "")
        details.extend({"rule_id": rule, "trial_number": number,
                        "bout_present": number not in missing_bout} for number, _ in observed)

    # Sources: check_baseline/check_cr and their empty-data guards. With no
    # selected CS rows the original script skipped both checks.
    selected_observed = [number for numbers in QC_BLOCKS.values() for number in numbers if number <= len(cs)]
    if not selected_observed:
        result["baseline_bouts"] = ("pass", "legacy_empty_cs_bypass")
        result["cr_bouts"] = ("pass", "legacy_empty_cs_bypass")
    else:
        cr_end_s = get_experiment_spec(experiment).conditioned_response_window.end_s
        for rule, start, end in (("baseline_bouts", -15.0, 0.0), ("cr_bouts", 0.0, cr_end_s)):
            low_blocks = []
            for block, numbers in QC_BLOCKS.items():
                observed = [(number, cs.iloc[number - 1]) for number in sorted(numbers) if number <= len(cs)]
                bouts = [number for number, event in observed
                         if _bout_in_window(times, moving, int(event["Beg"]), start, end)]
                details.append({"rule_id": rule, "block": block, "bout_trial_count": len(bouts),
                                "observed_trial_count": len(observed)})
                if len(bouts) < 3:
                    low_blocks.append(block)
            result[rule] = ("fail", "fewer_than_three:" + ",".join(low_blocks)) if low_blocks else ("pass", "")
    return result, details


def get_experiment_spec_for_events(experiment: str) -> dict[str, set[int]]:
    """Use declared US blocks; late test US events are not Re-Train events."""
    spec = get_experiment_spec(experiment)
    return {
        "train": {trial.trial_number for trial in spec.analysis_trials
                  if trial.alignment.value == "US" and "Train" in trial.block_10_name
                  and "Re-Train" not in trial.block_10_name},
        "retrain": {trial.trial_number for trial in spec.analysis_trials
                    if trial.alignment.value == "US" and "Re-Train" in trial.block_10_name},
    }


def evaluate_learner_inputs(outcomes: pd.DataFrame, *, metric_id: str) -> tuple[str, str, list[dict[str, Any]]]:
    """One merged pre-fit learner prerequisite, never a classifier label."""
    frame = outcomes.loc[(outcomes["alignment"] == "CS") & (outcomes["metric_id"] == metric_id)].copy()
    if frame.empty:
        return "not_evaluable", "no_cs_outcomes", []
    required = ("trial_number", "block_10_name", "baseline_total_activity", "response_total_activity")
    if any(column not in frame for column in required):
        return "not_evaluable", "missing_outcome_columns", []
    baseline = pd.to_numeric(frame["baseline_total_activity"], errors="coerce")
    response = pd.to_numeric(frame["response_total_activity"], errors="coerce")
    # Sources: prepare_data in all four 6_LearnersQuantification scripts; the
    # improved route logs raw means and WIP logs normalized vigor. Their merged
    # input requirement is finite positive baseline, response, and ratio.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = response / baseline
    valid = (
        frame["block_10_name"].notna()
        & np.isfinite(baseline) & np.isfinite(response)
        & (baseline > 0) & (response > 0)
        & np.isfinite(ratio) & (ratio > 0)
    )
    frame = frame.loc[valid].copy()
    numbers = set(pd.to_numeric(frame["trial_number"], errors="coerce").dropna().astype(int))
    counts = {name: len(numbers & trials) for name, trials in FIVE_TRIAL_BLOCKS.items()}
    details = [{"block": name, "valid_trial_count": count} for name, count in counts.items()]
    missing = []
    for epoch, blocks in LEARNER_EPOCHS.items():
        if sum(counts[name] for name in blocks) < 6:
            missing.append(f"{epoch}_fewer_than_six")
        missing.extend(f"{name}_fewer_than_three" for name in blocks if counts[name] < 3)
    return ("fail", ";".join(missing), details) if missing else ("pass", "", details)


def _technical_row(
    record: dict[str, Any], *, selected: bool, project_dir: Path,
    experiment: str, metric_recipe: str, policy: dict[str, Any],
    processing_status: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    recording_id = record.get("recording_id")
    condition_by_source = {
        item.source_name.lower(): item.condition_id
        for item in get_experiment_spec(experiment).conditions
    }
    source_condition = str(record.get("condition_id") or "").lower()
    reasons = []
    if record["status"] != "COMPLETE":
        reasons.append("inventory_" + record["status"].lower())
    if source_condition not in condition_by_source:
        reasons.append("invalid_condition")
    schema = record.get("tracking_schema")
    if schema is not None and not schema.get("ok"):
        reasons.append("invalid_tracking_header")
    if processing_status and processing_status != "ready":
        reasons.append("processing_" + processing_status)
    lineage: dict[str, str] = {}
    matched = None
    valid_fraction = None
    protocol_event_count = None
    protocol_outside_count = None
    if selected and recording_id and record["status"] == "COMPLETE":
        try:
            source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
            if source.requires_corrected_preprocess:
                lineage["corrected_preprocess"] = _verify_corrected_preprocess(project_dir, recording_id)
                summary = json.loads((project_dir / "Quality checks" / recording_id /
                                      "corrected_preprocessing_summary.json").read_text(encoding="utf-8"))
                matched = int(summary["matched_frame_count"])
                valid_fraction = float(summary["frame_valid_count"]) / matched if matched else 0.0
                timing = summary.get("protocol_timing", {})
                protocol_event_count = int(timing.get("event_count", 0))
                protocol_outside_count = int(timing.get("events_outside_acquisition", 0))
            lineage["temporal"] = _verify_temporal(project_dir, recording_id, experiment, source)
            verified = verify_candidate_trial_outcomes(project_dir, recording_id, metric_recipe=metric_recipe)
            lineage["trial_outcomes"] = sha256_file(verified.marker_path)
        except (ClassicalConditioningError, FileNotFoundError, ValueError, KeyError, OSError) as error:
            reasons.append("missing_or_invalid_derived_artifact")
            lineage["error"] = str(error)
    if matched is not None and matched < policy["min_matched_frames"]:
        reasons.append("too_few_matched_frames")
    if valid_fraction is not None and valid_fraction < policy["min_valid_frame_fraction"]:
        reasons.append("too_few_valid_frames")
    if protocol_event_count == 0:
        reasons.append("no_protocol_events")
    ready = selected and not reasons and bool(lineage.get("trial_outcomes"))
    row = {
        "recording_id": recording_id or "",
        "recording_name": record["recording_name"],
        "condition_id": condition_by_source.get(source_condition, ""),
        "source_condition": source_condition,
        "inventory_status": record["status"],
        "selected_for_run": selected,
        "processing_status": processing_status or "unknown",
        "tracking_header_ok": None if schema is None else bool(schema.get("ok")),
        "matched_frame_count": matched,
        "valid_frame_fraction": valid_fraction,
        "protocol_event_count": protocol_event_count,
        "protocol_events_outside_acquisition": protocol_outside_count,
        "technical_ready": ready,
        "technical_reason": ";".join(dict.fromkeys(reasons)),
        "primary_candidate": ready if policy["approval_status"] == "approved" else None,
        "policy_status": policy["approval_status"],
    }
    return row, lineage


def assess_discarding(
    raw_dir: Path, project_dir: Path, *, analysis_id: str, experiment: str,
    metric_id: str, metric_recipe: str = "tail-candidate-corrected",
    technical_policy_path: Path | None = None,
    disabled_rules: Iterable[str] = (),
    recording_ids: Iterable[str] | None = None,
    inventory: dict[str, Any] | None = None,
    processing_statuses: dict[str, str] | None = None,
) -> DiscardAssessmentResult:
    """Run both stages in order and replace one authenticated audit bundle."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError("analysis_id must be a path-safe identifier.")
    raw_dir = raw_dir.resolve()
    project_dir = project_dir.resolve()
    assert_project_dir_allowed(raw_dir, project_dir)
    if metric_id not in METRIC_IDS.values():
        raise ConfigurationError(f"Unknown assessment metric: {metric_id}")
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    disabled = frozenset(disabled_rules)
    if disabled - set(RULE_ORDER):
        raise ConfigurationError(f"Unknown disabled checks: {sorted(disabled - set(RULE_ORDER))}")
    policy = load_technical_policy(technical_policy_path)
    inventory = inventory or build_recording_inventory(raw_dir, hash_files=True, inspect_tracking_headers=True)
    selected = set(recording_ids) if recording_ids is not None else {
        str(record["recording_id"]) for record in inventory["records"] if record.get("recording_id")
    }
    records = list(inventory["records"])
    discovered = {record.get("recording_id") for record in records}
    for missing_id in sorted(selected - discovered):
        records.append({
            "recording_id": missing_id, "recording_name": missing_id,
            "condition_id": None, "status": "NOT_DISCOVERED",
            "tracking_schema": None,
        })
    if not records:
        raise ConfigurationError("No recordings were found or requested for discarding assessment.")
    statuses = processing_statuses or {}
    technical_rows: list[dict[str, Any]] = []
    exploratory_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    lineage: dict[str, Any] = {}
    for record in records:
        rid = record.get("recording_id")
        is_selected = rid in selected
        technical, hashes = _technical_row(
            record, selected=is_selected, project_dir=project_dir,
            experiment=experiment, metric_recipe=metric_recipe, policy=policy,
            processing_status=statuses.get(str(rid)),
        )
        technical_rows.append(technical)
        lineage[record["recording_name"]] = hashes
        outcomes = None
        checks: dict[str, tuple[str, str]] = {}
        if technical["technical_ready"]:
            try:
                base = project_dir / "Processed data" / str(rid)
                movement = pq.read_table(base / source.movement_artifact_name).to_pandas()
                protocol = pq.read_table(base / "stimulus_events.parquet").to_pandas()
                outcomes = pq.read_table(base / source.trial_outcomes_name).to_pandas()
                checks, bout_details = evaluate_legacy_bouts(movement, protocol, experiment=experiment)
                details.extend({"recording_name": record["recording_name"], **item} for item in bout_details)
            except (ClassicalConditioningError, FileNotFoundError, ValueError, KeyError, OSError) as error:
                lineage[record["recording_name"]]["behavior_error"] = str(error)
        # Source: 1_Preprocessing...Discarding.py::run_discard excluded fish
        # whose processed file could not be opened; here inputs stay in place.
        if outcomes is not None:
            checks["readable_fish"] = ("pass", "")
        elif is_selected and record["status"] == "COMPLETE":
            checks["readable_fish"] = ("fail", "processed_fish_unreadable_or_unavailable")
        else:
            checks["readable_fish"] = ("not_evaluable", "missing_authenticated_outcomes")
        # Sources: Stage 3 uses the list for display only; Stage 4 applies it;
        # Stage 5 defaults it off. This is provenance, not a second fish gate.
        checks["discard_propagation"] = ("pass", "same_preprocessing_list_not_reapplied")
        if outcomes is not None:
            learner_status, learner_reason, learner_details = evaluate_learner_inputs(outcomes, metric_id=metric_id)
            checks["learner_inputs"] = (learner_status, learner_reason)
            details.extend({"recording_name": record["recording_name"], "rule_id": "learner_inputs", **item}
                           for item in learner_details)
        cumulative = bool(technical["technical_ready"])
        for rule_id in RULE_ORDER:
            status, reason = checks.get(rule_id, ("not_evaluable", "technical_or_source_unavailable"))
            enabled = rule_id not in disabled
            if enabled and status != "pass":
                cumulative = False
            rule_rows.append({
                "recording_id": rid or "", "recording_name": record["recording_name"],
                "selected_for_run": is_selected,
                "rule_id": rule_id, "source": LEGACY_SOURCES[rule_id],
                "enabled": enabled, "status": status if enabled else "disabled",
                "reason": reason if enabled else "disabled_for_exploration",
                "cumulative_pass": cumulative,
            })
        current_rules = [row for row in rule_rows[-len(RULE_ORDER):] if row["enabled"]]
        failed = any(row["status"] == "fail" for row in current_rules)
        unevaluable = any(row["status"] == "not_evaluable" for row in current_rules)
        exploratory_rows.append({
            "recording_id": rid or "", "recording_name": record["recording_name"],
            "condition_id": technical["condition_id"],
            "selected_for_run": is_selected, "technical_ready": bool(technical["technical_ready"]),
            "exploratory_pass": cumulative,
            "exploratory_status": (
                "pass" if cumulative else "fail" if failed else
                "not_evaluable" if unevaluable or outcomes is None else "fail"
            ),
            "failed_or_unavailable_rules": ";".join(
                row["rule_id"] for row in rule_rows[-len(RULE_ORDER):]
                if row["enabled"] and row["status"] != "pass"
            ),
        })

    source_identity = {
        "analysis_id": analysis_id, "experiment": experiment, "metric_id": metric_id,
        "metric_recipe": metric_recipe, "policy": policy,
        "disabled_rules": sorted(disabled),
        "inventory_hash": inventory["records_sha256"], "lineage": lineage,
        "selected_recording_ids": sorted(selected), "processing_statuses": statuses,
    }
    assessment_hash = _digest(source_identity)
    # Input identity lives in the summary, not in a path suffix. A changed
    # assessment replaces the same derived paths; reviewed cohorts stay separate.
    output_dir = project_dir / "Processed data" / "Discarding" / analysis_id
    paths = {
        "technical": output_dir / "technical-assessment.parquet",
        "exploratory": output_dir / "exploratory-assessment.parquet",
        "rules": output_dir / "legacy-rule-results.parquet",
        "flow": output_dir / "discarding-flow.parquet",
        "details": output_dir / "rule-details.parquet",
        "summary": output_dir / "assessment-summary.json",
    }
    if paths["summary"].is_file():
        saved = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if saved.get("assessment_hash") == assessment_hash:
            if all(path.is_file() for path in paths.values()) and all(
                saved.get("artifacts", {}).get(name) == sha256_file(path)
                for name, path in paths.items() if name != "summary"
            ):
                return DiscardAssessmentResult(output_dir, paths["technical"], paths["exploratory"],
                                               paths["rules"], paths["flow"], paths["summary"], assessment_hash)
            raise ConfigurationError("Existing discarding assessment failed integrity checks.")
    technical_df = pd.DataFrame(technical_rows)
    exploratory_df = pd.DataFrame(exploratory_rows)
    rules_df = pd.DataFrame(rule_rows)
    flow = [{
        "step_index": 0, "rule_id": "technical",
        "selected_fish_count": len(selected),
        "cumulative_pass_count": int(technical_df.loc[
            technical_df["selected_for_run"], "technical_ready"
        ].sum()),
        "not_evaluable_count": 0,
    }]
    for index, rule_id in enumerate(RULE_ORDER):
        reached = rules_df.loc[
            (rules_df["rule_id"] == rule_id) & rules_df["selected_for_run"]
        ]
        flow.append({"step_index": index + 1, "rule_id": rule_id,
                     "selected_fish_count": len(selected),
                     "cumulative_pass_count": int(reached["cumulative_pass"].sum()),
                     "not_evaluable_count": int(reached["status"].eq("not_evaluable").sum())})
    flow_df = pd.DataFrame(flow)
    for frame in (technical_df, exploratory_df, rules_df, flow_df):
        frame["assessment_hash"] = assessment_hash
    details_df = pd.DataFrame(details)
    if details_df.empty:
        details_df = pd.DataFrame(columns=["recording_name", "rule_id"])
    details_df["assessment_hash"] = assessment_hash
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with artifact_staging(project_dir, prefix=f".{analysis_id}-discarding-") as stage:
        staged = {name: stage / path.name for name, path in paths.items()}
        for name, frame in (("technical", technical_df), ("exploratory", exploratory_df),
                            ("rules", rules_df), ("flow", flow_df), ("details", details_df)):
            frame.to_parquet(staged[name], index=False)
        artifact_hashes = {name: sha256_file(staged[name]) for name in staged if name != "summary"}
        write_json_atomic(staged["summary"], {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "assessment_hash": assessment_hash, "scientific_status": "exploratory",
            "primary_cohort_changed": False, "policy": policy,
            "selected_metric": metric_id, "enabled_rules": [r for r in RULE_ORDER if r not in disabled],
            "legacy_sources": LEGACY_SOURCES, "input_identity": source_identity,
            "technical_ready_count": int(technical_df["technical_ready"].sum()),
            "exploratory_pass_count": int(exploratory_df["exploratory_pass"].sum()),
            "artifacts": artifact_hashes,
        })
        publish_transaction(
            tuple((staged[name], path) for name, path in paths.items()),
            stage,
            overwrite=True,
        )
    return DiscardAssessmentResult(output_dir, paths["technical"], paths["exploratory"],
                                   paths["rules"], paths["flow"], paths["summary"], assessment_hash)
