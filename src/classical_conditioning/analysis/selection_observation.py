"""Source-bound historical selection projections, with no exclusion outputs.

Corrected outcomes are explicitly projections, not original normalized-vigor
tables. Pure legacy completeness functions are read from their actual sources;
we never import a script's filesystem, plotting or model-fitting entry points.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.analysis.observation import publish_observation

LEGACY_ROOT = Path(__file__).resolve().parents[3] / "legacy" / "scripts"
VARIANTS = {
    "nominal": "6_LearnersQuantification.py",
    "new": "6_LearnersQuantification_new.py",
    "improved": "6_LearnersQuantification_improved.py",
    "wip": "6_LearnersQuantification_WIP.py",
}


def legacy_settings(path: Path) -> tuple[ast.Module, dict[str, Any]]:
    """Read literal globals and instantiated AnalysisConfig coverage settings."""
    tree = ast.parse(path.read_text())
    globals_: dict[str, Any] = {}
    config: dict[str, Any] = {}
    for node in tree.body:
        value = getattr(node, "value", None)
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and value is not None:
            try:
                globals_[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
            if target.id == "analysis_cfg" and isinstance(value, ast.Call):
                for kw in value.keywords:
                    try:
                        config[kw.arg] = globals_[kw.value.id] if isinstance(kw.value, ast.Name) else ast.literal_eval(kw.value)
                    except (KeyError, ValueError, TypeError):
                        pass
    return tree, {**globals_, "analysis_cfg": config}


def _pure_function(tree: ast.Module, name: str, **globals_):
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"pd": pd, "np": np, "CategoricalDtype": pd.CategoricalDtype, **globals_}
    code = "from __future__ import annotations\n" + ast.unparse(node)
    exec(compile(code, f"<legacy-observation:{name}>", "exec"), namespace)
    return namespace[name]


def observe_normalized_input(project_dir: Path, output_dir: Path, input_path: Path,
                             required_blocks: list[str]) -> dict[str, Any]:
    """Audit a supplied post-scope, block-remapped legacy per-trial table.

    The caller names the blocks because their choice belongs to a particular
    historical run. No latest-file discovery, condition alias or inferred
    window/remapping is used. This is evidence, never a new model input.
    """
    if not required_blocks or len(set(required_blocks)) != len(required_blocks):
        raise ValueError("Supply the original distinct required blocks")
    digest = sha256_file(input_path)
    if input_path.suffix == ".parquet":
        frame = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        frame = pd.read_csv(input_path)
    else:
        raise ValueError("Legacy audit input must be Parquet or CSV, never a pickle")
    required = {"Fish", "Exp.", "Trial number", "Block name", "Normalized vigor"}
    if required - set(frame):
        raise ValueError(f"Missing legacy columns: {sorted(required - set(frame))}")
    source = LEGACY_ROOT / "5_NormalizedVigorPlotting.py"
    tree, settings = legacy_settings(source)
    if settings["APPLY_FISH_DISCARD"] or settings["age_filter"] != ["all"] or settings["setup_color_filter"] != ["all"]:
        raise ValueError("Non-default list/metadata filters require their original run inputs")
    baseline, response = _pure_function(tree, "select_baseline_response_columns")(frame)
    complete = frame[["Normalized vigor", "Trial number", "Block name", baseline, response]].notna().all(axis=1)
    counts = frame.loc[complete].groupby(["Fish", "Block name"], observed=True).size()
    passing = set(counts[counts >= settings["minimum_trials_per_fish_per_block"]].reset_index()["Fish"])
    lme = complete & frame["Fish"].isin(passing)
    # The actual boxplot path selects blocks before broad dropna and medians.
    stats_rows = frame["Block name"].isin(required_blocks) & frame.notna().all(axis=1)
    medians = (frame.loc[stats_rows].groupby(["Fish", "Exp.", "Block name"], observed=True)["Normalized vigor"]
               .median().reset_index())
    block_counts = medians.groupby(["Fish", "Exp."], observed=True)["Block name"].nunique()
    complete_pairs = set(block_counts[block_counts.eq(len(required_blocks))].index)
    trial = frame[["Fish", "Exp.", "Trial number", "Block name"]].copy()
    trial["input_row"] = np.arange(len(frame))
    trial["lme_required_fields_present"] = complete
    trial["lme_contributed"] = lme
    trial["lme_reason"] = np.where(lme, "", np.where(complete, "no_block_meets_minimum", "missing_required_fields"))
    trial["block_summary_contributed"] = stats_rows
    trial["wilcoxon_branch_contributed"] = stats_rows & pd.Series(
        [(fish, condition) in complete_pairs for fish, condition in zip(frame["Fish"], frame["Exp."])], index=frame.index)
    fish = frame[["Fish", "Exp."]].drop_duplicates().copy()
    fish["any_block_pass"] = fish["Fish"].isin(passing)
    fish["complete_required_blocks"] = [(f, c) in complete_pairs for f, c in zip(fish["Fish"], fish["Exp."])]
    if sha256_file(input_path) != digest:
        raise ValueError("Legacy audit input changed during observation")
    return publish_observation(project_dir, output_dir, tables={"trial-contributions": trial,
                               "fish-contributions": fish, "available-block-medians": medians},
                               identity={"input_path": str(input_path), "input_sha256": digest,
                                         "input_stage": "post_metadata_scope_and_block_remapping",
                                         "source_sha256": sha256_file(source), "required_blocks": required_blocks,
                                         "minimum_trials": settings["minimum_trials_per_fish_per_block"],
                                         "membership_decisions_applied": False})


def project_later_checks(outcomes: pd.DataFrame, metric_id: str) -> pd.DataFrame:
    """Return check evidence only; no selected rows or fish list is exposed."""
    rows: list[dict[str, Any]] = []
    normal_path = LEGACY_ROOT / "5_NormalizedVigorPlotting.py"
    tree, settings = legacy_settings(normal_path)
    minimum = settings["minimum_trials_per_fish_per_block"]
    frame = outcomes.loc[outcomes["alignment"].eq("CS") & outcomes["metric_id"].eq(metric_id)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["rule_id", "status", "reason"])
    required = ["trial_number", "block_10_name", "baseline_total_activity", "response_total_activity"]
    if set(required) - set(frame):
        raise ValueError("Projection is missing normalized-vigor input fields")
    numbers = pd.to_numeric(frame["trial_number"], errors="coerce")
    baseline = pd.to_numeric(frame["baseline_total_activity"], errors="coerce")
    response = pd.to_numeric(frame["response_total_activity"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = response / baseline
    # Legacy prepare_main_df uses dropna, not a finite/positive test.
    valid = (numbers.notna() & frame["block_10_name"].notna()
             & baseline.notna() & response.notna() & ratio.notna())
    block_names, selected_blocks, _, width = _pure_function(tree, "get_block_config")(frame["block_10_name"].nunique())
    if width == 1 or not block_names:
        # The archived block remapping has no comparable layout for this input.
        labels = pd.Series(None, index=frame.index, dtype=object)
    else:
        start = numbers.min()
        labels = numbers.map(lambda n: block_names[int((n - start) // width)]
                             if pd.notna(n) and 0 <= int((n - start) // width) < len(block_names) else None)

    def emit(rule, status, reason, observed, parameters, source=normal_path, enabled=True):
        rows.append({"rule_id": rule, "status": status, "reason": reason,
                     "observed": json.dumps(observed, sort_keys=True, allow_nan=False),
                     "parameters": json.dumps(parameters, sort_keys=True),
                     "source": str(source.relative_to(LEGACY_ROOT.parent.parent)),
                     "source_sha256": sha256_file(source), "enabled_in_legacy": enabled,
                     "scope": "corrected_outcome_projection", "affects_membership": False})

    emit("normalized_missing_rows", "pass" if valid.all() else "fail", "rows_missing_required_fields" if not valid.all() else "",
         {"input_rows": len(frame), "remaining_rows": int(valid.sum())}, {"operation": "dropna_required_fields"})
    counts = frame.loc[valid].groupby("block_10_name", observed=True).size()
    any_pass = bool((counts >= minimum).any())
    emit("normalized_any_block", "pass" if any_pass else "fail", "" if any_pass else "no_block_meets_minimum",
         {str(k): int(v) for k, v in counts.items()}, {"minimum_trials": minimum, "logic": "ANY", "block_unit": "10-trial"})
    # Block summary functions use broad dropna on their input, then medians;
    # unrelated legacy metadata is absent from corrected outcomes.
    emit("normalized_block_statistics", "not_evaluable", "original_pooled_table_required_for_broad_dropna",
         {}, {"operation": "dropna_all_columns_then_fish_block_median"})
    present = set(labels.loc[valid].dropna())
    emit("normalized_paired_blocks", "not_evaluable" if width == 1 else ("pass" if set(selected_blocks) <= present else "fail"),
         "projection_only_original_pooled_dropna_not_available", {"blocks_present": sorted(present)},
         {"required_blocks": selected_blocks, "logic": "ALL"})
    emit("normalized_window_nan_fraction", "not_evaluable", "original_window_nan_fractions_required",
         {}, {"maximum_nan_fraction": settings["MAX_NAN_FRAC_PER_WINDOW"]},
         enabled=settings["APPLY_MAX_NAN_FRAC_PER_WINDOW"])

    for variant, filename in VARIANTS.items():
        path = LEGACY_ROOT / filename
        learner_tree, learner_settings = legacy_settings(path)
        cfg = learner_settings["analysis_cfg"]
        needed = {"features_to_use", "pretrain_to_train_end_earlytest_blocks", "train_end_earlytest_to_late_test_blocks",
                  "min_pretrain_trials", "min_latetraindearlytest_trials", "min_late_test_trials", "min_trials_per_5trial_block_in_epoch"}
        if not needed <= cfg.keys():
            emit(f"learner_{variant}_coverage", "not_evaluable", "unresolved_instantiated_configuration", {}, {}, path)
            continue
        # Each script constructs five-trial blocks from its minimum trial.
        prepared = pd.DataFrame({"Trial number": numbers, "Block name": frame["block_10_name"],
                                 "Fish_ID": "observed-fish"})
        # _create_5_trial_blocks derives the experiment layout from block count;
        # reconstructing it from incomplete input would change its semantics.
        try:
            from classical_conditioning.config.experiments import get_experiment_spec
            spec = get_experiment_spec(str(frame["experiment_id"].iloc[0]))
            trial_blocks: dict[str, list[int]] = {}
            for trial in spec.analysis_trials:
                if trial.alignment.value == "CS":
                    trial_blocks.setdefault(trial.block_10_name, []).append(trial.trial_number)
            create = _pure_function(learner_tree, "_create_5_trial_blocks",
                                    EPOCH_BLOCK_TRIALS=learner_settings.get("EPOCH_BLOCK_TRIALS", 5),
                                    exp_config=SimpleNamespace(names_cs_blocks_10=list(trial_blocks),
                                                               trials_cs_blocks_10=list(trial_blocks.values())))
            prepared = create(prepared)
            eligible = valid.copy()
            if variant == "wip" and cfg.get("use_log_transform", True):
                eligible &= np.isfinite(ratio) & ratio.gt(0)
            with contextlib.redirect_stdout(io.StringIO()):
                retained = _pure_function(learner_tree, "_filter_fish_by_trials")(
                    prepared.loc[eligible].copy(), SimpleNamespace(**cfg))
            emit(f"learner_{variant}_coverage", "pass" if len(retained) else "fail",
                 "" if len(retained) else "insufficient_selected_epoch_or_block_rows",
                 {"input_rows": len(frame), "prepared_rows": int(eligible.sum()), "remaining_rows": len(retained)},
                 {key: cfg[key] for key in sorted(needed)}, path)
        except (ValueError, KeyError, NameError, IndexError) as error:
            emit(f"learner_{variant}_coverage", "not_evaluable", str(error), {}, cfg, path)
        emit(f"learner_{variant}_features", "not_evaluable", "deferred_until_model_runs", {},
             {"minimum_fish_with_all_features": learner_settings.get("MIN_FISH_WITH_ALL_FEATURES")}, path)
    return pd.DataFrame(rows)


def publish_selection_observation(
    project_dir: Path, output_dir: Path, *, inventory: pd.DataFrame,
    historical_rules: pd.DataFrame, details: pd.DataFrame,
    outcomes_by_recording: dict[str, pd.DataFrame], metric_id: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Attach independent later checks to the full inventory, without decisions."""
    inventory = inventory.copy()
    inventory["decision_status"] = "pending_review"
    # The old technical candidate field must never act as an inclusion decision.
    inventory = inventory.drop(columns=["primary_candidate"], errors="ignore")
    historical_rules = historical_rules.copy()
    historical_rules["enabled"] = historical_rules.get("enabled", True)
    if "evaluated_status" in historical_rules:
        historical_rules["status"] = historical_rules["evaluated_status"]
        historical_rules["reason"] = historical_rules["evaluated_reason"]
    historical_rules["affects_membership"] = False
    later = []
    window_evidence = []
    for row in inventory.itertuples(index=False):
        outcomes = outcomes_by_recording.get(row.recording_id)
        if outcomes is None or not ((outcomes["alignment"] == "CS") & (outcomes["metric_id"] == metric_id)).any():
            checks = pd.DataFrame([{"rule_id": rule, "status": "not_evaluable",
                                    "reason": "authenticated_outcomes_unavailable"}
                                   for rule in ("normalized_missing_rows", "normalized_any_block", "normalized_block_statistics",
                                                "normalized_paired_blocks", "normalized_window_nan_fraction",
                                                *(f"learner_{v}_{step}" for v in VARIANTS for step in ("coverage", "features")))])
        else:
            checks = project_later_checks(outcomes, metric_id)
        if outcomes is not None:
            columns = [name for name in outcomes if name in (
                "experiment_id", "recording_id", "fish_id", "condition_id", "trial_id", "alignment",
                "trial_number", "block_10_name", "metric_id", "vigor_sample_policy",
                "baseline_total_activity", "response_total_activity", "baseline_conditional_intensity", "conditional_intensity",
            ) or any(term in name for term in ("sample_count", "coverage", "bout_count", "valid_fraction"))]
            window_evidence.append(outcomes.loc[outcomes["metric_id"].eq(metric_id), columns].copy())
        checks["recording_id"] = row.recording_id
        checks["recording_name"] = row.recording_name
        later.append(checks)
    later_checks = pd.concat(later, ignore_index=True) if later else pd.DataFrame()
    later_checks["affects_membership"] = False
    later_checks["scope"] = "corrected_outcome_projection"
    # Keep identity columns explicit even for inventory cases with no outcomes.
    inventory["experiment_id"] = identity["experiment"]
    inventory["fish_id"] = inventory["recording_id"]
    inventory["identity_basis"] = "recording_id_convention_pending_review"
    return publish_observation(project_dir, output_dir, tables={
        "inventory": inventory, "historical-checks": historical_rules,
        "historical-details": details, "later-checks": later_checks,
        "window-evidence": pd.concat(window_evidence, ignore_index=True) if window_evidence else pd.DataFrame(columns=["recording_id", "trial_id"]),
    }, identity={**identity, "implementation_sha256": sha256_file(Path(__file__)),
                 "source_hashes": {p.name: sha256_file(p) for p in LEGACY_ROOT.glob("*.py")}})
