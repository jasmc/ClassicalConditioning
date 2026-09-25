"""Auditable, descriptive comparison of the four historical learner rules.

The algorithms themselves remain in legacy/scripts. This adapter
only translates authenticated cohort outcomes into their input schema and runs
each script in a separate process. It does not approve a canonical classifier.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from classical_conditioning.analysis.legacy_learner_worker import SCRIPTS
from classical_conditioning.exceptions import SchemaValidationError, ScientificValidationError


KEY = ["experiment_id", "condition_id", "fish_id"]
DEFAULT_METRIC = "legacy_distal_angular_speed"


# Stream the source file while hashing so large learner inputs stay bounded in memory.
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate_inputs(cohort_path: Path, outcomes_path: Path, outcomes: pd.DataFrame) -> str:
    """Verify the cohort/outcome pair against their published sidecars."""
    if cohort_path.parent != outcomes_path.parent:
        raise ScientificValidationError("Cohort and trial outcomes must share a cohort directory")
    project_dir = cohort_path.parent.parent.parent.parent
    summary_dir = project_dir / "Quality checks" / "Cohorts" / cohort_path.parent.name
    cohort_summary = summary_dir / "cohort-manifest-v1_summary.json"
    outcome_summary = summary_dir / "cohort-trial-outcomes_summary.json"
    if not cohort_summary.is_file() or not outcome_summary.is_file():
        raise ScientificValidationError("Authenticated cohort and outcome summaries are required")
    cohort_meta = json.loads(cohort_summary.read_text())
    outcome_meta = json.loads(outcome_summary.read_text())
    if cohort_meta["artifacts"]["manifest"]["sha256"] != _sha256(cohort_path):
        raise ScientificValidationError("Cohort manifest SHA-256 does not match its summary")
    if outcome_meta["artifacts"]["outcomes"]["sha256"] != _sha256(outcomes_path):
        raise ScientificValidationError("Trial outcomes SHA-256 does not match its summary")
    expected_hash = cohort_meta["logical_content_sha256"]
    if outcome_meta["cohort_hash"] != expected_hash:
        raise ScientificValidationError("Cohort and outcome summary hashes disagree")
    if set(outcomes["cohort_hash"].dropna()) != {expected_hash}:
        raise ScientificValidationError("Trial rows do not carry the authenticated cohort hash")
    return expected_hash


def build_legacy_input(cohort: pd.DataFrame, outcomes: pd.DataFrame, metric_id: str) -> pd.DataFrame:
    """Translate one metric's CS outcomes, retaining the frozen fish set."""
    for name, frame, columns in (
        ("cohort", cohort, KEY + ["primary_included"]),
        ("outcomes", outcomes, KEY + ["alignment", "metric_id", "trial_number",
                                  "baseline_total_activity", "response_total_activity"]),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise SchemaValidationError(f"{name} missing columns: {sorted(missing)}")
    if cohort.duplicated(KEY).any():
        raise SchemaValidationError("Duplicate cohort fish key")
    selected = cohort.loc[cohort["primary_included"].eq(True), KEY].copy()
    if selected.empty:
        raise ScientificValidationError("No primary-cohort fish")
    if selected["experiment_id"].nunique() != 1 or selected["experiment_id"].iloc[0] != "allDelay":
        raise ScientificValidationError("Historical comparison currently supports allDelay only")
    trial = outcomes.loc[outcomes["alignment"].eq("CS") & outcomes["metric_id"].eq(metric_id)].copy()
    if trial.empty:
        raise ScientificValidationError(f"No CS trial outcomes for {metric_id}")
    if trial.duplicated(KEY + ["trial_number"]).any():
        raise SchemaValidationError("Duplicate fish/trial outcome")
    trial = trial.merge(selected, on=KEY, how="inner", validate="many_to_one")
    if trial.empty:
        raise ScientificValidationError("No outcomes match the primary cohort")
    baseline = pd.to_numeric(trial["baseline_total_activity"], errors="coerce")
    response = pd.to_numeric(trial["response_total_activity"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = response / baseline
    return pd.DataFrame({
        "Exp.": trial["condition_id"].astype(str),
        "Fish": trial["fish_id"].astype(str),
        "Trial number": trial["trial_number"].astype(int),
        "Mean CR": response,
        "Mean 9s before": baseline,
        "Normalized vigor": normalized,
    }).reset_index(drop=True)


def compare_legacy_learners(
    cohort_path: Path, outcomes_path: Path, output_dir: Path,
    *, metric_id: str = DEFAULT_METRIC,
) -> dict[str, object]:
    """Run all four preserved algorithms and write one fish-level comparison."""
    cohort_path, outcomes_path = cohort_path.resolve(), outcomes_path.resolve()
    output_dir = output_dir.resolve()
    if not cohort_path.is_file() or not outcomes_path.is_file():
        raise FileNotFoundError("Both cohort and trial-outcome Parquet files are required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Comparison output directory is not empty: {output_dir}")
    cohort = pd.read_parquet(cohort_path)
    outcomes = pd.read_parquet(outcomes_path)
    cohort_hash = _authenticate_inputs(cohort_path, outcomes_path, outcomes)
    legacy_input = build_legacy_input(cohort, outcomes, metric_id)
    selected = cohort.loc[cohort["primary_included"].eq(True), KEY].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "translated-legacy-input.parquet"
    legacy_input.to_parquet(input_path, index=False)
    merged = selected.copy()
    status = []
    for variant, filename in SCRIPTS.items():
        result_path = output_dir / f"{variant}.parquet"
        command = [sys.executable, "-m", "classical_conditioning.analysis.legacy_learner_worker",
                   variant, str(input_path), str(result_path)]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        (output_dir / f"{variant}.log").write_text(run.stdout + "\n[stderr]\n" + run.stderr)
        if run.returncode:
            status.append({"variant": variant, "status": "failed", "exit_code": run.returncode,
                           "log": f"{variant}.log"})
            continue
        result = pd.read_parquet(result_path)
        if result.duplicated(["Fish_ID", "Condition"]).any():
            raise SchemaValidationError(f"{variant} returned duplicate fish keys")
        if not set(map(tuple, result[["Condition", "Fish_ID"]].to_numpy())).issubset(
            set(map(tuple, selected[["condition_id", "fish_id"]].to_numpy()))
        ):
            raise SchemaValidationError(f"{variant} returned fish outside the cohort")
        result = result.rename(columns={"Fish_ID": "fish_id", "Condition": "condition_id"})
        result["experiment_id"] = "allDelay"
        result["classification_status"] = "classified"
        merged = merged.merge(
            result[KEY + ["learner_primary", "classification_status"]].rename(columns={
                "learner_primary": f"{variant}_learner",
                "classification_status": f"{variant}_status",
            }), on=KEY, how="left", validate="one_to_one",
        )
        merged[f"{variant}_status"] = merged[f"{variant}_status"].fillna("unclassified")
        conditioned = result.loc[result["condition_id"].eq("delay")]
        reference = result.loc[result["condition_id"].eq("control")]
        status.append({"variant": variant, "status": "completed", "eligible_delay": len(conditioned),
                       "learner_delay": int(conditioned["learner_primary"].sum()),
                       "eligible_control": len(reference),
                       "control_flagged": int(reference["learner_primary"].sum()),
                       "unclassified_delay": int((selected["condition_id"].eq("delay")).sum() - len(conditioned)),
                       "learner_fish_ids": sorted(conditioned.loc[conditioned["learner_primary"], "fish_id"].tolist()),
                       "algorithm_sha256": _sha256(Path(__file__).resolve().parents[3] / "legacy" / "scripts" / filename),
                       "config_sha256": _sha256(result_path.with_suffix(".config.json")),
                       "result_sha256": _sha256(result_path)})
    merged.sort_values(KEY).to_csv(output_dir / "fish-comparison.csv", index=False)
    report: dict[str, object] = {
        "schema": "legacy-learner-comparison/1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "descriptive_classifier_characterization", "canonical_classifier": None,
        "experiment_id": "allDelay", "metric_id": metric_id,
        "input_translation": "Mean CR=response_total_activity; Mean 9s before=baseline_total_activity; Normalized vigor=response/baseline",
        "historical_equivalence": False,
        "caveat": "Historical rules on corrected outcome units; labels do not establish biological learning or validate a canonical classifier.",
        "cohort_path": str(cohort_path), "cohort_sha256": _sha256(cohort_path),
        "cohort_hash": cohort_hash,
        "outcomes_path": str(outcomes_path), "outcomes_sha256": _sha256(outcomes_path),
        "translated_input_sha256": _sha256(input_path),
        "primary_fish": len(selected),
        "primary_delay_fish": int(selected["condition_id"].eq("delay").sum()),
        "primary_control_fish": int(selected["condition_id"].eq("control").sum()),
        "runtime": {"python": sys.version.split()[0], **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "statsmodels", "scikit-learn", "seaborn")
        }},
        "variants": status,
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, check=False,
    )
    report["code_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(item["status"] != "completed" for item in status):
        raise ScientificValidationError(
            f"One or more legacy variants failed; inspect {output_dir / 'comparison.json'}"
        )
    return report


def render_legacy_learner_figures(
    comparison_dir: Path, *, include_individuals: bool = False,
) -> dict[str, object]:
    """Render the scripts' summary figures, checking the recomputed evidence."""
    comparison_dir = comparison_dir.resolve()
    report = json.loads((comparison_dir / "comparison.json").read_text())
    input_path = comparison_dir / "translated-legacy-input.parquet"
    if _sha256(input_path) != report["translated_input_sha256"]:
        raise ScientificValidationError("Translated legacy input changed since comparison")
    if any(item["status"] != "completed" for item in report["variants"]):
        raise ScientificValidationError("All four variants must complete before rendering")
    figure_root = comparison_dir / "figures"
    if figure_root.exists() and any(figure_root.iterdir()):
        raise FileExistsError(f"Figure directory is not empty: {figure_root}")
    figure_root.mkdir(parents=True, exist_ok=True)
    rendered = []
    for item in report["variants"]:
        variant = item["variant"]
        variant_dir = figure_root / variant
        variant_dir.mkdir()
        rerun_path = variant_dir / "rerun-result.parquet"
        command = [sys.executable, "-m", "classical_conditioning.analysis.legacy_learner_worker",
                   variant, str(input_path), str(rerun_path), "--figures"]
        if include_individuals:
            command.append("--individuals")
        run = subprocess.run(
            command,
            capture_output=True, text=True, check=False,
        )
        (variant_dir / "render.log").write_text(run.stdout + "\n[stderr]\n" + run.stderr)
        if run.returncode:
            rendered.append({"variant": variant, "status": "failed", "log": str(variant_dir / "render.log")})
            continue
        expected = pd.read_parquet(comparison_dir / f"{variant}.parquet")
        observed = pd.read_parquet(rerun_path)
        pd.testing.assert_frame_equal(expected, observed)
        paths = sorted(variant_dir.rglob("*.png"))
        if not paths:
            rendered.append({"variant": variant, "status": "no_figures", "log": str(variant_dir / "render.log")})
        else:
            rendered.append({"variant": variant, "status": "completed",
                             "figures": [str(path) for path in paths]})
    manifest: dict[str, object] = {
        "schema": "legacy-learner-figures/1.0",
        "analysis_mode": "descriptive_classifier_characterization",
        "source_comparison": str(comparison_dir / "comparison.json"),
        "source_comparison_sha256": _sha256(comparison_dir / "comparison.json"),
        "figure_scope": "historical script summary and optional individual composite plots; heatmap grids require original stage-5 figure inputs",
        "include_individuals": include_individuals,
        "variants": rendered,
    }
    (figure_root / "figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if any(item["status"] != "completed" for item in rendered):
        raise ScientificValidationError(f"One or more figure runs failed; inspect {figure_root / 'figures.json'}")
    return manifest
