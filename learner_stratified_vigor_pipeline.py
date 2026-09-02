"""Full 3 s-trace learner classification and CS-aligned log-vigor pipeline.

The pipeline reads one condition-level pickle at a time, writes compact
checkpoints, classifies fish with the Analysis 6 LogMedian classifier, and
plots fish-weighted temporal profiles for the configured 10-trial blocks.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as general_config


FISH_COLUMN_CANDIDATES = ("Fish", "Fish_ID")
TRIAL_COLUMN = "Trial number"
TIME_SECONDS_COLUMN = "Trial time (s)"
RAW_VIGOR_COLUMN = "Vigor (deg/ms)"
SCALED_VIGOR_COLUMN = "Scaled vigor (AU)"
BOUT_COLUMN = "Bout"
CONDITION_COLUMN = "Condition"
BLOCK_COLUMN = "Block name"
TIME_BIN_COLUMN = "Time relative to CS onset (s)"
REFERENCE_CONDITION = "control"
CONDITIONED_CONDITION = "3sTrace"
CLASSIFIER_FILENAME = "6_LearnersQuantification_LogMedian.py"
CATCH_TRIALS = (25, 39, 53, 59)
TIMING_WINDOWS = {
    "Baseline": (-5.0, 0.0),
    "Early_CS": (0.0, 4.0),
    "Late_CS": (6.0, 10.0),
    "Trace": (10.0, 13.0),
}

STRATUM_COLORS = {
    "Reference": "#00AEEF",
    "Conditioned learner": "#F15A29",
    "Conditioned non-learner": "#6D6E71",
    "Conditioned unclassified": "#B7B7B7",
}
STRATA = tuple(STRATUM_COLORS)


@dataclass(frozen=True)
class PipelineConfig:
    trace_path: str
    control_path: str
    output_dir: str
    time_min_s: float = -15.0
    time_max_s: float = 21.0
    time_bin_s: float = 0.5
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 10
    classifier_alpha: float = 0.05
    force: bool = False


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify learners in the full 3 s-trace dataset and plot vigor "
            "relative to CS onset in 10-trial blocks."
        )
    )
    parser.add_argument("--trace", type=Path, required=True, help="Trace condition CS-aligned pickle.")
    parser.add_argument("--control", type=Path, required=True, help="Control condition CS-aligned pickle.")
    parser.add_argument("--output", type=Path, required=True, help="Derived output directory.")
    parser.add_argument("--time-min", type=float, default=-15.0)
    parser.add_argument("--time-max", type=float, default=21.0)
    parser.add_argument("--time-bin", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=10)
    parser.add_argument("--classifier-alpha", type=float, default=0.05)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing per-condition checkpoints.",
    )
    return parser.parse_args(argv)


def validate_config(config: PipelineConfig) -> None:
    for label, value in (("trace", config.trace_path), ("control", config.control_path)):
        if not Path(value).is_file():
            raise FileNotFoundError(f"{label.capitalize()} input does not exist: {value}")
    if config.time_min_s >= config.time_max_s:
        raise ValueError("--time-min must be less than --time-max.")
    if config.time_bin_s <= 0:
        raise ValueError("--time-bin must be positive.")
    if config.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be at least 1.")
    if not 0 < config.classifier_alpha < 1:
        raise ValueError("--classifier-alpha must be between 0 and 1.")


@contextmanager
def legacy_pandas_array_pickle_compat():
    """Temporarily accept two-item extension-array states from the source pickles."""
    from pandas.core.arrays import Categorical
    from pandas.core.arrays.string_ import StringArray

    classes = (StringArray, Categorical)
    original_setstates = {array_class: array_class.__setstate__ for array_class in classes}

    def make_compatible_setstate(original_setstate):
        def compatible_setstate(array: Any, state: Any) -> None:
            if isinstance(state, tuple) and len(state) == 2:
                state = (*state, {})
            original_setstate(array, state)

        return compatible_setstate

    for array_class, original_setstate in original_setstates.items():
        array_class.__setstate__ = make_compatible_setstate(original_setstate)
    try:
        yield
    finally:
        for array_class, original_setstate in original_setstates.items():
            array_class.__setstate__ = original_setstate


def read_pickle_compat(path: Path) -> pd.DataFrame:
    """Read a pandas pickle and make version-skew failures actionable."""
    errors: list[tuple[str, Exception]] = []
    for compression in (None, "gzip"):
        try:
            with legacy_pandas_array_pickle_compat():
                data = pd.read_pickle(path, compression=compression)
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Expected a pandas DataFrame, got {type(data).__name__}.")
            return data
        except Exception as exc:
            label = "uncompressed" if compression is None else compression
            errors.append((label, exc))

    error_details = "; ".join(
        f"{label}: {type(error).__name__}: {error}" for label, error in errors
    )
    first_error = errors[0][1]
    raise RuntimeError(
        f"Could not read {path}. Run the pipeline through run_learner_stratified_vigor.ps1, "
        "which creates the pinned pandas compatibility environment. "
        f"Attempts: {error_details}"
    ) from first_error


def find_fish_column(data: pd.DataFrame) -> str:
    for column in FISH_COLUMN_CANDIDATES:
        if column in data.columns:
            return column
    raise KeyError(f"Missing fish identifier; expected one of {FISH_COLUMN_CANDIDATES}.")


def time_seconds(data: pd.DataFrame) -> pd.Series:
    if TIME_SECONDS_COLUMN in data.columns:
        return pd.to_numeric(data[TIME_SECONDS_COLUMN], errors="coerce")
    frame_column = general_config.time_trial_frame_label
    if frame_column in data.columns:
        return pd.to_numeric(data[frame_column], errors="coerce") / float(
            general_config.expected_framerate
        )
    raise KeyError(
        f"Missing time column; expected {TIME_SECONDS_COLUMN!r} or {frame_column!r}."
    )


def dense_numeric(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.SparseDtype):
        series = series.sparse.to_dense()
    return pd.to_numeric(series, errors="coerce")


def dense_bool(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.SparseDtype):
        series = series.sparse.to_dense()
    return series.fillna(False).astype(bool)


def trial_to_block_map() -> tuple[dict[int, str], list[str]]:
    experiment = get_experiment_config(ExperimentType.ALL_3S_TRACE.value)
    mapping: dict[int, str] = {}
    for name, trials in zip(experiment.names_cs_blocks_10, experiment.trials_cs_blocks_10):
        mapping.update({int(trial): name for trial in trials})
    return mapping, list(experiment.names_cs_blocks_10)


def add_block_names(data: pd.DataFrame) -> pd.DataFrame:
    mapping, names = trial_to_block_map()
    result = data.copy()
    result[BLOCK_COLUMN] = pd.Categorical(
        pd.to_numeric(result[TRIAL_COLUMN], errors="coerce").map(mapping),
        categories=names,
        ordered=True,
    )
    return result[result[BLOCK_COLUMN].notna()].copy()


def aggregate_trial_summary(
    data: pd.DataFrame,
    fish_column: str,
    times: pd.Series,
    condition: str,
) -> pd.DataFrame:
    required = {TRIAL_COLUMN, RAW_VIGOR_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing columns required for classification: {sorted(missing)}")

    fish = data[fish_column].astype("string")
    trials = pd.to_numeric(data[TRIAL_COLUMN], errors="coerce")
    raw_vigor = dense_numeric(data[RAW_VIGOR_COLUMN])
    baseline_mask = times.between(-float(general_config.baseline_window), 0)
    experiment = get_experiment_config(ExperimentType.ALL_3S_TRACE.value)
    response_mask = times.between(float(experiment.cr_window[0]), float(experiment.cr_window[1]))

    def window_median(mask: pd.Series, output_name: str) -> pd.Series:
        frame = pd.DataFrame(
            {
                "Fish": fish.loc[mask],
                TRIAL_COLUMN: trials.loc[mask],
                output_name: raw_vigor.loc[mask],
            }
        )
        return frame.groupby(["Fish", TRIAL_COLUMN], observed=True, sort=False)[output_name].median()

    baseline_name = f"Median {general_config.baseline_window} s before"
    baseline = window_median(baseline_mask, baseline_name)
    response = window_median(response_mask, "Median CR")
    summary = pd.concat([baseline, response], axis=1).reset_index()
    summary["Normalized vigor"] = summary["Median CR"] - summary[baseline_name]
    summary.insert(0, "Exp.", condition)
    summary = add_block_names(summary)
    finite = np.isfinite(summary["Normalized vigor"].to_numpy(dtype=float))
    summary.loc[~finite, "Normalized vigor"] = np.nan
    return summary


def aggregate_temporal_bins(
    data: pd.DataFrame,
    fish_column: str,
    times: pd.Series,
    condition: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    vigor_column = SCALED_VIGOR_COLUMN if SCALED_VIGOR_COLUMN in data.columns else RAW_VIGOR_COLUMN
    if vigor_column not in data.columns:
        raise KeyError(
            f"Missing temporal vigor; expected {SCALED_VIGOR_COLUMN!r} or {RAW_VIGOR_COLUMN!r}."
        )

    trials = pd.to_numeric(data[TRIAL_COLUMN], errors="coerce")
    in_window = times.between(config.time_min_s, config.time_max_s)
    selected_times = times.loc[in_window]
    bin_number = np.floor(
        (selected_times.to_numpy(dtype=float) - config.time_min_s) / config.time_bin_s
    ).astype(np.int32)
    bin_centers = config.time_min_s + (bin_number + 0.5) * config.time_bin_s

    vigor = dense_numeric(data.loc[in_window, vigor_column])
    if BOUT_COLUMN in data.columns:
        moving = dense_bool(data.loc[in_window, BOUT_COLUMN])
        conditional_vigor = vigor.where(moving)
    else:
        print(
            f"[warn] {BOUT_COLUMN!r} is absent. Movement probability will describe "
            "non-missing vigor coverage, not detected bouts."
        )
        moving = vigor.notna()
        conditional_vigor = vigor

    compact = pd.DataFrame(
        {
            "Fish_ID": data.loc[in_window, fish_column].astype("string").to_numpy(),
            TRIAL_COLUMN: trials.loc[in_window].to_numpy(),
            TIME_BIN_COLUMN: bin_centers,
            "Conditional vigor": conditional_vigor.to_numpy(dtype=float),
            "Moving": moving.to_numpy(dtype=np.float32),
        }
    )
    compact.dropna(subset=["Fish_ID", TRIAL_COLUMN, TIME_BIN_COLUMN], inplace=True)
    binned = (
        compact.groupby(
            ["Fish_ID", TRIAL_COLUMN, TIME_BIN_COLUMN],
            observed=True,
            sort=False,
        )
        .agg(
            **{
                "Conditional vigor": ("Conditional vigor", "median"),
                "Movement probability": ("Moving", "mean"),
            }
        )
        .reset_index()
    )
    binned.insert(0, CONDITION_COLUMN, condition)
    return add_block_names(binned)


def checkpoint_signature(
    source_path: Path,
    condition: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    source_stat = source_path.stat()
    return {
        "condition": condition,
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_stat.st_size,
        "source_modified_ns": source_stat.st_mtime_ns,
        "time_min_s": config.time_min_s,
        "time_max_s": config.time_max_s,
        "time_bin_s": config.time_bin_s,
        "baseline_window_s": float(general_config.baseline_window),
        "experiment": ExperimentType.ALL_3S_TRACE.value,
    }


def checkpoint_is_current(
    metadata_path: Path,
    expected_signature: dict[str, Any],
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        actual_signature = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual_signature == expected_signature


def process_condition(
    source_path: Path,
    condition: str,
    checkpoint_dir: Path,
    config: PipelineConfig,
) -> tuple[Path, Path]:
    trial_path = checkpoint_dir / f"{condition}_trial_summary.csv"
    temporal_path = checkpoint_dir / f"{condition}_temporal_bins.pkl.gz"
    metadata_path = checkpoint_dir / f"{condition}_checkpoint.json"
    signature = checkpoint_signature(source_path, condition, config)
    if (
        trial_path.exists()
        and temporal_path.exists()
        and checkpoint_is_current(metadata_path, signature)
        and not config.force
    ):
        print(f"[resume] Using completed {condition} checkpoints.")
        return trial_path, temporal_path

    print(f"[load] Reading full {condition} file: {source_path}")
    data = read_pickle_compat(source_path)
    fish_column = find_fish_column(data)
    if TRIAL_COLUMN not in data.columns:
        raise KeyError(f"{source_path.name} does not contain {TRIAL_COLUMN!r}.")
    times = time_seconds(data)
    print(
        f"[load] {condition}: {len(data):,} rows, "
        f"{data[fish_column].nunique(dropna=True):,} fish."
    )

    trial_summary = aggregate_trial_summary(data, fish_column, times, condition)
    temporal = aggregate_temporal_bins(data, fish_column, times, condition, config)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trial_summary.to_csv(trial_path, index=False)
    temporal.to_pickle(temporal_path, compression="gzip")
    metadata_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(f"[checkpoint] Saved {trial_path}")
    print(f"[checkpoint] Saved {temporal_path}")

    del temporal, trial_summary, times, data
    gc.collect()
    return trial_path, temporal_path


def load_analysis6_module(repository_root: Path) -> Any:
    module_path = repository_root / CLASSIFIER_FILENAME
    spec = importlib.util.spec_from_file_location("learners_analysis6_logmedian", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Analysis 6 from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_analysis6_classifier(
    trial_summary_path: Path,
    output_dir: Path,
    repository_root: Path,
    alpha: float,
) -> pd.DataFrame:
    """Run the Analysis 6 LogMedian classifier for all3sTrace."""
    module = load_analysis6_module(repository_root)
    experiment = get_experiment_config(ExperimentType.ALL_3S_TRACE.value)
    experiment.path_save = output_dir / "analysis6_runtime"

    module.EXPERIMENT = ExperimentType.ALL_3S_TRACE.value
    module.exp_config = experiment
    module.POOLED_DATA_PATH = trial_summary_path
    module.POOLED_DATA_NAN_FILTER = "auto"
    module.RESPONSE_COLUMN_NAME = "Median CR"
    module.APPLY_FISH_DISCARD = False
    module.RUN_EXPORT_RESULTS = False
    module.RUN_PLOT_TRAJECTORIES = False
    module.RUN_PLOT_FEATURE_SPACE = False
    module.RUN_PLOT_BLUP_CATERPILLAR = False
    module.RUN_PLOT_INDIVIDUALS_AND_GRID = False
    module.RUN_PLOT_BLUP_OVERLAY = False
    module.RUN_PLOT_HEATMAP_GRID = False

    late_train_early_test = [
        "Train 6",
        "Train 7",
        "Train 8",
        "Train 9",
        "Late Train",
        "Early Test",
    ]
    classifier_config = module.AnalysisConfig(
        csus="CS",
        random_seed=0,
        min_pretrain_trials=6,
        min_latetraindearlytest_trials=6,
        min_late_test_trials=6,
        min_trials_per_5trial_block_in_epoch=3,
        features_to_use=["acquisition", "extinction"],
        pretrain_to_train_end_earlytest_blocks=(
            ["Early Pre-Train", "Late Pre-Train"],
            late_train_early_test,
        ),
        train_end_earlytest_to_late_test_blocks=(
            late_train_early_test,
            ["Test 5", "Late Test"],
        ),
    )
    print(f"[classify] Running {CLASSIFIER_FILENAME} for all3sTrace.")
    return module.run_multivariate_lme_pipeline(
        classifier_config,
        alpha=alpha,
        use_per_fish_se_in_scoring=True,
    )


def prepare_analysis6_input(
    trial_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give Analysis 6 a condition-qualified subject key and retain its audit map."""
    required = {"Exp.", "Fish"}
    missing = required.difference(trial_summary.columns)
    if missing:
        raise KeyError(f"Missing Analysis 6 key columns: {sorted(missing)}")

    prepared = trial_summary.copy()
    prepared["Fish"] = prepared["Fish"].astype("string")
    prepared["Exp."] = prepared["Exp."].astype("string")
    if prepared[["Exp.", "Fish"]].isna().any().any():
        raise ValueError("Condition or fish identifier is missing from the trial summary.")

    key_map = (
        prepared[["Exp.", "Fish"]]
        .drop_duplicates()
        .rename(columns={"Exp.": CONDITION_COLUMN, "Fish": "Source_Fish_ID"})
    )
    key_map["Classifier_Fish_ID"] = (
        key_map[CONDITION_COLUMN].astype(str)
        + "::"
        + key_map["Source_Fish_ID"].astype(str)
    )
    if key_map["Classifier_Fish_ID"].duplicated().any():
        raise ValueError("Condition-qualified classifier fish keys are not unique.")

    prepared = prepared.merge(
        key_map,
        left_on=["Exp.", "Fish"],
        right_on=[CONDITION_COLUMN, "Source_Fish_ID"],
        how="left",
        validate="many_to_one",
    )
    prepared["Fish"] = prepared["Classifier_Fish_ID"]
    prepared.drop(
        columns=[CONDITION_COLUMN, "Source_Fish_ID", "Classifier_Fish_ID"],
        inplace=True,
    )
    return prepared, key_map


def restore_source_fish_ids(
    classification: pd.DataFrame,
    key_map: pd.DataFrame,
) -> pd.DataFrame:
    """Restore source fish IDs after classification with composite model keys."""
    restored = classification.rename(columns={"Fish_ID": "Classifier_Fish_ID"}).merge(
        key_map,
        on=[CONDITION_COLUMN, "Classifier_Fish_ID"],
        how="left",
        validate="one_to_one",
    )
    if restored["Source_Fish_ID"].isna().any():
        raise ValueError("Analysis 6 returned classifier fish keys absent from the key map.")
    restored["Fish_ID"] = restored["Source_Fish_ID"]
    restored.drop(columns="Source_Fish_ID", inplace=True)
    return restored


def create_manifest(
    classification: pd.DataFrame,
    all_fish: pd.DataFrame,
    classifier_run_id: str,
    alpha: float,
) -> pd.DataFrame:
    fish_table = (
        all_fish[["Exp.", "Fish"]]
        .rename(columns={"Exp.": CONDITION_COLUMN, "Fish": "Fish_ID"})
        .drop_duplicates()
    )
    evidence = classification.rename(columns={"Is_Learner": "Learner_Primary"}).copy()
    manifest = fish_table.merge(
        evidence,
        on=[CONDITION_COLUMN, "Fish_ID"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    manifest["Classification_Eligible"] = manifest["_merge"].eq("both")
    manifest["Learner_Primary"] = manifest["Learner_Primary"].astype("boolean")
    manifest["Learner_Status"] = "Unclassified"
    eligible = manifest["Classification_Eligible"]
    manifest.loc[eligible & manifest["Learner_Primary"].eq(True), "Learner_Status"] = "Learner"
    manifest.loc[eligible & manifest["Learner_Primary"].eq(False), "Learner_Status"] = "Non-learner"
    manifest["Analysis_Stratum"] = "Conditioned unclassified"
    manifest.loc[
        manifest[CONDITION_COLUMN].eq(REFERENCE_CONDITION), "Analysis_Stratum"
    ] = "Reference"
    manifest.loc[
        manifest[CONDITION_COLUMN].eq(CONDITIONED_CONDITION)
        & manifest["Learner_Status"].eq("Learner"),
        "Analysis_Stratum",
    ] = "Conditioned learner"
    manifest.loc[
        manifest[CONDITION_COLUMN].eq(CONDITIONED_CONDITION)
        & manifest["Learner_Status"].eq("Non-learner"),
        "Analysis_Stratum",
    ] = "Conditioned non-learner"
    manifest["Cohort_Role"] = np.where(
        manifest[CONDITION_COLUMN].eq(REFERENCE_CONDITION), "Reference", "Conditioned"
    )
    manifest["Classifier_Name"] = CLASSIFIER_FILENAME
    manifest["Classifier_Run_ID"] = classifier_run_id
    manifest["Classifier_Alpha"] = alpha
    manifest["Classification_Alignment"] = "CS"
    manifest.drop(columns="_merge", inplace=True)
    return manifest


def summarize_profiles(
    temporal: pd.DataFrame,
    outcome: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fish_profiles = (
        temporal.groupby(
            ["Analysis_Stratum", "Fish_ID", BLOCK_COLUMN, TIME_BIN_COLUMN],
            observed=True,
            sort=False,
        )[outcome]
        .median()
        .reset_index()
    )
    rng = np.random.default_rng(config.bootstrap_seed)
    profile_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for (stratum, block), group in fish_profiles.groupby(
        ["Analysis_Stratum", BLOCK_COLUMN], observed=True, sort=False
    ):
        pivot = group.pivot(index="Fish_ID", columns=TIME_BIN_COLUMN, values=outcome).sort_index(axis=1)
        values = pivot.to_numpy(dtype=float)
        if values.size == 0:
            continue
        contributing = np.sum(np.isfinite(values), axis=0)
        valid_columns = contributing > 0
        center = np.full(values.shape[1], np.nan)
        low = np.full(values.shape[1], np.nan)
        high = np.full(values.shape[1], np.nan)
        if valid_columns.any():
            valid_values = values[:, valid_columns]
            center[valid_columns] = np.nanmedian(valid_values, axis=0)
            sampled = rng.integers(
                0,
                values.shape[0],
                size=(config.bootstrap_iterations, values.shape[0]),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                boot = np.nanmedian(valid_values[sampled, :], axis=1)
            valid_indices = np.flatnonzero(valid_columns)
            for boot_column, output_index in enumerate(valid_indices):
                finite_boot = boot[:, boot_column]
                finite_boot = finite_boot[np.isfinite(finite_boot)]
                if finite_boot.size:
                    low[output_index], high[output_index] = np.percentile(
                        finite_boot,
                        [2.5, 97.5],
                    )
        for idx, time_value in enumerate(pivot.columns.to_numpy(dtype=float)):
            profile_rows.append(
                {
                    "Analysis_Stratum": stratum,
                    BLOCK_COLUMN: block,
                    TIME_BIN_COLUMN: time_value,
                    "Median": center[idx],
                    "CI_Low": low[idx],
                    "CI_High": high[idx],
                    "Contributing_Fish": int(contributing[idx]),
                }
            )
            coverage_rows.append(
                {
                    "Analysis_Stratum": stratum,
                    BLOCK_COLUMN: block,
                    TIME_BIN_COLUMN: time_value,
                    "Contributing_Fish": int(contributing[idx]),
                    "Stratum_Fish": int(values.shape[0]),
                    "Fraction_Contributing": float(contributing[idx] / values.shape[0]),
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(coverage_rows)


def summarize_catch_profiles(
    temporal: pd.DataFrame,
    outcome: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize each catch trial across fish without pooling time samples."""
    catch = temporal[temporal[TRIAL_COLUMN].isin(CATCH_TRIALS)].copy()
    observed_trials = set(pd.to_numeric(catch[TRIAL_COLUMN], errors="coerce").dropna().astype(int))
    missing_trials = set(CATCH_TRIALS).difference(observed_trials)
    if missing_trials:
        raise ValueError(f"Catch trials are missing from temporal data: {sorted(missing_trials)}")

    fish_profiles = (
        catch.groupby(
            ["Analysis_Stratum", "Fish_ID", TRIAL_COLUMN, TIME_BIN_COLUMN],
            observed=True,
            sort=False,
        )[outcome]
        .median()
        .reset_index()
    )
    rng = np.random.default_rng(config.bootstrap_seed)
    profile_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for (stratum, trial), group in fish_profiles.groupby(
        ["Analysis_Stratum", TRIAL_COLUMN], observed=True, sort=False
    ):
        pivot = group.pivot(index="Fish_ID", columns=TIME_BIN_COLUMN, values=outcome).sort_index(axis=1)
        values = pivot.to_numpy(dtype=float)
        if values.size == 0:
            continue
        center, low, high, contributing = profile_statistics(values, rng, config)
        for idx, time_value in enumerate(pivot.columns.to_numpy(dtype=float)):
            profile_rows.append(
                {
                    "Analysis_Stratum": stratum,
                    TRIAL_COLUMN: int(trial),
                    TIME_BIN_COLUMN: time_value,
                    "Median": center[idx],
                    "CI_Low": low[idx],
                    "CI_High": high[idx],
                    "Contributing_Fish": int(contributing[idx]),
                }
            )
            coverage_rows.append(
                {
                    "Analysis_Stratum": stratum,
                    TRIAL_COLUMN: int(trial),
                    TIME_BIN_COLUMN: time_value,
                    "Contributing_Fish": int(contributing[idx]),
                    "Stratum_Fish": int(values.shape[0]),
                    "Fraction_Contributing": float(contributing[idx] / values.shape[0]),
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(coverage_rows)


def profile_statistics(
    values: np.ndarray,
    rng: np.random.Generator,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a fish-level median and bootstrap interval at each time bin."""
    contributing = np.sum(np.isfinite(values), axis=0)
    valid_columns = contributing > 0
    center = np.full(values.shape[1], np.nan)
    low = np.full(values.shape[1], np.nan)
    high = np.full(values.shape[1], np.nan)
    if not valid_columns.any():
        return center, low, high, contributing

    valid_values = values[:, valid_columns]
    center[valid_columns] = np.nanmedian(valid_values, axis=0)
    sampled = rng.integers(
        0,
        values.shape[0],
        size=(config.bootstrap_iterations, values.shape[0]),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        boot = np.nanmedian(valid_values[sampled, :], axis=1)
    valid_indices = np.flatnonzero(valid_columns)
    for boot_column, output_index in enumerate(valid_indices):
        finite_boot = boot[:, boot_column]
        finite_boot = finite_boot[np.isfinite(finite_boot)]
        if finite_boot.size:
            low[output_index], high[output_index] = np.percentile(
                finite_boot,
                [2.5, 97.5],
            )
    return center, low, high, contributing


def robust_profile_ylim(
    profiles: pd.DataFrame,
    outcome: str,
    *,
    suppression: bool = False,
) -> tuple[float, float]:
    """Choose shared limits from central curves so sparse edge CIs do not dominate."""
    included = profiles.copy()
    if "Contributing_Fish" in included.columns:
        max_coverage = pd.to_numeric(
            included["Contributing_Fish"], errors="coerce"
        ).max()
        minimum_coverage = max(3, int(np.ceil(float(max_coverage) * 0.1)))
        included = included[
            pd.to_numeric(included["Contributing_Fish"], errors="coerce").ge(
                minimum_coverage
            )
        ]
    if TIME_BIN_COLUMN in included.columns:
        included = included[
            pd.to_numeric(included[TIME_BIN_COLUMN], errors="coerce").le(16)
        ]
    values = pd.to_numeric(included["Median"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    low, high = float(np.min(values)), float(np.max(values))
    if outcome == "Movement probability" and not suppression:
        return 0.0, min(1.0, max(0.1, high * 1.08))
    low = min(low, 0.0)
    high = max(high, 0.0)
    span = max(high - low, 0.05)
    padding = span * 0.12
    return low - padding, high + padding


def add_stimulus_timing(axis: plt.Axes, *, catch_trial: bool = False) -> None:
    axis.axvspan(0, 10, color="#E6E6E6", alpha=0.8, zorder=0)
    axis.axvspan(10, 13, color="#F7E7CE", alpha=0.55, zorder=0)
    axis.axvline(13, color="#333333", linestyle="--", linewidth=0.8, alpha=0.8)
    if catch_trial:
        axis.text(
            13,
            0.98,
            "expected US",
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=7,
            color="#333333",
        )


def add_panel_title(axis: plt.Axes, title: str) -> None:
    axis.text(
        0.01,
        0.98,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        zorder=10,
    )


def add_shared_figure_labels(
    fig: plt.Figure,
    handles: list[Any],
    labels: list[str],
    *,
    title: str,
    subtitle: str,
    ylabel: str,
) -> None:
    fig.suptitle(title, y=0.985, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=9)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.supxlabel("Time relative to CS onset (s)", y=0.035)
    if ylabel:
        fig.supylabel(ylabel, x=0.025)


def create_manual_layout_subplots(*args: Any, **kwargs: Any) -> tuple[plt.Figure, Any]:
    """Create a figure unaffected by the repository's global constrained layout."""
    with plt.rc_context(
        {
            "figure.autolayout": False,
            "figure.constrained_layout.use": False,
        }
    ):
        return plt.subplots(*args, layout=None, **kwargs)


def plot_profiles(
    profiles: pd.DataFrame,
    block_names: list[str],
    outcome: str,
    output_base: Path,
) -> None:
    fig, axes = create_manual_layout_subplots(
        3, 3, figsize=(12, 9), sharex=True, sharey=True
    )
    y_limits = robust_profile_ylim(profiles, outcome)
    for axis, block in zip(axes.flat, block_names):
        add_stimulus_timing(axis)
        for stratum in STRATA:
            current = profiles[
                profiles["Analysis_Stratum"].eq(stratum)
                & profiles[BLOCK_COLUMN].astype(str).eq(block)
            ].sort_values(TIME_BIN_COLUMN)
            if current.empty:
                continue
            x = current[TIME_BIN_COLUMN].to_numpy(dtype=float)
            y = current["Median"].to_numpy(dtype=float)
            low = current["CI_Low"].to_numpy(dtype=float)
            high = current["CI_High"].to_numpy(dtype=float)
            color = STRATUM_COLORS[stratum]
            axis.plot(x, y, color=color, linewidth=1.5, label=stratum)
            axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        add_panel_title(axis, block)
        axis.set_ylim(y_limits)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not handles:
        for axis in axes.flat[1:]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    add_shared_figure_labels(
        fig,
        handles,
        labels,
        title="3 s trace: learner-stratified profiles by 10-trial block",
        subtitle="Gray: CS (0-10 s); tan: trace interval (10-13 s); dashed: expected US onset",
        ylabel=outcome,
    )
    fig.subplots_adjust(top=0.75, bottom=0.09, left=0.08, right=0.98, hspace=0.34, wspace=0.12)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_catch_profiles(
    profiles: pd.DataFrame,
    outcome: str,
    output_base: Path,
) -> None:
    fig, axes = create_manual_layout_subplots(
        2, 2, figsize=(10, 7), sharex=True, sharey=True
    )
    y_limits = robust_profile_ylim(profiles, outcome)
    for axis, trial in zip(axes.flat, CATCH_TRIALS):
        add_stimulus_timing(axis, catch_trial=True)
        for stratum in STRATA:
            current = profiles[
                profiles["Analysis_Stratum"].eq(stratum)
                & profiles[TRIAL_COLUMN].eq(trial)
            ].sort_values(TIME_BIN_COLUMN)
            if current.empty:
                continue
            x = current[TIME_BIN_COLUMN].to_numpy(dtype=float)
            color = STRATUM_COLORS[stratum]
            axis.plot(
                x,
                current["Median"],
                color=color,
                linewidth=1.5,
                label=stratum,
            )
            axis.fill_between(
                x,
                current["CI_Low"],
                current["CI_High"],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        add_panel_title(axis, f"Catch trial {trial}")
        axis.set_xlim(-5, 16)
        axis.set_ylim(y_limits)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    add_shared_figure_labels(
        fig,
        handles,
        labels,
        title=f"3 s trace: {outcome.lower()} on training catch trials",
        subtitle="No US delivered; gray: CS; tan: trace interval; dashed: expected US onset",
        ylabel=outcome,
    )
    fig.subplots_adjust(top=0.72, bottom=0.11, left=0.09, right=0.98, hspace=0.28, wspace=0.12)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def create_catch_suppression_data(
    temporal: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """Express catch-trial responses as suppression from each trial's pre-CS baseline."""
    catch = temporal[temporal[TRIAL_COLUMN].isin(CATCH_TRIALS)].copy()
    baseline_start, baseline_end = TIMING_WINDOWS["Baseline"]
    baseline = (
        catch[catch[TIME_BIN_COLUMN].ge(baseline_start) & catch[TIME_BIN_COLUMN].lt(baseline_end)]
        .groupby(
            [CONDITION_COLUMN, "Analysis_Stratum", "Fish_ID", TRIAL_COLUMN],
            observed=True,
        )[outcome]
        .median()
        .rename("Pre_CS_Baseline")
        .reset_index()
    )
    catch = catch.merge(
        baseline,
        on=[CONDITION_COLUMN, "Analysis_Stratum", "Fish_ID", TRIAL_COLUMN],
        how="left",
        validate="many_to_one",
    )
    catch["Suppression"] = catch["Pre_CS_Baseline"] - catch[outcome]
    return catch


def calculate_catch_timing_metrics(
    temporal: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify whether suppression strengthens as the expected US approaches."""
    per_trial_rows: list[dict[str, Any]] = []
    for outcome in ("Conditional vigor", "Movement probability"):
        catch = create_catch_suppression_data(temporal, outcome)
        for keys, group in catch.groupby(
            [CONDITION_COLUMN, "Analysis_Stratum", "Fish_ID", TRIAL_COLUMN],
            observed=True,
        ):
            condition, stratum, fish_id, trial = keys
            row: dict[str, Any] = {
                CONDITION_COLUMN: condition,
                "Analysis_Stratum": stratum,
                "Fish_ID": fish_id,
                TRIAL_COLUMN: int(trial),
                "Outcome": outcome,
            }
            for window_name, (start, end) in TIMING_WINDOWS.items():
                values = group.loc[
                    group[TIME_BIN_COLUMN].ge(start) & group[TIME_BIN_COLUMN].lt(end),
                    outcome,
                ]
                row[window_name] = float(values.median()) if values.notna().any() else np.nan
            row["Suppression_Early_CS"] = row["Baseline"] - row["Early_CS"]
            row["Suppression_Late_CS"] = row["Baseline"] - row["Late_CS"]
            row["Suppression_Trace"] = row["Baseline"] - row["Trace"]
            row["Near_US_Strengthening"] = (
                row["Suppression_Trace"] - row["Suppression_Early_CS"]
            )
            anticipatory = group[
                group[TIME_BIN_COLUMN].ge(0) & group[TIME_BIN_COLUMN].lt(13)
            ][[TIME_BIN_COLUMN, "Suppression"]].dropna()
            row["Anticipatory_Suppression_Slope"] = (
                float(np.polyfit(anticipatory[TIME_BIN_COLUMN], anticipatory["Suppression"], 1)[0])
                if len(anticipatory) >= 3
                else np.nan
            )
            per_trial_rows.append(row)

    per_trial = pd.DataFrame(per_trial_rows)
    metric_columns = [
        "Baseline",
        "Early_CS",
        "Late_CS",
        "Trace",
        "Suppression_Early_CS",
        "Suppression_Late_CS",
        "Suppression_Trace",
        "Near_US_Strengthening",
        "Anticipatory_Suppression_Slope",
    ]
    per_fish = (
        per_trial.groupby(
            [CONDITION_COLUMN, "Analysis_Stratum", "Fish_ID", "Outcome"],
            observed=True,
        )[metric_columns]
        .median()
        .reset_index()
    )
    trial_counts = (
        per_trial.groupby(
            [CONDITION_COLUMN, "Analysis_Stratum", "Fish_ID", "Outcome"],
            observed=True,
        )[TRIAL_COLUMN]
        .nunique()
        .rename("Catch_Trials_Contributing")
        .reset_index()
    )
    per_fish = per_fish.merge(
        trial_counts,
        on=[CONDITION_COLUMN, "Analysis_Stratum", "Fish_ID", "Outcome"],
        validate="one_to_one",
    )

    rng = np.random.default_rng(config.bootstrap_seed)
    summary_rows: list[dict[str, Any]] = []
    evidence_metrics = (
        "Suppression_Early_CS",
        "Suppression_Late_CS",
        "Suppression_Trace",
        "Near_US_Strengthening",
        "Anticipatory_Suppression_Slope",
    )
    for (stratum, outcome), group in per_fish.groupby(
        ["Analysis_Stratum", "Outcome"], observed=True
    ):
        for metric in evidence_metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            sampled = rng.choice(
                values,
                size=(config.bootstrap_iterations, values.size),
                replace=True,
            )
            boot_medians = np.median(sampled, axis=1)
            ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
            summary_rows.append(
                {
                    "Analysis_Stratum": stratum,
                    "Outcome": outcome,
                    "Metric": metric,
                    "Fish_Count": int(values.size),
                    "Median": float(np.median(values)),
                    "CI_Low": float(ci_low),
                    "CI_High": float(ci_high),
                    "Fraction_Positive": float(np.mean(values > 0)),
                    "Interpretation": (
                        "Positive means suppression strengthens toward the expected US"
                        if metric in {"Near_US_Strengthening", "Anticipatory_Suppression_Slope"}
                        else "Positive means suppression relative to pre-CS baseline"
                    ),
                }
            )
    return per_fish, pd.DataFrame(summary_rows)


def plot_pooled_catch_suppression(
    temporal: pd.DataFrame,
    config: PipelineConfig,
    output_base: Path,
) -> pd.DataFrame:
    fig, axes = create_manual_layout_subplots(
        1, 2, figsize=(11, 4.5), sharex=True
    )
    panel_frames: list[pd.DataFrame] = []
    for axis, outcome in zip(axes, ("Conditional vigor", "Movement probability")):
        suppression = create_catch_suppression_data(temporal, outcome)
        suppression[BLOCK_COLUMN] = "All catch trials"
        profiles, _ = summarize_profiles(suppression, "Suppression", config)
        profiles["Outcome"] = outcome
        panel_frames.append(profiles)
        add_stimulus_timing(axis, catch_trial=True)
        for stratum in STRATA:
            current = profiles[profiles["Analysis_Stratum"].eq(stratum)].sort_values(
                TIME_BIN_COLUMN
            )
            if current.empty:
                continue
            x = current[TIME_BIN_COLUMN].to_numpy(dtype=float)
            color = STRATUM_COLORS[stratum]
            axis.plot(x, current["Median"], color=color, linewidth=1.6, label=stratum)
            axis.fill_between(
                x,
                current["CI_Low"],
                current["CI_High"],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set_xlim(-5, 16)
        axis.set_ylim(robust_profile_ylim(profiles, outcome, suppression=True))
        add_panel_title(axis, outcome)
        axis.set_ylabel("Suppression from pre-CS baseline")
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    add_shared_figure_labels(
        fig,
        handles,
        labels,
        title="3 s trace: anticipatory suppression pooled across catch trials",
        subtitle="Positive values indicate less movement or weaker bouts than the same trial's pre-CS baseline",
        ylabel="",
    )
    fig.subplots_adjust(top=0.69, bottom=0.16, left=0.09, right=0.98, wspace=0.22)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return pd.concat(panel_frames, ignore_index=True)


def run_pipeline(config: PipelineConfig) -> None:
    validate_config(config)
    repository_root = Path(__file__).resolve().parent
    output_dir = Path(config.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    manifest_dir = output_dir / "manifests"
    panel_dir = output_dir / "panel_data"
    figure_dir = output_dir / "figures"
    provenance_dir = output_dir / "provenance"
    for directory in (checkpoint_dir, manifest_dir, panel_dir, figure_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    condition_inputs = {
        REFERENCE_CONDITION: Path(config.control_path),
        CONDITIONED_CONDITION: Path(config.trace_path),
    }
    trial_paths: list[Path] = []
    temporal_paths: list[Path] = []
    for condition, source_path in condition_inputs.items():
        trial_path, temporal_path = process_condition(
            source_path, condition, checkpoint_dir, config
        )
        trial_paths.append(trial_path)
        temporal_paths.append(temporal_path)

    trial_summary = pd.concat([pd.read_csv(path) for path in trial_paths], ignore_index=True)
    trial_summary_path = checkpoint_dir / "all_conditions_trial_summary.csv"
    trial_summary.to_csv(trial_summary_path, index=False)
    analysis6_input, classifier_key_map = prepare_analysis6_input(trial_summary)
    analysis6_input_path = checkpoint_dir / "analysis6_input_trial_summary.csv"
    analysis6_input.to_csv(analysis6_input_path, index=False)
    classifier_key_map.to_csv(
        manifest_dir / "classifier_fish_key_map.csv",
        index=False,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    classifier_run_id = f"all3sTrace_analysis6_logmedian_{timestamp}"
    classification = run_analysis6_classifier(
        analysis6_input_path,
        output_dir,
        repository_root,
        config.classifier_alpha,
    )
    classification = restore_source_fish_ids(classification, classifier_key_map)
    classification.to_csv(manifest_dir / "analysis6_classification_raw.csv", index=False)
    manifest = create_manifest(
        classification,
        trial_summary,
        classifier_run_id,
        config.classifier_alpha,
    )
    manifest.to_csv(manifest_dir / "fish_classification_manifest.csv", index=False)

    sample_flow = (
        manifest.groupby(
            [CONDITION_COLUMN, "Cohort_Role", "Learner_Status", "Analysis_Stratum"],
            dropna=False,
        )
        .size()
        .rename("Fish_Count")
        .reset_index()
    )
    sample_flow.to_csv(manifest_dir / "classification_sample_flow.csv", index=False)

    temporal = pd.concat([pd.read_pickle(path) for path in temporal_paths], ignore_index=True)
    temporal = temporal.merge(
        manifest[
            [
                CONDITION_COLUMN,
                "Fish_ID",
                "Learner_Status",
                "Analysis_Stratum",
                "Classifier_Run_ID",
            ]
        ],
        on=[CONDITION_COLUMN, "Fish_ID"],
        how="left",
        validate="many_to_one",
    )
    if temporal["Analysis_Stratum"].isna().any():
        unmatched = temporal.loc[
            temporal["Analysis_Stratum"].isna(), [CONDITION_COLUMN, "Fish_ID"]
        ].drop_duplicates()
        unmatched.to_csv(manifest_dir / "unmatched_temporal_fish.csv", index=False)
        raise ValueError(
            f"{len(unmatched)} fish in temporal data did not match the classification manifest."
        )
    temporal.to_pickle(
        checkpoint_dir / "all_conditions_temporal_bins_labeled.pkl.gz",
        compression="gzip",
    )

    _, block_names = trial_to_block_map()
    for outcome, filename in (
        ("Conditional vigor", "conditional_vigor_by_10_trial_block"),
        ("Movement probability", "movement_probability_by_10_trial_block"),
    ):
        profiles, coverage = summarize_profiles(temporal, outcome, config)
        profiles.to_csv(panel_dir / f"{filename}.csv", index=False)
        coverage.to_csv(panel_dir / f"{filename}_coverage.csv", index=False)
        plot_profiles(profiles, block_names, outcome, figure_dir / filename)

    for outcome, filename in (
        ("Conditional vigor", "conditional_vigor_by_catch_trial"),
        ("Movement probability", "movement_probability_by_catch_trial"),
    ):
        profiles, coverage = summarize_catch_profiles(temporal, outcome, config)
        profiles.to_csv(panel_dir / f"{filename}.csv", index=False)
        coverage.to_csv(panel_dir / f"{filename}_coverage.csv", index=False)
        plot_catch_profiles(profiles, outcome, figure_dir / filename)

    timing_metrics, timing_summary = calculate_catch_timing_metrics(temporal, config)
    timing_metrics.to_csv(panel_dir / "catch_trial_timing_metrics_per_fish.csv", index=False)
    timing_summary.to_csv(panel_dir / "catch_trial_timing_summary.csv", index=False)
    pooled_suppression = plot_pooled_catch_suppression(
        temporal,
        config,
        figure_dir / "catch_trials_pooled_anticipatory_suppression",
    )
    pooled_suppression.to_csv(
        panel_dir / "catch_trials_pooled_anticipatory_suppression.csv",
        index=False,
    )

    source_info = {
        condition: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "content_hash": None,
            "content_hash_note": (
                "Not calculated to avoid rereading multi-gigabyte source files after "
                "validated checkpoints were written."
            ),
        }
        for condition, path in condition_inputs.items()
    }
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "descriptive_classifier_characterization",
        "circularity_notice": (
            "Learner labels and plotted behavior derive from the same experiment. "
            "These figures are descriptive, not independent confirmation."
        ),
        "classifier_run_id": classifier_run_id,
        "classifier": CLASSIFIER_FILENAME,
        "classifier_reason": (
            "The bf46bf7 baseline provides one canonical Analysis 6 implementation. "
            "It models median log-vigor differences directly, uses a directional joint "
            "statistic, and calibrates the control false-positive target at 0.05."
        ),
        "catch_trials": list(CATCH_TRIALS),
        "catch_trial_source": (
            "Existing 4_ScaledVigorPlotting_LogMedian.py training catch-trial "
            "definition."
        ),
        "timing_windows_s": TIMING_WINDOWS,
        "pipeline_config": asdict(config),
        "software": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "sources": source_info,
    }
    (provenance_dir / "run.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"[done] Outputs saved under {output_dir}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = PipelineConfig(
        trace_path=str(args.trace.resolve()),
        control_path=str(args.control.resolve()),
        output_dir=str(args.output.resolve()),
        time_min_s=args.time_min,
        time_max_s=args.time_max,
        time_bin_s=args.time_bin,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        classifier_alpha=args.classifier_alpha,
        force=args.force,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
