"""
Improved learner quantification pipeline (standalone)
=====================================================

This is a standalone analysis script that implements improved learner classification
for `6_LearnersQuantification_new.py`, without importing or loading that script.

Data Model:
- Loads pooled per-trial behavioral data (CSV or gzipped pickle) containing:
  - "Mean CR": mean raw vigor (deg/ms) in the conditioned response window
  - "Mean X s before": mean raw vigor in the baseline window  
  - "Normalized vigor": ratio of Mean CR / Mean baseline (already a relative measure)
- Preprocesses into consistent 5-trial block structure and filters fish by trial coverage
- Extracts per-fish BLUP features for learning epochs using control-anchored MixedLM

Statistical Approach:
The LME models (optionally log-transformed) normalized vigor by epoch:
    [log](Normalized vigor) ~ Epoch
When USE_LOG_TRANSFORM is True, the response is log(normalized vigor); BLUPs are then
log-fold changes (e.g. BLUP = -0.1 ≈ 10% decrease). Otherwise the model uses raw normalized
vigor and BLUPs are absolute changes in the ratio.

Classification Method:
- Computes directional z-scores per feature where "learning direction" = positive z
- Optionally uses per-fish SE for uncertainty-aware scoring (default is SE-free control-SD scaling)
- Computes joint statistic T (whitened Mahalanobis-like quadratic form on one-sided z-scores)
- Empirically calibrates threshold on control fish to achieve target false-positive rate (alpha)
- Classifies as learner if: T > threshold AND sufficient features point in learning direction

Features:
- "acquisition": Change from Pre-Train to Late Train + Early Test (expected: response suppression)
- "extinction": Change from Late Train + Early Test to Late Test (expected: spontaneous recovery)
  Note: The "extinction" feature actually measures response changes during the test phase,
  which reflects spontaneous recovery if the response increases, not extinction per se.

Outputs:
- `Fish_Learner_Classification_<CSUS>.csv` in the pooled data folder
"""

from __future__ import annotations

try:
    # If run inside IPython/Jupyter, enable inline plotting.
    from IPython import get_ipython  # type: ignore

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas.api.types import CategoricalDtype
from scipy.spatial import distance
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

import analysis_utils
import figure_saving
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config

# region Parameters
# ==============================================================================
# GENERAL RUNTIME SETTINGS
# ==============================================================================

# Set plotting style (shared across analysis scripts)
plotting_style.set_plot_style()

# ==============================================================================
# EXPERIMENT SELECTION
# ==============================================================================

EXPERIMENT: str = ExperimentType.ALL_DELAY.value
exp_config = get_experiment_config(EXPERIMENT)

# ==============================================================================
# PIPELINE CONTROL FLAGS
# ==============================================================================
RUN_PLOT_TRAJECTORIES: bool = False
RUN_PLOT_FEATURE_SPACE: bool = False
RUN_PLOT_BLUP_CATERPILLAR: bool = False
RUN_PLOT_INDIVIDUALS_AND_GRID: bool = True
RUN_PLOT_BLUP_OVERLAY: bool = True
RUN_PLOT_HEATMAP_GRID: bool = True
RUN_EXPORT_RESULTS: bool = True

# ==============================================================================
# CORE ANALYSIS CONFIGURATION PARAMETERS (used by analysis_cfg below)
# ==============================================================================

FEATURES_TO_USE: List[str] = ["acquisition", "extinction"]
PRETRAIN_TO_TRAIN_END_EARLYTEST_BLOCKS: Tuple[List[str], List[str]] = (
    ["Early Pre-Train", "Late Pre-Train"],
    ["Train 6", "Train 7", "Train 8", "Train 9", "Late Train", "Early Test"],
)
TRAIN_END_EARLYTEST_TO_LATE_TEST_BLOCKS: Tuple[List[str], List[str]] = (
    ["Train 6", "Train 7", "Train 8", "Train 9", "Late Train", "Early Test"],
    ["Test 5", "Late Test"],
)

MIN_PRETRAIN_TRIALS: int = 6
MIN_LATETRAINDEARLYTEST_TRIALS: int = 6
MIN_LATE_TEST_TRIALS: int = 6
MIN_TRIALS_PER_5TRIAL_BLOCK_IN_EPOCH: int = 3

# Active parameters
CI_MULTIPLIER_FOR_REPORTING: float = 1.96  # For display: 1.96 = 95% CI (z_{0.975})
ANALYSIS_RANDOM_SEED: Optional[int] = 0  # For reproducibility in any stochastic operations
Y_LIM_PLOT: Tuple[float, float] = (0.8, 1.2)  # Default y-axis limits for plots

# ==============================================================================
# DATA LOADING / COLUMN CONVENTIONS
# ==============================================================================
POOLED_DATA_REQUIRED_SUBSTRING: str = "NV per trial per fish"
BASELINE_COLUMN_SUBSTRING: str = "s before"
RESPONSE_COLUMN_NAME: str = "Mean CR"
EPOCH_BLOCK_TRIALS: int = 5
MIN_FISH_WITH_ALL_FEATURES: int = 10

# Which pooled data file to load (must match 5_NormalizedVigorPlotting output):
# - "nanFracFilt": require _nanFracFilt in filename (APPLY_MAX_NAN_FRAC_PER_WINDOW was True)
# - "no_nanFracFilt": require filename WITHOUT _nanFracFilt
# - "auto": accept either, prefer newest by mtime
POOLED_DATA_NAN_FILTER: str = "no_nanFracFilt"  # or "no_nanFracFilt" | "auto"

# Optional: explicit path to load (overrides search; set to None for auto-discovery)
POOLED_DATA_PATH: Optional[Path] = None

# ==============================================================================
# FIGURE EXPORT + OUTPUT NAMING
# ==============================================================================
FIG_DPI_INDIVIDUAL_FISH: int = 150
FIG_DPI_SUMMARY_GRID: int = 200
FIG_DPI_BLUP_OVERLAY: int = 200

FNAME_TRAJECTORIES: str = "Learner_Classification_Trajectories clean.png"
FNAME_FEATURE_SPACE: str = "Feature_Space_Scatter.png"
FNAME_BLUP_OVERLAY: str = "BLUP_Trajectories_Overlay.png"
FNAME_BLUP_CATERPILLAR_TEMPLATE: str = "BLUP_Caterpillar_{feat}.png"
DIR_INDIVIDUAL_PLOTS_COMBINED: str = "Individual_Fish_Plots_Combined"
FNAME_CLASSIFICATION_RESULTS_TEMPLATE: str = "Fish_Learner_Classification_{csus}.csv"

# Colors (aligned with other analysis scripts)
COLOR_PROB_LEARNER: str = "red"
COLOR_OUTLIER_WRONG_DIR: str = "gray"
COLOR_CTRL_NONLEARNER: str = "blue"
COLOR_EXP_NONLEARNER: str = "blue"
COLOR_BLUP_TRAJECTORY: str = "purple"
COLOR_GROUP_MEDIAN: str = "black"
COLOR_BASELINE_VIGOR: str = "green"
COLOR_RESPONSE_VIGOR: str = "deeppink"

# ==============================================================================
# IMPROVED CLASSIFIER SETTINGS
# ==============================================================================

# Target control false-positive rate for the *binary* learner label.
# The threshold is empirically calibrated so that ~alpha fraction of controls
# are classified as learners (false positives).
ALPHA_TARGET: float = 0.05

# Joint statistic: always use a whitened quadratic form (Mahalanobis-like) on
# positive-direction z-scores:  T = z_pos^T Σ^{-1} z_pos
# where Σ is estimated from control directional z-scores using Ledoit-Wolf shrinkage.
# (The old "independent" sum-of-squares option is intentionally removed.)

# Whether to use per-fish SE in the directional scoring.
# - False (default): ignore per-fish SE; scale by control-feature SD only (Mahalanobis-like).
# - True: include per-fish SE (and SE of control mean) in the denominator (uncertainty-aware).
USE_PER_FISH_SE_IN_SCORING: bool = True

# Numerical stability constant for division
EPS: float = 1e-12

# Fit LME on log(normalized vigor) when True (recommended: ratios are multiplicative,
# log stabilizes variance and yields BLUPs as log-fold changes).
USE_LOG_TRANSFORM: bool = True

# endregion


@dataclass
class FeatureConfig:
    name: str
    direction: str  # "negative" or "positive"
    description: str


@dataclass
class AnalysisConfig:
    """Configuration parameters for the learner classification analysis.
    
    Note on features:
    - "acquisition": Measures response change from Pre-Train (CS alone, no learning)
      to Late Train + Early Test (after CS-US pairings). Expected direction is "negative"
      meaning learned fish show DECREASED vigor (conditioned suppression) relative to controls.
    - "extinction": Measures response change from Late Train + Early Test to Late Test.
      Expected direction is "positive" meaning learned fish show INCREASED vigor 
      (spontaneous recovery of suppressed response) in late test.
      
    The feature directions assume a CONDITIONED SUPPRESSION paradigm where:
    - Learning = suppression of baseline activity during CS
    - Recovery = return toward baseline activity when CS-US contingency weakens
    """
    csus: str = "CS"
    output_format: str = "png"
    random_seed: Optional[int] = 10
    min_pretrain_trials: int = 10
    min_latetraindearlytest_trials: int = 15
    min_late_test_trials: int = 10
    min_trials_per_5trial_block_in_epoch: int = 3
    y_lim_plot: Tuple[float, float] = (0.8, 1.2)
    features_to_use: List[str] = field(default_factory=lambda: ["acquisition", "extinction"])
    ci_multiplier_for_reporting: float = 1.96  # For 95% CI display (1.96 = z_{0.975})
    feature_configs: Dict[str, FeatureConfig] = field(
        default_factory=lambda: {
            "acquisition": FeatureConfig(
                "Conditioned Suppression",  # More accurate name for the phenomenon
                "negative",  # Learning = DECREASED response (suppression)
                "Response change from Pre-Train to Late Train + Early Test (learned suppression)",
            ),
            "extinction": FeatureConfig(
                "Spontaneous Recovery",  # More accurate: this measures recovery, not extinction
                "positive",  # Learning = INCREASED response (recovery from suppression)
                "Response change from Late Train + Early Test to Late Test (recovery toward baseline)",
            ),
        }
    )
    pretrain_to_train_end_earlytest_blocks: Tuple[List[str], List[str]] = field(
        default_factory=lambda: (["Early Pre-Train", "Late Pre-Train"], ["Train 6", "Train 7", "Train 8", "Train 9", "Late Train", "Early Test"])
    )
    train_end_earlytest_to_late_test_blocks: Tuple[List[str], List[str]] = field(
        default_factory=lambda: (["Train 6", "Train 7", "Train 8", "Train 9", "Late Train", "Early Test"], ["Test 5", "Late Test"])
    )
    pretrain_blocks_5: List[str] = field(default_factory=lambda: ["Early Pre-Train", "Late Pre-Train"])
    earlytest_block: List[str] = field(default_factory=lambda: ["Late Train", "Early Test"])
    late_test_blocks_5: List[str] = field(default_factory=lambda: ["Test 5", "Late Test"])
    cond_types: List[str] = field(default_factory=list)
    use_log_transform: bool = True  # Fit LME on log(normalized vigor) when True

    def __post_init__(self) -> None:
        self.cond_types = list(exp_config.cond_types)
        # No additional validation needed here; scoring enforces direction-vote requirements
        # based on the number of selected features.


analysis_cfg = AnalysisConfig(
    features_to_use=FEATURES_TO_USE,
    pretrain_to_train_end_earlytest_blocks=PRETRAIN_TO_TRAIN_END_EARLYTEST_BLOCKS,
    train_end_earlytest_to_late_test_blocks=TRAIN_END_EARLYTEST_TO_LATE_TEST_BLOCKS,
    min_pretrain_trials=MIN_PRETRAIN_TRIALS,
    min_latetraindearlytest_trials=MIN_LATETRAINDEARLYTEST_TRIALS,
    min_late_test_trials=MIN_LATE_TEST_TRIALS,
    min_trials_per_5trial_block_in_epoch=MIN_TRIALS_PER_5TRIAL_BLOCK_IN_EPOCH,
    random_seed=ANALYSIS_RANDOM_SEED,
    y_lim_plot=Y_LIM_PLOT,
    ci_multiplier_for_reporting=CI_MULTIPLIER_FOR_REPORTING,
    use_log_transform=USE_LOG_TRANSFORM,
)


@dataclass
class BLUPResult:
    blup: float
    se: float
    ci_lower: float
    ci_upper: float
    fixed: float
    random: float


def _read_pickle_robust(path: Path) -> pd.DataFrame:
    """Robust pickle loader for pandas objects (handles version skew)."""
    path = Path(path)
    try:
        return pd.read_pickle(path, compression="gzip")
    except Exception:
        pass

    try:
        import gzip
        from pandas.compat import pickle_compat

        with gzip.open(path, "rb") as f:
            obj = pickle_compat.load(f)
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    except Exception:
        # Fall through: caller will try CSV, or raise.
        raise


def load_pooled_data(config: AnalysisConfig, path_pooled_data: Path) -> pd.DataFrame:
    """Load newest pooled dataset for the selected CS/US, preferring CSV if pickles fail."""

    # Early exit: explicit path override
    if POOLED_DATA_PATH is not None:
        p = Path(POOLED_DATA_PATH)
        if not p.exists():
            raise FileNotFoundError(f"POOLED_DATA_PATH does not exist: {p}")
        try:
            print(f"  Loading (explicit path): {p.name}")
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                df = _read_pickle_robust(p)
            if df is not None and not df.empty:
                print(f"  Read data from: {p.resolve()}")
                return df
            raise RuntimeError(f"File is empty or invalid: {p}")
        except Exception as e:
            raise RuntimeError(f"Failed to load POOLED_DATA_PATH: {e}") from e

    def _matches_csus(stem: str, csus: str) -> bool:
        # Supports "..._CS_allFish", "..._CS_selectedFish", and "..._CS" patterns.
        return stem.endswith(f"_{csus}") or (f"_{csus}_" in stem)

    # Search both the main pooled-data folder and common subfolders.
    # (Some pipelines move older/compatible pooled files into `old/`.)
    search_roots = [Path(path_pooled_data)]
    for rel in ["old", "old/all fish", "old/all_fish"]:
        p = Path(path_pooled_data) / rel
        if p.exists():
            search_roots.append(p)

    all_files: List[Path] = []
    for root in search_roots:
        all_files.extend([p for p in root.glob("*") if p.is_file()])

    candidates = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)
    candidates = [
        p
        for p in candidates
        if (POOLED_DATA_REQUIRED_SUBSTRING in p.stem)
        and _matches_csus(p.stem, str(config.csus))
        and p.suffix.lower() in {".csv", ".pkl", ".pickle"}
    ]

    if POOLED_DATA_NAN_FILTER == "nanFracFilt":
        candidates = [p for p in candidates if "_nanFracFilt" in p.stem]
    elif POOLED_DATA_NAN_FILTER == "no_nanFracFilt":
        candidates = [p for p in candidates if "_nanFracFilt" not in p.stem]
    # "auto": no extra filter

    if not candidates:
        raise FileNotFoundError(
            f"No matching pooled data file found (csv/pkl) for CS/US={config.csus} "
            f"with POOLED_DATA_NAN_FILTER={POOLED_DATA_NAN_FILTER}. "
            f"Try 'auto' to accept either nanFracFilt or non-nanFracFilt files."
        )

    # Prefer CSV when present (pickle compatibility is fragile across pandas versions).
    candidates = sorted(candidates, key=lambda p: (p.suffix.lower() != ".csv", -p.stat().st_mtime))

    last_err: Optional[Exception] = None
    for p in candidates:
        try:
            print(f"  Loading: {p.name}")
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                df = _read_pickle_robust(p)
            if df is not None and not df.empty:
                print(f"  Read data from: {p.resolve()}")
                return df
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to load pooled data. Last error: {last_err}")


def _create_5_trial_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Create 5-trial block structure from 10-trial blocks.
    
    This function subdivides the original 10-trial blocks into 5-trial blocks
    for finer-grained epoch analysis. Block names are hardcoded based on the
    expected experimental design (9 or 12 original blocks).
    
    LIMITATION: Block names are hardcoded and assume specific experimental designs.
    To support other designs, modify the block_names lists or make them configurable.
    
    Args:
        df: DataFrame with 'Block name' and 'Trial number' columns
        
    Returns:
        DataFrame with new 'Block name 5 trials' column
    """
    number_blocks_original = df["Block name"].nunique()
    number_trials_block = int(EPOCH_BLOCK_TRIALS)

    if number_blocks_original == 9:
        block_names = [
            "Early Pre-Train",
            "Late Pre-Train",
            "Early Train",
            "Train 2",
            "Train 3",
            "Train 4",
            "Train 5",
            "Train 6",
            "Train 7",
            "Train 8",
            "Train 9",
            "Late Train",
            "Early Test",
            "Test 2",
            "Test 3",
            "Test 4",
            "Test 5",
            "Late Test",
        ]
    elif number_blocks_original == 12:
        block_names = [
            "Early Pre-Train",
            "Late Pre-Train",
            "Early Train",
            "Train 2",
            "Train 3",
            "Train 4",
            "Train 5",
            "Train 6",
            "Train 7",
            "Train 8",
            "Train 9",
            "Late Train",
            "Early Test",
            "Test 2",
            "Test 3",
            "Test 4",
            "Test 5",
            "Late Test",
            "Early Re-Train",
            "Re-Train 2",
            "Re-Train 3",
            "Re-Train 4",
            "Re-Train 5",
            "Late Re-Train",
        ]
    else:
        raise ValueError(f"Unexpected number of blocks: {number_blocks_original}")

    # Keep only valid 10-trial blocks before splitting into 5-trial blocks.
    df = df[df["Block name"].isin(exp_config.names_cs_blocks_10)].copy()
    df["Block name"] = df["Block name"].astype(CategoricalDtype(categories=exp_config.names_cs_blocks_10, ordered=True))
    df.drop(columns="Block name", inplace=True)

    blocks = [
        range(x, x + number_trials_block)
        for x in range(
            exp_config.trials_cs_blocks_10[0][0],
            exp_config.trials_cs_blocks_10[-1][-1] + 1,
            number_trials_block,
        )
    ]

    df[f"Block name {number_trials_block} trials"] = ""
    for idx, trials in enumerate(blocks):
        mask = df["Trial number"].astype(int).isin(trials)
        df.loc[mask, f"Block name {number_trials_block} trials"] = block_names[idx]

    df[f"Block name {number_trials_block} trials"] = df[f"Block name {number_trials_block} trials"].astype(
        CategoricalDtype(categories=block_names, ordered=True)
    )
    return df


def _filter_fish_by_trials(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Filter fish based on minimum trial requirements for the selected features."""
    valid_fish_sets: List[set] = []

    def _valid_each_block(blocks: List[str]) -> set:
        """Fish with >= min_trials_per_5trial_block_in_epoch trials in EACH block in `blocks`."""
        if not blocks:
            return set(df["Fish_ID"].unique())
        counts = (
            df[df["Block name 5 trials"].isin(blocks)]
            .groupby(["Fish_ID", "Block name 5 trials"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        for b in blocks:
            if b not in counts.columns:
                counts[b] = 0
        mask = np.ones(len(counts), dtype=bool)
        for b in blocks:
            mask &= counts[b].to_numpy() >= int(config.min_trials_per_5trial_block_in_epoch)
        return set(counts.index[mask])

    if "acquisition" in config.features_to_use:
        pre_blocks, post_blocks = config.pretrain_to_train_end_earlytest_blocks
        valid_pre_each = _valid_each_block(pre_blocks)
        valid_post_each = _valid_each_block(post_blocks)

        pre_counts = df[df["Block name 5 trials"].isin(pre_blocks)].groupby("Fish_ID", observed=True).size()
        valid_pre = set(pre_counts[pre_counts >= config.min_pretrain_trials].index)

        post_counts = df[df["Block name 5 trials"].isin(post_blocks)].groupby("Fish_ID", observed=True).size()
        valid_post = set(post_counts[post_counts >= config.min_latetraindearlytest_trials].index)

        valid_acq = valid_pre & valid_post & valid_pre_each & valid_post_each
        valid_fish_sets.append(valid_acq)

        print(
            f"   Feature acquisition (Outcome Drop): {len(valid_acq)} fish with "
            f">= {config.min_pretrain_trials} trials in {pre_blocks} AND "
            f">= {config.min_latetraindearlytest_trials} trials in {post_blocks}"
            f" (and >= {config.min_trials_per_5trial_block_in_epoch} trials in EACH 5-trial block)"
        )

    if "extinction" in config.features_to_use:
        pre_blocks, post_blocks = config.train_end_earlytest_to_late_test_blocks
        valid_pre_each = _valid_each_block(pre_blocks)
        valid_post_each = _valid_each_block(post_blocks)

        pre_counts = df[df["Block name 5 trials"].isin(pre_blocks)].groupby("Fish_ID", observed=True).size()
        valid_pre = set(pre_counts[pre_counts >= config.min_latetraindearlytest_trials].index)

        post_counts = df[df["Block name 5 trials"].isin(post_blocks)].groupby("Fish_ID", observed=True).size()
        valid_post = set(post_counts[post_counts >= config.min_late_test_trials].index)

        valid_ext = valid_pre & valid_post & valid_pre_each & valid_post_each
        valid_fish_sets.append(valid_ext)

        print(
            f"   Feature extinction (Recovery Increase): {len(valid_ext)} fish with "
            f">= {config.min_latetraindearlytest_trials} trials in {pre_blocks} AND "
            f">= {config.min_late_test_trials} trials in {post_blocks}"
            f" (and >= {config.min_trials_per_5trial_block_in_epoch} trials in EACH 5-trial block)"
        )

    valid_fish = set.intersection(*valid_fish_sets) if valid_fish_sets else set(df["Fish_ID"].unique())
    print(
        f"  Fish filtering: {len(df['Fish_ID'].unique())} -> {len(valid_fish)} "
        f"(based on selected features: {config.features_to_use})"
    )
    return df[df["Fish_ID"].isin(valid_fish)]


def prepare_data(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Prepare data with proper block structure and filtering."""
    df = df.copy()

    df["Trial type"] = config.csus
    df = analysis_utils.identify_blocks_trials(df, exp_config.blocks_dict)
    df["Block name 10 trials"] = df["Block name"]

    df = _create_5_trial_blocks(df)

    baseline_candidates = [c for c in df.columns if BASELINE_COLUMN_SUBSTRING in c]
    if not baseline_candidates:
        raise KeyError(f"No baseline column matching substring {BASELINE_COLUMN_SUBSTRING!r}")
    baseline_col = baseline_candidates[0]
    response_col = RESPONSE_COLUMN_NAME

    df = df.rename(columns={"Exp.": "Condition", "Fish": "Fish_ID"})
    df = df[df["Condition"].isin(config.cond_types)]
    df.dropna(
        subset=[
            "Normalized vigor",
            "Trial number",
            "Block name 5 trials",
            "Block name 10 trials",
            baseline_col,
            response_col,
        ],
        inplace=True,
    )

    # When using log transform, we need strictly positive values for Normalized vigor.
    # Filter out non-positive or non-finite values early so downstream LME sees clean data.
    if config.use_log_transform:
        nv = df["Normalized vigor"].to_numpy(dtype=float)
        valid_nv = (nv > 0) & np.isfinite(nv)
        n_invalid = int((~valid_nv).sum())
        if n_invalid > 0:
            print(f"  [prepare_data] Dropping {n_invalid} rows with Normalized vigor <= 0 or non-finite (required for log transform)")
            df = df.loc[valid_nv].copy()

    df = _filter_fish_by_trials(df, config)

    df["Trial number"] = df["Trial number"].astype(int)
    df["Trial number"] = df["Trial number"] - df["Trial number"].min() + 1

    return df


def _extract_random_variance(effects: pd.Series, cov_re: pd.DataFrame, re_var: str) -> float:
    if not hasattr(cov_re, "iloc"):
        return 0.0
    try:
        re_var_idx = list(effects.index).index(re_var) if re_var in effects.index else 0
        arr = cov_re.to_numpy(dtype=float)
        return float(arr[re_var_idx, re_var_idx])
    except Exception:
        return 0.0


def get_blups_with_uncertainty(
    df: pd.DataFrame,
    formula: str,
    groups_col: str,
    re_var: str,
    method: str = "powell",
    ci_multiplier: float = 1.96,
) -> Dict[str, BLUPResult]:
    """Extract BLUPs with uncertainty estimates (best-effort)."""
    if df.empty or df[groups_col].nunique() < 2:
        return {}

    try:
        model = smf.mixedlm(formula, df, groups=df[groups_col], re_formula=f"~{re_var}")
        result = model.fit(reml=True, method=method)

        fixed_val = float(result.params[re_var])
        fixed_se = float(result.bse[re_var])
        scale = float(result.scale)
        G = result.cov_re.values

        try:
            G_inv = np.linalg.inv(G)
            use_cond_var = True
        except np.linalg.LinAlgError:
            G_inv = None
            use_cond_var = False

        try:
            re_idx = int(result.cov_re.columns.get_loc(re_var))
        except Exception:
            re_idx = 0

        blups: Dict[str, BLUPResult] = {}
        for group_id, effects in result.random_effects.items():
            random_val = float(effects.get(re_var, 0.0))
            blup_val = fixed_val + random_val

            cond_var = 0.0
            calculated = False
            if use_cond_var and hasattr(model, "row_indices"):
                idxs = model.row_indices.get(group_id)
                if idxs is not None and G_inv is not None:
                    try:
                        Z_i = model.exog_re[idxs]
                        precision = G_inv + (Z_i.T @ Z_i) / scale
                        v_post = np.linalg.inv(precision)
                        cond_var = float(v_post[re_idx, re_idx])
                        calculated = True
                    except Exception:
                        calculated = False

            if not calculated:
                cond_var = float(_extract_random_variance(effects, result.cov_re, re_var))

            total_se = float(np.sqrt(max(0.0, fixed_se**2 + cond_var)))
            blups[str(group_id)] = BLUPResult(
                blup=blup_val,
                se=total_se,
                ci_lower=blup_val - ci_multiplier * total_se,
                ci_upper=blup_val + ci_multiplier * total_se,
                fixed=fixed_val,
                random=random_val,
            )
        return blups
    except Exception as e:
        print(f"  [ERROR] BLUP extraction failed: {e}")
        return {}


def extract_change_feature(
    data: pd.DataFrame,
    config: AnalysisConfig,
    *,
    number_trials: int = 5,
    name_blocks: Tuple[List[str], List[str]],
) -> Dict[str, BLUPResult]:
    """Extract per-fish BLUP for epoch change using control-anchored mixed-effects model.
    
    Statistical Model:
    -----------------
    When config.use_log_transform is True (default), the LME fits log(normalized vigor):
        log(Normalized vigor) ~ Epoch
    so BLUPs are log-fold changes (e.g. BLUP = -0.1 ≈ 10% decrease). Otherwise:
        Normalized vigor ~ Epoch
    and BLUPs are absolute changes in the ratio.
    - Normalized vigor = Mean CR / Mean baseline (already a relative measure)
    - Epoch = 0 (pre-epoch blocks) or 1 (post-epoch blocks)
    
    Control-Anchoring:
    -----------------
    1. Fit model on control fish only to get fixed Epoch effect (μ_ctrl)
    2. Adjust all responses: Vigor_Adj = response - μ_ctrl × Epoch  (response = log(NV) or NV)
    3. Fit second model on adjusted data to get individual random Epoch deviations
    4. Final BLUP = μ_ctrl + random_effect_i (individual deviation from control)
    
    This centers the control group at zero deviation, making experimental fish
    deviations interpretable as "excess change beyond control expectation."
    
    LIMITATION: 5-trial block aggregation loses trial-level temporal information.
    Consider trial-by-trial analysis if acquisition dynamics are important.
    
    Args:
        data: DataFrame with Normalized vigor, Fish_ID, Block name columns
        config: Analysis configuration
        number_trials: Block size (default 5)
        name_blocks: Tuple of (pre_blocks, post_blocks) names
        
    Returns:
        Dict mapping Fish_ID to BLUPResult with BLUP, SE, CI bounds
    """
    print(f"\n  [B/C] Extracting Epoch Change: {name_blocks[0]} -> {name_blocks[1]} (Control-Anchored)...")

    frames: List[pd.DataFrame] = []
    for fish in data["Fish_ID"].unique():
        sub = data[data["Fish_ID"] == fish]
        pre = (
            sub[sub[f"Block name {number_trials} trials"].isin(name_blocks[0])]
            .sort_values("Trial number")
            .assign(Epoch=0)
        )
        test = (
            sub[sub[f"Block name {number_trials} trials"].isin(name_blocks[1])]
            .sort_values("Trial number")
            .assign(Epoch=1)
        )
        if not pre.empty and not test.empty:
            frames.append(pd.concat([pre, test]))

    if not frames:
        print("  [WARN] Insufficient data")
        return {}

    df_B = pd.concat(frames)

    # Optionally fit on log(normalized vigor) for multiplicative effects and variance stabilization
    if config.use_log_transform:
        # Log is only valid for strictly positive values; drop non-positive or non-finite
        nv = df_B["Normalized vigor"].to_numpy(dtype=float)
        valid = (nv > 0) & np.isfinite(nv)
        if not np.all(valid):
            n_dropped = int((~valid).sum())
            df_B = df_B.loc[valid].copy()
            if df_B.empty or df_B["Fish_ID"].nunique() < 2:
                print(f"  [WARN] After dropping {n_dropped} rows with non-positive/non-finite Normalized vigor, insufficient data for log-scale LME. Returning empty.")
                return {}
        df_B["Log_Normalized_Vigor"] = np.log(df_B["Normalized vigor"])
        response_col = "Log_Normalized_Vigor"
        formula_response = "Log_Normalized_Vigor ~ Epoch"
    else:
        response_col = "Normalized vigor"
        formula_response = "Q('Normalized vigor') ~ Epoch"

    ref_cond = config.cond_types[0]
    df_ctrl = df_B[df_B["Condition"] == ref_cond].copy()
    if df_ctrl["Fish_ID"].nunique() < 3:
        print("  [WARN] Not enough control fish for anchoring, using standard method")
        return get_blups_with_uncertainty(
            df_B,
            formula_response,
            "Fish_ID",
            "Epoch",
            ci_multiplier=config.ci_multiplier_for_reporting,
        )

    print(f"    Calculating anchor from {df_ctrl['Fish_ID'].nunique()} {ref_cond} fish")
    try:
        model_ctrl = smf.mixedlm(
            formula_response,
            df_ctrl,
            groups=df_ctrl["Fish_ID"],
            re_formula="~Epoch",
        )
        res_ctrl = model_ctrl.fit(reml=True, method="powell")
        ctrl_epoch_effect = float(res_ctrl.params["Epoch"])
        ctrl_epoch_se = float(res_ctrl.bse.get("Epoch", np.nan))
        anchor_unit = "log-scale" if config.use_log_transform else "ratio"
        print(f"    Control Epoch Anchor: {ctrl_epoch_effect:.6f} ({anchor_unit})")
    except Exception as e:
        print(f"  [WARN] Control model failed ({e}), using standard method")
        return get_blups_with_uncertainty(
            df_B,
            formula_response,
            "Fish_ID",
            "Epoch",
            ci_multiplier=config.ci_multiplier_for_reporting,
        )

    df_B["Vigor_Adj"] = df_B[response_col] - (ctrl_epoch_effect * df_B["Epoch"])

    try:
        model = smf.mixedlm(
            "Q('Vigor_Adj') ~ 1",
            df_B,
            groups=df_B["Fish_ID"],
            re_formula="~Epoch",
        )
        result = model.fit(reml=True, method="powell")
        random_effects_cov = result.cov_re

        scale = float(result.scale)
        G = result.cov_re.values
        try:
            G_inv = np.linalg.inv(G)
            use_cond_var = True
        except np.linalg.LinAlgError:
            G_inv = None
            use_cond_var = False

        try:
            re_idx = int(result.cov_re.columns.get_loc("Epoch"))
        except Exception:
            re_idx = 0

        blups: Dict[str, BLUPResult] = {}
        for group_id, effects in result.random_effects.items():
            random_epoch = float(effects.get("Epoch", 0.0))
            final_blup = float(ctrl_epoch_effect + random_epoch)

            cond_var = float(_extract_random_variance(effects, random_effects_cov, "Epoch"))

            if use_cond_var and G_inv is not None and hasattr(model, "row_indices"):
                idxs = model.row_indices.get(group_id)
                if idxs is not None:
                    try:
                        Z_i = model.exog_re[idxs]
                        precision = G_inv + (Z_i.T @ Z_i) / scale
                        v_post = np.linalg.inv(precision)
                        cond_var = float(v_post[re_idx, re_idx])
                    except Exception:
                        pass

            fixed_se = float(ctrl_epoch_se) if np.isfinite(ctrl_epoch_se) else 0.0
            cond_var = float(cond_var) if np.isfinite(cond_var) else 0.0
            total_se = float(np.sqrt(fixed_se**2 + max(cond_var, 0.0)))
            if not np.isfinite(total_se) or total_se <= 0:
                total_se = 0.01

            blups[str(group_id)] = BLUPResult(
                blup=final_blup,
                se=total_se,
                ci_lower=final_blup - config.ci_multiplier_for_reporting * total_se,
                ci_upper=final_blup + config.ci_multiplier_for_reporting * total_se,
                fixed=float(ctrl_epoch_effect),
                random=float(random_epoch),
            )

        print(f"  [OK] Extracted for {len(blups)} fish (anchored to control)")
        return blups
    except Exception as e:
        print(f"  [ERROR] Anchored BLUP extraction failed: {e}")
        return get_blups_with_uncertainty(
            df_B,
            formula_response,
            "Fish_ID",
            "Epoch",
            ci_multiplier=config.ci_multiplier_for_reporting,
        )


def get_robust_mahalanobis(X: np.ndarray, X_ref: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Mahalanobis distances using Ledoit-Wolf shrinkage covariance.
    
    This function is used for DIAGNOSTIC purposes only in the improved pipeline.
    The actual learner classification uses directional z-scores and empirical calibration,
    not Mahalanobis distances.
    
    Ledoit-Wolf shrinkage is used because:
    - Small sample sizes make sample covariance estimates unreliable
    - Shrinkage toward a structured target (scaled identity) improves conditioning
    - The shrinkage intensity is automatically selected to minimize MSE
    
    The shrinkage intensity (alpha) is reported when verbose=True. Values close to 1
    indicate heavy shrinkage (small sample or high dimensionality), while values
    close to 0 indicate the sample covariance is trusted.
    
    Args:
        X: Feature matrix for all fish (n_fish x n_features)
        X_ref: Feature matrix for reference (control) fish
        verbose: If True, print shrinkage diagnostics
        
    Returns:
        Tuple of (distances array, inverse covariance matrix)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("--- Robust Covariance Estimation (diagnostic) ---")
        print(f"{'='*60}\n")

    mu_ref = np.mean(X_ref, axis=0)
    lw = LedoitWolf()
    cov_shrunk = lw.fit(X_ref).covariance_
    
    if verbose:
        # Report Ledoit-Wolf shrinkage intensity for validation
        shrinkage = lw.shrinkage_
        print(f"  Ledoit-Wolf shrinkage intensity: {shrinkage:.4f}")
        print(f"    (0 = sample covariance, 1 = scaled identity)")
        print(f"  Reference sample size: {X_ref.shape[0]}")
        print(f"  Number of features: {X_ref.shape[1]}")

    try:
        inv_cov = np.linalg.inv(cov_shrunk)
        if verbose:
            print(f"  Covariance matrix inverted successfully")
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_shrunk, rcond=1e-10)
        if verbose:
            print(f"  [WARN] Using pseudo-inverse due to singular covariance")

    dists: List[float] = []
    for x in X:
        try:
            dists.append(float(distance.mahalanobis(x, mu_ref, inv_cov)))
        except Exception:
            # Fallback to standardized Euclidean if Mahalanobis fails
            dists.append(float(np.sqrt(np.sum(((x - mu_ref) / np.std(X_ref, axis=0)) ** 2))))
    return np.array(dists, dtype=float), inv_cov


@dataclass
class ClassificationResult:
    """Result container used by plotting helpers."""

    fish_ids: np.ndarray
    conditions: np.ndarray
    features: np.ndarray
    features_se: np.ndarray
    feature_names: List[str]
    distances: np.ndarray
    votes_point: np.ndarray
    votes_probabilistic: np.ndarray
    p_learning: np.ndarray  # here: directional score p_dir = Phi(z_dir) (probability-like only when SE-aware)
    z_dir: np.ndarray       # directional z (learning direction positive)
    is_outlier: np.ndarray
    is_learner_point: np.ndarray
    is_learner_probabilistic: np.ndarray
    threshold: float
    threshold_ci: Tuple[float, float]
    mu_ctrl: np.ndarray
    T_scores: Optional[np.ndarray] = None      # joint T statistic per fish
    p_empirical: Optional[np.ndarray] = None   # empirical p-value per fish


def _get_fish_style(
    fish: str,
    fish_list: List[str],
    result: ClassificationResult,
    is_ref: bool,
) -> Tuple[str, Dict[str, Any]]:
    """Determine plot style for a fish (colors align across scripts)."""
    try:
        idx = list(result.fish_ids).index(fish)
    except ValueError:
        return COLOR_OUTLIER_WRONG_DIR, {"alpha": 0.3, "linewidth": 1}

    if bool(result.is_learner_probabilistic[idx]):
        return COLOR_PROB_LEARNER, {"alpha": 0.85, "linewidth": 2.2, "zorder": 4}
    if bool(result.is_outlier[idx]):
        return COLOR_OUTLIER_WRONG_DIR, {"alpha": 0.5, "linewidth": 1.3, "zorder": 3}

    color = COLOR_CTRL_NONLEARNER if is_ref else COLOR_EXP_NONLEARNER
    return color, {"alpha": 0.3, "linewidth": 1.0, "zorder": 1}


def plot_behavioral_trajectories(
    data: pd.DataFrame,
    result: ClassificationResult,
    config: AnalysisConfig,
    save_path: Optional[Path] = None,
    include_axis_key_panel: bool = True,
) -> Figure:
    """Behavioral trajectory summary (Normalized vigor per 10-trial block)."""
    block_order = (
        data.groupby("Block name 10 trials", observed=True)["Trial number"].mean().sort_values().index
    )
    df_viz = (
        data.groupby(["Fish_ID", "Block name 10 trials", "Condition"], observed=True)["Normalized vigor"]
        .median()
        .reset_index()
    )

    # Avoid clipping
    y_lo, y_hi = float(config.y_lim_plot[0]), float(config.y_lim_plot[1])
    try:
        y_min = float(np.nanmin(df_viz["Normalized vigor"].to_numpy(dtype=float)))
        y_max = float(np.nanmax(df_viz["Normalized vigor"].to_numpy(dtype=float)))
        y_lo2 = min(y_lo, y_min)
        y_hi2 = max(y_hi, y_max)
        pad = max(0.01, 0.02 * (y_hi2 - y_lo2))
        y_lim_used = (y_lo2 - pad, y_hi2 + pad)
    except Exception:
        y_lim_used = (y_lo, y_hi)

    present_conditions = df_viz["Condition"].unique()
    unique_conds = [c for c in np.unique(result.conditions) if c in present_conditions]
    ref_cond = config.cond_types[0]

    n_panels = len(unique_conds) + (1 if include_axis_key_panel else 0)
    fig_w = (6 * len(unique_conds)) + (4.2 if include_axis_key_panel else 0)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, 6), sharex=True, sharey=True, facecolor="white"
    )
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)

    if include_axis_key_panel:
        cond_axes = axes[:-1]
        key_ax = axes[-1]
    else:
        cond_axes = axes
        key_ax = None

    fig.suptitle(
        "Learner Classification (Improved): Behavioral Trajectories",
        fontsize=5 + 14,
        fontweight="bold",
        y=0.98,
    )

    fish_list = list(result.fish_ids)
    vote_details = ["Feature direction details (z_dir>0 / p_dir>0.5):"]

    for ax, cond in zip(cond_axes, unique_conds):
        ax.set_facecolor("white")
        df_cond = df_viz[df_viz["Condition"] == cond]
        is_ref = (cond == ref_cond)

        fish_in_cond = df_cond["Fish_ID"].unique()
        fish_in_result = [f for f in fish_in_cond if f in fish_list]
        fish_not_in_result = [f for f in fish_in_cond if f not in fish_list]

        fish_indices = [fish_list.index(f) for f in fish_in_result]
        status_score = []
        for idx in fish_indices:
            score = 0
            if bool(result.is_outlier[idx]):
                score = 1
            if bool(result.is_learner_probabilistic[idx]):
                score = 2
            status_score.append(score)
        sorted_fish = [x for _, x in sorted(zip(status_score, fish_in_result))]

        for fish in fish_not_in_result:
            fish_data = df_cond[df_cond["Fish_ID"] == fish]
            color = COLOR_CTRL_NONLEARNER if is_ref else COLOR_EXP_NONLEARNER
            sns.lineplot(
                data=fish_data,
                x="Block name 10 trials",
                y="Normalized vigor",
                color=color,
                ax=ax,
                legend=False,
                alpha=0.2,
                linewidth=0.5,
                zorder=0,
            )

        for fish in sorted_fish:
            fish_data = df_cond[df_cond["Fish_ID"] == fish]
            color, kwargs = _get_fish_style(fish, fish_list, result, is_ref)
            sns.lineplot(
                data=fish_data,
                x="Block name 10 trials",
                y="Normalized vigor",
                color=color,
                ax=ax,
                legend=False,
                **kwargs,
            )

            idx = fish_list.index(fish)
            if bool(result.is_outlier[idx]) or bool(result.is_learner_probabilistic[idx]):
                feat_str: List[str] = []
                for f_idx, feat in enumerate(config.features_to_use):
                    z = float(result.z_dir[idx, f_idx])
                    p_dir = float(result.p_learning[idx, f_idx])
                    mark = "Y" if z > 0 else "N"
                    feat_str.append(f"{feat}:{mark}({p_dir:.0%})")
                symbol = "*" if bool(result.is_learner_probabilistic[idx]) else "o"
                vote_details.append(f"{symbol} {fish}: {' '.join(feat_str)}")

        grp = (
            df_cond.groupby("Block name 10 trials", observed=True)["Normalized vigor"]
            .median()
            .reset_index()
        )
        sns.lineplot(
            data=grp,
            x="Block name 10 trials",
            y="Normalized vigor",
            color=COLOR_GROUP_MEDIAN,
            marker="o",
            markersize=6,
            ax=ax,
            linewidth=2.5,
            zorder=10,
            label="Group Median",
        )

        cond_mask = (result.conditions == cond)
        n_fish_in_result = len(fish_in_result)
        n_fish_total = len(fish_in_cond)
        n_learn = int((cond_mask & result.is_learner_probabilistic).sum())
        n_out = int((cond_mask & result.is_outlier).sum())
        ax.set_title(
            f"{str(cond).capitalize()} (n={n_fish_in_result}/{n_fish_total})\n"
            f"Improved learners: {n_learn}  |  T>thr: {n_out}",
            fontsize=5 + 11,
            fontweight="bold",
        )

        ax.set_xlabel("Block", fontsize=5 + 10)
        ax.set_ylabel("Normalized Vigor", fontsize=5 + 10)
        ax.axhline(1.0, linestyle=":", color="black", alpha=0.5)
        ax.set_xticklabels(block_order, rotation=45, ha="right", fontsize=5 + 9)
        ax.grid(alpha=0.3)
        ax.set_ylim(y_lim_used)

    legend_elements = [
        Line2D([0], [0], color=COLOR_PROB_LEARNER, lw=2, label="* Improved Learner"),
        Line2D([0], [0], color=COLOR_OUTLIER_WRONG_DIR, lw=1.5, label="o T > threshold"),
        Line2D([0], [0], color=COLOR_CTRL_NONLEARNER, lw=1, label="Ctrl. (Non-learner)"),
        Line2D([0], [0], color=COLOR_EXP_NONLEARNER, lw=1, label="Exp. (Non-learner)"),
        Line2D([0], [0], color=COLOR_GROUP_MEDIAN, lw=2.5, marker="o", label="Group Median"),
    ]

    if include_axis_key_panel and key_ax is not None:
        key_ax.set_facecolor("white")
        key_ax.set_title("Axis / Legend Key", fontsize=5 + 11, fontweight="bold")
        key_ax.set_xlabel("Block (10-trial blocks)", fontsize=5 + 10)
        key_ax.set_ylabel("Normalized vigor (median per block)", fontsize=5 + 10)
        key_ax.set_xticks([])
        key_ax.grid(alpha=0.15)
        key_ax.set_ylim(y_lim_used)
        key_ax.legend(handles=legend_elements, loc="upper left", fontsize=5 + 9, framealpha=0.95)

        if len(vote_details) > 1:
            key_ax.text(
                0.02,
                0.02,
                "\n".join(vote_details),
                transform=key_ax.transAxes,
                fontsize=5 + 7,
                va="bottom",
                ha="left",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="lightgray"),
            )
        plt.tight_layout()
    else:
        cond_axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=5 + 9, framealpha=0.9)
        plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        figure_saving.save_figure(fig, save_path, frmt="png", dpi=int(FIG_DPI_SUMMARY_GRID), bbox_inches="tight")
        print(f"  Saved: {save_path.name}")

    return fig


def _get_grid_info_text(fish_idx: int, result: ClassificationResult, config: AnalysisConfig) -> str:
    """Compact annotation for summary grids with detailed per-fish results."""
    cond = str(result.conditions[fish_idx])
    cond_abbr = cond[:3].upper()

    # Classification status
    if bool(result.is_learner_probabilistic[fish_idx]):
        status = "LEARNER"
    elif bool(result.is_outlier[fish_idx]):
        status = "T>thr"
    else:
        status = "non-lrn"

    # Votes
    v = int(result.votes_probabilistic[fish_idx])
    n_feat = len(config.features_to_use)

    # T score and p-value for this fish
    T_fish = float(result.T_scores[fish_idx]) if result.T_scores is not None else np.nan
    p_fish = float(result.p_empirical[fish_idx]) if result.p_empirical is not None else np.nan
    thr = float(result.threshold)

    # Per-feature details: BLUP, z_dir, direction check
    feat_lines: List[str] = []
    for feat_idx, feat in enumerate(config.features_to_use):
        blup = float(result.features[fish_idx, feat_idx])
        z = float(result.z_dir[fish_idx, feat_idx])
        dir_ok = "+" if z > 0 else "-"
        feat_abbr = feat[:3]
        feat_lines.append(f"{feat_abbr}:{blup:+.2f} z:{z:+.1f}{dir_ok}")

    # Build text block
    lines = [
        f"{cond_abbr} | {status}",
        f"T:{T_fish:.2f} p:{p_fish:.3f}",
        f"V:{v}/{n_feat} thr:{thr:.2f}",
    ] + feat_lines
    return "\n".join(lines)


def plot_feature_space(
    result: ClassificationResult,
    config: AnalysisConfig,
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Scatter plot of fish in feature space with control-mean crosshair."""
    n_features = len(config.features_to_use)
    if n_features < 2:
        return None

    feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_pairs = len(feature_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 7), facecolor="white")
    if n_pairs == 1:
        axes = [axes]

    fig.suptitle(
        "Individual Fish in Feature Space\n(BLUPs with Control Means) - Improved",
        fontsize=5 + 14,
        fontweight="bold",
        y=0.98,
    )

    for ax, (idx_x, idx_y) in zip(axes, feature_pairs):
        feat_x = config.features_to_use[idx_x]
        feat_y = config.features_to_use[idx_y]
        cfg_x = config.feature_configs[feat_x]
        cfg_y = config.feature_configs[feat_y]

        for i in range(len(result.fish_ids)):
            if bool(result.is_learner_probabilistic[i]):
                color, marker, size, zorder = COLOR_PROB_LEARNER, "*", 200, 5
            elif bool(result.is_outlier[i]):
                color, marker, size, zorder = COLOR_OUTLIER_WRONG_DIR, "s", 80, 3
            else:
                is_ref = str(result.conditions[i]) == str(config.cond_types[0])
                color = COLOR_CTRL_NONLEARNER if is_ref else COLOR_EXP_NONLEARNER
                marker, size, zorder = "o", 40, 2

            ax.scatter(
                float(result.features[i, idx_x]),
                float(result.features[i, idx_y]),
                c=color,
                marker=marker,
                s=size,
                zorder=zorder,
                edgecolors="black",
                linewidth=0.5,
            )

            xerr = float(config.ci_multiplier_for_reporting) * float(result.features_se[i, idx_x])
            yerr = float(config.ci_multiplier_for_reporting) * float(result.features_se[i, idx_y])
            ax.errorbar(
                float(result.features[i, idx_x]),
                float(result.features[i, idx_y]),
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.6,
                alpha=0.35,
                zorder=float(zorder) - 0.1,
            )

            if bool(result.is_learner_probabilistic[i]):
                ax.annotate(
                    str(result.fish_ids[i]),
                    (float(result.features[i, idx_x]), float(result.features[i, idx_y])),
                    fontsize=5 + 7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

        ax.axvline(float(result.mu_ctrl[idx_x]), color="black", linestyle="--", alpha=0.5)
        ax.axhline(float(result.mu_ctrl[idx_y]), color="black", linestyle="--", alpha=0.5)

        x_arrow = "<-" if cfg_x.direction == "negative" else "->"
        y_arrow = "v" if cfg_y.direction == "negative" else "^"
        ax.set_xlabel(f"{cfg_x.name} ({feat_x})\n{x_arrow} Learning direction", fontsize=5 + 10)
        ax.set_ylabel(f"{cfg_y.name} ({feat_y})\n{y_arrow} Learning direction", fontsize=5 + 10)
        ax.set_title(f"{cfg_x.name} vs {cfg_y.name}", fontsize=5 + 11)
        ax.grid(alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color=COLOR_PROB_LEARNER, lw=2.5, label="* Improved Learner"),
        Line2D([0], [0], color=COLOR_OUTLIER_WRONG_DIR, lw=1.0, label="T > threshold"),
        Line2D([0], [0], color=COLOR_EXP_NONLEARNER, lw=1, label="Exp. (Non-learner)"),
        Line2D([0], [0], color=COLOR_CTRL_NONLEARNER, lw=1, label="Ctrl. (Non-learner)"),
        Line2D([0], [0], color="black", linestyle="--", label="Control Mean"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=5 + 10)
    plt.tight_layout(rect=(0, 0.12, 1, 0.95))

    if save_path is not None:
        save_path = Path(save_path)
        figure_saving.save_figure(fig, save_path, frmt="png", dpi=int(FIG_DPI_SUMMARY_GRID), bbox_inches="tight")
        print(f"  Saved: {save_path.name}")
    return fig


def plot_blup_trajectory_overlay(
    result: ClassificationResult,
    config: AnalysisConfig,
    split_by_condition: bool = True,
    save_path: Optional[Path] = None,
    alpha: float = 0.18,
    lw: float = 1.0,
    show_uncertainty_bands: bool = False,
) -> Figure:
    """Overlay per-fish BLUP trajectories in a single plot."""

    def _feat_index(feat_code: str) -> Optional[int]:
        try:
            return config.features_to_use.index(feat_code)
        except ValueError:
            return None

    j_acq = _feat_index("acquisition")
    j_ext = _feat_index("extinction")
    if j_acq is None and j_ext is None:
        raise ValueError("No BLUP trajectory features available.")

    if j_acq is not None and j_ext is not None:
        x = np.array([0, 1, 2], dtype=float)
        x_labels = ["Pre-train", "Late Train + Early Test", "Late Test"]
    elif j_acq is not None:
        x = np.array([0, 1], dtype=float)
        x_labels = ["Pre-train", "Late Train + Early Test"]
    else:
        x = np.array([0, 1], dtype=float)
        x_labels = ["Late Train + Early Test", "Late Test"]

    if split_by_condition:
        conds_order = [c for c in config.cond_types if c in np.unique(result.conditions)]
        if not conds_order:
            conds_order = list(np.unique(result.conditions))
        n_panels = len(conds_order)
    else:
        conds_order = ["all"]
        n_panels = 1

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.5 * n_panels, 5.5),
        facecolor="white",
        sharex=True,
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        "BLUP Trajectories Overlay (purple)\n(per-fish cumulative epoch effects) - Improved",
        fontsize=5 + 14,
        fontweight="bold",
        y=0.98,
    )

    panel_y: Dict[str, List[np.ndarray]] = {cond: [] for cond in conds_order}

    for i in range(len(result.fish_ids)):
        cond = str(result.conditions[i])
        if j_acq is not None and j_ext is not None:
            acq = float(result.features[i, j_acq])
            ext = float(result.features[i, j_ext])
            acq_se = float(result.features_se[i, j_acq])
            ext_se = float(result.features_se[i, j_ext])
            y = np.array([0.0, acq, acq + ext], dtype=float)
            y_se = np.array([0.0, acq_se, np.sqrt(acq_se**2 + ext_se**2)], dtype=float)
        elif j_acq is not None:
            acq = float(result.features[i, j_acq])
            acq_se = float(result.features_se[i, j_acq])
            y = np.array([0.0, acq], dtype=float)
            y_se = np.array([0.0, acq_se], dtype=float)
        else:
            ext = float(result.features[i, j_ext])  # type: ignore[index]
            ext_se = float(result.features_se[i, j_ext])  # type: ignore[index]
            y = np.array([0.0, ext], dtype=float)
            y_se = np.array([0.0, ext_se], dtype=float)

        if split_by_condition:
            if cond not in conds_order:
                continue
            ax = axes[conds_order.index(cond)]
            panel_y[cond].append(y)
        else:
            ax = axes[0]
            panel_y["all"].append(y)

        if bool(result.is_learner_probabilistic[i]):
            a_i = min(0.55, alpha * 2.5)
        elif bool(result.is_outlier[i]):
            a_i = min(0.40, alpha * 2.0)
        else:
            a_i = alpha

        ax.plot(x, y, "--", color=COLOR_BLUP_TRAJECTORY, alpha=a_i, linewidth=lw)
        if show_uncertainty_bands:
            ci_mult = float(config.ci_multiplier_for_reporting)
            ax.fill_between(x, y - ci_mult * y_se, y + ci_mult * y_se, color=COLOR_BLUP_TRAJECTORY, alpha=a_i * 0.15)

    for ax, cond in zip(axes, conds_order):
        ax.axhline(0.0, linestyle=":", color=COLOR_BLUP_TRAJECTORY, alpha=0.35, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=5 + 9)
        ax.set_xlabel("Epoch", fontsize=5 + 10)
        ax.grid(alpha=0.25)

        if cond != "all":
            n_cond = int((result.conditions == cond).sum())
            ax.set_title(f"{str(cond).capitalize()} (n={n_cond})", fontsize=5 + 11, fontweight="bold")
        else:
            ax.set_title("All fish", fontsize=5 + 11, fontweight="bold")

        ys = panel_y.get(cond, [])
        if ys:
            y_med = np.median(np.vstack(ys), axis=0)
            ax.plot(x, y_med, "o-", color=COLOR_GROUP_MEDIAN, linewidth=2.2, markersize=6, alpha=0.85, label="Median", zorder=10)
            ax.legend(loc="best", fontsize=5 + 9, framealpha=0.9)

    axes[0].set_ylabel("BLUP (Delta Log Response) - cumulative", fontsize=5 + 10)
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))

    if save_path is not None:
        save_path = Path(save_path)
        figure_saving.save_figure(fig, save_path, frmt="png", dpi=int(FIG_DPI_BLUP_OVERLAY), bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path.name}")
    return fig


def plot_blup_caterpillar(
    result: ClassificationResult,
    config: AnalysisConfig,
    feat_code: str,
    save_path: Optional[Path] = None,
    sort_by_blup: bool = True,
) -> Figure:
    """Per-fish BLUP ± CI caterpillar plot for one feature."""
    if feat_code not in config.features_to_use:
        raise ValueError(f"Feature {feat_code!r} not in config.features_to_use")

    j = config.features_to_use.index(feat_code)
    feat_cfg = config.feature_configs[feat_code]
    mu = float(result.mu_ctrl[j])

    blup_all = result.features[:, j].astype(float)
    se_all = result.features_se[:, j].astype(float)
    order = np.argsort(blup_all) if sort_by_blup else np.arange(len(result.fish_ids))

    fish = result.fish_ids[order]
    blup = blup_all[order]
    se = se_all[order]
    ci_mult = float(config.ci_multiplier_for_reporting)
    lo = blup - ci_mult * se
    hi = blup + ci_mult * se

    colors: List[str] = []
    for idx in order:
        if bool(result.is_learner_probabilistic[idx]):
            colors.append(COLOR_PROB_LEARNER)
        elif bool(result.is_outlier[idx]):
            colors.append(COLOR_OUTLIER_WRONG_DIR)
        else:
            is_ref = str(result.conditions[idx]) == str(config.cond_types[0])
            colors.append(COLOR_CTRL_NONLEARNER if is_ref else COLOR_EXP_NONLEARNER)

    fig_h = max(4, 0.18 * len(fish))
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="white")
    y = np.arange(len(fish))
    for yi, l, h, c in zip(y, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, alpha=0.75, lw=1.2)
    ax.scatter(blup, y, c=colors, s=25, edgecolors="black", linewidths=0.4, zorder=3)
    ax.axvline(mu, color="black", linestyle="--", alpha=0.7, label=f"Control mean = {mu:.3f}")

    dir_text = "learning: BLUP < control mean" if feat_cfg.direction == "negative" else "learning: BLUP > control mean"
    ax.set_title(f"{feat_cfg.name} ({feat_code}) - {dir_text}", fontsize=5 + 12, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([str(f) for f in fish], fontsize=5 + 7)
    ax.set_xlabel(f"BLUP ± {ci_mult:.2f}×SE (feature scale)", fontsize=5 + 10)
    ax.grid(alpha=0.25)

    legend_elements = [
        Line2D([0], [0], color=COLOR_PROB_LEARNER, lw=2, label="* Improved Learner"),
        Line2D([0], [0], color=COLOR_OUTLIER_WRONG_DIR, lw=1.5, label="T > threshold"),
        Line2D([0], [0], color=COLOR_CTRL_NONLEARNER, lw=1.5, label="Ctrl. (Non-learner)"),
        Line2D([0], [0], color=COLOR_EXP_NONLEARNER, lw=1.5, label="Exp. (Non-learner)"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Control mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=5 + 9, framealpha=0.95)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        figure_saving.save_figure(fig, save_path, frmt="png", dpi=int(FIG_DPI_SUMMARY_GRID), bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path.name}")
    return fig


def save_combined_plots_and_grid(
    data: pd.DataFrame,
    result: ClassificationResult,
    config: AnalysisConfig,
    output_dir: Path,
    condition: Optional[str] = None,
    filename_suffix: str = "",
) -> None:
    """Save per-fish combined plots and summary grids."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _set_ylim_no_clip(
        ax: Axes,
        y_values: Iterable[float],
        default_lim: Sequence[float],
        pad_frac: float = 0.02,
    ) -> Tuple[float, float]:
        y_lo, y_hi = float(default_lim[0]), float(default_lim[1])
        try:
            arr = np.asarray(list(y_values), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                y_min = float(arr.min())
                y_max = float(arr.max())
                y_lo2 = min(y_lo, y_min)
                y_hi2 = max(y_hi, y_max)
                pad = max(0.01, float(pad_frac) * (y_hi2 - y_lo2))
                y_lo, y_hi = (y_lo2 - pad, y_hi2 + pad)
        except Exception:
            pass
        ax.set_ylim((y_lo, y_hi))
        return (y_lo, y_hi)

    def _feat_index(feat_code: str) -> Optional[int]:
        try:
            return config.features_to_use.index(feat_code)
        except ValueError:
            return None

    def _blup_ci(idx: int, feat_code: str) -> Optional[Tuple[float, float, float]]:
        j = _feat_index(feat_code)
        if j is None:
            return None
        blup = float(result.features[idx, j])
        se = float(result.features_se[idx, j])
        ci_low = blup - float(config.ci_multiplier_for_reporting) * se
        ci_high = blup + float(config.ci_multiplier_for_reporting) * se
        return blup, ci_low, ci_high

    def _get_epoch_point(fish_block_data: pd.DataFrame, block_names: List[str]) -> Optional[Tuple[float, float]]:
        if not block_names:
            return None
        sub = fish_block_data[fish_block_data["Block name 5 trials"].isin(block_names)]
        if sub.empty:
            return None
        present = set(sub["Block name 5 trials"].astype(str).unique())
        if any(str(b) not in present for b in block_names):
            return None
        x = float(sub["Trial number"].mean())
        y = float(sub["Normalized vigor"].median())
        return x, y

    def _annotate_key_blocks(ax: Axes, fish_block_data: pd.DataFrame, color: str) -> None:
        pt_pre = _get_epoch_point(fish_block_data, config.pretrain_blocks_5)
        pt_mid = _get_epoch_point(fish_block_data, config.earlytest_block)
        pt_late = _get_epoch_point(fish_block_data, config.late_test_blocks_5)
        for pt in [pt_pre, pt_mid, pt_late]:
            if pt is None:
                continue
            x, y = pt
            ax.scatter([x], [y], marker="o", s=85, c=color, edgecolors="black", linewidths=0.9, zorder=11, alpha=0.98)

    # Fish selection
    if condition is not None:
        fish_indices = np.where(result.conditions == condition)[0]
    else:
        fish_indices = np.arange(len(result.fish_ids))

    df_trials = data.copy()
    if condition is not None:
        df_trials = df_trials[df_trials["Condition"] == condition]

    baseline_col = [c for c in df_trials.columns if BASELINE_COLUMN_SUBSTRING in c]
    baseline_col = baseline_col[0] if baseline_col else None
    response_col = RESPONSE_COLUMN_NAME if RESPONSE_COLUMN_NAME in df_trials.columns else None

    norm_baseline_col = "Normalized_Baseline" if baseline_col else None
    norm_response_col = "Normalized_Response" if response_col else None

    if baseline_col or response_col:
        pretrain_mask = df_trials["Block name 5 trials"].isin(config.pretrain_blocks_5)
        for fish_id in df_trials["Fish_ID"].unique():
            fish_mask = df_trials["Fish_ID"] == fish_id
            fish_pretrain_mask = fish_mask & pretrain_mask
            if baseline_col and baseline_col in df_trials.columns:
                m = float(df_trials.loc[fish_pretrain_mask, baseline_col].mean())
                df_trials.loc[fish_mask, norm_baseline_col] = df_trials.loc[fish_mask, baseline_col] / m if m > 0 else 1.0
            if response_col and response_col in df_trials.columns:
                m = float(df_trials.loc[fish_pretrain_mask, response_col].mean())
                df_trials.loc[fish_mask, norm_response_col] = df_trials.loc[fish_mask, response_col] / m if m > 0 else 1.0

    agg_dict: Dict[str, Any] = {"Normalized vigor": "median", "Trial number": "mean"}
    if norm_baseline_col and norm_baseline_col in df_trials.columns:
        agg_dict[norm_baseline_col] = "median"
    if norm_response_col and norm_response_col in df_trials.columns:
        agg_dict[norm_response_col] = "median"

    df_blocks = (
        df_trials.groupby(["Fish_ID", "Block name 5 trials"], observed=True)
        .agg(agg_dict)
        .reset_index()
        .sort_values("Trial number")
    )

    cond_label = f" ({condition})" if condition else ""
    print(f"  Generating combined individual plots{cond_label} in {output_dir.name}...")

    # Individual plots
    for idx in fish_indices:
        fish = str(result.fish_ids[idx])
        fish_trial_data = df_trials[df_trials["Fish_ID"] == fish]
        fish_block_data = df_blocks[df_blocks["Fish_ID"] == fish]
        cond = str(result.conditions[idx])

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        if bool(result.is_learner_probabilistic[idx]):
            color, lw = COLOR_PROB_LEARNER, 2.5
            status = "* IMPROVED LEARNER"
        elif bool(result.is_outlier[idx]):
            color, lw = COLOR_OUTLIER_WRONG_DIR, 1.5
            status = "o T > threshold"
        else:
            color = COLOR_CTRL_NONLEARNER if cond == config.cond_types[0] else COLOR_EXP_NONLEARNER
            lw = 1.0
            status = "Non-learner"

        ax.scatter(
            fish_trial_data["Trial number"],
            fish_trial_data["Normalized vigor"],
            c=color,
            alpha=0.3,
            s=15,
            edgecolors="none",
            label="Trials",
        )
        ax.plot(
            fish_block_data["Trial number"],
            fish_block_data["Normalized vigor"],
            "-",
            color=color,
            linewidth=lw,
            alpha=0.95,
            label="Block Median (NV)",
        )
        if norm_baseline_col and norm_baseline_col in fish_block_data.columns:
            ax.plot(
                fish_block_data["Trial number"],
                fish_block_data[norm_baseline_col],
                "-",
                color=COLOR_BASELINE_VIGOR,
                linewidth=lw * 0.8,
                alpha=0.75,
                label="Baseline Vigor",
            )
        if norm_response_col and norm_response_col in fish_block_data.columns:
            ax.plot(
                fish_block_data["Trial number"],
                fish_block_data[norm_response_col],
                "-",
                color=COLOR_RESPONSE_VIGOR,
                linewidth=lw * 0.8,
                alpha=0.75,
                label="Response Vigor",
            )

        _annotate_key_blocks(ax, fish_block_data, color)

        ax.axhline(1.0, linestyle=":", color="black", alpha=0.3, label="Baseline")
        ax.set_xlabel("Trial Number", fontsize=5 + 9)
        ax.set_ylabel("Normalized Vigor", fontsize=5 + 9)

        y_values_for_ylim = [fish_trial_data["Normalized vigor"], fish_block_data["Normalized vigor"]]
        if norm_baseline_col and norm_baseline_col in fish_block_data.columns:
            y_values_for_ylim.append(fish_block_data[norm_baseline_col])
        if norm_response_col and norm_response_col in fish_block_data.columns:
            y_values_for_ylim.append(fish_block_data[norm_response_col])
        _set_ylim_no_clip(ax, pd.concat(y_values_for_ylim, ignore_index=True).to_list(), tuple(config.y_lim_plot))

        ax.set_title(f"{fish} ({cond})\n{status}", fontsize=5 + 10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5 + 8, loc="upper right")

        # Text box: BLUPs, threshold, T, votes, per-feature z/p_dir for learner vs non-learner
        n_feat = len(config.features_to_use)
        v = int(result.votes_probabilistic[idx])
        T_fish = float(result.T_scores[idx]) if result.T_scores is not None else np.nan
        p_fish = float(result.p_empirical[idx]) if result.p_empirical is not None else np.nan
        thr = float(result.threshold)
        info_lines = [
            f"T = {T_fish:.3f}   threshold T* = {thr:.3f}   {'T > T*' if T_fish > thr else 'T ≤ T*'}",
            f"Votes (learning dir): {v}/{n_feat}  {'(all +)' if v >= n_feat else '(not all +)'}   p_emp = {p_fish:.4f}",
            "",
        ]
        for feat_idx, feat in enumerate(config.features_to_use):
            blup_ci = _blup_ci(int(idx), feat)
            blup_str = f"{result.features[idx, feat_idx]:+.3f}"
            if blup_ci is not None:
                blup_str = f"{blup_ci[0]:+.3f} [{blup_ci[1]:+.3f}, {blup_ci[2]:+.3f}]"
            z = float(result.z_dir[idx, feat_idx])
            p_dir = float(result.p_learning[idx, feat_idx])
            direction = "+" if z > 0 else "-"
            info_lines.append(f"  {feat}: BLUP {blup_str}  z_dir={z:+.2f}  p_dir={p_dir:.2%} {direction}")
        info_text = "\n".join(info_lines)
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=6, va="top", ha="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="gray"),
        )

        # Secondary y-axis: BLUP trajectory (draw whenever we have BLUP values; use block x if available else abstract 0,1,2)
        acq_ci = _blup_ci(int(idx), "acquisition")
        ext_ci = _blup_ci(int(idx), "extinction")
        if acq_ci is not None and ext_ci is not None:
            acq_blup, _, _ = acq_ci
            ext_blup, _, _ = ext_ci
            blup_pre = 0.0
            blup_mid = acq_blup
            blup_late = acq_blup + ext_blup
            pt_pre = _get_epoch_point(fish_block_data, config.pretrain_blocks_5)
            pt_mid = _get_epoch_point(fish_block_data, config.earlytest_block)
            pt_late = _get_epoch_point(fish_block_data, config.late_test_blocks_5)
            if pt_pre is not None and pt_mid is not None and pt_late is not None:
                x_pre, _ = pt_pre
                x_mid, _ = pt_mid
                x_late, _ = pt_late
                x_vals = [x_pre, x_mid, x_late]
            else:
                # Fallback: align to left/mid/right of trial range so BLUP is always visible
                x_min = float(fish_block_data["Trial number"].min()) if not fish_block_data.empty else 0.0
                x_max = float(fish_block_data["Trial number"].max()) if not fish_block_data.empty else 1.0
                x_vals = [x_min, (x_min + x_max) / 2.0, x_max]
            ax2 = ax.twinx()
            ax2.plot(
                x_vals,
                [blup_pre, blup_mid, blup_late],
                "s--",
                color=COLOR_BLUP_TRAJECTORY,
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=COLOR_BLUP_TRAJECTORY,
                markeredgewidth=1.5,
                linewidth=lw,
                alpha=0.85,
                label="BLUP trajectory",
                zorder=15,
            )
            ax2.axhline(0.0, linestyle=":", color=COLOR_BLUP_TRAJECTORY, alpha=0.4)
            ax2.set_ylabel("BLUP (Delta Log Response)", fontsize=5 + 9, color=COLOR_BLUP_TRAJECTORY)
            ax2.tick_params(axis="y", labelcolor=COLOR_BLUP_TRAJECTORY)
            blup_range = max(abs(blup_pre), abs(blup_mid), abs(blup_late), 0.05) * 1.5
            ax2.set_ylim(-blup_range, blup_range)
            ax2.legend(fontsize=5 + 7, loc="lower right")

        save_path = output_dir / f"{fish}_combined{filename_suffix}.png"
        figure_saving.save_figure(fig, save_path, frmt="png", dpi=int(FIG_DPI_INDIVIDUAL_FISH), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Summary grid
    print(f"  Generating combined summary grid{cond_label}...")
    n_fish = len(fish_indices)
    if n_fish == 0:
        print(f"  [WARN] No fish to plot for condition: {condition}")
        return

    n_cols = 6
    n_rows = (n_fish + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows + 1.5),
        facecolor="white",
        squeeze=False,
    )
    title_suffix = f" - {str(condition).capitalize()}" if condition else ""
    fig.suptitle(f"Combined Summary Grid{title_suffix}", fontsize=5 + 16, y=0.99)

    for plot_idx, idx in enumerate(fish_indices):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]
        fish = str(result.fish_ids[idx])
        cond = str(result.conditions[idx])

        fish_trial_data = df_trials[df_trials["Fish_ID"] == fish]
        fish_block_data = df_blocks[df_blocks["Fish_ID"] == fish]

        if bool(result.is_learner_probabilistic[idx]):
            color = COLOR_PROB_LEARNER
        elif bool(result.is_outlier[idx]):
            color = COLOR_OUTLIER_WRONG_DIR
        else:
            color = COLOR_CTRL_NONLEARNER if cond == config.cond_types[0] else COLOR_EXP_NONLEARNER

        ax.scatter(
            fish_trial_data["Trial number"],
            fish_trial_data["Normalized vigor"],
            c=color,
            alpha=0.22,
            s=10,
            edgecolors="none",
        )
        ax.plot(
            fish_block_data["Trial number"],
            fish_block_data["Normalized vigor"],
            "-",
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )
        if norm_baseline_col and norm_baseline_col in fish_block_data.columns:
            ax.plot(
                fish_block_data["Trial number"],
                fish_block_data[norm_baseline_col],
                "-",
                color=COLOR_BASELINE_VIGOR,
                linewidth=0.9,
                alpha=0.65,
            )
        if norm_response_col and norm_response_col in fish_block_data.columns:
            ax.plot(
                fish_block_data["Trial number"],
                fish_block_data[norm_response_col],
                "-",
                color=COLOR_RESPONSE_VIGOR,
                linewidth=0.9,
                alpha=0.65,
            )

        # BLUP trajectory (acquisition + extinction) on twin axis
        acq_ci = _blup_ci(int(idx), "acquisition")
        ext_ci = _blup_ci(int(idx), "extinction")
        if acq_ci is not None and ext_ci is not None:
            acq_blup, _, _ = acq_ci
            ext_blup, _, _ = ext_ci
            blup_pre, blup_mid, blup_late = 0.0, acq_blup, acq_blup + ext_blup
            pt_pre = _get_epoch_point(fish_block_data, config.pretrain_blocks_5)
            pt_mid = _get_epoch_point(fish_block_data, config.earlytest_block)
            pt_late = _get_epoch_point(fish_block_data, config.late_test_blocks_5)
            if pt_pre is not None and pt_mid is not None and pt_late is not None:
                x_vals = [pt_pre[0], pt_mid[0], pt_late[0]]
            else:
                x_min = float(fish_block_data["Trial number"].min()) if not fish_block_data.empty else 0.0
                x_max = float(fish_block_data["Trial number"].max()) if not fish_block_data.empty else 1.0
                x_vals = [x_min, (x_min + x_max) / 2.0, x_max]
            ax2 = ax.twinx()
            ax2.plot(x_vals, [blup_pre, blup_mid, blup_late], "s--", color=COLOR_BLUP_TRAJECTORY, markersize=4, markeredgecolor=COLOR_BLUP_TRAJECTORY, linewidth=0.9, alpha=0.85, zorder=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_ylim(
                -max(abs(blup_pre), abs(blup_mid), abs(blup_late), 0.05) * 1.5,
                max(abs(blup_pre), abs(blup_mid), abs(blup_late), 0.05) * 1.5,
            )

        ax.axhline(1.0, linestyle=":", color="black", alpha=0.25)
        ax.set_title(f"{fish}", fontsize=5 + 9, color=color, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.12)

        # Border strength by class
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5 if bool(result.is_learner_probabilistic[idx]) else (1.6 if bool(result.is_outlier[idx]) else 0.6))

        ax.text(
            0.02,
            0.02,
            _get_grid_info_text(int(idx), result, config),
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray"),
        )

    for plot_idx in range(n_fish, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row, col].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=COLOR_PROB_LEARNER, lw=2.5, label="* Improved Learner"),
        Line2D([0], [0], color=COLOR_OUTLIER_WRONG_DIR, lw=2, label="T > threshold"),
        Line2D([0], [0], color=COLOR_BASELINE_VIGOR, lw=1.2, label="Baseline Vigor (norm)"),
        Line2D([0], [0], color=COLOR_RESPONSE_VIGOR, lw=1.2, label="Response Vigor (norm)"),
        Line2D([0], [0], color=COLOR_BLUP_TRAJECTORY, lw=1.2, linestyle="--", label="BLUP trajectory"),
        Line2D([0], [0], color=COLOR_EXP_NONLEARNER, lw=1, label="Non-learner"),
    ]
    fig.legend(handles=legend_elements, loc="lower left", ncol=3, bbox_to_anchor=(0.02, 0.01), fontsize=5 + 10, frameon=True)

    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    plt.subplots_adjust(wspace=0.22, hspace=0.35)

    cond_suffix = f"_{condition}" if condition else ""
    grid_path = output_dir.parent / f"All_Fish_Combined_Summary_Grid{cond_suffix}{filename_suffix}.png"
    figure_saving.save_figure(fig, grid_path, frmt="png", dpi=int(FIG_DPI_SUMMARY_GRID), bbox_inches="tight")
    # plt.close(fig)
    print(f"  Saved: {grid_path.name}")


def save_heatmap_grid(
    result: ClassificationResult,
    config: AnalysisConfig,
    path_heatmap_fig_cs: Path,
    output_dir: Path,
    condition: Optional[str] = None,
    filename_suffix: str = "",
    heatmap_variant: str = "scaled",
) -> None:
    """Grid figure with pre-saved heatmaps (SVG/PNG) for each fish. heatmap_variant: 'scaled' or 'raw'."""
    import io

    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    if condition is not None:
        fish_indices = np.where(result.conditions == condition)[0]
    else:
        fish_indices = np.arange(len(result.fish_ids))

    n_fish = len(fish_indices)
    if n_fish == 0:
        print(f"  [WARN] No fish to plot for condition: {condition}")
        return

    heatmap_files = list(Path(path_heatmap_fig_cs).glob("*.svg")) + list(Path(path_heatmap_fig_cs).glob("*.png"))
    fish_to_heatmap: Dict[str, Path] = {}
    for fpath in heatmap_files:
        parts = fpath.stem.split("_")
        if len(parts) >= 2:
            fish_id = "_".join(parts[:2]).lower()
            if fish_id not in fish_to_heatmap or fpath.suffix.lower() == ".png":
                fish_to_heatmap[fish_id] = fpath

    n_cols = 6
    n_rows = (n_fish + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows + 1.5),
        facecolor="white",
        squeeze=False,
    )
    title_suffix = f" - {str(condition).capitalize()}" if condition else ""
    title_label = "Raw Vigor Heatmaps" if heatmap_variant == "raw" else "Scaled Vigor Heatmaps"
    fig.suptitle(f"{title_label} (aligned to CS){title_suffix}", fontsize=5 + 16, y=0.99)

    def _crop_border(img: np.ndarray, frac: float = 0.02) -> np.ndarray:
        if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
            return img
        h, w = int(img.shape[0]), int(img.shape[1])
        dy = int(round(h * float(frac)))
        dx = int(round(w * float(frac)))
        if (h - 2 * dy) < 2 or (w - 2 * dx) < 2:
            return img
        return img[dy : h - dy, dx : w - dx, ...]

    def _load_image(img_path: Path) -> Optional[np.ndarray]:
        try:
            if img_path.suffix.lower() == ".svg":
                try:
                    import importlib

                    cairosvg = importlib.import_module("cairosvg")
                    png_data = cairosvg.svg2png(url=str(img_path))  # type: ignore[attr-defined]
                    img = Image.open(io.BytesIO(png_data))
                    return _crop_border(np.array(img), frac=0.02)
                except ImportError:
                    try:
                        import importlib

                        renderPM = importlib.import_module("reportlab.graphics.renderPM")
                        svglib = importlib.import_module("svglib.svglib")
                        drawing = svglib.svg2rlg(str(img_path))  # type: ignore[attr-defined]
                        png_data = renderPM.drawToString(drawing, fmt="PNG")  # type: ignore[attr-defined]
                        img = Image.open(io.BytesIO(png_data))
                        return _crop_border(np.array(img), frac=0.02)
                    except ImportError:
                        print(f"    [WARN] Cannot load SVG (install cairosvg or svglib): {img_path.name}")
                        return None
            img = Image.open(img_path)
            return _crop_border(np.array(img), frac=0.02)
        except Exception as e:
            print(f"    [WARN] Failed to load image {img_path.name}: {e}")
            return None

    missing_fish: List[str] = []
    loaded_count = 0

    for plot_idx, idx in enumerate(fish_indices):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]
        fish = str(result.fish_ids[idx])
        fish_lower = fish.lower()
        cond = str(result.conditions[idx])

        if bool(result.is_learner_probabilistic[idx]):
            border_color = COLOR_PROB_LEARNER
            border_lw = 2.5
        elif bool(result.is_outlier[idx]):
            border_color = COLOR_OUTLIER_WRONG_DIR
            border_lw = 1.5
        else:
            border_color = COLOR_CTRL_NONLEARNER if cond == config.cond_types[0] else COLOR_EXP_NONLEARNER
            border_lw = 1.0

        heatmap_path = fish_to_heatmap.get(fish_lower)
        if heatmap_path is not None:
            img_array = _load_image(heatmap_path)
            if img_array is not None:
                ax.imshow(img_array)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                loaded_count += 1
            else:
                ax.text(0.5, 0.5, f"Load Error\n{fish}", ha="center", va="center", transform=ax.transAxes, fontsize=5 + 8, color="red")
                ax.axis("off")
        else:
            missing_fish.append(fish)
            ax.text(0.5, 0.5, f"No heatmap\n{fish}", ha="center", va="center", transform=ax.transAxes, fontsize=5 + 8, color="gray")
            ax.axis("off")

        ax.set_title(f"{fish}", fontsize=5 + 9, color=border_color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(border_lw)

    for plot_idx in range(n_fish, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row, col].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=COLOR_PROB_LEARNER, lw=2.5, label="* Improved Learner"),
        Line2D([0], [0], color=COLOR_OUTLIER_WRONG_DIR, lw=1.5, label="T > threshold"),
        Line2D([0], [0], color=COLOR_EXP_NONLEARNER, lw=1, label="Non-learner"),
    ]
    fig.legend(handles=legend_elements, loc="lower left", ncol=3, bbox_to_anchor=(0.02, 0.01), fontsize=5 + 10, frameon=True)
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    plt.subplots_adjust(wspace=0.08, hspace=0.14)

    cond_suffix = f"_{condition}" if condition else ""
    grid_name_prefix = "Heatmap_Grid_Raw_CS" if heatmap_variant == "raw" else "Heatmap_Grid_CS"
    grid_path = Path(output_dir) / f"{grid_name_prefix}{cond_suffix}{filename_suffix}.png"
    figure_saving.save_figure(fig, grid_path, frmt="png", dpi=int(FIG_DPI_SUMMARY_GRID), bbox_inches="tight")
    # plt.close(fig)

    print(f"  Saved heatmap grid: {grid_path.name}")
    print(f"    Loaded: {loaded_count}/{n_fish} heatmaps")
    if missing_fish:
        print(f"    Missing heatmaps for: {missing_fish[:10]}{'...' if len(missing_fish) > 10 else ''}")


@dataclass
class ImprovedScores:
    mu_ctrl_full: np.ndarray
    se_mu_ctrl: np.ndarray
    z_dir: np.ndarray
    p_dir: np.ndarray
    T: np.ndarray
    T_ctrl: np.ndarray
    threshold_T: float
    p_empirical: np.ndarray
    votes_dir: np.ndarray
    is_learner: np.ndarray


def _se_of_mean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    if n <= 1:
        return np.full(x.shape[1], np.nan, dtype=float)
    return np.std(x, axis=0, ddof=1) / np.sqrt(n)


def _build_reference_means_loocv(X: np.ndarray, is_ctrl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    is_ctrl = np.asarray(is_ctrl, dtype=bool)
    X_ctrl = X[is_ctrl]
    n_ctrl = int(X_ctrl.shape[0])
    if n_ctrl < 2:
        raise ValueError("Need at least 2 control fish for LOO reference means.")

    sum_ctrl = np.sum(X_ctrl, axis=0)
    mu_full = sum_ctrl / n_ctrl
    mu_ref = np.broadcast_to(mu_full, X.shape).copy()
    mu_loo_ctrl = (sum_ctrl - X_ctrl) / (n_ctrl - 1)
    mu_ref[is_ctrl] = mu_loo_ctrl
    return mu_full, mu_ref


def _directional_z(
    X: np.ndarray,
    X_se: np.ndarray,
    mu_ref: np.ndarray,
    se_mu_ctrl: np.ndarray,
    directions: Sequence[str],
    *,
    use_per_fish_se: bool,
    sigma_ctrl: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    X_se = np.asarray(X_se, dtype=float)
    mu_ref = np.asarray(mu_ref, dtype=float)
    se_mu_ctrl = np.asarray(se_mu_ctrl, dtype=float)
    if bool(use_per_fish_se):
        # Uncertainty-aware denominator (fish SE + control-mean SE)
        denom = np.sqrt(np.maximum(X_se, 0.0) ** 2 + np.maximum(se_mu_ctrl, 0.0) ** 2) + float(EPS)
    else:
        # SE-free denominator: scale by control-feature SD (per feature, constant across fish)
        if sigma_ctrl is None:
            raise ValueError("sigma_ctrl must be provided when use_per_fish_se is False.")
        sigma_ctrl = np.asarray(sigma_ctrl, dtype=float).reshape(1, -1)
        sigma_ctrl = np.where(np.isfinite(sigma_ctrl) & (sigma_ctrl > float(EPS)), sigma_ctrl, 1.0)
        denom = np.broadcast_to(sigma_ctrl, X.shape) + float(EPS)

    z_dir = np.empty_like(X, dtype=float)
    for j, d in enumerate(directions):
        if str(d).lower().startswith("neg"):
            z_dir[:, j] = (mu_ref[:, j] - X[:, j]) / denom[:, j]
        else:
            z_dir[:, j] = (X[:, j] - mu_ref[:, j]) / denom[:, j]

    p_dir = norm.cdf(z_dir)
    return z_dir, p_dir


def _joint_statistic(z_dir: np.ndarray, z_dir_ctrl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute joint test statistic from directional z-scores.
    
    Only positive z-scores (features in learning direction) contribute to T.
    This makes the test one-sided: we're testing for ABOVE-control learning,
    not just any deviation from control.
    
    Args:
        z_dir: Directional z-scores for all fish (n_fish x n_features)
        z_dir_ctrl: Directional z-scores for control fish only
              
    Returns:
        Tuple of (T for all fish, T for controls only)
        
    Notes:
        - Whitening uses Ledoit-Wolf shrinkage covariance estimated from control z-scores.
        - This is closely related to a (squared) Mahalanobis distance, but applied to
          one-sided evidence only: z_pos = max(0, z_dir).
    """
    z_pos = np.clip(np.asarray(z_dir, dtype=float), 0.0, None)
    z_pos_ctrl = np.clip(np.asarray(z_dir_ctrl, dtype=float), 0.0, None)

    # Whitened: use Ledoit-Wolf shrinkage covariance from control z-scores
    lw = LedoitWolf()
    cov = lw.fit(np.asarray(z_dir_ctrl, dtype=float)).covariance_
    inv_cov = np.linalg.pinv(cov, rcond=1e-10)
    # Mahalanobis-like statistic using only positive (learning direction) z-scores
    T = np.sum((z_pos @ inv_cov) * z_pos, axis=1)
    T_ctrl = np.sum((z_pos_ctrl @ inv_cov) * z_pos_ctrl, axis=1)
    return T, T_ctrl


def _empirical_p_values(T: np.ndarray, T_ctrl: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    T_ctrl = np.asarray(T_ctrl, dtype=float)
    return (1.0 + (T_ctrl[None, :] >= T[:, None]).sum(axis=1)) / (1.0 + len(T_ctrl))


def score_and_classify(
    X: np.ndarray,
    X_se: np.ndarray,
    is_ctrl: np.ndarray,
    directions: Sequence[str],
    *,
    alpha: float = ALPHA_TARGET,
    use_per_fish_se_in_scoring: bool = USE_PER_FISH_SE_IN_SCORING,
) -> ImprovedScores:
    """Score fish and classify as learners using uncertainty-aware directional test.
    
    Algorithm:
    1. Compute control mean and control variability for each feature
    2. For each fish, compute directional score z_dir where positive means "learning direction":
       - Default (SE-free): z_dir = (X - μ_ctrl) / SD_ctrl   (sign flipped for negative-direction features)
       - Optional (SE-aware): z_dir = (X - μ_ctrl) / sqrt(SE_fish² + SE_ctrl_mean²)
    3. Flip sign based on learning direction so positive z always means "learning"
    4. Compute joint statistic T from positive z-scores only
    5. Calibrate threshold empirically on controls to achieve target false-positive rate
    6. Classify as learner if T > threshold AND ALL selected features are positive (in learning direction)
    
    Note on threshold calibration:
    - Uses quantile of control T distribution (no bootstrap CI)
    - This provides direct control of false-positive rate on controls
    
    Args:
        X: Feature matrix (n_fish x n_features), BLUP values
        X_se: Standard error matrix (same shape as X)
        is_ctrl: Boolean mask for control fish
        directions: List of "positive" or "negative" per feature (learning direction)
        alpha: Target false-positive rate for controls (default 0.05)
        use_per_fish_se_in_scoring: If True, include per-fish SE in z_dir; if False, scale by control SD only.
        
    Returns:
        ImprovedScores dataclass with z-scores, T values, threshold, classifications
    """
    X = np.asarray(X, dtype=float)
    X_se = np.asarray(X_se, dtype=float)
    is_ctrl = np.asarray(is_ctrl, dtype=bool)

    X_ctrl = X[is_ctrl]
    if X_ctrl.shape[0] < 2:
        raise ValueError("Need at least 2 control fish.")

    se_mu = _se_of_mean(X_ctrl)
    se_mu = np.where(np.isfinite(se_mu), se_mu, 0.0)
    sigma_ctrl = np.std(X_ctrl, axis=0, ddof=1)
    sigma_ctrl = np.where(np.isfinite(sigma_ctrl), sigma_ctrl, 0.0)

    mu_full, mu_ref = _build_reference_means_loocv(X, is_ctrl)
    z_dir, p_dir = _directional_z(
        X,
        X_se,
        mu_ref,
        se_mu,
        directions=directions,
        use_per_fish_se=bool(use_per_fish_se_in_scoring),
        sigma_ctrl=sigma_ctrl,
    )
    z_dir_ctrl = z_dir[is_ctrl]
    required_votes = int(z_dir.shape[1])

    T, T_ctrl = _joint_statistic(z_dir=z_dir, z_dir_ctrl=z_dir_ctrl)
    threshold_T = float(np.quantile(T_ctrl, 1.0 - float(alpha)))

    p_emp = _empirical_p_values(T, T_ctrl)
    votes_dir = (z_dir > 0).sum(axis=1).astype(int)
    # Learner = above threshold AND in learning direction (outliers with clear learning trajectory count as learners)
    is_learner = (T > float(threshold_T)) & (votes_dir >= required_votes)
    # Outlier = above threshold but wrong/mixed direction (plotted as "T > threshold" only when not in learning direction)

    return ImprovedScores(
        mu_ctrl_full=mu_full,
        se_mu_ctrl=se_mu,
        z_dir=z_dir,
        p_dir=p_dir,
        T=T,
        T_ctrl=T_ctrl,
        threshold_T=threshold_T,
        p_empirical=p_emp,
        votes_dir=votes_dir,
        is_learner=is_learner,
    )


def run_multivariate_lme_pipeline(
    config: AnalysisConfig,
    *,
    alpha: float = ALPHA_TARGET,
    use_per_fish_se_in_scoring: bool = USE_PER_FISH_SE_IN_SCORING,
) -> pd.DataFrame:
    (
        _path_lost_frames,
        _path_summary_exp,
        _path_summary_beh,
        _path_processed_data,
        _path_cropped_exp_with_bout_detection,
        _path_tail_angle_fig_cs,
        _path_tail_angle_fig_us,
        path_raw_vigor_fig_cs,
        _path_raw_vigor_fig_us,
        path_scaled_vigor_fig_cs,
        _path_scaled_vigor_fig_us,
        _path_normalized_fig_cs,
        _path_normalized_fig_us,
        path_pooled_vigor_fig,
        _path_analysis_protocols,
        _path_orig_pkl,
        _path_all_fish,
        path_pooled_data,
    ) = file_utils.create_folders(exp_config.path_save)

    print(f"\nExperiment: {EXPERIMENT}")
    print(f"Conditions: {config.cond_types}")

    raw = load_pooled_data(config, path_pooled_data)
    data = prepare_data(raw, config)

    blup_dicts: Dict[str, Dict[str, BLUPResult]] = {}
    if "acquisition" in config.features_to_use:
        blup_dicts["acquisition"] = extract_change_feature(
            data,
            config,
            number_trials=int(EPOCH_BLOCK_TRIALS),
            name_blocks=config.pretrain_to_train_end_earlytest_blocks,
        )
    if "extinction" in config.features_to_use:
        blup_dicts["extinction"] = extract_change_feature(
            data,
            config,
            number_trials=int(EPOCH_BLOCK_TRIALS),
            name_blocks=config.train_end_earlytest_to_late_test_blocks,
        )

    common_fish_sets = [set(blup_dicts[feat].keys()) for feat in config.features_to_use]
    common_fish = sorted(list(set.intersection(*common_fish_sets)))
    if len(common_fish) < int(MIN_FISH_WITH_ALL_FEATURES):
        per_feat = ", ".join(f"{feat}: {len(blup_dicts[feat])} fish" for feat in config.features_to_use)
        raise ValueError(
            f"Only {len(common_fish)} fish have all features (need at least {MIN_FISH_WITH_ALL_FEATURES}). "
            f"Per-feature counts: {per_feat}. "
            f"Check that BLUP extraction succeeded and that Normalized vigor is strictly positive when using log transform."
        )

    X = np.array([[blup_dicts[feat][f].blup for feat in config.features_to_use] for f in common_fish], dtype=float)
    X_se = np.array([[blup_dicts[feat][f].se for feat in config.features_to_use] for f in common_fish], dtype=float)

    cond_map = data.drop_duplicates("Fish_ID").set_index("Fish_ID")["Condition"].to_dict()
    conds = np.array([cond_map.get(f, "unknown") for f in common_fish])

    ref_cond = config.cond_types[0]
    is_ctrl = conds == ref_cond

    # Diagnostic distances (not used for improved classification decision)
    distances, _ = get_robust_mahalanobis(X, X[is_ctrl], verbose=True)

    directions: List[str] = [config.feature_configs[feat].direction for feat in config.features_to_use]
    scores = score_and_classify(
        X=X,
        X_se=X_se,
        is_ctrl=is_ctrl,
        directions=directions,
        alpha=alpha,
        use_per_fish_se_in_scoring=bool(use_per_fish_se_in_scoring),
    )

    out = pd.DataFrame(
        {
            "Fish_ID": np.array(common_fish),
            "Condition": conds,
            "Mahalanobis_Distance_DIAG": distances,
            "T_joint": scores.T,
            "p_empirical": scores.p_empirical,
            "votes_dir": scores.votes_dir,
            "Is_Learner": scores.is_learner,
        }
    )
    for j, feat in enumerate(config.features_to_use):
        out[f"Feature_{feat}_BLUP"] = X[:, j]
        out[f"Feature_{feat}_SE"] = X_se[:, j]
        out[f"Feature_{feat}_z_dir"] = scores.z_dir[:, j]
        out[f"Feature_{feat}_p_dir"] = scores.p_dir[:, j]

    ctrl_rate = float(np.mean(scores.is_learner[is_ctrl])) if is_ctrl.any() else np.nan
    exp_rate = float(np.mean(scores.is_learner[~is_ctrl])) if (~is_ctrl).any() else np.nan
    print("\n" + "=" * 60)
    print("--- Improved Classifier Summary ---")
    print("=" * 60)
    print(f"  Reference condition: {ref_cond} (n_ctrl={int(is_ctrl.sum())})")
    scoring_mode = "SE-aware" if bool(use_per_fish_se_in_scoring) else "control-SD (SE-free)"
    print(f"  Method: whitened one-sided quadratic form; scoring={scoring_mode}")
    required_votes = int(len(config.features_to_use))
    print(f"  alpha={alpha:.3f}, required_votes={required_votes}/{required_votes}")
    print(f"  Calibrated threshold T*: {scores.threshold_T:.4f}")
    print(f"  Observed control learner rate: {ctrl_rate * 100:.1f}% (target {alpha * 100:.1f}%)")
    print(f"  Observed experimental learner rate: {exp_rate * 100:.1f}%")
    print("=" * 60 + "\n")

    if RUN_EXPORT_RESULTS:
        fname = FNAME_CLASSIFICATION_RESULTS_TEMPLATE.format(csus=config.csus)
        results_path = Path(path_pooled_data) / (Path(fname).stem + "_wip" + Path(fname).suffix)
        out.to_csv(results_path, index=False)
        print(f"  Saved improved classification results: {results_path.name}")

    # Build plotting-compatible result object
    feature_names = [config.feature_configs[feat].name for feat in config.features_to_use]
    # Outlier = T > threshold but NOT in learning direction (so "T>thr" label only for wrong/mixed direction)
    is_outlier_T = (scores.T > float(scores.threshold_T)) & (scores.votes_dir < required_votes)
    result_obj = ClassificationResult(
        fish_ids=np.array(common_fish),
        conditions=conds,
        features=X,
        features_se=X_se,
        feature_names=feature_names,
        distances=distances,
        votes_point=scores.votes_dir,
        votes_probabilistic=scores.votes_dir,
        p_learning=scores.p_dir,
        z_dir=scores.z_dir,
        is_outlier=is_outlier_T,
        is_learner_point=np.zeros_like(scores.is_learner, dtype=bool),
        is_learner_probabilistic=scores.is_learner,
        threshold=float(scores.threshold_T),
        threshold_ci=(float(scores.threshold_T), float(scores.threshold_T)),
        mu_ctrl=scores.mu_ctrl_full,
        T_scores=scores.T,
        p_empirical=scores.p_empirical,
    )

    # Optional plotting (save with _wip in filename)
    if RUN_PLOT_TRAJECTORIES:
        traj_path = Path(path_pooled_vigor_fig) / (Path(FNAME_TRAJECTORIES).stem + "_wip" + Path(FNAME_TRAJECTORIES).suffix)
        fig = plot_behavioral_trajectories(data, result_obj, config, save_path=traj_path)
        plt.close(fig)

    if RUN_PLOT_FEATURE_SPACE and len(config.features_to_use) > 1:
        feat_path = Path(path_pooled_vigor_fig) / (Path(FNAME_FEATURE_SPACE).stem + "_wip" + Path(FNAME_FEATURE_SPACE).suffix)
        fig = plot_feature_space(result_obj, config, save_path=feat_path)
        if fig is not None:
            plt.close(fig)

    if RUN_PLOT_BLUP_OVERLAY:
        overlay_path = Path(path_pooled_vigor_fig) / (Path(FNAME_BLUP_OVERLAY).stem + "_wip" + Path(FNAME_BLUP_OVERLAY).suffix)
        fig = plot_blup_trajectory_overlay(
            result_obj,
            config,
            split_by_condition=True,
            save_path=overlay_path,
            show_uncertainty_bands=True,
        )
        plt.close(fig)

    if RUN_PLOT_BLUP_CATERPILLAR:
        for feat_code in config.features_to_use:
            fname = FNAME_BLUP_CATERPILLAR_TEMPLATE.format(feat=str(feat_code))
            cater_path = Path(path_pooled_vigor_fig) / (Path(fname).stem + "_wip" + Path(fname).suffix)
            fig = plot_blup_caterpillar(result_obj, config, feat_code=str(feat_code), save_path=cater_path, sort_by_blup=True)
            plt.close(fig)

    if RUN_PLOT_INDIVIDUALS_AND_GRID:
        combined_indiv_dir = Path(path_pooled_vigor_fig) / DIR_INDIVIDUAL_PLOTS_COMBINED
        for cond in config.cond_types:
            cond_dir = combined_indiv_dir / str(cond)
            save_combined_plots_and_grid(data, result_obj, config, cond_dir, condition=str(cond), filename_suffix="_wip")

    if RUN_PLOT_HEATMAP_GRID:
        heatmap_grid_dir = Path(path_pooled_vigor_fig) / "Heatmap_Grids"
        for cond in config.cond_types:
            save_heatmap_grid(result_obj, config, Path(path_scaled_vigor_fig_cs), heatmap_grid_dir, condition=str(cond), filename_suffix="_wip", heatmap_variant="scaled")
            save_heatmap_grid(result_obj, config, Path(path_raw_vigor_fig_cs), heatmap_grid_dir, condition=str(cond), filename_suffix="_wip", heatmap_variant="raw")

    return out


if __name__ == "__main__":
    run_multivariate_lme_pipeline(
        analysis_cfg,
        alpha=ALPHA_TARGET,
        use_per_fish_se_in_scoring=USE_PER_FISH_SE_IN_SCORING,
    )

