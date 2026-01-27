"""
Multivariate Learner Classification Pipeline
===========================================

PURPOSE:
    Identify individual learners in behavioral experiments by extracting subject-level features
    from trial data using mixed-effects models, and classifying them via robust multivariate statistics.

WORKFLOW:
    1. Load and preprocess trial-level behavioral data, including block/trial structure and filtering.
    2. Extract behavioral features for each subject using Linear Mixed-Effects (LME) models:
        - Feature B: Outcome Drop (change from Pre-Train (10 trials) to Train End + Early Test)
        - Feature C: Recovery Increase (change from Train End + Early Test to Late Test (last 10 trials))
       Features are "anchored" to the control group to improve robustness.
    3. Compute BLUPs (Best Linear Unbiased Predictors) and uncertainty (SE, CI) for each feature.
    4. Build a multivariate feature matrix and calculate Mahalanobis distances from the control group,
       using Ledoit-Wolf shrinkage for robust covariance estimation.
    5. Determine a classification threshold (percentile of control distances, with bootstrap CI).
    6. Classify subjects as "learners" using two voting systems:
        - Point Estimate: BLUP passes directional test vs. control mean
        - Conservative: CI passes directional test vs. control mean (stricter)
    7. Visualize results: individual and summary plots, feature space, diagnostics, and export results.

KEY OUTPUTS:
    - Per-subject classification (learner/non-learner, outlier status, votes)
    - Individual and summary behavioral plots with feature pass/fail details
    - Diagnostic plots (normality, correlation, PCA, distance distributions)
    - Exported results and summary tables

ASSUMPTIONS:
    - Multivariate normality (tested via Shapiro-Wilk and Mardia's tests)
    - Homogeneity of variance (robustified via shrinkage)
    - Sufficient sample size per group (≥10 recommended)
    - Linear relationships for LME modeling

CUSTOMIZATION:
    - Select features: config.features_to_use = ['acquisition'], ['extinction'], or any combination
    - Adjust voting thresholds: config.voting_threshold_point, config.voting_threshold_conservative
    - Set classification percentile: config.threshold_percentile (e.g., 75 for upper quartile)
    - Modify block definitions and trial requirements in AnalysisConfig

REFERENCES:
    - Mixed-effects modeling: statsmodels.formula.api.mixedlm
    - Robust covariance: sklearn.covariance.LedoitWolf
    - Mahalanobis distance: scipy.spatial.distance
    - Bootstrap CI: numpy.random, np.percentile
"""
# %%
# region Imports & Configuration

import itertools
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas.api.types import CategoricalDtype
from scipy import stats
from scipy.spatial import distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

# Add the repository root containing shared modules to the Python path.
if "__file__" in globals():
    module_root = Path(__file__).resolve()
else:
    module_root = Path.cwd()

import analysis_utils
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config

# Configuration
pd.set_option("mode.copy_on_write", True)
# pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

# Set plotting style (shared across analysis scripts)
plotting_style.set_plot_style()
sns.color_palette("colorblind")

# endregion

# region Parameters
# ==============================================================================
# EXPERIMENT SELECTION
# ==============================================================================
EXPERIMENT = ExperimentType.ALL_DELAY.value
config = get_experiment_config(EXPERIMENT)
exp_config = config

# ==============================================================================
# PIPELINE CONTROL FLAGS
# ==============================================================================
# Toggle analysis stages to mirror Pipeline_Analysis.py behavior.
RUN_PIPELINE = True
RUN_DIAGNOSTICS = True
RUN_PLOT_DIAGNOSTICS = False
RUN_PLOT_TRAJECTORIES = False
RUN_PLOT_FEATURE_SPACE = False
RUN_PLOT_BLUP_CATERPILLAR = False
RUN_PLOT_INDIVIDUALS = True
RUN_PLOT_BLUP_OVERLAY = True
RUN_EXPORT_DETAILED_SUMMARY = False
RUN_EXPORT_RESULTS = False

# ==============================================================================
# ANALYSIS CONFIGURATION
# ==============================================================================

@dataclass
class FeatureConfig:
    """Configuration for a single feature."""
    name: str
    direction: str  # 'negative' or 'positive'
    description: str


@dataclass
class AnalysisConfig:
    """Configuration parameters for the analysis."""
    csus: str = 'CS'
    output_format: str = 'png'
    n_bootstrap: int = 1000
    random_seed: Optional[int] = 0
    threshold_percentile: float = 50
    min_latepretrain_trials: int = 10
    min_earlytest_trials: int = 15
    min_late_test_trials: int = 10
    min_trials_per_5trial_block_in_epoch: int = 3
    y_lim_plot: Tuple[float, float] = (0.8, 1.2)
    
    # Feature selection: which features to use for classification
    # Options: any combination of ['acquisition', 'extinction']
    features_to_use: List[str] = field(default_factory=lambda: ['acquisition', 'extinction'])
    
    # Voting thresholds
    voting_threshold_point: int = 3        # For point estimate method
    voting_threshold_conservative: int = 3  # For conservative CI method
    
    # IMPORTANT:
    # - se_multiplier_for_voting controls the conservative voting rule ("k × SE"), and does NOT have to be 1.96.
    # - ci_multiplier_for_reporting controls what you print/plot as a "CI" (typically 1.96 for ~95%).
    se_multiplier_for_voting: float = 1.96
    ci_multiplier_for_reporting: float = 1.96

    # Feature definitions
    feature_configs: Dict[str, FeatureConfig] = field(default_factory=lambda: {
        'acquisition': FeatureConfig('Outcome Drop', 'negative',
                          'Response change when outcome removed (Pre-Train (10 trials) -> Train End + Early Test)'),
        'extinction': FeatureConfig('Recovery Increase', 'positive',
                          'Response change when no outcome (Train End + Early Test -> Late Test (last 10 trials))')
    })
    
    # Block definitions
    pretrain_blocks_5: List[str] = field(default_factory=lambda: ['Early Pre-Train', 'Late Pre-Train'])
    # earlytest_block: List[str] = field(default_factory=lambda: ['Late Train', 'Early Test'])
    late_test_blocks_5: List[str] = field(default_factory=lambda: ['Test 5', 'Late Test'])
    pretrain_block: List[str] = field(default_factory=lambda: ['Early Pre-Train', 'Late Pre-Train'])
    earlytest_block: List[str] = field(default_factory=lambda: ['Late Train', 'Early Test'])
    latetest_block: List[str] = field(default_factory=lambda: ['Test 5', 'Late Test'])
    pretrain_to_train_end_earlytest_blocks: Tuple[List[str], List[str]] = field(
        default_factory=lambda: (['Early Pre-Train', 'Late Pre-Train'], ['Late Train', 'Early Test'])
    )
    train_end_earlytest_to_late_test_blocks: Tuple[List[str], List[str]] = field(
        default_factory=lambda: (['Late Train', 'Early Test'], ['Test 5', 'Late Test'])
    )
    
    # Derived from experiment type
    cr_window: List[int] = field(default_factory=list)
    cond_types: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.cr_window = list(exp_config.cr_window)
        self.cond_types = list(exp_config.cond_types)
        
        # Validate and adjust voting thresholds
        n_features = len(self.features_to_use)
        if self.voting_threshold_point > n_features:
            print(f"⚠️  Adjusting voting_threshold_point: {self.voting_threshold_point} → {n_features}")
            self.voting_threshold_point = n_features
        if self.voting_threshold_conservative > n_features:
            print(f"⚠️  Adjusting voting_threshold_conservative: {self.voting_threshold_conservative} → {n_features}")
            self.voting_threshold_conservative = n_features
    
    def print_summary(self):
        """Print configuration summary."""
        print(f"\n{'='*60}")
        print("ANALYSIS CONFIGURATION")
        print(f"{'='*60}")
        print(f"Features selected: {self.features_to_use}")
        for feat in self.features_to_use:
            cfg = self.feature_configs[feat]
            print(f"   {feat}: {cfg.name} (expect {cfg.direction})")
        print(f"Voting threshold (Point):        {self.voting_threshold_point}/{len(self.features_to_use)}")
        print(f"Voting threshold (Conservative): {self.voting_threshold_conservative}/{len(self.features_to_use)}")
        print(f"Distance percentile:             {self.threshold_percentile}th")
        print(f"Voting rule (k×SE):              k={self.se_multiplier_for_voting}")
        print(f"Reporting CI multiplier:         z={self.ci_multiplier_for_reporting}")
        print(f"Bootstrap seed:                  {self.random_seed}")
        print(f"Min trials per 5-trial block:    {self.min_trials_per_5trial_block_in_epoch}")
        print(f"{'='*60}\n")


@dataclass
class BLUPResult:
    """Container for BLUP extraction results."""
    blup: float
    se: float
    ci_lower: float
    ci_upper: float
    fixed: float
    random: float


@dataclass
class ClassificationResult:
    """Container for classification results."""
    fish_ids: np.ndarray
    conditions: np.ndarray
    features: np.ndarray          # Shape: (n_fish, n_features)
    features_se: np.ndarray       # Shape: (n_fish, n_features)
    feature_names: List[str]
    distances: np.ndarray
    votes_point: np.ndarray
    votes_conservative: np.ndarray
    is_outlier: np.ndarray
    is_learner_point: np.ndarray
    is_learner_conservative: np.ndarray
    threshold: float
    threshold_ci: Tuple[float, float]
    mu_ctrl: np.ndarray           # Control group means


# ==============================================================================
# ANALYSIS CONFIGURATION (DEFAULTS)
# ==============================================================================
analysis_cfg = AnalysisConfig(
    # Modify these to experiment with different settings.
    features_to_use=['acquisition', 'extinction'],
    pretrain_to_train_end_earlytest_blocks=(['Early Pre-Train', 'Late Pre-Train'], ['Late Train', 'Early Test']),
    train_end_earlytest_to_late_test_blocks=(['Late Train', 'Early Test'], ['Test 5', 'Late Test']),
    min_latepretrain_trials=6,
    min_earlytest_trials=6,
    min_late_test_trials=6,
    voting_threshold_point=2,
    voting_threshold_conservative=2,
    threshold_percentile=75,
    se_multiplier_for_voting=0.5,
    ci_multiplier_for_reporting=1.96,
    random_seed=0,
    min_trials_per_5trial_block_in_epoch=3,
    n_bootstrap=1000,
    y_lim_plot=(0.8, 1.2),
)

# endregion


# region Pipeline Functions
# ============================================================================
# Helper Functions
# ============================================================================

def get_blups_with_uncertainty(
    df: pd.DataFrame,
    formula: str,
    groups_col: str,
    re_var: str,
    method: str = 'powell',
    ci_multiplier: float = 1.96
) -> Dict[str, BLUPResult]:
    """
    Extract BLUPs (Best Linear Unbiased Predictors) with uncertainty estimates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing response, predictors, and grouping variable
    formula : str
        Model formula (e.g., "Log_Response ~ Log_Baseline + Epoch")
    groups_col : str
        Column name for random effect grouping (e.g., "Fish_ID")
    re_var : str
        Random effect variable name (e.g., "Epoch")
    method : str
        Optimization method for mixed model fitting
    ci_multiplier : float
        Multiplier for SE to compute CIs (1.96 for 95% CI)
        
    Returns
    -------
    Dict[str, BLUPResult]
        Dictionary mapping group IDs to BLUP results with uncertainty
    """
    if df.empty or df[groups_col].nunique() < 2:
        return {}
    
    try:
        # Fit a mixed-effects model so we can separate population-level change (fixed effect)
        # from per-fish deviations (random effects) on the epoch term.
        model = smf.mixedlm(formula, df, groups=df[groups_col], re_formula=f"~{re_var}")
        result = model.fit(reml=True, method=method)
        
        # Fixed effect represents the average epoch change across all fish.
        fixed_val = result.params[re_var]
        fixed_se = result.bse[re_var]
        
        # Prepare for conditional variance (posterior variance) calculation
        # Var(u|y) = (G^-1 + Z'R^-1Z)^-1
        scale = result.scale
        G = result.cov_re.values

        G_inv = None
        
        # Attempt to invert G (Random Effects Covariance)
        try:
            G_inv = np.linalg.inv(G)
            use_cond_var = True
        except np.linalg.LinAlgError:
            # If G is singular, fallback to population variance
            use_cond_var = False
            
        # Find index of the random effect variable in the covariance matrix
        try:
            re_idx = result.cov_re.columns.get_loc(re_var)
        except KeyError:
            re_idx = 0
            
        blups = {}
        for group_id, effects in result.random_effects.items():
            # Random effect is the fish-specific deviation from the population mean.
            random_val = effects.get(re_var, 0)
            blup_val = fixed_val + random_val
            
            # Calculate Conditional Variance (Prediction Error Variance)
            cond_var = 0.0
            calculated = False
            
            if use_cond_var and hasattr(model, 'row_indices'):
                idxs = model.row_indices.get(group_id)
                if idxs is not None:
                    try:
                        Z_i = model.exog_re[idxs]
                        # Posterior Precision = G^-1 + (1/sigma^2) * Z'Z
                        if G_inv is None:
                            raise ValueError("G_inv unavailable")
                        precision = G_inv + (Z_i.T @ Z_i) / scale
                        v_post = np.linalg.inv(precision)
                        cond_var = v_post[re_idx, re_idx]
                        calculated = True
                    except (np.linalg.LinAlgError, ValueError):
                        pass
            
            if not calculated:
                # Fallback: Use population variance if conditional variance cannot be computed.
                cond_var = _extract_random_variance(effects, result.cov_re, re_var)
            
            # Total SE combines fixed-effect uncertainty and random-effect prediction error.
            total_se = np.sqrt(fixed_se**2 + cond_var)
            
            blups[group_id] = BLUPResult(
                blup=blup_val,
                se=total_se,
                ci_lower=blup_val - ci_multiplier * total_se,
                ci_upper=blup_val + ci_multiplier * total_se,
                fixed=fixed_val,
                random=random_val
            )
        
        return blups
    
    except Exception as e:
        print(f"⚠️  BLUP Extraction Error: {e}")
        return {}


def _extract_random_variance(
    effects: pd.Series,
    cov_re: pd.DataFrame,
    re_var: str
) -> float:
    """Extract variance of random effect from covariance matrix."""
    if not hasattr(cov_re, 'iloc'):
        return 0.0
    try:
        # Map the requested random effect name to the covariance diagonal entry.
        re_var_idx = list(effects.index).index(re_var) if re_var in effects.index else 0
        arr = cov_re.to_numpy(dtype=float)
        return float(arr[re_var_idx, re_var_idx])
    except (IndexError, ValueError):
        return 0.0
    except Exception:
        return 0.0

def _get_fish_style(
    fish: str,
    fish_list: List[str],
    result: ClassificationResult,
    is_ref: bool
) -> Tuple[str, Dict[str, Any]]:
    """Determine plot style for a fish."""
    try:
        idx = list(result.fish_ids).index(fish)
    except ValueError:
        return 'gray', {'alpha': 0.3, 'linewidth': 1}

    # Encode learner/outlier status into color and emphasis for plotting.
    if result.is_learner_conservative[idx]:
        return 'red', {'alpha': 0.8, 'linewidth': 2, 'zorder': 4}
    elif result.is_learner_point[idx]:
        return 'orange', {'alpha': 0.6, 'linewidth': 1.5, 'zorder': 3}
    elif result.is_outlier[idx]:
        return 'goldenrod', {'alpha': 0.4, 'linewidth': 1, 'zorder': 2}
    else:
        color = 'lightgray' if is_ref else 'lightblue'
        return color, {'alpha': 0.3, 'linewidth': 1, 'zorder': 1}

def check_multivariate_normality(
    X: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> bool:
    """
    Test multivariate normality using Shapiro-Wilk and Mardia's tests.
    
    Returns True if normality assumption is satisfied.
    """
    n, p = X.shape
    
    if verbose:
        print(f"\n{'='*60}")
        print("MULTIVARIATE NORMALITY DIAGNOSTICS")
        print(f"{'='*60}")
    
    # Univariate Shapiro-Wilk Tests
    shapiro_results = []
    if verbose:
        print("\n1. Univariate Shapiro-Wilk Tests:")
    
    for i, name in enumerate(feature_names):
        # Shapiro-Wilk tests each feature for univariate normality.
        stat, pval = stats.shapiro(X[:, i])
        shapiro_results.append(pval)
        if verbose:
            sig = "❌ Non-normal" if pval < 0.05 else "✓ Normal"
            print(f"   {name:20s}: W={stat:.4f}, p={pval:.4f} {sig}")
    
    # Mardia's Tests (only for multivariate case)
    mardia_ok = True
    if p > 1:
        if verbose:
            print("\n2. Mardia's Multivariate Tests:")
        
        # Mardia's tests evaluate multivariate skewness and kurtosis.
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            m = X_centered @ inv_cov @ X_centered.T
            
            # Skewness
            skewness = np.sum(m**3) / (n**2)
            skew_stat = (n * skewness) / 6
            skew_df = p * (p + 1) * (p + 2) / 6
            skew_pval = 1 - stats.chi2.cdf(skew_stat, skew_df)
            
            if verbose:
                skew_sig = "❌ Reject" if skew_pval < 0.05 else "✓ OK"
                print(f"   Skewness:  β={skewness:.4f}, χ²={skew_stat:.2f}, p={skew_pval:.4f} {skew_sig}")
            
            # Kurtosis
            d = np.diag(m)
            kurtosis = np.sum(d**2) / n
            expected_kurt = p * (p + 2)
            kurt_stat = (kurtosis - expected_kurt) / np.sqrt(8 * p * (p + 2) / n)
            kurt_pval = 2 * (1 - stats.norm.cdf(abs(kurt_stat)))
            
            if verbose:
                kurt_sig = "❌ Reject" if kurt_pval < 0.05 else "✓ OK"
                print(f"   Kurtosis:  β={kurtosis:.4f}, Z={kurt_stat:.2f}, p={kurt_pval:.4f} {kurt_sig}")
            
            mardia_ok = (skew_pval > 0.05) and (kurt_pval > 0.05)
            
        except np.linalg.LinAlgError:
            if verbose:
                print("   ⚠️  Singular covariance - cannot compute Mardia's test")
            mardia_ok = False
    
    univariate_ok = all(p > 0.05 for p in shapiro_results)
    normality_ok = mardia_ok and univariate_ok
    
    if verbose:
        print(f"\n{'='*60}")
        status = "✓ SATISFIED" if normality_ok else "❌ VIOLATED"
        print(f"OVERALL: Multivariate normality {status}")
        print(f"{'='*60}\n")
    
    return bool(normality_ok)


def analyze_feature_correlation(
    X: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> Tuple[np.ndarray, List[Tuple]]:
    """Analyze correlation structure of features."""
    if verbose:
        print(f"\n{'='*60}")
        print("FEATURE CORRELATION ANALYSIS")
        print(f"{'='*60}\n")
    
    # Correlation matrix summarizes redundancy across features.
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    if verbose:
        print("Correlation Matrix:")
        corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
        print(corr_df.round(3))
    
    high_corr_pairs = []
    # Flag highly correlated pairs that may inflate Mahalanobis distances.
    high_corr = np.where((np.abs(corr_matrix) > 0.7) & (corr_matrix != 1))
    
    if len(high_corr[0]) > 0 and verbose:
        print("\n⚠️  HIGH CORRELATIONS DETECTED (|r| > 0.7):")
        for i, j in zip(high_corr[0], high_corr[1]):
            if i < j:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i,j]))
                print(f"   {feature_names[i]:20s} ↔ {feature_names[j]:20s}: r={corr_matrix[i,j]:.3f}")
    
    if verbose:
        print(f"\n{'='*60}\n")
    
    return corr_matrix, high_corr_pairs


def perform_pca(
    X: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> Tuple[PCA, np.ndarray]:
    """Perform Principal Component Analysis."""
    if verbose:
        print(f"\n{'='*60}")
        print("PRINCIPAL COMPONENT ANALYSIS")
        print(f"{'='*60}\n")
    
    # PCA is a diagnostic tool to check variance structure and feature redundancy.
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    if verbose:
        print("Variance Explained:")
        cumulative = 0
        for i, var in enumerate(pca.explained_variance_ratio_):
            cumulative += var
            print(f"   PC{i+1}: {var*100:5.1f}% (cumulative: {cumulative*100:5.1f}%)")
        
        # Loadings indicate how each feature contributes to each component.
        print("\nLoadings:")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(feature_names))],
            index=feature_names
        )
        print(loadings.round(3))
        print(f"\n{'='*60}\n")
    
    return pca, X_pca


def get_robust_mahalanobis(
    X: np.ndarray,
    X_ref: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Calculate Mahalanobis distances with robust covariance estimation.
    Uses Ledoit-Wolf shrinkage for stable covariance estimation.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("ROBUST COVARIANCE ESTIMATION")
        print(f"{'='*60}\n")
    
    # Use the reference group's mean and covariance as the baseline distribution.
    mu_ref = np.mean(X_ref, axis=0)
    n_ref, p = X_ref.shape
    
    if verbose:
        print(f"Reference group: n={n_ref}, p={p}, n/p={n_ref/p:.2f}")
    
    # Standard covariance diagnostics
    cov_standard = np.cov(X_ref, rowvar=False)
    if p > 1:
        condition_number = np.linalg.cond(cov_standard)
        if verbose:
            print(f"Condition number: {condition_number:.2e}")
    
    # Ledoit-Wolf shrinkage stabilizes covariance when n is small relative to p.
    lw = LedoitWolf()
    cov_shrunk = lw.fit(X_ref).covariance_
    shrinkage = lw.shrinkage_
    
    if verbose:
        print(f"Ledoit-Wolf shrinkage: δ = {shrinkage:.4f}")
    
    # Invert the covariance; fall back to pseudoinverse if needed.
    try:
        inv_cov = np.linalg.inv(cov_shrunk)
        method_used = "Direct inversion (Ledoit-Wolf)"
        if verbose:
            print("✓ Successfully inverted covariance")
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_shrunk, rcond=1e-10)
        method_used = "Moore-Penrose pseudoinverse"
        if verbose:
            print("⚠️  Using pseudoinverse")
    
    if verbose:
        print(f"{'='*60}\n")
    
    # Calculate Mahalanobis distances for all fish in feature space.
    distances = []
    for x in X:
        try:
            d = distance.mahalanobis(x, mu_ref, inv_cov)
            distances.append(d)
        except:
            d = np.sqrt(np.sum(((x - mu_ref) / np.std(X_ref, axis=0))**2))
            distances.append(d)
    
    return np.array(distances), inv_cov, cov_shrunk, method_used


def bootstrap_threshold(
    distances_ref: np.ndarray,
    percentile: float = 95,
    n_boot: int = 1000,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap confidence interval for distance threshold."""
    rng = np.random.default_rng(seed)
    n = len(distances_ref)
    # Resample the control distances to estimate the percentile threshold distribution.
    boot_thresholds = np.array([
        np.percentile(rng.choice(distances_ref, size=n, replace=True), percentile)
        for _ in range(n_boot)
    ])
    
    # Use the median bootstrap percentile as the robust threshold estimate.
    thresh_median = float(np.median(boot_thresholds))
    thresh_ci = (float(np.percentile(boot_thresholds, 2.5)), float(np.percentile(boot_thresholds, 97.5)))
    
    if verbose:
        thresh_point = float(np.percentile(distances_ref, percentile))
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION THRESHOLD ({percentile}th percentile)")
        print(f"{'='*60}")
        print(f"Point estimate:    {thresh_point:.4f}")
        print(f"Bootstrap median:  {thresh_median:.4f}")
        print(f"Bootstrap 95% CI:  [{thresh_ci[0]:.4f}, {thresh_ci[1]:.4f}]")
        print(f"{'='*60}\n")
    
    return thresh_median, thresh_ci


def classify_learners(
    X: np.ndarray,
    X_se: np.ndarray,
    distances: np.ndarray,
    threshold: float,
    mu_ctrl: np.ndarray,
    config: 'AnalysisConfig'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify subjects using point estimate and conservative voting.
    
    Returns
    -------
    Tuple: (is_outlier, votes_point, votes_conservative, is_learner_point, is_learner_conservative)
    """
    n_samples = len(X)
    n_features = len(config.features_to_use)
    
    votes_point = np.zeros(n_samples, dtype=int)
    votes_conservative = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        for feat_idx, feat in enumerate(config.features_to_use):
            # Compare each fish's BLUP to the control mean in the expected direction.
            feat_cfg = config.feature_configs[feat]
            blup = X[i, feat_idx]
            se = X_se[i, feat_idx]
            ctrl_mean = mu_ctrl[feat_idx]
            
            if feat_cfg.direction == 'negative':
                # Point: BLUP < control mean
                if blup < ctrl_mean:
                    votes_point[i] += 1
                # Conservative: upper CI < control mean
                if (blup + config.se_multiplier_for_voting * se) < ctrl_mean:
                    votes_conservative[i] += 1
            else:  # positive direction
                # Point: BLUP > control mean
                if blup > ctrl_mean:
                    votes_point[i] += 1
                # Conservative: lower CI > control mean
                if (blup - config.se_multiplier_for_voting * se) > ctrl_mean:
                    votes_conservative[i] += 1
    
    # Outliers must exceed the Mahalanobis threshold and pass voting criteria.
    is_outlier = distances > threshold
    is_learner_point = is_outlier & (votes_point >= config.voting_threshold_point)
    is_learner_conservative = is_outlier & (votes_conservative >= config.voting_threshold_conservative)
    
    return is_outlier, votes_point, votes_conservative, is_learner_point, is_learner_conservative


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_data(config: AnalysisConfig, path_pooled_data: Path) -> pd.DataFrame:
    """Load and validate the most recent pooled dataset for the selected CS/US."""
    # Use the newest pooled file matching the requested CS/US label.
    paths = sorted(
        [*Path(path_pooled_data).glob("*.pkl")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    paths = [p for p in paths
             if "NV per trial per fish" in p.stem
             and p.stem.split("_")[-1] == config.csus]
    
    if not paths:
        raise FileNotFoundError("No matching pooled data file found.")
    
    print(f"Loading: {paths[0].name}")
    data = pd.read_pickle(paths[0], compression='gzip')
    
    if data.empty:
        raise ValueError("Loaded dataframe is empty.")
    
    return data


def prepare_data(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Prepare data with proper block structure and filtering."""
    df = df.copy()
    
    # Identify blocks and trials using experiment metadata.
    df["Trial type"] = config.csus
    df = analysis_utils.identify_blocks_trials(df, exp_config.blocks_dict)
    df["Block name 10 trials"] = df["Block name"]
    
    # Create 5-trial blocks to match the epoch definitions used for features.
    df = _create_5_trial_blocks(df)
    
    # Rename columns into consistent analysis labels.
    baseline_col = [c for c in df.columns if 's before' in c][0]
    response_col = 'Mean CR'
    
    df = df.rename(columns={'Exp.': 'Condition', 'Fish': 'Fish_ID'})
    df = df[df['Condition'].isin(config.cond_types)]
    df.dropna(subset=['Normalized vigor', 'Trial number', 'Block name 5 trials', 
                      'Block name 10 trials', baseline_col, response_col], inplace=True)
    
    # Filter fish by minimum trial requirements for each selected feature.
    df = _filter_fish_by_trials(df, config)
    
    # Log-transform baseline/response to stabilize variance for mixed models.
    df['Log_Baseline'] = np.log(df[baseline_col] + 1)
    df['Log_Response'] = np.log(df[response_col] + 1)
    df['Trial number'] = df['Trial number'].astype(int)
    df['Trial number'] = df['Trial number'] - df['Trial number'].min() + 1
    df['Normalized vigor plot'] = df['Normalized vigor']
    
    return df


def _create_5_trial_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Create 5-trial block structure."""
    number_blocks_original = df['Block name'].nunique()
    number_trials_block = 5
    
    # Map experiment-specific block counts into standardized 5-trial block names.
    if number_blocks_original == 9:
        block_names = [
            'Early Pre-Train', 'Late Pre-Train', 'Early Train', 'Train 2', 
            'Train 3', 'Train 4', 'Train 5', 'Train 6', 'Train 7', 'Train 8', 
            'Train 9', 'Late Train', 'Early Test', 'Test 2', 'Test 3', 
            'Test 4', 'Test 5', 'Late Test'
        ]
    elif number_blocks_original == 12:
        block_names = [
            'Early Pre-Train', 'Late Pre-Train', 'Early Train', 'Train 2', 
            'Train 3', 'Train 4', 'Train 5', 'Train 6', 'Train 7', 'Train 8', 
            'Train 9', 'Late Train', 'Early Test', 'Test 2', 'Test 3', 
            'Test 4', 'Test 5', 'Late Test', 'Early Re-Train', 'Re-Train 2', 
            'Re-Train 3', 'Re-Train 4', 'Re-Train 5', 'Late Re-Train'
        ]
    else:
        raise ValueError(f"Unexpected number of blocks: {number_blocks_original}")
    
    # Keep only valid 10-trial blocks before splitting into 5-trial blocks.
    df = df[df["Block name"].isin(exp_config.names_cs_blocks_10)].copy()
    df["Block name"] = df["Block name"].astype(
        CategoricalDtype(categories=exp_config.names_cs_blocks_10, ordered=True)
    )
    df.drop(columns="Block name", inplace=True)
    
    # Build contiguous 5-trial ranges using the original 10-trial definitions.
    blocks = [
        range(x, x + number_trials_block)
        for x in range(
            exp_config.trials_cs_blocks_10[0][0],
            exp_config.trials_cs_blocks_10[-1][-1] + 1,
            number_trials_block
        )
    ]
    
    df[f'Block name {number_trials_block} trials'] = ''
    for idx, trials in enumerate(blocks):
        # Assign each trial to a 5-trial block label.
        mask = df['Trial number'].astype(int).isin(trials)
        df.loc[mask, f'Block name {number_trials_block} trials'] = block_names[idx]
    
    df[f'Block name {number_trials_block} trials'] = df[f'Block name {number_trials_block} trials'].astype(
        CategoricalDtype(categories=block_names, ordered=True)
    )
    
    return df

def _filter_fish_by_trials(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Filter fish based on minimum trial requirements for the selected features.
    Only fish with enough trials for all selected features are retained.
    """
    valid_fish_sets = []

    # Keep only fish that meet the minimum trial counts needed to estimate each selected feature.
    # Use the same block definitions that are later used in extract_change_feature() so filtering is consistent.

    def _valid_each_block(blocks: List[str]) -> set:
        """Fish with >= min_trials_per_5trial_block_in_epoch trials in EACH block in `blocks`."""
        # This enforces coverage across all blocks, avoiding fish with missing epochs.
        if not blocks:
            return set(df['Fish_ID'].unique())
        counts = (
            df[df['Block name 5 trials'].isin(blocks)]
            .groupby(['Fish_ID', 'Block name 5 trials'], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        # Ensure all requested block columns exist
        for b in blocks:
            if b not in counts.columns:
                counts[b] = 0
        mask = np.ones(len(counts), dtype=bool)
        for b in blocks:
            # Require at least the configured minimum per block.
            mask &= counts[b].to_numpy() >= int(config.min_trials_per_5trial_block_in_epoch)
        return set(counts.index[mask])

    if 'acquisition' in config.features_to_use:
        # Feature "acquisition" (Outcome Drop): pre-train -> train end + early test
        pre_blocks, post_blocks = config.pretrain_to_train_end_earlytest_blocks

        valid_pre_each = _valid_each_block(pre_blocks)
        valid_post_each = _valid_each_block(post_blocks)

        pre_mask = df['Block name 5 trials'].isin(pre_blocks)
        pre_counts = df[pre_mask].groupby('Fish_ID', observed=True).size()
        valid_pre = set(pre_counts[pre_counts >= config.min_latepretrain_trials].index)

        post_mask = df['Block name 5 trials'].isin(post_blocks)
        post_counts = df[post_mask].groupby('Fish_ID', observed=True).size()
        valid_post = set(post_counts[post_counts >= config.min_earlytest_trials].index)

        valid_acquisition = valid_pre & valid_post & valid_pre_each & valid_post_each
        valid_fish_sets.append(valid_acquisition)

        print(
            f"   Feature acquisition (Outcome Drop): {len(valid_acquisition)} fish with "
            f">= {config.min_latepretrain_trials} trials in {pre_blocks} AND "
            f">= {config.min_earlytest_trials} trials in {post_blocks}"
            f" (and >= {config.min_trials_per_5trial_block_in_epoch} trials in EACH 5-trial block)"
        )

    if 'extinction' in config.features_to_use:
        # Feature "extinction" (Recovery Increase): train end + early test -> late test
        pre_blocks, post_blocks = config.train_end_earlytest_to_late_test_blocks

        valid_pre_each = _valid_each_block(pre_blocks)
        valid_post_each = _valid_each_block(post_blocks)

        pre_mask = df['Block name 5 trials'].isin(pre_blocks)
        pre_counts = df[pre_mask].groupby('Fish_ID', observed=True).size()
        valid_pre = set(pre_counts[pre_counts >= config.min_earlytest_trials].index)

        post_mask = df['Block name 5 trials'].isin(post_blocks)
        post_counts = df[post_mask].groupby('Fish_ID', observed=True).size()
        valid_post = set(post_counts[post_counts >= config.min_late_test_trials].index)

        valid_extinction = valid_pre & valid_post & valid_pre_each & valid_post_each
        valid_fish_sets.append(valid_extinction)

        print(
            f"   Feature extinction (Recovery Increase): {len(valid_extinction)} fish with "
            f">= {config.min_earlytest_trials} trials in {pre_blocks} AND "
            f">= {config.min_late_test_trials} trials in {post_blocks}"
            f" (and >= {config.min_trials_per_5trial_block_in_epoch} trials in EACH 5-trial block)"
        )


    # Intersect all sets to get fish with enough trials for all selected features
    if valid_fish_sets:
        valid_fish = set.intersection(*valid_fish_sets)
    else:
        valid_fish = set(df['Fish_ID'].unique())

    print(f"Fish filtering: {len(df['Fish_ID'].unique())} → {len(valid_fish)} (based on selected features: {config.features_to_use})")
    return df[df['Fish_ID'].isin(valid_fish)]



# ============================================================================
# Feature Extraction
# ============================================================================
def extract_change_feature(
    data: pd.DataFrame,
    config: AnalysisConfig,
    number_trials: int = 5,
    name_blocks: Tuple[List[str], List[str]] = (['Early Pre-Train', 'Late Pre-Train'], ['Train 9', 'Late Train', 'Early Test'])
) -> Dict[str, BLUPResult]:
    """
    Extract Feature B/C: Epoch change (Control-Anchored).
    Noisy fish are shrunk toward Control's epoch difference.
    """
    print(f"\n[B/C] Extracting Epoch Change: {name_blocks[0]} → {name_blocks[1]} (Control-Anchored)...")
    
    # Build a per-fish dataset with two epochs (pre vs test) to estimate the change.
    frames = []
    for fish in data['Fish_ID'].unique():
        sub = data[data['Fish_ID'] == fish]
        pre = sub[sub[f'Block name {number_trials} trials'].isin(name_blocks[0])].sort_values('Trial number').assign(Epoch=0)
        test = sub[sub[f'Block name {number_trials} trials'].isin(name_blocks[1])].sort_values('Trial number').assign(Epoch=1)
        
        if not pre.empty and not test.empty:
            frames.append(pd.concat([pre, test]))
    
    if not frames:
        print("   ⚠️  Insufficient data")
        return {}
    
    df_B = pd.concat(frames)
    
    # 1. Estimate the control group's epoch effect to anchor the change metric.
    ref_cond = config.cond_types[0]
    df_ctrl = df_B[df_B['Condition'] == ref_cond].copy()
    
    if df_ctrl['Fish_ID'].nunique() < 3:
        print("   ⚠️  Not enough control fish for anchoring, using standard method")
        return get_blups_with_uncertainty(
            df_B, "Log_Response ~ Log_Baseline + Epoch",
            "Fish_ID", "Epoch", ci_multiplier=config.ci_multiplier_for_reporting
        )
    
    print(f"   ...calculating anchor from {df_ctrl['Fish_ID'].nunique()} {ref_cond} fish")
    
    try:
        # Fit control-only LME to get the population-level epoch change.
        model_ctrl = smf.mixedlm(
            "Log_Response ~ Log_Baseline + Epoch", 
            df_ctrl, 
            groups=df_ctrl["Fish_ID"], 
            re_formula="~Epoch"
        )
        res_ctrl = model_ctrl.fit(reml=True, method='powell')
        ctrl_epoch_effect = res_ctrl.params['Epoch']
        ctrl_epoch_se = res_ctrl.bse.get('Epoch', np.nan)
        print(f"   ...Control Epoch Anchor: {ctrl_epoch_effect:.6f}")
    except Exception as e:
        print(f"   ⚠️  Control model failed ({e}), using standard method")
        return get_blups_with_uncertainty(
            df_B, "Log_Response ~ Log_Baseline + Epoch",
            "Fish_ID", "Epoch", ci_multiplier=config.ci_multiplier_for_reporting
        )

    # 2. Remove the control epoch effect so the model only learns fish-specific deviations.
    df_B['Log_Response_Adj'] = df_B['Log_Response'] - (ctrl_epoch_effect * df_B['Epoch'])

    # 3. Fit an LME without a fixed epoch term; random effects capture per-fish change.
    try:
        model = smf.mixedlm(
            "Log_Response_Adj ~ Log_Baseline",  # No Epoch in fixed effects!
            df_B, 
            groups=df_B["Fish_ID"], 
            re_formula="~Epoch"
        )
        result = model.fit(reml=True, method='powell')
        
        random_effects_cov = result.cov_re

        # Conditional variance for the random Epoch effect (prediction error variance)
        # Var(u|y) = (G^-1 + (1/sigma^2) Z'Z)^-1
        scale = result.scale
        G = result.cov_re.values
        G_inv = None
        use_cond_var = False
        try:
            G_inv = np.linalg.inv(G)
            use_cond_var = True
        except np.linalg.LinAlgError:
            use_cond_var = False

        try:
            re_idx = result.cov_re.columns.get_loc('Epoch')
        except Exception:
            re_idx = 0
        
        blups = {}
        for group_id, effects in result.random_effects.items():
            # Combine control anchor with the fish-specific random epoch deviation.
            random_epoch = effects.get('Epoch', 0)
            final_blup = ctrl_epoch_effect + random_epoch

            # Default: population variance fallback
            cond_var = _extract_random_variance(effects, random_effects_cov, 'Epoch')

            # Try to compute group-specific conditional variance if possible
            if use_cond_var and G_inv is not None and hasattr(model, 'row_indices'):
                idxs = model.row_indices.get(group_id)
                if idxs is not None:
                    try:
                        Z_i = model.exog_re[idxs]
                        # Posterior precision combines prior covariance with fish-specific data.
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
            
            blups[group_id] = BLUPResult(
                blup=final_blup,
                se=total_se,
                ci_lower=final_blup - config.ci_multiplier_for_reporting * total_se,
                ci_upper=final_blup + config.ci_multiplier_for_reporting * total_se,
                fixed=ctrl_epoch_effect,
                random=random_epoch
            )
            
        print(f"   ✓ Extracted for {len(blups)} fish (anchored to control)")
        return blups

    except Exception as e:
        print(f"⚠️  Anchored BLUP Extraction Error: {e}")
        return get_blups_with_uncertainty(
            df_B, "Log_Response ~ Log_Baseline + Epoch",
            "Fish_ID", "Epoch", ci_multiplier=config.ci_multiplier_for_reporting
        )



# ============================================================================
# Visualization
# ============================================================================

def plot_behavioral_trajectories(
    data: pd.DataFrame,
    result: ClassificationResult,
    config: AnalysisConfig,
    save_path: Optional[Path] = None
) -> Figure:
    """Create behavioral trajectory visualization."""
    # Order blocks by mean trial number so plots follow the experimental timeline.
    block_order = (data.groupby('Block name 10 trials', observed=True)['Trial number']
                   .mean().sort_values().index)
    
    # Collapse to per-fish median vigor per block for a clean trajectory plot.
    df_viz = (data.groupby(['Fish_ID', 'Block name 10 trials', 'Condition'], observed=True)
              ['Normalized vigor'].median().reset_index())
    
    # Filter result conditions to only those present in the data passed
    present_conditions = df_viz['Condition'].unique()
    unique_conds = [c for c in np.unique(result.conditions) if c in present_conditions]
    
    n_conds = len(unique_conds)
    ref_cond = config.cond_types[0]
    
    fig, axes = plt.subplots(1, n_conds, figsize=(6*n_conds, 6), 
                             sharex=True, sharey=True, facecolor='white')
    if n_conds == 1:
        axes = [axes]
    
    fig.suptitle('Learner Classification: Behavioral Trajectories', 
                 fontsize=5+14, fontweight='bold', y=0.98)
    
    fish_list = list(result.fish_ids)
    
    # Prepare detailed vote info for text box
    vote_details = ["Vote Details (Point/Cons):"]
    
    for i, (ax, cond) in enumerate(zip(axes, unique_conds)):
        ax.set_facecolor('white')
        df_cond = df_viz[df_viz['Condition'] == cond]
        is_ref = (cond == ref_cond)
        
        # Sort fish to draw learners on top - only include fish that are in result
        fish_in_cond = df_cond['Fish_ID'].unique()
        fish_in_result = [f for f in fish_in_cond if f in fish_list]
        fish_not_in_result = [f for f in fish_in_cond if f not in fish_list]
        
        fish_indices = [fish_list.index(f) for f in fish_in_result]
        
        # Sort by status: Non-learner < Outlier < Point < Conservative
        status_score = []
        for idx in fish_indices:
            score = 0
            if result.is_outlier[idx]: score = 1
            if result.is_learner_point[idx]: score = 2
            if result.is_learner_conservative[idx]: score = 3
            status_score.append(score)
            
        sorted_fish = [x for _, x in sorted(zip(status_score, fish_in_result))]
        
        # Plot fish not in result first (background, faded)
        for fish in fish_not_in_result:
            fish_data = df_cond[df_cond['Fish_ID'] == fish]
            color = 'lightgray' if is_ref else 'lightblue'
            sns.lineplot(data=fish_data, x='Block name 10 trials', y='Normalized vigor',
                        color=color, ax=ax, legend=False, alpha=0.2, linewidth=0.5, zorder=0)
        
        # Plot fish in result
        for fish in sorted_fish:
            fish_data = df_cond[df_cond['Fish_ID'] == fish]
            color, kwargs = _get_fish_style(fish, fish_list, result, is_ref)
            sns.lineplot(data=fish_data, x='Block name 10 trials', y='Normalized vigor',
                        color=color, ax=ax, legend=False, **kwargs)
            
            # Collect vote details for learners/outliers
            idx = fish_list.index(fish)
            if result.is_learner_point[idx] or result.is_outlier[idx]:
                feat_str = []
                for f_idx, feat in enumerate(config.features_to_use):
                    cfg = config.feature_configs[feat]
                    val = result.features[idx, f_idx]
                    se = result.features_se[idx, f_idx]
                    mu = result.mu_ctrl[f_idx]
                    
                    # Check pass/fail
                    if cfg.direction == 'negative':
                        p_pass = val < mu
                        c_pass = (val + config.se_multiplier_for_voting * se) < mu
                    else:
                        p_pass = val > mu
                        c_pass = (val - config.se_multiplier_for_voting * se) > mu
                    
                    p_mark = "✓" if p_pass else "✗"
                    c_mark = "✓" if c_pass else "✗"
                    feat_str.append(f"{feat}:{p_mark}{c_mark}")
                
                symbol = "★" if result.is_learner_conservative[idx] else ("●" if result.is_learner_point[idx] else "○")
                vote_details.append(f"{symbol} {fish}: {' '.join(feat_str)}")

        # Group median
        grp = df_cond.groupby('Block name 10 trials', observed=True)['Normalized vigor'].median().reset_index()
        sns.lineplot(data=grp, x='Block name 10 trials', y='Normalized vigor',
                    color='black', marker='o', markersize=6, ax=ax,
                    linewidth=2.5, zorder=10, label='Group Median')
        
        # Labels
        cond_mask = (result.conditions == cond)
        n_fish_in_result = len(fish_in_result)
        n_fish_total = len(fish_in_cond)
        n_learn_p = (cond_mask & result.is_learner_point).sum()
        n_learn_c = (cond_mask & result.is_learner_conservative).sum()
        
        ax.set_title(f"{cond.capitalize()} (n={n_fish_in_result}/{n_fish_total})\nLearners: {n_learn_p} point, {n_learn_c} conservative",
                    fontsize=5+11, fontweight='bold')
        ax.set_xlabel('Block', fontsize=5+10)
        ax.set_ylabel('Normalized Vigor' if i == 0 else '', fontsize=5+10)
        ax.axhline(1.0, linestyle=':', color='black', alpha=0.5)
        ax.set_xticklabels(block_order, rotation=45, ha='right', fontsize=5+9)
        ax.grid(alpha=0.3)
        ax.set_ylim(config.y_lim_plot)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=1, label='Non-Learner'),
        Line2D([0], [0], color='black', lw=2.5, marker='o', label='Group Median')
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=5+9, framealpha=0.9)
    
    # Add text box with vote details
    if len(vote_details) > 1:
        # Split into columns if too long
        text_str = "\n".join(vote_details)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
        fig.text(0.99, 0.5, text_str, fontsize=5+8, va='center', ha='right', 
                 bbox=props, fontfamily='monospace')
        plt.subplots_adjust(right=0.8)
    # else:
        # plt.tight_layout()
    
    # if save_path:
    #     # fig.savefig(str(save_path), dpi=300, facecolor='white', bbox_inches='tight')
    #     # print(f"✓ Saved: {save_path.name}")
    
    return fig


def plot_diagnostics(
    X: np.ndarray,
    feature_names: List[str],
    corr_matrix: np.ndarray,
    pca: PCA,
    X_pca: np.ndarray,
    distances: np.ndarray,
    ctrl_distances: np.ndarray,
    conds: np.ndarray,
    ref_cond: str,
    thresh_median: float,
    thresh_ci: Tuple[float, float],
    save_path: Optional[Path] = None
) -> Figure:
    """
    Create comprehensive diagnostic plots for the multivariate analysis.
    Includes: Correlation matrix, PCA scree plot, PCA biplot, and Distance distribution.
    """
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    gs = fig.add_gridspec(2, 3)
    
    # 1. Correlation Matrix Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=feature_names, yticklabels=feature_names, ax=ax1, square=True)
    ax1.set_title("Feature Correlations", fontsize=5+12, fontweight='bold')
    
    # 2. PCA Scree Plot
    ax2 = fig.add_subplot(gs[0, 1])
    explained_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(explained_var)
    x_ticks = range(1, len(explained_var) + 1)
    
    ax2.bar(x_ticks, explained_var, alpha=0.6, label='Individual')
    ax2.plot(x_ticks, cum_var, 'r-o', label='Cumulative')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance Explained (%)')
    ax2.set_title("PCA Variance Explained", fontsize=5+12, fontweight='bold')
    ax2.set_xticks(x_ticks)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. PCA Biplot (PC1 vs PC2)
    ax3 = fig.add_subplot(gs[0, 2])
    is_ref = (conds == ref_cond)
    
    # Plot points
    ax3.scatter(X_pca[is_ref, 0], X_pca[is_ref, 1], c='gray', alpha=0.6, label=ref_cond)
    ax3.scatter(X_pca[~is_ref, 0], X_pca[~is_ref, 1], c='steelblue', alpha=0.6, label='Experimental')
    
    # Plot loading vectors
    loadings = pca.components_.T
    scale = np.max(np.abs(X_pca)) * 0.8  # Scale vectors to fit plot
    
    for i, feature in enumerate(feature_names):
        # ax3.arrow(0, 0, loadings[i, 0] * scale, loadings[i,  1] * scale, 
        #          color='r', alpha=0.8, head_width=scale*0.05)
        ax3.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15, 
                feature, color='r', ha='center', va='center', fontsize=5+9)
                
    ax3.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    ax3.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    ax3.set_title("PCA Biplot", fontsize=5+12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    # 4. Mahalanobis Distance Distribution (Histogram)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Plot distributions
    sns.histplot(distances[is_ref], color='gray', label=f'{ref_cond} (Reference)', 
                kde=True, stat='density', alpha=0.4, ax=ax4, bins=15)
    sns.histplot(distances[~is_ref], color='steelblue', label='Experimental', 
                kde=True, stat='density', alpha=0.4, ax=ax4, bins=15)
    
    # Add theoretical Chi-square distribution
    df = len(feature_names)
    x = np.linspace(0, max(distances.max(), stats.chi2.ppf(0.99, df)), 100)
    ax4.plot(x, stats.chi2.pdf(x**2, df) * 2 * x, 'k--', lw=2, label=f"Theoretical $\\chi^2_{{{df}}}$")
    
    # Add threshold lines
    ax4.axvline(thresh_median, color='red', linestyle='-', lw=2, label='Threshold (Median)')
    ax4.axvspan(thresh_ci[0], thresh_ci[1], color='red', alpha=0.1, label='Threshold 95% CI')
    
    ax4.set_xlabel('Mahalanobis Distance')
    ax4.set_title("Distance Distribution & Threshold", fontsize=5+12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # plt.tight_layout()
    
    if save_path:
        # fig.savefig(str(save_path), dpi=300, facecolor='white', bbox_inches='tight')
        # print(f"✓ Saved: {save_path.name}")
        pass
        
    return fig


def _get_grid_info_text(
    fish_idx: int,
    result: ClassificationResult,
    config: AnalysisConfig
) -> str:
    """Generate compact feature info text for grid subplot."""
    lines = []
    
    # Condition abbreviation
    cond = result.conditions[fish_idx]
    cond_abbr = cond[:3].upper()
    
    # Distance and outlier status
    dist = result.distances[fish_idx]
    outlier_mark = "O" if result.is_outlier[fish_idx] else ""
    
    # Feature pass/fail summary
    feat_status = []
    for feat_idx, feat in enumerate(config.features_to_use):
        cfg = config.feature_configs[feat]
        blup = result.features[fish_idx, feat_idx]
        se = result.features_se[fish_idx, feat_idx]
        mu = result.mu_ctrl[feat_idx]
        
        if cfg.direction == 'negative':
            passed_pt = blup < mu
            passed_cons = (blup + config.se_multiplier_for_voting * se) < mu
        else:
            passed_pt = blup > mu
            passed_cons = (blup - config.se_multiplier_for_voting * se) > mu
        
        # Compact status: P=Point, C=Conservative
        # e.g. "✓✓" (both), "✓✗" (point only), "✗✗" (neither)
        pt_mark = "✓" if passed_pt else "✗"
        cons_mark = "✓" if passed_cons else "✗"
        feat_status.append(f"{feat}:{pt_mark}{cons_mark}")
    
    # Votes
    vp = result.votes_point[fish_idx]
    vc = result.votes_conservative[fish_idx]
    n_feat = len(config.features_to_use)
    
    lines.append(f"{cond_abbr} D:{dist:.1f}{outlier_mark}")
    lines.append(" ".join(feat_status))
    lines.append(f"V:{vp}/{n_feat}P {vc}/{n_feat}C")
    
    return "\n".join(lines)


def plot_feature_space(
    result: ClassificationResult,
    config: AnalysisConfig,
    save_path: Optional[Path] = None
) -> Optional[Figure]:
    """Create feature space scatter plots for all pairs of features."""
    n_features = len(config.features_to_use)
    if n_features < 2:
        return None
        
    feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_pairs = len(feature_pairs)
    
    # Increase height to accommodate legend and info text
    fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 7), facecolor='white')
    if n_pairs == 1:
        axes = [axes]
        
    fig.suptitle('Individual Fish in Feature Space\n(BLUPs with Control Means)', fontsize=5+14, fontweight='bold', y=0.98)
    
    for ax, (idx_x, idx_y) in zip(axes, feature_pairs):
        feat_x = config.features_to_use[idx_x]
        feat_y = config.features_to_use[idx_y]
        cfg_x = config.feature_configs[feat_x]
        cfg_y = config.feature_configs[feat_y]
        
        # Plot points
        for i in range(len(result.fish_ids)):
            # Determine style
            if result.is_learner_conservative[i]:
                color, marker, size, zorder = 'red', '*', 200, 5
            elif result.is_learner_point[i]:
                color, marker, size, zorder = 'orange', 'o', 100, 4
            elif result.is_outlier[i]:
                color, marker, size, zorder = 'goldenrod', 's', 80, 3
            else:
                is_ref = result.conditions[i] == config.cond_types[0]
                color = 'gray' if is_ref else 'lightblue'
                marker, size, zorder = 'o', 40, 2
                
            ax.scatter(result.features[i, idx_x], result.features[i, idx_y],
                      c=color, marker=marker, s=size, zorder=zorder,
                      edgecolors='black', linewidth=0.5)

            # Uncertainty cross (BLUP ± CI) for this fish
            x = float(result.features[i, idx_x])
            y = float(result.features[i, idx_y])
            xerr = float(config.ci_multiplier_for_reporting) * float(result.features_se[i, idx_x])
            yerr = float(config.ci_multiplier_for_reporting) * float(result.features_se[i, idx_y])
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.6,
                alpha=0.35,
                zorder=float(zorder) - 0.1,
            )
            
            # Label learners
            if result.is_learner_point[i]:
                ax.annotate(result.fish_ids[i], 
                           (result.features[i, idx_x], result.features[i, idx_y]),
                           fontsize=5+7, xytext=(3, 3), textcoords='offset points')

        # Add control means
        ax.axvline(result.mu_ctrl[idx_x], color='black', linestyle='--', alpha=0.5)
        ax.axhline(result.mu_ctrl[idx_y], color='black', linestyle='--', alpha=0.5)
        
        # Axis labels with learning direction
        x_arrow = "←" if cfg_x.direction == 'negative' else "→"
        y_arrow = "↓" if cfg_y.direction == 'negative' else "↑"
        
        ax.set_xlabel(f"{cfg_x.name} ({feat_x})\n{x_arrow} Learning direction", fontsize=5+10)
        ax.set_ylabel(f"{cfg_y.name} ({feat_y})\n{y_arrow} Learning direction", fontsize=5+10)
        ax.set_title(f'{cfg_x.name} vs {cfg_y.name}', fontsize=5+11)
        ax.grid(alpha=0.3)
        
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=2.5, label='★ Conservative Learner'),
        Line2D([0], [0], color='orange', lw=2, label='● Point Learner'),
        Line2D([0], [0], color='goldenrod', lw=1, label='○ Outlier (Wrong Direction)'),
        Line2D([0], [0], color='steelblue', lw=1, label='Exp. (Non-learner)'),
        Line2D([0], [0], color='gray', lw=1, label='Ctrl. (Non-Learner)'),
        Line2D([0], [0], color='black', linestyle='--', label='Control Mean')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, 0.02), fontsize=5+10, frameon=True)
    
    plt.tight_layout(rect=(0, 0.12, 1, 0.95))
    
    # if save_path:
    #     fig.savefig(str(save_path), dpi=300, facecolor='white', bbox_inches='tight')
    #     print(f"✓ Saved: {save_path.name}")
        
    return fig


def plot_blup_trajectory_overlay(
    result: ClassificationResult,
    config: AnalysisConfig,
    split_by_condition: bool = True,
    save_path: Optional[Path] = None,
    alpha: float = 0.18,
    lw: float = 1.0,
) -> Figure:
    """Overlay per-fish BLUP trajectories (the purple dashed lines) in a single plot.

    This visualizes exactly what the per-fish secondary-axis "purple" BLUP trajectories show,
    but stacked across all fish to make comparison easier.

    Notes
    -----
    - If both 'acquisition' and 'extinction' are present, the trajectory is cumulative:
        Pre-train: 0
        Late Train + Early Test: acquisition BLUP
        Late Test: acquisition BLUP + extinction BLUP
    - If only one of the features is present, only the relevant two epochs are plotted.
    """

    def _feat_index(feat_code: str) -> Optional[int]:
        # Map feature code to its column index, returning None if absent.
        try:
            return config.features_to_use.index(feat_code)
        except ValueError:
            return None

    j_acq = _feat_index("acquisition")
    j_ext = _feat_index("extinction")

    if j_acq is None and j_ext is None:
        raise ValueError(
            "No BLUP trajectory features available. Expected 'acquisition' and/or 'extinction' in config.features_to_use."
        )

    # Define standardized epoch x-axis (so all fish overlay cleanly).
    if j_acq is not None and j_ext is not None:
        x = np.array([0, 1, 2], dtype=float)
        x_labels = ["Pre-train", "Late Train + Early Test", "Late Test"]
    elif j_acq is not None:
        x = np.array([0, 1], dtype=float)
        x_labels = ["Pre-train", "Late Train + Early Test"]
    else:  # j_ext is not None only
        x = np.array([0, 1], dtype=float)
        x_labels = ["Late Train + Early Test", "Late Test"]

    # Choose subplot layout
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
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        "BLUP Trajectories Overlay (purple)\n(per-fish cumulative epoch effects)",
        fontsize=5 + 14,
        fontweight="bold",
        y=0.98,
    )

    # Collect for y-limits and panel medians
    panel_y = {cond: [] for cond in conds_order}

    for i, fish in enumerate(result.fish_ids):
        cond = str(result.conditions[i])

        # Build y trajectory
        if j_acq is not None and j_ext is not None:
            acq = float(result.features[i, j_acq])
            ext = float(result.features[i, j_ext])
            y = np.array([0.0, acq, acq + ext], dtype=float)
        elif j_acq is not None:
            acq = float(result.features[i, j_acq])
            y = np.array([0.0, acq], dtype=float)
        else:
            ext = float(result.features[i, j_ext])
            y = np.array([0.0, ext], dtype=float)

        if split_by_condition:
            if cond not in conds_order:
                continue
            ax = axes[conds_order.index(cond)]
            panel_y[cond].append(y)
        else:
            ax = axes[0]
            panel_y["all"].append(y)

        # Slight emphasis for learners, but keep the "all purple" look
        if bool(result.is_learner_conservative[i]):
            a_i, lw_i = min(0.55, alpha * 2.5), max(1.8, lw * 1.8)
        elif bool(result.is_learner_point[i]):
            a_i, lw_i = min(0.40, alpha * 2.0), max(1.4, lw * 1.4)
        else:
            a_i, lw_i = alpha, lw

        ax.plot(
            x,
            y,
            "--",
            color="purple",
            alpha=a_i,
            linewidth=lw_i,
        )

    # Decorate panels
    for ax, cond in zip(axes, conds_order):
        ax.axhline(0.0, linestyle=":", color="purple", alpha=0.35, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=5 + 9)
        ax.set_xlabel("Epoch", fontsize=5 + 10)
        ax.grid(alpha=0.25)

        if cond != "all":
            n_cond = int((result.conditions == cond).sum())
            ax.set_title(f"{str(cond).capitalize()} (n={n_cond})", fontsize=5 + 11, fontweight="bold")
        else:
            ax.set_title("All fish", fontsize=5 + 11, fontweight="bold")

        # Median trajectory for readability
        ys = panel_y.get(cond, [])
        if len(ys) > 0:
            y_med = np.median(np.vstack(ys), axis=0)
            ax.plot(
                x,
                y_med,
                "o-",
                color="black",
                linewidth=2.2,
                markersize=6,
                alpha=0.85,
                label="Median",
                zorder=10,
            )
            ax.legend(loc="best", fontsize=5 + 9, framealpha=0.9)

    axes[0].set_ylabel("BLUP (Δ Log Response) — cumulative", fontsize=5 + 10)

    plt.tight_layout(rect=(0, 0.02, 1, 0.95))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved: {save_path.name}")

    return fig


# def plot_blup_caterpillar(
#     result: ClassificationResult,
#     config: AnalysisConfig,
#     feat_code: str,
#     save_path: Optional[Path] = None,
#     sort_by_blup: bool = True,
# ) -> plt.Figure:
#     """Plot per-fish BLUP ± CI (caterpillar plot) for one feature.

#     This is the most literal visualization of what the classifier uses:
#     - BLUP point estimate per fish
#     - Uncertainty via result.features_se and config.ci_multiplier
#     - Control mean reference line (result.mu_ctrl)
#     """
#     if feat_code not in config.features_to_use:
#         raise ValueError(f"Feature '{feat_code}' not in config.features_to_use={config.features_to_use}")

#     j = config.features_to_use.index(feat_code)
#     feat_cfg = config.feature_configs[feat_code]
#     mu = float(result.mu_ctrl[j])

#     blup_all = result.features[:, j].astype(float)
#     se_all = result.features_se[:, j].astype(float)

#     if sort_by_blup:
#         order = np.argsort(blup_all)
#     else:
#         order = np.arange(len(result.fish_ids))

#     fish = result.fish_ids[order]
#     blup = blup_all[order]
#     se = se_all[order]
#     lo = blup - float(config.ci_multiplier) * se
#     hi = blup + float(config.ci_multiplier) * se

#     # Colors by status
#     colors: List[str] = []
#     for idx in order:
#         if bool(result.is_learner_conservative[idx]):
#             colors.append("red")
#         elif bool(result.is_learner_point[idx]):
#             colors.append("orange")
#         elif bool(result.is_outlier[idx]):
#             colors.append("goldenrod")
#         else:
#             is_ref = str(result.conditions[idx]) == str(config.cond_types[0])
#             colors.append("gray" if is_ref else "steelblue")

#     # Figure sizing: keep readable for many fish
#     fig_h = max(4, 0.18 * len(fish))
#     fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="white")

#     y = np.arange(len(fish))
#     for yi, l, h, c in zip(y, lo, hi, colors):
#         ax.plot([l, h], [yi, yi], color=c, alpha=0.75, lw=1.2)

#     ax.scatter(blup, y, c=colors, s=25, edgecolors="black", linewidths=0.4, zorder=3)
#     ax.axvline(mu, color="black", linestyle="--", alpha=0.7, label=f"Control mean = {mu:.3f}")

#     # Directional hint
#     if feat_cfg.direction == "negative":
#         dir_text = "learning: BLUP < control mean"
#     else:
#         dir_text = "learning: BLUP > control mean"

#     ax.set_title(f"{feat_cfg.name} ({feat_code}) — {dir_text}", fontsize=5 + 12, fontweight="bold")
#     ax.set_yticks(y)
#     ax.set_yticklabels([str(f) for f in fish], fontsize=5 + 7)
#     ax.set_xlabel("BLUP (feature scale used in extraction)", fontsize=5 + 10)
#     ax.grid(alpha=0.25)
#     ax.legend(loc="lower right", fontsize=5 + 9)
#     plt.tight_layout()

#     if save_path is not None:
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save_path, dpi=200, bbox_inches="tight")
#         print(f"✓ Saved: {save_path.name}")

#     return fig

def save_combined_plots_and_grid(
    data: pd.DataFrame,
    result: ClassificationResult,
    config: AnalysisConfig,
    output_dir: Path,
    condition: Optional[str] = None
) -> None:
    """Generate individual fish plots and summary grid with overlaid trial data and block trajectories.

    The line plots medians of 5-trial blocks.

    Additionally shows the 3 key 10-trial epochs (as 'o' markers using the median across the two 5-trial blocks):
        - Pre-train (Early + Late Pre-Train)
        - Late Train + Early Test
        - Late Test (Test 5 + Late Test)

    And overlays feature arrows (BLUP + CI) between those epochs:
        - acquisition: Pre-train -> Late Train + Early Test
        - extinction:  Late Train + Early Test -> Late Test
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _feat_index(feat_code: str) -> Optional[int]:
        # Map feature code to its index in the result arrays.
        try:
            return config.features_to_use.index(feat_code)
        except ValueError:
            return None

    def _blup_ci(idx: int, feat_code: str) -> Optional[Tuple[float, float, float]]:
        """Return (blup, ci_low, ci_high) from result arrays, or None if feature missing."""
        # CI is based on the reporting multiplier (typically ~95%).
        j = _feat_index(feat_code)
        if j is None:
            return None
        blup = float(result.features[idx, j])
        se = float(result.features_se[idx, j])
        ci_low = blup - float(config.ci_multiplier_for_reporting) * se
        ci_high = blup + float(config.ci_multiplier_for_reporting) * se
        return blup, ci_low, ci_high

    def _get_epoch_point(fish_block_data: pd.DataFrame, block_names: List[str]) -> Optional[Tuple[float, float]]:
        # Require at least min_trials_per_5trial_block_in_epoch points in EACH constituent 5-trial block.
        if not block_names:
            return None
        sub = fish_block_data[fish_block_data["Block name 5 trials"].isin(block_names)]
        if sub.empty:
            return None
        # Use the median vigor for the epoch to align with the BLUP definition.
        # fish_block_data is already aggregated to one row per 5-trial block, so treat presence as "enough".
        # The strict trial-count requirement is enforced earlier in _filter_fish_by_trials().
        present = set(sub["Block name 5 trials"].astype(str).unique())
        if any(str(b) not in present for b in block_names):
            return None
        x = float(sub["Trial number"].mean())
        y = float(sub["Normalized vigor"].median())
        return x, y

    def _annotate_key_blocks_and_features(ax: Axes, idx: int, fish_block_data: pd.DataFrame, color: str) -> None:
        """Show 10-trial epoch medians and feature (BLUP) arrows between them."""

        # 10-trial epoch medians (each epoch is represented by 2x 5-trial blocks)
        pt_pre = _get_epoch_point(fish_block_data, config.pretrain_blocks_5)
        pt_mid = _get_epoch_point(fish_block_data, config.earlytest_block)  # Late Train + Early Test
        pt_late = _get_epoch_point(fish_block_data, config.late_test_blocks_5)

        # Plot key epoch medians so the BLUP arrows have clear anchors on the behavior trace.
        epoch_points: List[Tuple[str, Optional[Tuple[float, float]]]] = [
            ("Pre-train", pt_pre),
            ("Late Train + Early Test", pt_mid),
            ("Late Test", pt_late),
        ]

        for label, pt in epoch_points:
            if pt is None:
                continue
            x, y = pt
            ax.scatter(
                [x],
                [y],
                marker="o",
                s=85,
                c=color,
                edgecolors="black",
                linewidths=0.9,
                zorder=11,
                alpha=0.98,
            )
            # ax.annotate(
            #     label,
            #     (x, y),
            #     xytext=(0, 10),
            #     textcoords="offset points",
            #     ha="center",
            #     va="bottom",
            #     fontsize=5+7,
            #     fontweight="bold",
            #     color="black",
            #     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="lightgray"),
            #     zorder=12,
            # )

        # # Feature arrows + BLUP annotations
        # # acquisition: Pre-train -> Late Train + Early Test
        # acq_ci = _blup_ci(idx, "acquisition")
        # if pt_pre is not None and pt_mid is not None and acq_ci is not None:
        #     (x0, y0), (x1, y1) = pt_pre, pt_mid
        #     blup, ci_lo, ci_hi = acq_ci
        #     ax.annotate(
        #         "",
        #         xy=(x1, y1),
        #         xytext=(x0, y0),
        #         # arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.95),
        #         zorder=10,
        #     )
        #     xm = 0.5 * (x0 + x1)
        #     ym = max(y0, y1) + 0.04 * (config.y_lim_plot[1] - config.y_lim_plot[0])
        #     ax.text(
        #         xm,
        #         ym,
        #         f"acq BLUP: {blup:.3f}\n[{ci_lo:.3f}, {ci_hi:.3f}]",
        #         ha="center",
        #         va="bottom",
        #         fontsize=5+7,
        #         fontfamily="monospace",
        #         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.78, edgecolor=color),
        #         zorder=12,
        #     )

        # # extinction: Late Train + Early Test -> Late Test
        # ext_ci = _blup_ci(idx, "extinction")
        # if pt_mid is not None and pt_late is not None and ext_ci is not None:
        #     (x0, y0), (x1, y1) = pt_mid, pt_late
        #     blup, ci_lo, ci_hi = ext_ci
        #     ax.annotate(
        #         "",
        #         xy=(x1, y1),
        #         xytext=(x0, y0),
        #         # arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.95),
        #         zorder=10,
        #     )
        #     xm = 0.5 * (x0 + x1)
        #     ym = max(y0, y1) + 0.04 * (config.y_lim_plot[1] - config.y_lim_plot[0])
        #     ax.text(
        #         xm,
        #         ym,
        #         f"ext BLUP: {blup:.3f}\n[{ci_lo:.3f}, {ci_hi:.3f}]",
        #         ha="center",
        #         va="bottom",
        #         fontsize=5+7,
        #         fontfamily="monospace",
        #         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.78, edgecolor=color),
        #         zorder=12,
        #     )

    # Filter result indices if condition is specified
    if condition is not None:
        cond_mask = result.conditions == condition
        fish_indices = np.where(cond_mask)[0]
    else:
        fish_indices = np.arange(len(result.fish_ids))

    df_trials = data.copy()
    if condition is not None:
        df_trials = df_trials[df_trials["Condition"] == condition]

    # Calculate block medians for overlay
    df_blocks = (
        df_trials.groupby(["Fish_ID", "Block name 5 trials"], observed=True)
        .agg({"Normalized vigor": "median", "Trial number": "mean"})
        .reset_index()
        .sort_values("Trial number")
    )

    # NOTE:
    # - The line plots medians of ALL 5-trial blocks.
    # - The 'o' markers for the 10-trial epochs (Pre-train, Late Train + Early Test, Late Test)
    #   are added by _annotate_key_blocks_and_features().

    # 1. Individual Plots
    cond_label = f" ({condition})" if condition else ""
    print(f"Generating combined individual plots{cond_label} in {output_dir.name}...")

    for idx in fish_indices:
        fish = str(result.fish_ids[idx])
        fish_trial_data = df_trials[df_trials["Fish_ID"] == fish]
        fish_block_data = df_blocks[df_blocks["Fish_ID"] == fish]
        cond = str(result.conditions[idx])

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

        # Style
        if result.is_learner_conservative[idx]:
            color, lw = "red", 2.5
            status = "★ CONSERVATIVE LEARNER"
        elif result.is_learner_point[idx]:
            color, lw = "orange", 2
            status = "● POINT LEARNER"
        elif result.is_outlier[idx]:
            color, lw = "goldenrod", 1.5
            status = "○ OUTLIER (Wrong Direction)"
        else:
            color = "gray" if cond == config.cond_types[0] else "steelblue"
            lw = 1
            status = "Non-learner"

        # A. Plot individual trials (Scatter)
        ax.scatter(
            fish_trial_data["Trial number"],
            fish_trial_data["Normalized vigor"],
            c=color,
            alpha=0.3,
            s=15,
            edgecolors="none",
            label="Trials",
        )

        # B. Plot block medians as a line (NO markers)
        ax.plot(
            fish_block_data["Trial number"],
            fish_block_data["Normalized vigor"],
            "-",
            color=color,
            linewidth=lw,
            alpha=0.95,
            label="Block Median",
        )

        # C. Add epoch medians ('o') + feature BLUP arrows
        _annotate_key_blocks_and_features(ax, int(idx), fish_block_data, color)

        ax.axhline(1.0, linestyle=":", color="black", alpha=0.3, label="Baseline")
        ax.set_xlabel("Trial Number", fontsize=5+9)
        ax.set_ylabel("Normalized Vigor", fontsize=5+9)
        ax.set_ylim(config.y_lim_plot)
        ax.set_title(f"{fish} ({cond})\n{status}", fontsize=5+10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5+8, loc="upper right")

        # D. Secondary y-axis: BLUP predictions in log-response scale
        # The BLUPs are epoch-change effects estimated on Log_Response.
        # We plot the predicted trajectory: starting from control-mean anchor,
        # then applying acquisition BLUP, then extinction BLUP.
        pt_pre = _get_epoch_point(fish_block_data, config.pretrain_blocks_5)
        pt_mid = _get_epoch_point(fish_block_data, config.earlytest_block)
        pt_late = _get_epoch_point(fish_block_data, config.late_test_blocks_5)

        acq_ci = _blup_ci(int(idx), "acquisition")
        ext_ci = _blup_ci(int(idx), "extinction")

        if pt_pre is not None and pt_mid is not None and pt_late is not None and acq_ci is not None and ext_ci is not None:
            acq_blup, _, _ = acq_ci
            ext_blup, _, _ = ext_ci

            # X positions from epoch points
            x_pre, _ = pt_pre
            x_mid, _ = pt_mid
            x_late, _ = pt_late

            # BLUP trajectory: anchor at 0, then cumulative changes
            # Pre-train epoch = 0 (reference)
            # After acquisition (mid) = 0 + acq_blup
            # After extinction (late) = 0 + acq_blup + ext_blup
            blup_pre = 0.0
            blup_mid = acq_blup
            blup_late = acq_blup + ext_blup

            ax2 = ax.twinx()
            ax2.plot(
                [x_pre, x_mid, x_late],
                [blup_pre, blup_mid, blup_late],
                "s--",
                color="purple",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor="purple",
                markeredgewidth=1.5,
                linewidth=2,
                alpha=0.85,
                label="BLUP trajectory",
                zorder=15,
            )
            ax2.axhline(0.0, linestyle=":", color="purple", alpha=0.4)
            ax2.set_ylabel("BLUP (Δ Log Response)", fontsize=5+9, color="purple")
            ax2.tick_params(axis="y", labelcolor="purple")
            # Set symmetric y-limits around 0 for clarity
            blup_range = max(abs(blup_pre), abs(blup_mid), abs(blup_late), 0.05) * 1.5
            ax2.set_ylim(-blup_range, blup_range)
            ax2.legend(fontsize=5+7, loc="lower right")

        # Add detailed info box
        info_lines = [f"Condition: {cond}"]
        info_lines.append(f"Distance: {float(result.distances[idx]):.2f} (thresh: {float(result.threshold):.2f})")
        info_lines.append(f"Outlier: {'Yes' if bool(result.is_outlier[idx]) else 'No'}")
        info_lines.append("")

        for feat_idx, feat in enumerate(config.features_to_use):
            feat_cfg = config.feature_configs[feat]
            val = float(result.features[idx, feat_idx])
            se = float(result.features_se[idx, feat_idx])
            mu = float(result.mu_ctrl[feat_idx])

            # Check pass/fail
            if feat_cfg.direction == "negative":
                p_pass = val < mu
                c_pass = (val + config.se_multiplier_for_voting * se) < mu
            else:
                p_pass = val > mu
                c_pass = (val - config.se_multiplier_for_voting * se) > mu

            p_mark = "✓" if p_pass else "✗"
            c_mark = "✓" if c_pass else "✗"
            feat_name = feat_cfg.name.split()[0]
            info_lines.append(f"{feat_name}: {val:.3f} {'<' if feat_cfg.direction == 'negative' else '>'} {mu:.3f}")
            info_lines.append(f"   Point:{p_mark} Cons:{c_mark}")

        info_lines.append("")
        info_lines.append(
            f"Votes: {int(result.votes_point[idx])}/{len(config.features_to_use)}P, "
            f"{int(result.votes_conservative[idx])}/{len(config.features_to_use)}C"
        )

        info_text = "\n".join(info_lines)
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="lightgray")
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=5+6,
            verticalalignment="top",
            bbox=props,
            fontfamily="monospace",
        )

        fig.savefig(output_dir / f"Fish_{fish}_{cond}_combined.png", dpi=150)
        plt.close(fig)

    # 2. Summary Grid
    print(f"Generating combined summary grid{cond_label}...")
    n_fish = len(fish_indices)

    if n_fish == 0:
        print(f"   ⚠️  No fish to plot for condition: {condition}")
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
    fig.suptitle(
        f"All Fish Combined Trajectories{title_suffix}\n(Trials + Block Medians)",
        fontsize=5+16,
        y=0.99,
    )

    for plot_idx, idx in enumerate(fish_indices):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]

        fish = str(result.fish_ids[idx])
        fish_trial_data = df_trials[df_trials["Fish_ID"] == fish]
        fish_block_data = df_blocks[df_blocks["Fish_ID"] == fish]
        cond = str(result.conditions[idx])

        # Style
        if result.is_learner_conservative[idx]:
            color = "red"
            lw = 2
        elif result.is_learner_point[idx]:
            color = "orange"
            lw = 1.5
        elif result.is_outlier[idx]:
            color = "goldenrod"
            lw = 1
        else:
            color = "gray" if cond == config.cond_types[0] else "steelblue"
            lw = 1

        # Plot trials
        ax.scatter(
            fish_trial_data["Trial number"],
            fish_trial_data["Normalized vigor"],
            c=color,
            alpha=0.2,
            s=30,
            edgecolors="none",
        )

        # Plot block medians (line only)
        ax.plot(
            fish_block_data["Trial number"],
            fish_block_data["Normalized vigor"],
            "-",
            color=color,
            linewidth=lw,
            alpha=0.9,
        )

        # Also show epoch medians ('o') + feature BLUP arrows (small) in grid
        _annotate_key_blocks_and_features(ax, idx, fish_block_data, color)

        # Add BLUP trajectory on secondary axis (simplified for grid)
        pt_pre = _get_epoch_point(fish_block_data, config.pretrain_blocks_5)
        pt_mid = _get_epoch_point(fish_block_data, config.earlytest_block)
        pt_late = _get_epoch_point(fish_block_data, config.late_test_blocks_5)
        acq_ci = _blup_ci(idx, "acquisition")
        ext_ci = _blup_ci(idx, "extinction")

        if pt_pre is not None and pt_mid is not None and pt_late is not None and acq_ci is not None and ext_ci is not None:
            acq_blup, _, _ = acq_ci
            ext_blup, _, _ = ext_ci
            x_pre, _ = pt_pre
            x_mid, _ = pt_mid
            x_late, _ = pt_late

            blup_pre = 0.0
            blup_mid = acq_blup
            blup_late = acq_blup + ext_blup

            ax2 = ax.twinx()
            ax2.plot(
                [x_pre, x_mid, x_late],
                [blup_pre, blup_mid, blup_late],
                "s--",
                color="purple",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="purple",
                markeredgewidth=1,
                linewidth=1.5,
                alpha=0.8,
                zorder=15,
            )
            ax2.axhline(0.0, linestyle=":", color="purple", alpha=0.3, linewidth=0.8)
            ax2.set_yticks([])
            blup_range = max(abs(blup_pre), abs(blup_mid), abs(blup_late), 0.05) * 1.5
            ax2.set_ylim(-blup_range, blup_range)

        ax.axhline(1.0, linestyle=":", color="black", alpha=0.3)
        ax.set_ylim(config.y_lim_plot)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title
        ax.set_title(f"{fish}", fontsize=5+9, color=color, fontweight="bold")

        # Info text
        lines: List[str] = []
        cond_abbr = str(cond)[:3].upper()
        dist = float(result.distances[idx])
        outlier_mark = "O" if bool(result.is_outlier[idx]) else ""

        feat_status = []
        for feat_idx, feat in enumerate(config.features_to_use):
            cfg = config.feature_configs[feat]
            blup = float(result.features[idx, feat_idx])
            se = float(result.features_se[idx, feat_idx])
            mu = float(result.mu_ctrl[feat_idx])

            if cfg.direction == "negative":
                passed_pt = blup < mu
                passed_cons = (blup + config.se_multiplier_for_voting * se) < mu
            else:
                passed_pt = blup > mu
                passed_cons = (blup - config.se_multiplier_for_voting * se) > mu

            pt_mark = "✓" if passed_pt else "✗"
            cons_mark = "✓" if passed_cons else "✗"
            name_short = cfg.name.split()[0][:5]
            feat_status.append(f"{name_short}:{pt_mark}{cons_mark}")

        vp = int(result.votes_point[idx])
        vc = int(result.votes_conservative[idx])
        n_feat = len(config.features_to_use)

        lines.append(f"{cond_abbr} D:{dist:.1f}{outlier_mark}")
        lines.append(" ".join(feat_status))
        lines.append(f"V:{vp}/{n_feat}P {vc}/{n_feat}C")

        # Add compact feature BLUP+CI lines (if present)
        acq_ci = _blup_ci(idx, "acquisition")
        if acq_ci is not None:
            blup, ci_lo, ci_hi = acq_ci
            lines.append(f"acq:{blup:.2f}[{ci_lo:.2f},{ci_hi:.2f}]")
        ext_ci = _blup_ci(idx, "extinction")
        if ext_ci is not None:
            blup, ci_lo, ci_hi = ext_ci
            lines.append(f"ext:{blup:.2f}[{ci_lo:.2f},{ci_hi:.2f}]")

        info_text = "\n".join(lines)
        props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray", pad=0.3)
        ax.text(
            0.02,
            0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=5+6,
            verticalalignment="bottom",
            bbox=props,
            fontfamily="monospace",
        )

        # Border
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(
                2.5
                if bool(result.is_learner_conservative[idx])
                else (2 if bool(result.is_learner_point[idx]) else 0.5)
            )

    # Hide empty subplots
    for plot_idx in range(n_fish, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row, col].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="red", lw=2.5, label="★ Conservative Learner"),
        Line2D([0], [0], color="orange", lw=2, label="● Point Learner"),
        Line2D([0], [0], color="goldenrod", lw=1, label="○ Outlier (Wrong Direction)"),
    ]

    if condition is None:
        legend_elements.append(Line2D([0], [0], color="steelblue", lw=1, label="Exp. (Non-learner)"))
        legend_elements.append(Line2D([0], [0], color="gray", lw=1, label="Ctrl. (Non-learner)"))
    elif condition == config.cond_types[0]:
        legend_elements.append(Line2D([0], [0], color="gray", lw=1, label=f"{str(condition).capitalize()} (Non-learner)"))
    else:
        legend_elements.append(
            Line2D([0], [0], color="steelblue", lw=1, label=f"{str(condition).capitalize()} (Non-Learner)")
        )

    info_legend = (
        "Info Box Legend:\n"
        "  Name: Feature pass/fail (✓/✗)\n"
        "  D: Mahalanobis distance, O=Outlier\n"
        "  V: Votes (P=Point, C=Conservative)\n"
        "  acq/ext: BLUP and CI for the 10-trial epoch changes"
    )

    fig.legend(handles=legend_elements, loc="lower left", ncol=3, bbox_to_anchor=(0.02, 0.01), fontsize=5+10, frameon=True)

    fig.text(
        0.75,
        0.01,
        info_legend,
        fontsize=5+9,
        fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.96))

    cond_suffix = f"_{condition}" if condition else ""
    grid_path = output_dir.parent / f"All_Fish_Combined_Summary_Grid{cond_suffix}.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {grid_path.name}")

def save_individual_plots_and_grid(
    data: pd.DataFrame,
    result: ClassificationResult,
    config: AnalysisConfig,
    output_dir: Path,
    condition: Optional[str] = None
) -> None:
    """Wrapper to call the combined plotting function."""
    save_combined_plots_and_grid(data, result, config, output_dir, condition)

def save_individual_trial_plots_and_grid(
    data: pd.DataFrame,
    result: ClassificationResult,
    config: AnalysisConfig,
    output_dir: Path,
    condition: Optional[str] = None
) -> None:
    """Wrapper to call the combined plotting function (for backward compatibility)."""
    save_combined_plots_and_grid(data, result, config, output_dir, condition)

def create_detailed_summary_df(
    result: ClassificationResult,
    blup_dicts: Dict[str, Dict[str, BLUPResult]],
    config: AnalysisConfig
) -> pd.DataFrame:
    """Create comprehensive DataFrame with all metrics per fish."""
    rows = []
    for i, fish in enumerate(result.fish_ids):
        # Start a per-fish summary row with core classification metrics.
        row = {
            'Fish_ID': fish,
            'Condition': result.conditions[i],
            'Mahalanobis_Distance': result.distances[i],
            'Is_Outlier': result.is_outlier[i],
            'Vote_Point': result.votes_point[i],
            'Vote_Conservative': result.votes_conservative[i],
            'Is_Learner_Point': result.is_learner_point[i],
            'Is_Learner_Conservative': result.is_learner_conservative[i]
        }
        
        # Add feature details
        for feat_idx, feat in enumerate(config.features_to_use):
            # From result arrays
            row[f'{feat}_BLUP'] = result.features[i, feat_idx]
            row[f'{feat}_SE'] = result.features_se[i, feat_idx]

            # Reporting CI (typically ~95%) computed from SE
            blup_val = float(row[f'{feat}_BLUP'])
            se_val = float(row[f'{feat}_SE'])
            row[f'{feat}_CI_lower'] = blup_val - float(config.ci_multiplier_for_reporting) * se_val
            row[f'{feat}_CI_upper'] = blup_val + float(config.ci_multiplier_for_reporting) * se_val
            
            # From BLUP dicts (more detail)
            if feat in blup_dicts and fish in blup_dicts[feat]:
                b_res = blup_dicts[feat][fish]
                row[f'{feat}_Fixed'] = b_res.fixed
                row[f'{feat}_Random'] = b_res.random
                
            # Pass/Fail status
            cfg = config.feature_configs[feat]
            mu = result.mu_ctrl[feat_idx]
            
            if cfg.direction == 'negative':
                pass_point = row[f'{feat}_BLUP'] < mu
                pass_cons_vote = (row[f'{feat}_BLUP'] + config.se_multiplier_for_voting * row[f'{feat}_SE']) < mu
                pass_cons_ci95 = row[f'{feat}_CI_upper'] < mu
            else:
                pass_point = row[f'{feat}_BLUP'] > mu
                pass_cons_vote = (row[f'{feat}_BLUP'] - config.se_multiplier_for_voting * row[f'{feat}_SE']) > mu
                pass_cons_ci95 = row[f'{feat}_CI_lower'] > mu
                
            row[f'{feat}_Pass_Point'] = pass_point
            row[f'{feat}_Pass_ConservativeVoting'] = pass_cons_vote
            row[f'{feat}_Pass_ConservativeCI'] = pass_cons_ci95
            
        rows.append(row)
        
    return pd.DataFrame(rows)


def print_fish_details(fish_id: str, summary_df: pd.DataFrame, config: AnalysisConfig, mu_ctrl: np.ndarray):
    """Interactive helper to inspect a specific fish."""
    if fish_id not in summary_df['Fish_ID'].values:
        print(f"Fish {fish_id} not found.")
        return
        
    row = summary_df[summary_df['Fish_ID'] == fish_id].iloc[0]
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {fish_id}")
    print(f"{'='*60}")
    print(f"Condition: {row['Condition']}")
    print(f"Distance:  {row['Mahalanobis_Distance']:.4f}")
    print(f"Outlier:   {'YES' if row['Is_Outlier'] else 'NO'}")
    print(f"Learner:   {'★ CONSERVATIVE' if row['Is_Learner_Conservative'] else ('● POINT' if row['Is_Learner_Point'] else 'NO')}")
    
    print(
        f"\n{'Feature':<15} {'BLUP':>8} {'CI (~95%)':>20} {'Ref':>8} {'Point':>5} {'VoteC':>5} {'CI_C':>5}"
    )
    print("-" * 70)
    
    for i, feat in enumerate(config.features_to_use):
        name = config.feature_configs[feat].name
        blup = row[f'{feat}_BLUP']
        ci = f"[{row[f'{feat}_CI_lower']:.2f}, {row[f'{feat}_CI_upper']:.2f}]"
        ref = mu_ctrl[i]
        pp = "✓" if row[f'{feat}_Pass_Point'] else "✗"
        pv = "✓" if row.get(f'{feat}_Pass_ConservativeVoting', False) else "✗"
        pc = "✓" if row.get(f'{feat}_Pass_ConservativeCI', False) else "✗"
        
        print(f"{name:<15} {blup:>8.4f} {ci:>20} {ref:>8.4f} {pp:>5} {pv:>5} {pc:>5}")
    print(f"{'='*60}\n")


def print_learning_vs_performance_safeguards(
    data: pd.DataFrame,
    result: ClassificationResult,
    config: AnalysisConfig,
) -> None:
    """Quick checks to separate learning signatures from performance/batch confounds.

    Prints:
    - Baseline (Log_Baseline) and pre-train vigor comparisons for learners vs non-learners
    - A simple "batch" breakdown based on Fish_ID prefix (e.g., 20240612_07 -> 20240612)

    This is intentionally light-weight (no decisions made here), just a sanity check.
    """
    if data.empty:
        return

    # Build per-fish table from raw trial-level data
    df = data.copy()
    if 'Fish_ID' not in df.columns or 'Condition' not in df.columns:
        return
    if 'Log_Baseline' not in df.columns or 'Normalized vigor' not in df.columns:
        print("⚠️  Safeguards skipped: required columns missing (Log_Baseline / Normalized vigor)")
        return

    fish_set = set(map(str, result.fish_ids))
    df = df[df['Fish_ID'].astype(str).isin(fish_set)].copy()

    # Learner label (use conservative learner as the primary "learner" definition)
    learner_map = {str(f): bool(result.is_learner_conservative[i]) for i, f in enumerate(result.fish_ids)}

    def _batch_from_fish_id(fid: str) -> str:
        # Extract batch/date prefix so batch effects can be inspected quickly.
        # Common format: YYYYMMDD_##
        token = str(fid).split('_')[0]
        if token.isdigit() and len(token) == 8:
            return token
        return token

    fish_stats = (
        df.groupby(['Fish_ID', 'Condition'], observed=True)
        .agg(
            baseline_median=('Log_Baseline', 'median'),
            vigor_median=('Normalized vigor', 'median'),
            pretrain_vigor_median=(
                'Normalized vigor',
                lambda s: float(np.median(s))  # placeholder, overwritten below
            ),
        )
        .reset_index()
    )

    # Recompute pretrain vigor median using the proper pretrain blocks (if available)
    if 'Block name 5 trials' in df.columns:
        pre = df[df['Block name 5 trials'].isin(config.pretrain_blocks_5)].copy()
        pre_vigor = (
            pre.groupby(['Fish_ID', 'Condition'], observed=True)['Normalized vigor']
            .median()
            .reset_index()
            .rename(columns={'Normalized vigor': 'pretrain_vigor_median'})
        )
        fish_stats = fish_stats.drop(columns=['pretrain_vigor_median']).merge(
            pre_vigor,
            on=['Fish_ID', 'Condition'],
            how='left',
        )

    fish_stats['Fish_ID'] = fish_stats['Fish_ID'].astype(str)
    fish_stats['Is_Learner'] = fish_stats['Fish_ID'].map(learner_map).fillna(False)
    fish_stats['Batch'] = fish_stats['Fish_ID'].map(_batch_from_fish_id)

    print(f"\n{'='*60}")
    print("LEARNING vs PERFORMANCE SAFEGUARDS")
    print(f"{'='*60}")
    print(f"Learner definition used: conservative (votes >= {config.voting_threshold_conservative})")
    print(f"Voting k×SE: k={config.se_multiplier_for_voting}; Reporting CI z={config.ci_multiplier_for_reporting}")

    # Compare baseline/vigor by learner status within each condition
    for cond in sorted(fish_stats['Condition'].unique()):
        sub = fish_stats[fish_stats['Condition'] == cond].copy()
        if sub.empty:
            continue
        n_learn = int(sub['Is_Learner'].sum())
        n_total = len(sub)
        print(f"\nCondition: {cond}  (learners {n_learn}/{n_total})")

        def _summ(col: str) -> str:
            # Summarize learner vs non-learner medians and optionally a Mann-Whitney U test.
            a = sub.loc[sub['Is_Learner'], col].dropna().astype(float)
            b = sub.loc[~sub['Is_Learner'], col].dropna().astype(float)
            if len(a) < 2 or len(b) < 2:
                return "(not enough data for comparison)"
            # Nonparametric: robust in small n
            try:
                stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                return (
                    f"learn median={np.median(a):.3f} vs non={np.median(b):.3f}; "
                    f"U={stat:.1f}, p={p:.4f}"
                )
            except Exception:
                return f"learn median={np.median(a):.3f} vs non={np.median(b):.3f}"

        print(f"  Log_Baseline median:   {_summ('baseline_median')}")
        print(f"  Pretrain vigor median: {_summ('pretrain_vigor_median')}")
        print(f"  Overall vigor median:   {_summ('vigor_median')}")

    # Batch clustering check
    batch_tab = pd.crosstab(fish_stats['Batch'], fish_stats['Is_Learner'])
    if not batch_tab.empty:
        batch_tab = batch_tab.rename(columns={False: 'NonLearner', True: 'Learner'})
        batch_tab['Total'] = batch_tab.sum(axis=1)
        batch_tab['LearnerRate'] = batch_tab.get('Learner', 0) / batch_tab['Total']
        batch_tab = batch_tab.sort_values('LearnerRate', ascending=False)

        print(f"\nBatch clustering (from Fish_ID prefix):")
        with pd.option_context('display.max_rows', 30, 'display.width', 120):
            print(batch_tab[['Learner', 'NonLearner', 'Total', 'LearnerRate']].round({'LearnerRate': 3}))

    print(f"{'='*60}\n")


def print_classification_summary(
    result: ClassificationResult,
    config: AnalysisConfig
) -> None:
    """Print detailed classification results summary."""
    ref_cond = config.cond_types[0]
    unique_conds = np.unique(result.conditions)
    is_ctrl = result.conditions == ref_cond
    n_features = len(config.features_to_use)
    
    print(f"\n{'='*60}")
    print("DIRECTIONAL VOTING")
    print(f"{'='*60}")
    print(f"\nExpected learning signatures (control means):")
    for i, feat in enumerate(config.features_to_use):
        cfg = config.feature_configs[feat]
        direction = "<" if cfg.direction == 'negative' else ">"
        print(f"   Feature {feat} ({cfg.name}): {direction} {result.mu_ctrl[i]:.4f}")
    
    print(f"\nVoting distribution (Point Estimate):")
    for v in range(n_features + 1):
        n = (result.votes_point == v).sum()
        pct = n / len(result.votes_point) * 100
        print(f"   {v}/{n_features} features: {n:3d} fish ({pct:5.1f}%)")
    
    print(f"\nVoting distribution (Conservative):")
    for v in range(n_features + 1):
        n = (result.votes_conservative == v).sum()
        pct = n / len(result.votes_conservative) * 100
        print(f"   {v}/{n_features} features: {n:3d} fish ({pct:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\nMETHOD 1: Point Estimate (≥{config.voting_threshold_point} features)")
    print("─" * 60)
    for cond in unique_conds:
        mask = result.conditions == cond
        n_total = mask.sum()
        n_learners = (mask & result.is_learner_point).sum()
        pct = n_learners / n_total * 100 if n_total > 0 else 0
        label = "(Reference)" if cond == ref_cond else "(Experimental)"
        print(f"   {cond:15s} {label:15s}: {n_learners:3d} / {n_total:3d} ({pct:5.1f}%)")
    
    print(f"\nMETHOD 2: Conservative (≥{config.voting_threshold_conservative} features)")
    print("─" * 60)
    for cond in unique_conds:
        mask = result.conditions == cond
        n_total = mask.sum()
        n_learners = (mask & result.is_learner_conservative).sum()
        pct = n_learners / n_total * 100 if n_total > 0 else 0
        label = "(Reference)" if cond == ref_cond else "(Experimental)"
        print(f"   {cond:15s} {label:15s}: {n_learners:3d} / {n_total:3d} ({pct:5.1f}%)")
    
    # False positive rates
    ctrl_fp_point = (is_ctrl & result.is_learner_point).sum()
    ctrl_fp_cons = (is_ctrl & result.is_learner_conservative).sum()
    n_ctrl = is_ctrl.sum()
    
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVE ANALYSIS")
    print(f"Expected FP rate: ~{100 - config.threshold_percentile:.0f}%")
    print(f"Observed (Point):       {ctrl_fp_point}/{n_ctrl} ({ctrl_fp_point/n_ctrl*100:.1f}%)")
    print(f"Observed (Conservative): {ctrl_fp_cons}/{n_ctrl} ({ctrl_fp_cons/n_ctrl*100:.1f}%)")
    
    print("="*60)











# endregion


# region Main
# ============================================================================
# Main Analysis Pipeline
# ============================================================================

# Function: Entry point for the multivariate LME learner classification pipeline.
def run_multivariate_lme_pipeline(config: AnalysisConfig) -> Tuple[ClassificationResult, Dict[str, Any]]:
    """
    Run the full multivariate learner classification pipeline.

    Returns:
        (result, results_dict): classification object and a flat dict for export.
    """
    # Set up experiment directories used for loading pooled data and saving plots.
    (
        _path_lost_frames,
        _path_summary_exp,
        _path_summary_beh,
        _path_processed_data,
        _path_cropped_exp_with_bout_detection,
        _path_tail_angle_fig_cs,
        _path_tail_angle_fig_us,
        _path_raw_vigor_fig_cs,
        _path_raw_vigor_fig_us,
        _path_scaled_vigor_fig_cs,
        _path_scaled_vigor_fig_us,
        _path_normalized_fig_cs,
        _path_normalized_fig_us,
        path_pooled_vigor_fig,
        _path_analysis_protocols,
        _path_orig_pkl,
        _path_all_fish,
        path_pooled_data,
    ) = file_utils.create_folders(exp_config.path_save)

    config.print_summary()

    print("\n" + "=" * 80)
    print("MULTIVARIATE LEARNER CLASSIFICATION PIPELINE")
    print("=" * 80)
    print(f"Experiment: {EXPERIMENT}")
    print(f"Conditions: {config.cond_types}")

    # Step 1: Load and prepare data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 80)

    # Load pooled per-trial data and enforce block/trial filters.
    raw_data = load_data(config, path_pooled_data)
    data = prepare_data(raw_data, config)
    print(f"Data prepared: {len(data)} trials, {data['Fish_ID'].nunique()} fish")

    # Step 2: Feature extraction
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE EXTRACTION (BLUPS WITH UNCERTAINTY)")
    print("=" * 80)

    # Extract BLUP-based features for the selected learning epochs.
    blup_dicts: Dict[str, Dict[str, BLUPResult]] = {}
    if 'acquisition' in config.features_to_use:
        blup_dicts['acquisition'] = extract_change_feature(
            data, config, number_trials=5, name_blocks=config.pretrain_to_train_end_earlytest_blocks
        )
    if 'extinction' in config.features_to_use:
        blup_dicts['extinction'] = extract_change_feature(
            data, config, number_trials=5, name_blocks=config.train_end_earlytest_to_late_test_blocks
        )

    # Find common fish with all selected features.
    common_fish_sets = [set(blup_dicts[feat].keys()) for feat in config.features_to_use]
    common_fish = sorted(list(set.intersection(*common_fish_sets)))

    if len(common_fish) < 10:
        raise ValueError(f"Only {len(common_fish)} fish have all features. Need at least 10.")

    print(f"{len(common_fish)} fish with complete feature sets")

    # Build the feature matrix used for multivariate classification.
    X = np.array([[blup_dicts[feat][f].blup for feat in config.features_to_use] for f in common_fish])
    X_se = np.array([[blup_dicts[feat][f].se for feat in config.features_to_use] for f in common_fish])

    cond_map = data.drop_duplicates('Fish_ID').set_index('Fish_ID')['Condition'].to_dict()
    conds = np.array([cond_map.get(f, 'unknown') for f in common_fish])

    ref_cond = config.cond_types[0]
    is_ctrl = conds == ref_cond

    print(f"   Reference: {ref_cond} (n={is_ctrl.sum()})")
    print(f"   Experimental: (n={(~is_ctrl).sum()})")

    # Step 3: Statistical diagnostics
    print("\n" + "=" * 80)
    print("STEP 3: STATISTICAL DIAGNOSTICS")
    print("=" * 80)

    feature_names = [config.feature_configs[feat].name for feat in config.features_to_use]

    corr_matrix = None
    pca = None
    X_pca = None
    if RUN_DIAGNOSTICS or RUN_PLOT_DIAGNOSTICS:
        _ = check_multivariate_normality(X, feature_names)
        corr_matrix, _ = analyze_feature_correlation(X, feature_names)
        pca, X_pca = perform_pca(X, feature_names)

    # Step 4: Mahalanobis distance calculation
    print("\n" + "=" * 80)
    print("STEP 4: DISTANCE CALCULATION")
    print("=" * 80)

    # Compute robust Mahalanobis distances against the control-group covariance.
    X_ctrl = X[is_ctrl]
    distances, _, _, _ = get_robust_mahalanobis(X, X_ctrl)

    # Step 5: Bootstrap threshold
    ctrl_distances = distances[is_ctrl]
    thresh_median, thresh_ci = bootstrap_threshold(
        ctrl_distances,
        percentile=config.threshold_percentile,
        n_boot=config.n_bootstrap,
        seed=config.random_seed,
    )

    # Step 6: Classification
    print("\n" + "=" * 80)
    print("STEP 5: CLASSIFICATION")
    print("=" * 80)

    # Classify learners using distance outliers + directional voting.
    mu_ctrl = np.mean(X_ctrl, axis=0)
    is_outlier, votes_point, votes_conservative, is_learner_point, is_learner_conservative = classify_learners(
        X, X_se, distances, thresh_median, mu_ctrl, config
    )

    # Create result object
    result = ClassificationResult(
        fish_ids=np.array(common_fish),
        conditions=conds,
        features=X,
        features_se=X_se,
        feature_names=feature_names,
        distances=distances,
        votes_point=votes_point,
        votes_conservative=votes_conservative,
        is_outlier=is_outlier,
        is_learner_point=is_learner_point,
        is_learner_conservative=is_learner_conservative,
        threshold=thresh_median,
        threshold_ci=thresh_ci,
        mu_ctrl=mu_ctrl,
    )

    # Print summary tables
    print_classification_summary(result, config)
    print_learning_vs_performance_safeguards(data, result, config)

    # Step 7: Visualization
    print("\n" + "=" * 80)
    print("STEP 6: VISUALIZATION")
    print("=" * 80)

    if RUN_PLOT_DIAGNOSTICS and corr_matrix is not None and pca is not None and X_pca is not None:
        diag_path = path_pooled_vigor_fig / "Multivariate_Diagnostics.png"
        plot_diagnostics(
            X,
            feature_names,
            corr_matrix,
            pca,
            X_pca,
            distances,
            ctrl_distances,
            conds,
            ref_cond,
            thresh_median,
            thresh_ci,
            save_path=diag_path,
        )

    if RUN_PLOT_TRAJECTORIES:
        traj_path = path_pooled_vigor_fig / "Learner_Classification_Trajectories.png"
        plot_behavioral_trajectories(data, result, config, save_path=traj_path)

    if RUN_PLOT_FEATURE_SPACE and len(config.features_to_use) > 1:
        feat_plot_path = path_pooled_vigor_fig / "Feature_Space_Scatter.png"
        plot_feature_space(result, config, save_path=feat_plot_path)

    if RUN_PLOT_INDIVIDUALS:
        combined_indiv_dir = path_pooled_vigor_fig / "Individual_Fish_Plots_Combined"
        for cond in config.cond_types:
            cond_dir = combined_indiv_dir / cond
            save_combined_plots_and_grid(data, result, config, cond_dir, condition=cond)

    if RUN_PLOT_BLUP_OVERLAY:
        blup_overlay_path = path_pooled_vigor_fig / "BLUP_Trajectories_Overlay.png"
        plot_blup_trajectory_overlay(result, config, split_by_condition=True, save_path=blup_overlay_path)

    # Step 8: Optional exports
    summary_df = None
    if RUN_EXPORT_DETAILED_SUMMARY:
        summary_df = create_detailed_summary_df(result, blup_dicts, config)
        detailed_path = path_pooled_data / f"Fish_Detailed_Summary_{config.csus}.csv"
        summary_df.to_csv(detailed_path, index=False)
        print(f"Saved detailed summary: {detailed_path.name}")

    results_dict = {
        'Fish_ID': result.fish_ids,
        'Condition': result.conditions,
        'Mahalanobis_Distance': result.distances,
        'Vote_Point': result.votes_point,
        'Vote_Conservative': result.votes_conservative,
        'Is_Outlier': result.is_outlier,
        'Is_Learner_Point': result.is_learner_point,
        'Is_Learner_Conservative': result.is_learner_conservative,
    }

    # Add features dynamically
    for i, feat in enumerate(config.features_to_use):
        results_dict[f'Feature_{feat}'] = result.features[:, i]
        results_dict[f'Feature_{feat}_SE'] = result.features_se[:, i]

    if RUN_EXPORT_RESULTS:
        results_path = path_pooled_data / f"Fish_Learner_Classification_{config.csus}.csv"
        pd.DataFrame(results_dict).to_csv(results_path, index=False)
        print(f"Saved classification results: {results_path.name}")

    return result, results_dict


if __name__ == "__main__":
    if RUN_PIPELINE:
        try:
            run_multivariate_lme_pipeline(analysis_cfg)
        except Exception as e:
            print(f"Error in Multivariate LME pipeline: {e}")
# endregion
