"""Standing author policy: analytical vigor exists only on valid bout frames.

Raw tracking/derivative artifacts remain source measurements. Analytical copies
mask no-bout periods with NaN before any mean, median, scale or log operation.
An empty bout window remains undefined, rather than becoming zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VIGOR_SAMPLE_POLICY = "valid-bout-frames-only-v1"


def mask_bout_vigor(values, valid, moving) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid, moving = np.asarray(valid, dtype=bool), np.asarray(moving, dtype=bool)
    if values.ndim != 1 or values.shape != valid.shape or values.shape != moving.shape:
        raise ValueError("Vigor and shared movement masks must be aligned one-dimensional arrays")
    return np.where(valid & moving & np.isfinite(values), values, np.nan)


def bout_only_trial_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Use authenticated moving-only fields when reading older outcome artifacts.

The legacy total_activity column names remain compatibility aliases, but their
active values now mean conditional bout vigor. Never edit the source artifact.
"""
    required = {"baseline_conditional_intensity", "conditional_intensity"}
    missing = required.difference(outcomes.columns)
    if missing:
        if "vigor_sample_policy" in outcomes and outcomes.vigor_sample_policy.eq(VIGOR_SAMPLE_POLICY).all():
            return outcomes.copy()
        raise ValueError("Cannot apply bout-only vigor policy: missing conditional bout fields " + str(sorted(missing)))
    result = outcomes.copy()
    result["baseline_total_activity"] = result["baseline_conditional_intensity"]
    result["response_total_activity"] = result["conditional_intensity"]
    result["vigor_sample_policy"] = VIGOR_SAMPLE_POLICY
    return result


def bout_only_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Adapt older authenticated bin tables without reading or copying frames."""
    if "Conditional intensity mean" not in profiles:
        raise ValueError("Bout-only profiles require Conditional intensity mean")
    result = profiles.copy()
    already_bout_only = "Vigor sample policy" in profiles and profiles["Vigor sample policy"].eq(VIGOR_SAMPLE_POLICY).all()
    if not already_bout_only and "Scaled total activity" in result:
        # Its two frame/bin scaling layers cannot be recovered from bin means.
        result["Scaled total activity"] = np.nan
        result["Scaled vigor requires rebuild"] = True
    result["Total activity mean"] = result["Conditional intensity mean"]
    result["Vigor sample policy"] = VIGOR_SAMPLE_POLICY
    return result
