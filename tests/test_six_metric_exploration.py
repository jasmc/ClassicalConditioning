from __future__ import annotations

import numpy as np
import pandas as pd

from classical_conditioning.analysis.six_metric_exploration import (
    SixMetricExplorationConfig,
    build_trial_log_suppression_scores,
    summarize_conditions,
    summarize_fish_learning_effects,
)
def _row(
    *, fish: str, condition: str, block: str, baseline: float, response: float,
    baseline_conditional: float = 2.0, conditional: float = 1.0,
) -> dict[str, object]:
    return {
        "recording_id": fish, "fish_id": fish, "condition_id": condition,
        "alignment": "CS", "trial_id": f"{fish}:{block}", "trial_number": 1,
        "block_10_name": block, "metric_id": "legacy_distal_angular_speed",
        "baseline_valid_sample_count": 5, "response_valid_sample_count": 5,
        "baseline_total_activity": baseline, "response_total_activity": response,
        "baseline_conditional_intensity": baseline_conditional,
        "conditional_intensity": conditional,
    }


def test_log_scores_keep_total_and_bout_semantics_separate() -> None:
    frame = pd.DataFrame([
        _row(fish="f1", condition="delay", block="Pre-train", baseline=4, response=2),
        _row(fish="f2", condition="control", block="Pre-train", baseline=0, response=2),
        _row(fish="f3", condition="delay", block="Pre-train", baseline=4, response=2, baseline_conditional=np.nan),
    ])
    scores = build_trial_log_suppression_scores(frame)
    total = scores.loc[(scores.fish_id == "f1") & (scores.score_type == "total_activity_log_suppression")].iloc[0]
    assert total.included
    assert total.log_suppression == np.log(2)
    zero = scores.loc[(scores.fish_id == "f2") & (scores.score_type == "total_activity_log_suppression")].iloc[0]
    assert not zero.included and zero.exclusion_reason == "nonpositive_baseline"
    bout = scores.loc[(scores.fish_id == "f3") & (scores.score_type == "bout_conditional_log_suppression")].iloc[0]
    assert not bout.included and bout.exclusion_reason == "nonfinite_window_value"


def test_fish_level_condition_contrast_is_delay_minus_control() -> None:
    rows = []
    for fish, condition, late_response in (("d1", "delay", 1.0), ("d2", "delay", 1.0), ("c1", "control", 2.0), ("c2", "control", 2.0)):
        rows.extend([
            _row(fish=fish, condition=condition, block="Pre-train", baseline=2, response=2),
            _row(fish=fish, condition=condition, block="Train 1", baseline=2, response=2),
            _row(fish=fish, condition=condition, block="Train 5", baseline=2, response=late_response),
            _row(fish=fish, condition=condition, block="Test 1", baseline=2, response=late_response),
            _row(fish=fish, condition=condition, block="Test 2", baseline=2, response=late_response),
            _row(fish=fish, condition=condition, block="Test 3", baseline=2, response=late_response),
        ])
    config = SixMetricExplorationConfig(n_bootstrap=99)
    effects = summarize_fish_learning_effects(build_trial_log_suppression_scores(pd.DataFrame(rows)), config)
    summary = summarize_conditions(effects, config)
    value = summary.loc[
        (summary.score_type == "total_activity_log_suppression")
        & (summary.summary_type == "delay_minus_control"), "estimate"
    ].iloc[0]
    assert value > 0
