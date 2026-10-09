# Standing author decision: ignore no-bout vigor everywhere

Approved scientific sampling rule, 2026-10-09, from Joaquim in this conversation:

> whenever there is no bout (meaning, no movement, meaning vigor is nan) those values must be ignored all the time and everywhere

All active analytical vigor arrays are masked by authenticated shared movement state before averaging, medians, normalization or logarithms. Valid non-bout frames are NaN, never zero-filled. Missing/invalid frames remain NaN. Empty bout windows are undefined and excluded; they cannot become a zero response or a finite normalized observation. Logarithms additionally require positive finite bout vigor. Finite zeros within detected bouts may enter arithmetic means, but cannot enter logs.

Physical acquisition coverage, bout counts and movement probability remain separate outcomes. Their denominators describe physical observations, not the set of samples entering vigor. Raw tracking and derivative artifacts remain immutable source measurements; masking occurs in analytical copies. Historical/frozen figure exports and statistical records remain preserved as superseded evidence and are not active choices under this rule.

Compatibility column names `baseline_total_activity`, `response_total_activity` and `Total activity mean` now refer to conditional bout-frame means in active analytical readers. Authenticated older outcome/profile tables are adapted using their stored moving-only conditional-intensity fields. Old two-layer scaled values cannot be repaired from bin means and are rejected for active rendering until rebuilt from masked frames. Old learner score caches require rebuilding before active review.

The active D/E comparison offers two estimators of bout intensity: arithmetic-mean ratio and legacy-style log-median difference. This instruction approves their shared sampling rule, not a final A/B selection, new panel freeze, or new whole-figure freeze. F has no authenticated inputs.

[Updated panel review](PANEL_REVIEW_COMMENTS.md) · [active comparison](../reviews/figure2_bout_only_ratio_vs_log_20261009/comparison.html).
