# B: statistical review and processing, 2026-10-09

All 24 existing B tests reproduce from the saved fish/block values. Tests are two-sided; Holm24 covers D/E together for B. Fish are the statistical units. Paired comparisons use fish with both blocks; control counts therefore vary by contrast.

## Main findings

- Delay PT to ET: within-Delay Wilcoxon Holm24 p=0.00002745; direct Delay/control comparison of paired changes p=0.001630. Median-change difference is -0.10176 log units, pointwise bootstrap 95% interval [-0.15086,-0.06140]. ET Delay/control Mann–Whitney p=0.0001818.
- Delay ET to LT: within-Delay p=0.00009159; direct Delay/control change p=0.0007403. Difference of median changes +0.08523, pointwise interval [0.04312,0.13977]. This supports an increase after ET. PT to LT change is not detectably different between groups (p=1 after Holm24); absence of significance does not establish equivalence.
- Trace PT to ET: within-Trace p=0.01357; direct Trace/control change p=0.8283. ET Trace/control p=0.3287. Thus a within-Trace decrease is supported, but the corrected tests do not establish a Trace-specific reduction relative to controls.

## Reduction relative to the zero reference

The existing stars compare blocks or conditions. They do not test B against zero. A separate exploratory audit adds 12 two-sided zero-reference Wilcoxon tests (2 panels x 2 conditions x 3 blocks), Holm12, plus 12 exact sign tests as a symmetry-free sensitivity analysis. The existing figure stars are unchanged. A conservative Holm36 sensitivity combines the 24 original tests and 12 zero-reference Wilcoxon tests.

- Delay ET: median B=-0.07521, exp(B)=0.92755, corresponding to about 7.2% lower typical bout-frame vigor than the trial baselines. 27/29 fish are below zero. Zero-reference Wilcoxon Holm12 p=0.000007555; sign-test Holm12 p=0.00001948. The zero-reference Wilcoxon remains significant with Holm36 (p=0.000022665).
- Trace ET: median B=-0.01103, approximately 1.1% lower on the multiplicative scale. Zero-reference Wilcoxon Holm12 p=0.5298; sign-test Holm12 p=1. This does not establish a reduction below baseline at the population level.
- No other group/block is significant against zero after Holm12. These post hoc tests are exploratory; family definitions must be settled before a confirmatory manuscript claim.

## Criticism and limits

- Wilcoxon signed-rank tests rely on symmetry of differences for a location interpretation. Delay paired changes are skewed (PT to ET skew=-1.19; ET to LT=1.49). Sign-test sensitivities still support both changes (Holm12 p=0.0001828 and 0.001141), so the qualitative conclusion is robust, but signed-rank p-values should not be described as assumption-free median tests.
- Mann–Whitney tests distributions/ranks, not a pure median difference without additional shape assumptions. The reported median-change differences and bootstrap intervals are descriptive effect estimates; their intervals are pointwise and unadjusted, whereas the hypothesis tests use Holm correction.
- No-bout exclusions can be informative. Selected trial exclusions: Delay control 33/420, Delay 7/435; Trace control 5/285, Trace 7/600. Neither B nor A measures time spent still or bout frequency. Conclusions apply to observed bout intensity.
- D requires at least 3 eligible trials per fish/block; E requires only 1. E may therefore have less reliable block estimates. B requires only one positive valid bout frame in each trial window; these sampling rules warrant a prespecified coverage sensitivity analysis before final inference.
- All bout frames are pooled within each window. Longer bouts contribute more frames: B describes typical bout-frame intensity, not the median of equally weighted individual bouts.
- No day/batch adjustment is used. Whether fish are independent at the experimental-unit level must be checked against the design; the current rank tests do not model shared day/tank effects.
- B was chosen for interpretation, not independently prespecified before examining A/B results. Alternative-outcome selection and these added zero tests should be disclosed as exploratory.

## Processing from raw vigor to B

- Start with corrected measured-time vigor, `legacy_distal_angular_speed_rad_per_ms`: absolute wrapped change in the summed local tail angles divided by the measured frame interval (rad/ms).
- Match each frame exactly to the authenticated shared movement state. Exclude invalid frames and no-bout frames with NaN.
- Retain only finite, strictly positive bout vigor for logarithms. Zero-valued bout samples are excluded; no pseudocount or zero filling is used.
- Align to each trial's CS onset. Baseline is [-15,0) s; response is [0,9) s for Delay or [0,13) s for Trace. Endpoints are left-inclusive, right-exclusive.
- Apply natural logarithms to eligible frame values; calculate median ln(vigor) separately in each window. No additional historical smoothing or downsampling is added.
- Trial B = median ln(response bout-frame vigor) minus median ln(baseline bout-frame vigor). Both windows require positive eligible samples and must pass the current trial eligibility rule. Empty windows yield NaN and the trial is excluded.
- Within each fish, take the median of eligible trial B values for PT10–14, ET65–69, and LT90–94. Require at least 3 eligible trials for D, 1 for E.
- Plot each eligible fish value, then the equal-fish group median and IQR. Zero means unchanged log-scale typical intensity; negative means reduced intensity. The main whiskers are IQR, not confidence intervals.

Sources for test interpretation: [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), [SciPy Mann–Whitney](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html). Numerical evidence: `log_tests.csv`, `log_paired_diagnostics.csv`, `direct_change_effects_bootstrap.csv`, `B_zero_reference_audit.csv`, `B_combined36_sensitivity.csv`, `B_trial_exclusion_audit.csv`.
