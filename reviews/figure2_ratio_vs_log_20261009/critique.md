# Figure 2 D/E: ratio versus legacy log-median outcome

Review date: 2026-10-09. These are alternative exploratory outcomes. The selected ratio panels and their selection/freeze records are preserved. No scientific choice or new freeze is made here.

## Recommendation

**Keep A, the current mean-activity ratio, as the primary D/E outcome for population conditioning/suppression; use B, the legacy log-median outcome, as a secondary analysis of movement intensity.** This recommendation follows the intended behavioral question, not which version produces more stars. A incorporates valid stationary frames, so it can reflect less movement as well as weaker movement. B conditions on positive detected movement and therefore asks what happens to typical vigor when a fish moves. It cannot by itself measure suppression through stillness. If typical within-bout intensity is the desired primary question, B is the relevant outcome and the claim/caption should be changed accordingly.

## What was held fixed and what changed

- Paper metric: `legacy_distal_angular_speed`, corrected frame column `legacy_distal_angular_speed_rad_per_ms`.
- Authenticated source cohorts: 29 Delay + 28 controls for D; 40 Trace + 19 controls for E. Identical source cohorts do not imply identical eligible fish/block counts after the outcome changes.
- CS alignment; measured-time half-open baseline [-15,0) s; response [0,9) s for D and [0,13) s for E; PT 10–14, ET 65–69, LT 90–94; minimum eligible trials D=3, E=1.
- A is the median of each fish/block's eligible trial ratios of **mean raw response activity / mean raw baseline activity**, including valid stationary frames. No-change reference=1.
- B computes the **median natural log of positive valid moving-frame vigor in response minus the corresponding median in baseline**, then the median of eligible trial differences within fish/block. No-change reference=0. Logs have no additive pseudocount. A multiplicative unit conversion cancels in the log difference.
- B reproduces the historical **outcome operation** on current corrected data. It does not restore historical rolling-median smoothing, row downsampling, interpolation, detector, inclusive endpoint overlap, or optional >90% NaN exclusion. It is not an exact reproduction of old generated figures. The historical files and SHA-256 records are in `../figure2_def_legacy_log_audit_20261009/`.
- B also requires current trial-ratio eligibility, plus at least one finite positive valid moving sample in each window. A trial with no eligible movement in either window has an undefined intensity comparison, not zero/no change. This extra eligibility is recorded per trial. The old conditional-vigor recipe's missingness cannot silently be interpreted as successful suppression or no response.

## Criticism of A: current ratio

1. It answers the total mean-activity question clearly and retains stillness, but combines movement frequency/duty and intensity. A ratio below one does not identify which component changed. Independent movement-probability and conditional-intensity outcomes are the remedy for a mechanistic claim.
2. Mean window values are sensitive to unusually large valid movements. Fish/block medians make the trial aggregation robust, but do not repair an influential frame within each window mean.
3. Dividing by a small/noisy baseline can amplify noise. The scale is asymmetric around one and paired ratio differences can be skewed. Positivity alone is not a precision or coverage guarantee. Inspect baseline level, measured coverage and influential fish before final inference; do not choose an arbitrary pseudocount to recover stars.
4. Current D and E minimum-trial rules differ. E can be represented by a single trial; per-block sample sizes and contributing-trial counts must accompany the plot. Ratios do not eliminate missingness bias or provide an extinction/onset estimate.

## Criticism of B: legacy log median

1. The zero-centered scale expresses multiplicative changes symmetrically, and the median within each window is less sensitive to extreme positive values. It is directly useful for typical movement-intensity modulation.
2. Dropping stationary/nonpositive samples changes the biological target. A fish can suppress the number of bouts without changing its retained moving-frame vigor, producing strong total-activity suppression and little B change. Median log differences are not simply log(A); changing the estimator and movement mask can change sample membership and results.
3. No movement makes intensity undefined. Extra exclusions are potentially informative: fish with the strongest suppression may disappear from B. Even retained B blocks may use a different set of trials than A. The comparison table records both outcomes by fish/block.
4. It weights moving samples, not bouts equally; long bouts contribute more samples. Short bouts, detector thresholds and the positive-value mask affect it. The median's robustness does not authenticate the detector or upstream metric.
5. Logs are not automatically required by rank tests and do not automatically validate signed-rank assumptions. The earlier legacy LME's additional log(x+1) is not copied into B's rank tests; applying it to already logged window values would be a separate, problematic transformation.

## Statistics and adequacy

Both versions have all 24 comparisons: six paired Wilcoxon within-condition block comparisons, three independent Mann–Whitney block comparisons, and three independent Mann–Whitney comparisons of matched within-fish changes per panel. Pairing is explicitly by fish. Data are not clipped before testing. Exact Wilcoxon is used only without zeros/absolute-difference ties; otherwise the recorded approximation is used. Recording day is not loaded into tests and no LMM is fitted. All 24 original ratio raw and Holm-adjusted p-values reproduced to tolerance 1e-12.

Holm24 is applied separately per version for consistency with the existing review family. This does not control a family selected across two alternative outcomes. These versions are neither independent replications nor an opportunity to select the smaller p-value. Paper primary/secondary estimands and families remain a scientific decision.

Wilcoxon concerns symmetric within-fish differences under its null; Mann–Whitney compares independent distributions/ranks, not automatically medians. Pairwise change distributions include asymmetry (see skewness and paired sign-test sensitivities in `*_paired_diagnostics.csv`). These descriptive sample diagnostics do not establish or categorically refute assumptions. The separate exact sign-test/Holm12 sensitivity asks about direction of paired change, discards its magnitude, and does not replace the displayed main tests or their Holm24 family.

Six direct condition-by-change contrasts across D/E are shown on companion plots for each version. Comparing a conditioned group's stars with control's ns does not test different changes. There are 2 direct-change results below .05 across the two versions under their respective Holm24 families; inspect the saved results rather than equating ns with equivalence or no conditioning.

`direct_change_effects_bootstrap.csv` adds rank-biserial effects and differences of median changes with 5,000 fish resamples, seed10. Each resampled fish brings its paired change intact. Percentile95 intervals are pointwise descriptive intervals, not simultaneous intervals; they need not agree with multiplicity-adjusted stars. Differences of medians are descriptive location effects and are not the Mann–Whitney test's universal estimand. The resampling preserves observed missingness and cannot remove informative-missingness bias.

`sensitivity_log_existing_block_ratio_tests.csv` isolates a log transform of the *existing fish/block ratios*. Monotone logging leaves same-block Mann–Whitney raw ranks/p-values unchanged; paired absolute-difference ranks and ranks of between-group changes can change. Therefore one must recompute paired/change tests, even for a pure transformation. This auxiliary sensitivity is not B's historical estimator.

### Significant main comparisons

- ratio, D, within, delay, PT versus ET: raw p=2.78093e-05, Holm24 p=0.000667423.
- ratio, D, within, delay, ET versus LT: raw p=0.00166633, Holm24 p=0.0366593.
- ratio, E, within, trace, PT versus ET: raw p=0.000140394, Holm24 p=0.00322907.
- log, D, between, delay vs control, ET: raw p=8.26503e-06, Holm24 p=0.000181831.
- log, D, within, delay, PT versus ET: raw p=1.14366e-06, Holm24 p=2.74479e-05.
- log, D, change, delay vs control, PT versus ET: raw p=8.1477e-05, Holm24 p=0.00162954.
- log, D, within, delay, ET versus LT: raw p=3.98234e-06, Holm24 p=9.15937e-05.
- log, D, change, delay vs control, ET versus LT: raw p=3.5254e-05, Holm24 p=0.000740333.
- log, E, within, trace, PT versus ET: raw p=0.000714402, Holm24 p=0.0135736.

### Trial sample flow

```
panel condition  trials  current_eligible  log_eligible  extra_missing  missing_baseline_positive_samples  missing_response_positive_samples
    D   control     420               420           387             33                                 25                                 27
    D     delay     435               435           428              7                                  5                                  4
    E   control     285               280           280              0                                  5                                  5
    E     trace     600               600           593              7                                  3                                  4
```

### Eligible fish/block summaries

```
version panel condition block  n_fish       q25    median       q75
  ratio     D   control    PT      28  0.976148  1.014804  1.056212
  ratio     D   control    ET      28  0.924243  0.992699  1.026195
  ratio     D   control    LT      28  0.985658  1.018453  1.072013
  ratio     D     delay    PT      29  1.000934  1.023052  1.116108
  ratio     D     delay    ET      29  0.860415  0.938422  0.965196
  ratio     D     delay    LT      29  0.970900  0.995676  1.070196
  ratio     E   control    PT      19  0.996912  1.012081  1.033477
  ratio     E   control    ET      19  1.003030  1.015869  1.026120
  ratio     E   control    LT      18  0.994899  1.010880  1.029948
  ratio     E     trace    PT      40  0.994918  1.014347  1.045078
  ratio     E     trace    ET      40  0.961414  0.990652  1.017931
  ratio     E     trace    LT      40  0.977744  1.005067  1.020104
    log     D   control    PT      28 -0.023177  0.004656  0.031129
    log     D   control    ET      24 -0.022525  0.004162  0.029867
    log     D   control    LT      26 -0.012300  0.008347  0.025433
    log     D     delay    PT      29 -0.038669  0.006830  0.051908
    log     D     delay    ET      29 -0.149123 -0.075213 -0.023592
    log     D     delay    LT      29 -0.031496  0.010915  0.039370
    log     E   control    PT      19 -0.013767  0.018732  0.050433
    log     E   control    ET      19 -0.007564  0.012451  0.026120
    log     E   control    LT      18 -0.017444  0.017330  0.032767
    log     E     trace    PT      40 -0.023497 -0.010709  0.066695
    log     E     trace    ET      40 -0.057399 -0.011033  0.013681
    log     E     trace    LT      40 -0.026444 -0.007104  0.017083
```

## Artifacts and verification

- `comparison.html`: A/B side-by-side D/E, all-test toggle, direct-change companions and complete results.
- `*_fish_blocks.parquet`, `*_log_trials.parquet`, `paired_version_comparison.csv`: plotted values, window sample counts and eligibility.
- `all_statistics.csv`, `*_tests.csv`, `*_fish_changes.csv`, `direct_change_effects_bootstrap.csv`: all tests, pairing, effects and uncertainty.
- `inputs.json`, `method.json`, figure sidecars and `verification.json`: authenticated provenance, recipe and checks. F remains placeholder/inconclusive; no missing inputs or tests are invented.
- Calculation checks cover exact window exclusion at response end, positive-only logarithms, unit invariance and undefined no-movement response. Geometry checks and PNG readback check legibility, annotation/headings, zero/one references and lack of point clipping.

References: [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), [SciPy Mann–Whitney](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html), and [Gelman and Stern, 2006](https://doi.org/10.1198/000313006X152649).
