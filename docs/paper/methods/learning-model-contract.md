# Implemented learning-model contract

This is the mathematical contract of the supported exploratory learning-onset route, not an approved common estimator for the paper. The [scientific decisions](../decisions.md) and exact [Figure 2 selected scopes](../figures/figure-02/README.md) govern accepted outcomes and inference. In particular, the frozen G phase-aware LogMedian LMM is a separate selected analysis; these formulas do not replace it.

Commands, parameters, diagnostics and output schemas are in the [operational manual](../../running-analysis/learning-models.md). Remaining inferential design belongs to [Plan 06](../../../Plans/06_ANALYSIS_AND_STATISTICS.md).

## Statistical specification

### Model input

For the selected metric and outcome:

```text
log_response = log(response + offset)
log_baseline = log(pre-event baseline + offset)
cr_score     = log_baseline - log_response
trial_scaled = (trial_number - pooled trial mean) / pooled trial SD
fish_key     = experiment_id + "::" + fish_id
```

For `conditional-intensity`, the offset is zero and eligibility requires a
finite positive intensity. `cr_score` and response/baseline trajectories are
descriptive and support the fish-level robustness analysis. The primary LMEs
model `log_response` and adjust for `log_baseline` directly.

### Block model

```text
log_response ~ log_baseline
             + condition
             + block
             + condition:block

groups = fish_key
random effects = configured random-effects formula
```

The analysis publishes:

- all fixed-effect coefficients;
- one joint Wald test over every condition-by-block interaction term; and
- planned control-minus-test contrasts for every block, expressed as change
  from the configured pre-training block.

The block learning contrast is:

```text
D(block) = [control - test in block]
           - [control - test in pretraining]
```

Positive values mean stronger test-condition suppression relative to control.
Intervals use the fixed-effect covariance and normal approximation. Holm
correction is applied across the planned block family. A block is marked
supported only when its lower confidence bound exceeds `delta_min` and its
Holm-adjusted p-value is below `1 - confidence_level`.

### Primary longitudinal model

```text
log_response ~ log_baseline
             + condition
             + cubic_spline(trial_scaled, df=spline_df)
             + condition:cubic_spline(trial_scaled, df=spline_df)

groups = fish_key
random effects = configured random-effects formula
```

At every scheduled trial, the program constructs a named design-matrix
contrast rather than guessing coefficient names:

```text
D(t) = [control - test at trial t]
       - mean([control - test] over pretraining trials)
```

The baseline covariate is evaluated at the model-input mean. Positive `D(t)`
has the same stronger-test-learning direction as the block contrast.

### Fish-bootstrap simultaneous band

Each bootstrap replicate resamples complete fish trajectories with replacement
within condition and refits the longitudinal model. The replicate statistic is
the maximum absolute deviation between its complete trial-contrast curve and
the original curve. The configured confidence quantile of those maxima is the
single critical distance added to and subtracted from every original trial
estimate. The result is a bootstrap simultaneous band over the scheduled trial
family. Its nominal coverage depends on the model and bootstrap assumptions;
it is not a guarantee that the true trajectory lies inside the band 95% of
the time in this particular experiment.

### Onset rule

Onset is the earliest run of `persistence_trials` consecutive scheduled trials
for which:

```text
simultaneous lower confidence bound > delta_min
```

The comparison is strict. A lower bound equal to the threshold does not count.
If no run exists, the result records `threshold_never_exceeded` or
`threshold_not_persistent`; it does not substitute the lowest p-value.

Bootstrap onset quantiles are conditional on bootstrap replicates that
localized an onset. Always report the localized and nonlocalized proportions
with those quantiles.

### Categorical-trial sensitivity

```text
log_response ~ log_baseline + condition * C(trial_number)
```

This checks whether spline smoothing creates the apparent curve. It produces
pointwise contrasts only and is not the primary onset estimator.

### Fish-level robustness analysis

For each fish, the median pre-training `cr_score` is subtracted from the median
`cr_score` in the late blocks (`late - pretraining`). The analysis compares mean
fish changes as test minus control using a two-sided condition-label permutation
test and a within-condition fish bootstrap interval. It is a deliberately
simple condition-preserving robustness analysis, not a replacement for the
LMEs.
