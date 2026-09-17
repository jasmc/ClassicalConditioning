# Learning-onset analysis

The supported learning-onset route starts from one frozen cohort and produces
condition-aware block and trial analyses. It is implemented but remains
non-paper-authoritative until the paper cohort, metric, outcome, effect
threshold, and diagnostic configuration are approved and the analysis is run.

## Commands

Build the one-time cohort population table:

```powershell
classical-conditioning build-cohort-trial-outcomes `
  --project-dir "<SAVE-DIR>" `
  --cohort-id <COHORT-ID>
```

Run the analysis. The test condition and minimum meaningful effect are
deliberately required rather than silently assumed:

```powershell
classical-conditioning learning-onset `
  --project-dir "<SAVE-DIR>" `
  --cohort-id <COHORT-ID> `
  --analysis-id <ANALYSIS-ID> `
  --metric tail_length_weighted_angular_l1 `
  --outcome total-activity `
  --test-condition delay `
  --delta-min <APPROVED-EFFECT>
```

Render the accepted analysis only after its block, longitudinal, and
leave-one-fish-out diagnostics pass:

```powershell
classical-conditioning figure-learning-onset `
  --project-dir "<SAVE-DIR>" `
  --analysis-id <ANALYSIS-ID> `
  --mode publication
```

Render residual diagnostics independently of the publication gate:

```powershell
classical-conditioning figure-learning-diagnostics `
  --project-dir "<SAVE-DIR>" `
  --analysis-id <ANALYSIS-ID> `
  --mode static
```

## Statistical outputs

The block model is:

```text
log_response ~ log_baseline + condition * block
```

It publishes a joint condition-by-block Wald test and planned
control-minus-test changes relative to the pre-training contrast. Holm
correction is applied across the named block family.

The trial model is:

```text
log_response ~ log_baseline + condition * spline(scaled_trial)
```

It publishes the control-minus-test change from the average pre-training
contrast at every scheduled trial. Fish are resampled within condition and the
maximum absolute contrast deviation supplies a simultaneous confidence band.
The band and onset inference fail unless at least 100 refits and 80% of all
requested bootstrap refits succeed; both thresholds are configurable and saved.
An optional categorical-trial mixed model is also fitted as a smoothing
sensitivity analysis. It is recorded separately and does not replace the
prespecified spline model.

Onset is the first run of consecutive scheduled trials whose lower simultaneous
bound exceeds the configured `delta_min`. The default persistence length is
three trials. No qualifying run produces an explicit non-localized result.

The robustness analysis calculates one late-minus-pre-training suppression
change per fish and compares test versus control using condition-label
permutation and a fish bootstrap interval.

## Figure

The final figure has three panels:

1. fish and condition response/baseline trajectories across CS trials;
2. the adjusted test-versus-control learning contrast and simultaneous band;
3. the planned block contrasts and confidence intervals.

An onset marker appears only when onset is localized and the required block,
longitudinal, simultaneous-band, fish-level robustness, and leave-one-fish-out
diagnostic gates pass. Rank-deficient fits fail these gates.

## Saved analysis tables

- `learning-model-input.parquet`
- `analysis-eligibility.parquet`
- `block-global-test.parquet`
- `block-model-coefficients.parquet`
- `block-contrasts.parquet`
- `longitudinal-model-coefficients.parquet`
- `trial-contrasts.parquet`
- `adjusted-trajectories.parquet`
- `categorical-trial-contrasts.parquet`
- `learning-onset.parquet`
- `fish-learning-effects.parquet`
- `fish-robustness.parquet`
- `bootstrap-trial-contrasts.parquet`
- `bootstrap-onsets.parquet`
- `learning-model-diagnostics.parquet`
- `learning-model-sensitivity.parquet`
- `learning-model-residuals.parquet`
- `learning-model-coverage.parquet`
- `leave-one-fish-out.parquet`
- `figure-fish-trajectories.parquet`
- `figure-group-trajectories.parquet`

Fish identity is scoped by experiment and fish ID throughout fitting,
resampling, counting, influence analysis, and plotting. These tables are the
panel data and inference record. The completion marker
authenticates all of them together with the cohort hash and analysis
configuration hash.
