# Learning-onset LME pipeline and parameter reference

This is the operational reference for the condition-aware learning-onset
mixed-effects analysis. It documents how a user-selected activity metric flows
into the analysis, every public parameter, the statistical estimands, model
fits, resampling, diagnostic gates, outputs, and the implementation choices
that are intentionally fixed rather than command-line parameters.

The route is implemented but does not make a result paper-authoritative. A
paper claim additionally requires an approved cohort, outcome, metric, effect
threshold, and diagnostic review.

## Non-negotiable metric-selection contract

The LME analysis depends on the metric named by the user with `--metric`.
`metric_id` is not a label added after fitting.

The dependency is enforced twice:

1. eligibility retains only cohort-outcome rows whose `metric_id` equals the
   requested metric; and
2. model-input construction again selects only that metric before fitting,
   resampling, leave-one-fish-out analysis, or figure-data construction.

Consequently, changing `--metric` can change response values, eligibility,
sample coverage, coefficients, contrasts, onset, robustness results,
diagnostics, and all three figure panels. If the requested metric is absent,
the command fails with `No <alignment> outcomes for metric '<metric>'` instead
of silently substituting another metric.

Use a distinct `--analysis-id` for every metric and parameter specification.
The summary records the complete configuration and its SHA-256 hash, but a
metric-specific analysis ID makes comparisons and accidental reuse easier to
recognize.

Do not confuse these two settings:

| Setting | Stage | Meaning |
| --- | --- | --- |
| `--metric-recipe` | `build-cohort-trial-outcomes` | Selects the authenticated upstream trial-outcome artifact family, such as `tail-candidate-corrected`. One family can contain several metric IDs. |
| `--metric` | `learning-onset` | Selects exactly one `metric_id` from the cohort trial-outcome table for inference. |

## End-to-end pipeline

```text
immutable raw acquisition triplets
  -> lossless intake
  -> corrected per-frame preprocessing
  -> candidate activity metrics
  -> shared movement/bout state
  -> per-fish, per-trial outcomes for every metric
  -> reviewed and frozen cohort manifest
  -> authenticated cohort trial-outcome table
  -> metric/outcome/alignment-specific eligibility
  -> metric-specific LME model input
       |-> block LME and planned block contrasts
       |-> spline longitudinal LME and trial contrasts
       |-> optional categorical-trial sensitivity LME
       |-> fish bootstrap simultaneous band and onset distribution
       |-> fish-level permutation/bootstrap robustness analysis
       |-> optimizer/random-structure sensitivity fits
       `-> leave-one-fish-out refits and residual/coverage diagnostics
  -> authenticated tables and QC summary
  -> learning-onset figure when required diagnostic gates pass
  -> residual-diagnostics figure for model review
```

### 1. Produce per-recording trial outcomes

The corrected candidate route must already have produced authenticated
per-recording trial outcomes. These tables contain one outcome row for each
fish, alignment, scheduled trial, and metric. The LME never reads raw tracking
data directly.

### 2. Freeze the biological population

```powershell
uv run classical-conditioning freeze-cohort `
  --project-dir "<PROJECT>" `
  --input "<REVIEWED-COHORT.csv>" `
  --cohort-id <COHORT-ID> `
  --policy-id <POLICY-ID>
```

The cohort manifest defines fish membership before inference. Analysis
eligibility may exclude unusable fish-trial observations, but it must not
silently redefine cohort membership.

### 3. Build the cohort trial-outcome boundary

```powershell
uv run classical-conditioning build-cohort-trial-outcomes `
  --project-dir "<PROJECT>" `
  --cohort-id <COHORT-ID> `
  --metric-recipe tail-candidate-corrected
```

This stage authenticates every included fish's trial-outcome artifacts, applies
the frozen cohort, concatenates retained rows, and publishes sample-flow and
lineage records. The result normally contains several metric IDs.

### 4. Run one metric-specific LME analysis

Minimal example:

```powershell
uv run classical-conditioning learning-onset `
  --project-dir "<PROJECT>" `
  --cohort-id <COHORT-ID> `
  --analysis-id <METRIC-SPECIFIC-ANALYSIS-ID> `
  --metric tail_length_weighted_angular_l1 `
  --outcome total-activity `
  --test-condition delay `
  --delta-min 0
```

Inspect all current public options with:

```powershell
uv run classical-conditioning learning-onset --help
```

### 5. Review diagnostics before rendering

Read these first:

- `Quality checks/Analyses/<analysis-id>/learning-onset_summary.json`;
- `learning-model-diagnostics.parquet`;
- `learning-model-coverage.parquet`;
- `learning-model-residuals.parquet`;
- `learning-model-sensitivity.parquet`; and
- `leave-one-fish-out.parquet`.

### 6. Render figures from authenticated analysis artifacts

```powershell
uv run classical-conditioning figure-learning-diagnostics `
  --project-dir "<PROJECT>" `
  --analysis-id <ANALYSIS-ID> `
  --mode static

uv run classical-conditioning figure-learning-onset `
  --project-dir "<PROJECT>" `
  --analysis-id <ANALYSIS-ID> `
  --mode static
```

The current renderer requires every publication-required diagnostic gate to
pass in both `static` and `publication` modes. The residual-diagnostics figure
can still be rendered independently to investigate a failed analysis. The
September 2026 allDelay PNG shows an exploratory-only banner that the current
source does not draw. From the screenshot and current code, it is a previously
generated image; rerendering that failed analysis with the current source
would be rejected because its leave-one-fish-out gate failed. Figure mode
controls export format, not the scientific acceptance rule.

The one-command `run-allDelay-full-windows.ps1` launcher accepts `-MetricId`
and passes it to `learning-onset`. Its default preserves the original
`tail_length_weighted_angular_l1` technical run. Supply a different
`-LearningAnalysisId` for a different metric. If the existing analysis ID has
a saved metric that differs from `-MetricId`, the launcher stops rather than
reusing it. Use the direct `learning-onset` command to change the other model
parameters in the tables below.

## Complete public parameter reference

Every field of `LearningOnsetConfig` is exposed by the `learning-onset` CLI.
All resolved values are saved under `config` in
`learning-onset_summary.json` and contribute to `config_sha256`.

### Identity, data selection, and comparison

| CLI option | Default | Constraint | What it changes |
| --- | ---: | --- | --- |
| `--project-dir` | required | Existing derived-data project | Input/output project root. |
| `--cohort-id` | required | Frozen authenticated cohort | Biological population and cohort hash. |
| `--analysis-id` | required | Portable identifier | Output namespace; use a new ID for every specification. |
| `--metric` | required | Must occur in cohort outcomes | Selects the metric rows entering the complete analysis. |
| `--outcome` | `total-activity` | `total-activity` or `conditional-intensity` | Chooses response and baseline columns derived from the selected metric. |
| `--alignment` | `CS` | `CS` or `US` | Selects scheduled trials aligned to the named event. |
| `--control-condition` | `control` | Must differ from test | Fixed-effect reference and control side of contrasts. |
| `--test-condition` | required | Must differ from control | Condition interpreted as the learned/test group. |
| `--pretraining-block` | `Pre-train` | Named configured block | Reference block for block change, trial change, fish effects, and onset. |
| `--late-block` | `Test 2`, `Test 3` | Repeatable named block | Blocks pooled for fish-level robustness; the final named block is the primary leave-one-fish-out block. |

### Outcome eligibility and transformation

| CLI option | Default | Constraint | What it changes |
| --- | ---: | --- | --- |
| `--min-baseline-samples` | `1` | Positive integer | Minimum valid samples required in the baseline window. |
| `--min-response-samples` | `1` | Positive integer | Minimum valid samples required in the response window. |
| `--activity-offset` | `1e-6` | Finite and positive | Offset in `log(activity + offset)` for `total-activity`; ignored for conditional intensity. |

Eligibility also rejects nonfinite responses/baselines, nonpositive ratio
baselines, missing block assignments, and undefined or nonpositive conditional
intensity. Every decision is retained in `analysis-eligibility.parquet`.

### Effect definition and uncertainty

| CLI option | Default | Constraint | What it changes |
| --- | ---: | --- | --- |
| `--delta-min` | required | Finite scientific threshold | Lower-bound threshold for a supported learning contrast. Zero requires a positive contrast; a positive value requires a larger effect. The code also accepts negative values, which weaken the criterion and need explicit scientific justification. |
| `--persistence-trials` | `3` | Positive integer | Consecutive scheduled trials required to localize onset. |
| `--confidence-level` | `0.95` | Greater than 0.5 and less than 1 | Coefficient/contrast intervals, simultaneous-band quantile, onset quantiles, and fish-bootstrap interval. |

Choose `delta_min`, persistence, confidence level, primary metric, outcome, and
cohort before examining the inferential result.

### Primary mixed-model specification

| CLI option | Default | Constraint | What it changes |
| --- | ---: | --- | --- |
| `--spline-df` | `5` | Integer at least 3 | Flexibility of the condition-by-trial spline in the primary longitudinal model. |
| `--random-effects-formula` | `1 + trial_scaled` | Valid statsmodels formula | Requested within-fish random structure for block, longitudinal, bootstrap, sensitivity, and leave-one-out fits. |
| `--disable-random-intercept-fallback` | off | Flag | Prevents retrying a failed requested random structure with `1`. |
| `--optimizer` | `lbfgs` | Supported statsmodels optimizer name | Optimizer for primary and resampled mixed models. |

A fallback changes model structure and is recorded through
`requested_random_effects`, `used_random_effects`, and `fallback_used`; it is
not hidden.

### Sensitivity analyses

| CLI option | Default | Effect |
| --- | ---: | --- |
| `--skip-categorical-sensitivity` | off | Skips the optional condition-by-categorical-trial model. |
| `--sensitivity-optimizer` | `powell` | Alternate optimizer for diagnostic refits; use `none` to disable. |
| `--skip-random-intercept-sensitivity` | off | Skips diagnostic refits using random intercept only. |

Sensitivity fits are recorded separately and do not replace the prespecified
primary result.

### Resampling and reproducibility

| CLI option | Default | Constraint | What it changes |
| --- | ---: | --- | --- |
| `--bootstrap` | `499` | Nonnegative | Requested whole-fish longitudinal bootstrap refits. Zero disables the simultaneous band and therefore prevents accepted onset inference. |
| `--min-successful-bootstrap` | `100` | Positive; no greater than requested count when `--bootstrap` is nonzero | Minimum successful refits required to accept the simultaneous band. |
| `--min-bootstrap-success-fraction` | `0.8` | Greater than 0 and at most 1 | Additional successful-refit fraction required for the band. |
| `--permutations` | `9999` | At least 99 | Condition-label permutations for the fish-level robustness p-value. |
| `--seed` | `20260917` | Integer | Reproducible fish resampling, permutations, and robustness bootstrap. |

The fish-level robustness confidence interval uses at least 999 bootstrap
draws: `max(--bootstrap, 999)`. This is distinct from the longitudinal
simultaneous-band bootstrap acceptance count.

### Output replacement

| CLI option | Default | Effect |
| --- | ---: | --- |
| `--overwrite` | off | Replaces the complete existing analysis artifact set for the same analysis ID. Without it, any existing output causes a failure. |

Do not use `--overwrite` merely to compare metrics or model choices. Preserve
the prior analysis and choose a new analysis ID.

## Relation to the archived block-comparison analysis

The legacy route was **not only** a set of block-to-block tests. Its archived
implementation first reduced trials to fish-level block medians and ran
Mann–Whitney comparisons between conditions within each selected block,
Mann–Whitney comparisons between successive blocks within a condition, and
paired Wilcoxon comparisons across successive blocks for fish with complete
block data. It also fit a global condition-by-block mixed model, separate
mixed models for condition-specific mean and slope differences within each
block, and separate per-trial mixed models. These were distinct result families;
none was a simultaneous-band onset rule.

The current route retains a planned block comparison but estimates each block's
**change in the test-versus-control difference from pre-training** in one
cohort-wide mixed model. Its primary trial-scale question instead comes from
one longitudinal spline mixed model and a whole-fish bootstrap simultaneous
band across the scheduled trials. This avoids treating a sequence of
independently significant blocks or trials as a localized learning onset.
The old and new effect sizes and p-values are therefore not interchangeable;
the legacy route is a reproducibility benchmark, not a second implementation
of the current estimand.

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

## Required diagnostics and publication gates

The primary block and longitudinal fits pass only when they converge, have a
full-rank fixed-effect design, and their random-effect covariance does not meet
the implementation's singularity rule. The accepted simultaneous band must
also meet both successful-bootstrap thresholds.

Leave-one-fish-out analysis refits both primary models after omitting every
fish. It fails the required gate when a refit fails, a localized onset
disappears, onset moves by more than the persistence window, or the primary
late-block effect changes sign. The primary late block is the last
`--late-block` supplied.

Fish-level robustness must also be estimable. Optimizer, random-intercept, and
categorical-trial sensitivities are published for review but are not themselves
required gates.

Residual-versus-fitted and normal-quantile plots are review evidence; the code
does not automatically convert visual residual quality into an accepted paper
claim.

## Output contract

Processed analysis tables are written below:

```text
Processed data/Analyses/<analysis-id>/
```

They include model input, eligibility, block global test, block coefficients,
block contrasts, longitudinal coefficients, trial contrasts, adjusted
trajectories, categorical sensitivity contrasts, onset summary, fish effects,
fish robustness, bootstrap trial curves, bootstrap onset results, and the
fish/group trajectory tables used by the figure.

The exact analysis-table filenames are `learning-model-input.parquet`,
`analysis-eligibility.parquet`, `block-global-test.parquet`,
`block-model-coefficients.parquet`, `block-contrasts.parquet`,
`longitudinal-model-coefficients.parquet`, `trial-contrasts.parquet`,
`adjusted-trajectories.parquet`, `categorical-trial-contrasts.parquet`,
`learning-onset.parquet`, `fish-learning-effects.parquet`,
`fish-robustness.parquet`, `bootstrap-trial-contrasts.parquet`,
`bootstrap-onsets.parquet`, `figure-fish-trajectories.parquet`, and
`figure-group-trajectories.parquet`.

Quality evidence is written below:

```text
Quality checks/Analyses/<analysis-id>/
```

It includes the complete JSON summary, diagnostics, model sensitivity,
residuals, coverage, and leave-one-fish-out results. A completion marker below
`Metadata/` authenticates every table hash together with the cohort and
configuration hashes. Figure rendering reloads and verifies this complete
artifact set before plotting.
The QC table filenames are `learning-model-diagnostics.parquet`,
`learning-model-sensitivity.parquet`, `learning-model-residuals.parquet`,
`learning-model-coverage.parquet`, and `leave-one-fish-out.parquet`.

## Fixed implementation decisions

The following choices are visible in the saved formulas, diagnostics, or this
reference but are not public parameters. Changing one is a method/code change
that requires tests, documentation, and a new recipe or analysis specification:

| Decision | Current value |
| --- | --- |
| Mixed-model engine | `statsmodels.formula.api.mixedlm` |
| Estimation | Maximum likelihood (`reml=False`) |
| Longitudinal spline degree | Cubic (`degree=3`) |
| Spline intercept | Excluded from spline basis; model formula supplies intercept |
| Grouping unit | `experiment_id::fish_id` |
| Block order | Pre-train; Train 1–5; Test 1–3 |
| Block multiplicity correction | Holm step-down |
| Global block test | Joint Wald chi-square over all condition-by-block interactions |
| Random-covariance singularity thresholds | minimum eigenvalue `<= 1e-8` or `<= 1e-6 ×` maximum eigenvalue |
| Trial simultaneous statistic | Maximum absolute unstudentized contrast deviation |
| Bootstrap sampling | Complete fish, within condition, with replacement |
| Onset comparison | Strict lower bound `> delta_min` |
| Primary influence block | Last configured late block |
| Figure group trajectory | Equal-fish median with fish IQR |
| Artifact compression | Lossless Zstandard Parquet |

These fixed choices should not be altered after viewing a preferred result.
If a scientifically justified alternative is needed, preserve the original
analysis, implement the alternative explicitly, and compare the two as named
specifications.

## Review checklist

Before interpreting or comparing an LME result, confirm:

1. the frozen cohort ID/hash and inclusion policy are the intended ones;
2. `config.metric_id` is the chosen primary metric and every model-input row has
   that same metric ID;
3. outcome, alignment, conditions, pre-training block, late blocks, and effect
   direction match the question;
4. `delta_min`, persistence, confidence level, spline flexibility, and random
   structure were selected before reviewing results;
5. coverage and eligibility exclusions are acceptable and condition-balanced;
6. required diagnostics all pass before using publication mode;
7. block, longitudinal, categorical-sensitivity, and fish-level results are
   described as different estimands rather than interchangeable tests; and
8. a nonlocalized onset is reported as nonlocalized, even when some block or
   pointwise contrasts are positive.
