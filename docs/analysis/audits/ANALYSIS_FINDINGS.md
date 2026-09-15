# Major Analysis Issues

## Scope

This document audits the current Python implementation against the analysis described in the associated Learning paper. It is based on inspection of the code and manuscript methods, not on a complete reanalysis of the underlying experimental data.

The audit therefore identifies issues that could affect results and conclusions, but it does not quantify their actual effect sizes or determine whether the biological conclusions remain significant after correction.

## Executive summary

The most serious concern is that several foundational calculations in the executable pipeline do not match the manuscript:

- Vigor is calculated from one cumulative tail-angle signal rather than as the sum of absolute angular speeds across all tail segments.
- Spatial smoothing and a configured bout-strength threshold are not implemented.
- Fish exclusions are inconsistently applied across analysis stages.
- Scaled vigor uses a different baseline interval and transformation from the method described in the paper.
- Immobility is treated as missing data, even though reduced movement is the biological outcome of interest.
- Inclusion criteria condition on movement during the response interval and can therefore select animals based on the outcome.

These discrepancies affect preprocessing, cohort composition, behavioral metrics, and downstream statistics. The raw data should be reprocessed after the canonical definitions are resolved and tested. Existing figures should not be treated as final until that validation is complete.

## Severity levels

- **Critical:** Can directly alter the analyzed signal, included cohort, or central biological result.
- **High:** Can materially alter effect estimates, uncertainty, or statistical inference.
- **Medium:** Important for robustness, interpretation, or reproducibility, but less likely to reverse the primary result alone.

## Critical issues

### 1. The implemented vigor calculation does not match the manuscript

The manuscript defines vigor as the **sum of the absolute angular speeds across all tail segments**.

The code instead:

1. Cumulatively sums segment angles across the tail.
2. Selects one endpoint/cumulative tail-angle column.
3. Takes the absolute frame-to-frame derivative of that single signal.

Relevant implementation:

- `1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py`, tail-angle processing and vigor calculation
- `analysis_utils.py`, `calculate_vigor_fast`

These quantities are not equivalent:

```text
abs(sum(delta_angle_i)) != sum(abs(delta_angle_i))
```

Opposing angular changes in different tail segments can cancel in the implemented signal. The current calculation may therefore underestimate complex tail movements compared with the metric described in the manuscript.

**Potential impact:** Every bout, vigor measurement, heatmap, normalized-vigor value, inclusion decision, and statistical result can be affected.

**Required action:**

1. Decide which vigor definition is scientifically intended.
2. Implement it explicitly.
3. Add synthetic tests with known segment-level angular changes.
4. Reprocess all raw recordings.
5. Update the manuscript to match the tested implementation.

---

### 2. Core filtering and bout-detection steps are incomplete or inconsistent with the methods

Several important discrepancies appear in `analysis_utils.py`:

- Spatial smoothing across tail segments is commented out and is not applied.
- The second bout-strength threshold is explicitly left unimplemented.
- The `thr2` parameter is passed to `find_beg_and_end_of_bouts` but is never used.
- The rolling maximum/minimum window operation does not clearly match the description in the manuscript.
- The manuscript describes a 10 ms temporal smoothing window, approximately 7 frames at 700 FPS, while the code uses 10 frames.

The function includes the comment:

```python
# Filter by max angle (thr2) logic would go here...
```

This means that the configured secondary threshold does not contribute to bout detection.

**Potential impact:** Changes which frames count as movement, which trials have valid behavior, which fish pass inclusion criteria, and all downstream vigor summaries.

**Required action:**

1. Define the intended spatial and temporal filters mathematically.
2. Clarify the intended rolling-minimum and rolling-maximum windows.
3. Implement the secondary threshold or remove it from the documented method.
4. Test bout boundaries against manually annotated traces.
5. Record sensitivity to reasonable threshold choices.

---

### 3. Fish exclusions are not consistently applied

`3_FishGrouping.py` loads `Discarded_fish_IDs.txt`, but the IDs appear to be used only when generating included/discarded heatmap grids. The fish files themselves are still processed and concatenated into condition-level datasets.

Downstream defaults are inconsistent:

- `3_FishGrouping.py`: `APPLY_FISH_DISCARD = True`, but pooled rows are not filtered during grouping.
- `4_ScaledVigorPlotting.py`: `APPLY_FISH_DISCARD = True`.
- `5_NormalizedVigorPlotting.py`: `APPLY_FISH_DISCARD = False`.

As configured, scaled-vigor plots and normalized-vigor statistics can use different fish cohorts. Normalized-vigor results may also include animals reported as excluded in the manuscript.

**Potential impact:** Sample sizes, medians, confidence intervals, effect sizes, and statistical tests may differ across manuscript panels without an explicit scientific reason.

**Required action:**

1. Filter excluded fish before condition-level files are created, or keep all data but apply one immutable cohort manifest downstream.
2. Generate a table listing every fish and its inclusion status and reason.
3. Store an exclusion-list hash with each generated dataset.
4. Assert expected fish counts before every analysis.
5. Ensure every manuscript panel identifies the cohort it uses.

---

### 4. Scaled vigor is not calculated as described

The manuscript describes scaled vigor as a proportional change from the median of the immediately preceding 15-second baseline:

```text
(vigor at time t - baseline median) / baseline median
```

The grouping implementation instead:

- Selects samples with trial time less than `-15 s`.
- Calculates the 10th and 90th percentiles of those samples.
- Applies percentile min-max scaling:

```text
(vigor at time t - P10) / (P90 - P10)
```

For a trial spanning approximately `-45 s` to `+45 s`, the condition `time < -15 s` selects approximately `-45 s` to `-15 s`, not the documented immediate `-15 s` to `0 s` baseline.

Some plotting functions later subtract the median from the immediate pre-stimulus interval, but this occurs after the earlier percentile transformation.

**Potential impact:** Scaled-vigor heatmaps and time courses do not represent the metric described in the paper. Differences in early versus immediate baseline activity can change apparent response profiles.

**Required action:**

1. Select one canonical scaling formula.
2. Use an explicit bounded baseline mask such as `-15 <= time < 0`.
3. Avoid multiple undocumented normalization stages.
4. Recreate scaled-vigor outputs from canonical per-fish data.
5. Label visualization-only transformations separately from analytical variables.

## High-priority issues

### 5. Immobility is treated as missing data

Before grouping, vigor outside detected bouts is converted to `NaN`. Normalized vigor then computes means using only non-NaN samples.

Consequences:

- A fish that stops moving does not receive a vigor of zero; it contributes missing data.
- A trial containing one short bout can contribute alongside a trial with sustained movement.
- The metric captures bout intensity conditional on movement, not total motor output.
- Learning-related changes in bout probability and movement duration are omitted.
- Missingness is related to the outcome being studied and is therefore informative.

There is optional code to invalidate trials with excessive missingness, but it is disabled:

```python
MAX_NAN_FRAC_PER_WINDOW = 0.90
APPLY_MAX_NAN_FRAC_PER_WINDOW = False
```

Even if enabled at 90%, a window can be accepted when only 10% of its samples contain movement.

**Potential impact:** Reported motor suppression may represent weaker bouts among moving fish while missing changes in movement probability or duration.

**Required action:** Analyze at least three complementary outcomes:

1. Probability of any movement
2. Fraction of time moving or bout count
3. Bout vigor conditional on movement

A hurdle or two-part model is more appropriate than treating immobility as ordinary missingness.

---

### 6. Inclusion criteria condition on the behavioral outcome

The exclusion procedure requires movement in the conditioned-response interval in at least three trials in selected blocks.

The hypothesized learned response is suppression of movement. A strong learner that becomes immobile in the response interval can therefore fail the criterion because its vigor cannot be calculated.

This creates post-treatment selection:

```text
condition -> learning -> response-window movement -> inclusion
```

Applying the same rule to control and conditioned groups does not remove the problem when the groups are expected to differ in response-window movement.

**Potential impact:** Strongly suppressing fish may be preferentially excluded, biasing results toward animals that remain active. The direction and magnitude of the resulting bias are uncertain.

**Required action:**

1. Separate health/tracking criteria from behavioral-outcome criteria.
2. Do not require response-window movement for the primary intention-to-analyze cohort.
3. Use a two-part outcome model that can retain immobile trials.
4. Report sensitivity analyses with and without the current engagement criterion.

---

### 7. Minimum-trial filtering contains a logic error

`prepare_main_df` counts observations for each fish and block:

```python
counts = df.groupby(["Fish_ID", "Block_name"]).size()
```

It then extracts fish IDs from every fish-block combination that passes the threshold and keeps all rows for those fish:

```python
valid_fish = counts[counts >= minimum_trials_per_fish_per_block]...
df = df[df["Fish_ID"].isin(valid_fish)]
```

A fish is therefore retained if it has enough trials in **any one block**, not every required block.

**Potential impact:** Fish with sparse or outcome-dependent missing data in other phases can enter mixed-effects and trajectory analyses.

**Required action:** Require all prespecified blocks to meet the threshold, for example by evaluating the minimum per-block count for each fish or by constructing an explicit fish-by-block completeness table.

---

### 8. Bootstrap settings do not match the manuscript

The manuscript states that 95% confidence intervals use 1,000 bootstrap iterations with seed 10.

Current defaults include:

- `4_ScaledVigorPlotting.py`: `n_boot = 10`
- `5_NormalizedVigorPlotting.py`: `n_boot = 100`
- Learner-analysis scripts: other values

Ten bootstrap iterations are inadequate for estimating a stable 95% interval. One hundred is also relatively unstable for final publication figures.

**Potential impact:** Confidence bands may be noisy, irreproducible, and inconsistent with the reported method.

**Required action:** Use at least 1,000 iterations for exploratory figures and preferably 5,000-10,000 for final intervals, with a documented fixed random seed.

---

### 9. The bootstrap sampling unit needs verification

Plots appear to bootstrap trial-level observations while each fish contributes repeated trials. Unless fish identity is explicitly supplied as the resampling unit, ordinary row resampling treats correlated trials as independent.

The biological replication unit is the fish, not the fish-trial row.

**Potential impact:** Confidence intervals can be too narrow and create an exaggerated impression of precision.

**Required action:** Use either:

- A fish-level bootstrap, or
- A hierarchical bootstrap that resamples fish first and trials within fish second.

The resampling unit and interval type should be stated in the methods and figure legends.

---

### 10. Mixed-effects inference is fragile

Concerns in the current mixed-effects analysis include:

- The default random-effects formula is `~Log_Baseline`, rather than a clearly justified random intercept and/or trial slope.
- Fit exceptions are converted into strings and the pipeline continues.
- A returned result is not explicitly checked for convergence, boundary variance, or singular covariance.
- The global interaction section extracts individual interaction-term p-values rather than conducting a joint test of all interaction terms.
- Per-block and per-trial models create several families of comparisons.
- Separate models for each trial are less efficient than a prespecified longitudinal model.

**Potential impact:** Model uncertainty or failed assumptions may be hidden, and inferential error rates may be difficult to interpret.

**Required action:**

1. Define one primary longitudinal model before examining results.
2. Justify the random-effects structure biologically and statistically.
3. Check and report convergence, singularity, and variance estimates.
4. Use a joint likelihood-ratio or Wald test for multi-parameter interactions.
5. Limit post-hoc contrasts to a prespecified family.
6. Compare results with a robust or nonparametric sensitivity analysis.

---

### 11. Ratio outcomes are sensitive to baseline noise

Normalized vigor divides the response-window mean by the baseline-window mean. A low or sparsely sampled baseline can produce a very large and unstable ratio.

The mixed-effects analysis partially avoids this by modeling log response with log baseline as a covariate, but block-level Mann-Whitney and Wilcoxon analyses still use the ratio.

**Potential impact:** Increased heteroskedasticity, outlier sensitivity, and coupling between the numerator and a noisy denominator.

**Required action:**

- Prefer modeling response-window behavior with baseline as a covariate.
- Consider a log response-to-baseline contrast only when both quantities are adequately estimated.
- Include movement/sample availability in the model.
- Analyze movement occurrence separately.

## Data-integrity and reproducibility issues

### H1. Historical per-fish pickles warp within-trial time via reciprocal Original-frame rate

Local stage-1 gzip pickles for `20221115_04` and `20221116_12` advance
`Original frame number` within each trial at `expected/predicted`
(reciprocal). Current `analysis_utils.interpolate_data` and package
`interpolate_legacy` advance absolute acquisition frames at
`predicted/expected` on the expected-rate grid. CS onset Original frames
still agree closely. Under naive trial-time alignment, vigor looks divergent
because lag grows linearly away from onset exactly as predicted by that rate
ratio; after rate-warp resampling, CS1 vigor correlation recovers to ~0.99.
Treat this as a historical pickle/timebase artifact, not as a target for
`legacy-paper-v1`.

---

### 12. Lost-frame detection may not implement the documented exclusion rule

The manuscript states that recordings with at least one missing frame were excluded.

The code estimates frame loss using accumulated timing drift divided by an inter-frame interval multiplied by a buffer size of 700 frames. A single missing frame does not obviously cross that threshold.

**Potential impact:** Timing discontinuities may be missed, affecting interpolation, angular derivatives, and stimulus alignment.

**Required action:**

1. Check expected versus observed `FrameID` sequences directly.
2. Compare timestamp gaps against integer multiples of the expected interval.
3. Build test cases containing exactly one, several, and sustained missing frames.
4. Confirm that the implemented rule matches the manuscript.

---

### 13. The manuscript and executable code have drifted apart

Observed discrepancies include:

- Vigor definition
- Spatial filtering
- Bout thresholding
- Baseline interval
- Scaled-vigor formula
- Temporal smoothing window
- Bootstrap count
- Fish-exclusion behavior

This is more serious than incomplete documentation: the manuscript cannot currently be assumed to describe the analysis that produced the outputs.

The repository README also follows intended/manuscript definitions in some places and should be revised after the canonical implementation is selected.

**Required action:** Create a single analysis specification containing formulas, units, windows, thresholds, inclusion criteria, and statistical models. Test the code against that specification and generate methods text from the same source where practical.

---

### 14. Results are not tied to immutable configurations

Analysis choices are controlled by editable module-level constants:

- Experiment selection
- Exclusion handling
- Conditioned-response windows
- Missingness thresholds
- Plotting stages
- Statistical formulas and optimizers
- Bootstrap counts
- Fish subsets

Generated files do not record a complete configuration, input hashes, code commit, package versions, or exclusion-list hash. Some loaders choose the first or most recently modified matching file, which can silently select a different artifact.

**Potential impact:** It may be impossible to prove which settings and inputs generated a manuscript panel.

**Required action:**

1. Move run settings into versioned configuration files.
2. Give every run a unique identifier.
3. Save the resolved configuration with every output.
4. Record the Git commit, package versions, input file hashes, and cohort manifest.
5. Replace first/latest-file discovery with explicit artifact paths.

---

### 15. Learner classification is not ready for confirmatory use

The repository contains four divergent learner-classification scripts. The nominal version suppresses warnings globally and exposes many adjustable thresholds and feature-selection choices.

This creates substantial researcher flexibility and makes it unclear which implementation is canonical.

**Potential impact:** Individual learner labels may be unstable, optimistic, or specific to the selected thresholds and implementation.

**Required action:**

1. Select one canonical implementation.
2. Freeze features and thresholds before evaluation.
3. Evaluate classification with cross-validation or an independent dataset.
4. Quantify false-positive rates and label stability.
5. Report uncertainty rather than only binary labels.

Keeping learner classification outside the current minimal manuscript is appropriate until these steps are complete.

## Recommended remediation sequence

### Phase 1: Preserve and specify

1. Freeze the current raw data, processed data, figures, and exclusion files.
2. Record the commit and environment associated with existing outputs.
3. Write a canonical analysis specification covering:
   - Vigor
   - Filtering
   - Bout detection
   - Trial alignment
   - Baseline and response windows
   - Scaling
   - Missing data
   - Inclusion criteria
   - Statistical models

### Phase 2: Correct and test preprocessing

4. Implement synthetic reference tests for vigor and bout detection.
5. Validate frame-loss detection with known gaps.
6. Compare automated bouts with manually annotated recordings.
7. Resolve every code-versus-method discrepancy.
8. Reprocess all recordings from raw data.

### Phase 3: Correct cohort and outcome handling

9. Build an immutable fish-level cohort manifest.
10. Separate health/tracking exclusions from response-dependent criteria.
11. Analyze movement occurrence, duration/count, and conditional vigor separately.
12. Fix the all-block minimum-trial criterion.
13. Generate sample-size and missing-data flow tables for every condition and phase.

### Phase 4: Rebuild inference

14. Use fish-level or hierarchical bootstrap intervals.
15. Increase final bootstrap iterations to at least 1,000, preferably 5,000-10,000.
16. Define one primary longitudinal model and a limited set of planned contrasts.
17. Validate convergence and model assumptions.
18. Perform sensitivity analyses for cohort, missingness, scaling, and vigor definitions.

### Phase 5: Reconcile reporting

19. Regenerate all figures and statistical tables.
20. Compare corrected results with the frozen original outputs.
21. Update the manuscript methods, sample sizes, results, and figure legends.
22. Update `README.md` to describe the tested implementation.
23. Archive the code, configuration, environment, cohort manifest, and final outputs together.

## Minimum sensitivity analyses

Before relying on the central conclusions, compare:

1. All technically valid fish versus the current quality-controlled cohort
2. Current response-window engagement exclusions versus no outcome-dependent exclusions
3. Immobility as missing versus a two-part movement model
4. Current endpoint vigor versus summed segment-wise absolute angular speed
5. Current percentile scaling versus the documented baseline-relative scaling
6. Alternative reasonable bout thresholds and smoothing windows
7. Row-level versus fish-level/hierarchical bootstrap intervals
8. Ratio tests versus baseline-adjusted longitudinal models

## Overall assessment

The experimental question and overall pipeline structure are clear, and the unpaired controls provide an appropriate conceptual basis for testing associative learning. However, the current implementation has major inconsistencies at the earliest analytical stages.

The central delay- and short-trace-conditioning conclusions may remain valid after correction, but that cannot be established from code inspection alone. Because the discrepancies affect vigor, bout detection, scaling, missingness, and cohort selection, the appropriate next step is to correct and test the canonical analysis and then reprocess the raw data before finalizing the manuscript.
