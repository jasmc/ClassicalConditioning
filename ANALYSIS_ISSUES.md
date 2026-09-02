# Major Analysis Issues

## Documentation baseline

This audit describes:

- `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
- Source feature commits `505e29a` and `88fe4fc`
- Cherry-picked equivalents `4d6939f` and `f372101`
- Working-tree LogMedian compatibility adaptations dated 2026-08-28

It supersedes the earlier audit of `b0dfcb3`.

## Scope

This is a code and methods audit, not a reanalysis of experimental data. It
identifies risks but does not quantify their effects on conclusions.

## Critical issues

## 1. Preprocessing vigor does not match the manuscript definition

The manuscript describes vigor as the sum of absolute angular speeds across
tail segments.

Step 1 cumulatively sums tail angles, selects one cumulative tail-angle column,
and computes the absolute derivative of that one signal.

In general:

```text
abs(sum(delta_angle_i)) != sum(abs(delta_angle_i))
```

Opposing segment changes can cancel.

Potential impact:

- Bout detection
- Trial validity
- Vigor magnitude
- LogMedian values
- Normalized vigor
- Classification
- Every downstream figure and statistic

Required action:

- Approve one mathematical definition.
- Add synthetic segment-level tests.
- Reprocess raw data if the implementation changes.

## 2. Filtering and bout detection remain incomplete

In `analysis_utils.py`:

- Spatial filtering is intentionally simplified and not fully implemented.
- The secondary bout threshold parameter `thr2` is accepted but unused.
- The function contains a placeholder for the second filtering stage.

Potential impact:

- Which frames are marked as bouts
- Movement probability
- Conditional vigor
- Exclusion criteria
- Learner features

Required action:

- Specify and implement both filters and both bout thresholds.
- Compare with manually annotated traces.
- Add threshold and boundary tests.

## 3. Excluded fish can re-enter step 3

`3_FishGrouping_LogMedian.py` recursively discovers:

```python
paths.orig_pkl.glob("**/*.pkl")
```

The loading path does not exclude files under an `Excluded` subdirectory.
Therefore fish moved by step 1 can still be rediscovered and grouped.

Later `APPLY_FISH_DISCARD` flags are disabled by default in steps 4-6.

Potential impact:

- Reported cohorts can include excluded animals.
- Sample sizes can differ from manuscript counts.
- Learner labels and group profiles can use the wrong cohort.

Required action:

- Immediately exclude `Excluded` paths in step 3.
- Replace path-location semantics with one validated cohort manifest.

## 4. The LogMedian baseline window is inconsistent across stages

Step 3 calculates its baseline median with:

```text
trial_time < -15 s
```

For a roughly -45 to +45 second trial, that selects approximately -45 to -15
seconds.

Step 5 and the learner-stratified pipeline use:

```text
-15 to 0 s
```

Step 4 may then subtract the immediate pre-stimulus median from the already
baseline-centered LogMedian signal.

Potential impact:

- The `Scaled vigor (AU)` reference differs from the trial-summary reference.
- Time profiles and scalar trial summaries are not directly comparable.
- Baseline drift can appear as a response difference.

Required action:

- Approve one analytical baseline window.
- Separate analytical normalization from display centering.
- Rename transformed fields so their meaning is explicit.

## High-priority issues

## 5. Vigor is conditional on movement

Step 3 sets vigor outside bouts to missing. Median CR and baseline values are
therefore based only on moving samples.

The metric does not measure total movement suppression. A fish can learn by:

- Reducing probability of movement
- Reducing time moving
- Reducing bout count
- Reducing vigor within bouts

The learner-stratified pipeline improves interpretation by plotting movement
probability alongside conditional vigor, but the classifier still uses the
conditional median-vigor summary.

Required action:

- Treat movement occurrence and conditional vigor as separate outcomes.
- Consider a two-part or hurdle analysis.
- Do not interpret conditional vigor alone as total motor output.

## 6. Response-related inclusion criteria can bias the cohort

Step 1 can require movement within the CR interval. Suppression of movement is
the expected learned response, so a strongly suppressing fish can become
ineligible for the very reason being studied.

Required action:

- Separate technical/health exclusions from response-dependent engagement
  criteria.
- Report sensitivity analyses with and without behavioral engagement filters.

## 7. Missingness filtering is disabled in step 5

Current defaults include:

```python
APPLY_MAX_NAN_FRAC_PER_WINDOW = False
```

Trials with very limited movement can therefore yield median estimates based on
few samples.

Required action:

- Report valid sample and bout coverage per trial.
- Approve a minimum coverage policy.
- Test conclusions across reasonable coverage thresholds.

## 8. Bootstrap settings are inconsistent

Step 4 uses 1,000 bootstrap iterations. Step 5 uses 100. The manuscript methods
state 1,000 in relevant contexts.

The biological resampling unit must also be explicit. Trial rows or time
samples must not be treated as independent fish.

The learner-stratified pipeline correctly aggregates trials within fish first
and uses fish-level bootstrap resampling for its profiles.

Required action:

- Use fish-level or hierarchical bootstrap as appropriate.
- Use an approved iteration count and fixed seed.
- Record both in provenance.

## 9. Statistical model validation remains limited

Step 5 and step 6 use mixed-effects models, but the code includes broad
exception handling and does not consistently fail on:

- Non-convergence
- Singular covariance
- Boundary estimates
- Unstable random effects

Required action:

- Save model inputs and diagnostics.
- Require convergence for inferential output.
- Predefine the primary model and comparison families.
- Report effect sizes, not only p-values.

## 10. Learner analysis remains descriptive and circular

Learner labels are derived from acquisition and recovery behavior from the same
experiment used in the stratified profiles.

The integrated pipeline explicitly records:

```text
analysis_mode = descriptive_classifier_characterization
```

and includes a circularity notice. This is correct.

Do not present learner versus non-learner differences from those same data as
independent validation.

Required action for confirmatory claims:

- Held-out trials
- Held-out phase
- Cross-fitting
- Or an independent cohort

## 11. Learner classification is sensitive to a small control reference

The classifier estimates a directional covariance and empirical threshold from
control fish. With a modest control sample:

- The empirical tail is coarse.
- Leave-one-out control scores are correlated with the fitted covariance.
- A nominal 5% target does not guarantee exactly 5% out-of-sample false
  positives.

Required action:

- Report observed control classification rate.
- Bootstrap label stability.
- Evaluate threshold sensitivity.
- Validate on an independent or cross-fitted control sample where possible.

## Integration and reproducibility issues

## 12. The learner feature commits were incompatible with `bf46bf7` as written

The transplanted pipeline originally:

- Imported removed file `6_LearnersQuantification_WIP.py`.
- Created mean baseline and mean CR values.
- Used response/baseline ratios.
- Labeled the conditioned group `trace`.

The LogMedian baseline instead requires:

- `6_LearnersQuantification_LogMedian.py`
- Median baseline and median CR
- Log-space subtraction
- Condition ID `3sTrace`

These have been adapted in the current working tree and tests were updated.

## 13. Standalone LogMedian classifier had an inconsistent response constant

`6_LearnersQuantification_LogMedian.py` documented and received `Median CR`,
but its constant was `Mean CR`.

The current working tree changes it to:

```python
RESPONSE_COLUMN_NAME = "Median CR"
```

## 14. Artifact selection and identity remain fragile

Several baseline scripts search for matching files and select the first or
newest candidate. Filenames encode scientific settings such as:

- Alignment
- selected versus all fish
- missingness filtering
- LogMedian status

Required action:

- Pass explicit input paths.
- Add schema versions and run IDs.
- Save resolved configuration with each artifact.

## 15. Pickle is the dominant canonical format

Pickle is Python- and pandas-version-sensitive and can deserialize arbitrary
objects. The integrated learner pipeline contains compatibility code for older
pandas extension-array states, demonstrating this risk.

Required action:

- Migrate canonical tables to Parquet after schema validation.
- Preserve existing pickles as immutable legacy inputs.
- Use CSV only as a human-readable export.

## 16. Checkpoint provenance does not hash large source files

Learner-pipeline checkpoint signatures use path, size, and modification time.
Final provenance explicitly leaves source content hashes null to avoid
rereading multi-gigabyte files.

This is efficient but cannot prove content identity if a file is modified
without a detectable metadata change.

Required action:

- Calculate source hashes once and cache them.
- Or use upstream artifact manifests containing trusted hashes.

## 17. Tracked data-flow wording was corrected in this working tree

At `bf46bf7`, `0_Pipeline_Data_Flow_Description.txt` described the conditioned
response as anticipatory tail movements. The 2026-08-28 working-tree adaptation
changes this to predictive suppression, corrects the direction of the
LogMedian normalized-vigor subtraction, and adds step 6 plus the optional
learner-stratified stage.

Remaining action:

- Verify the edited wording against the approved scientific analysis
specification before committing it as final methods documentation.

## 18. Automated coverage is narrow

The integrated learner pipeline has useful synthetic unit tests. The broader
stage-1 through stage-6 pipeline lacks a comprehensive automated suite.

Missing high-value tests include:

- Segment-wise vigor
- Spatial filtering
- Secondary bout threshold
- Single-frame loss
- Trial alignment
- Cohort exclusion
- LogMedian baseline calculation
- Mixed-model convergence
- End-to-end paper fixture

## Recommended order of correction

1. Canonical vigor specification and tests
2. Filtering and bout-detection implementation
3. Frame-loss validation
4. Full raw-data reprocessing
5. Cohort manifest and exclusion fix
6. Baseline-window unification
7. Movement occurrence plus conditional-vigor analysis
8. Missingness thresholds and coverage reporting
9. Fish-level bootstrap and statistical model validation
10. Learner-label stability and held-out validation
11. Artifact migration and provenance
12. Final figure regeneration

## Overall assessment

`bf46bf7` is a substantially cleaner baseline than `b0dfcb3`: it consolidates
the analysis around LogMedian files, removes several legacy modules and
competing learner scripts, and introduces shared pipeline/heatmap utilities.

The later learner-stratified work adds valuable explicit inputs, manifests,
fish-first temporal aggregation, movement probability, catch-trial timing,
tests, and a circularity warning.

However, foundational preprocessing, cohort, and baseline inconsistencies
remain. Final paper claims should be regenerated after those issues are
resolved and the complete paper dataset is reprocessed.
