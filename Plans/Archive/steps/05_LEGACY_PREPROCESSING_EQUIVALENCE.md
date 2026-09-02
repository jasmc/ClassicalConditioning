# Step 05 — Legacy Preprocessing Equivalence

**Status:** Complete for local fixtures (pickle timebase classified)  
**Change class:** Behavior-preserving only  
**Depends on:** Steps 02-04  
**Unlocks:** Trustworthy comparison of corrected preprocessing and metrics

**Completed (scoped):** 2026-08-31  
**Archive note:** Detailed plan archived after local-fixture pickle timebase
classification. Do not invert `legacy-paper-v1` interpolate to match historical
pickles. See
[Archived pickle handoff](../Notes/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md).

## Objective

Extract the existing point-15 preprocessing into testable functions and prove
that the migrated path reproduces current processed tables before any
scientific correction.

## Frozen legacy recipe

Suggested identities:

```text
tail representation: legacy-cumulative-angle-v1
filter: legacy-temporal-filter-v1
activity metric: legacy-distal-angular-speed-v1
bout metric: legacy-rolling-range-v1
movement detector: legacy-bout-detector-v1
trial segmentation: legacy-segmentation-v1
scaling: legacy-percentile-scaling-v1
analysis recipe: legacy-paper-v1
```

Names should be finalized during implementation, but their scientific behavior
must not be silently changed.

## Migration order

1. Frame validation decision adapter
2. Camera/tracking synchronization
3. Existing interpolation
4. Radian-to-degree conversion
5. Existing cumulative angle transformation
6. Existing temporal filter and edge dropping
7. Point-15 selection
8. Existing absolute frame-to-frame angular speed
9. Existing rolling bout-strength metric
10. Existing threshold, duration, and interbout handling
11. Stimulus annotation
12. Trial segmentation
13. Block annotation
14. Existing scaled-vigor calculation
15. Output dtype/category organization

## Process for each operation

- [ ] Write a current-behavior contract from code and baseline evidence.
- [ ] Create a synthetic or approved real characterization fixture.
- [ ] Extract a pure function without changing operations or order.
- [ ] Compare the old and extracted operation on identical input.
- [ ] Integrate through a compatibility wrapper.
- [ ] Compare with the frozen legacy result and write only canonical output.
- [ ] Record known scientific discrepancy without correcting it.

## Core functions

```python
synchronize_legacy_v1(...)
interpolate_legacy_v1(...)
build_cumulative_angles_legacy_v1(...)
filter_angles_legacy_v1(...)
calculate_distal_vigor_legacy_v1(...)
calculate_bout_metric_legacy_v1(...)
detect_bouts_legacy_v1(...)
segment_trials_legacy_v1(...)
calculate_scaled_vigor_legacy_v1(...)
```

Names with `legacy` make accidental use in corrected recipes visible.

## Characterization cases

- Constant signal
- One moving point
- Multiple points moving together
- Opposing local point movements
- Missing first/last samples
- One dropped frame
- Timing jitter
- Bout touching a boundary
- Short bout
- Short interbout gap
- No movement
- Zero/constant baseline
- Sparse movement baseline
- CS- and US-aligned trials
- Catch trial

## Equivalence checks

Compare:

- retained frame and sample set;
- time columns;
- all retained angle columns;
- vigor;
- rolling bout metric before it is dropped;
- bout, bout start, and bout end masks;
- event annotations;
- trial, block, and phase assignments;
- scaled vigor;
- categories, sparse fields, and missingness;
- output row and fish/trial counts.

Use exact comparisons after canonical normalization unless a documented
dependency-level reason requires a defined tolerance.

## Outputs

For every pilot recording:

```text
processed-samples-legacy.parquet
processed-samples-legacy.json
legacy-equivalence-report.json
```

## Required tests

- Unit tests for every extracted operation
- Characterization tests against baseline
- Contract test for processed sample schema
- Integration test from synthetic raw recording through processed samples
- Property test that permitted input row normalization does not alter output
- Difference reporter test that localizes first divergence by operation

## Exit gate

The migrated legacy recipe reproduces the accepted baseline for representative
recordings and the synthetic end-to-end fixture. Every discrepancy is either
resolved as a regression or documented and approved as an unavoidable
serialization/numerical-environment difference.

## Pilot progress

Implemented in `3709776`:

- frozen `legacy-paper-v1` configuration;
- source-hash verification;
- legacy camera/tracking discard behavior;
- legacy reference frame, interpolation, filtering, vigor, bouts, trial
  segmentation, blocks, and scaling;
- lossless Parquet output plus authenticated summary/completion marker;
- streaming Parquet comparison utility.

Pilot `20221115_04`:

```text
final rows: 10,836,172
CS trials: 94
US trials: 78
```

Local pickle equivalence for available fish is classified (exit for Step 05
pickle gate). Optional only:

- speed up lag diagnosis; extend warp-aware checks beyond CS1;
- keep pickle comparison reports under Quality checks as evidence.

Local pickle-vs-Parquet reports (`compare-legacy-pickle`) for
`20221115_04` and `20221116_12`:

- row counts match (`10,836,172`);
- after storage-type normalization, `Trial type`, `Trial number`,
  `Trial time (frame) [700 FPS]`, and `Block name` match exactly;
- **Original-frame / within-trial timebase (historical pickle artifact):**
  pickle `Original frame number` advances at `expected/predicted`
  (reciprocal). Parquet advances at `predicted/expected`, matching current
  `analysis_utils.interpolate_data` / `interpolate_legacy`. CS onset
  Original frames agree within about one frame. Naive trial-time vigor
  correlation looks poor (CS1 ~0.39 / ~0.15) because within-trial lag grows
  as `trial_time * (expected/predicted - predicted/expected)`. After
  rate-warp resampling, CS1 vigor correlation recovers to ~0.996 / ~0.988
  and measured lags track that prediction (corr ~0.999). Do **not** invert
  interpolate in `legacy-paper-v1` to match the pickle;
- sparse CS/US marker and bout-flag differences follow from the warped
  timebase under naive trial-time alignment.

Do not treat naive pickle row equality as the sole Step 05 gate. Legacy
artifacts can be wrong; classify before changing frozen recipes.

Added since the pilot recipe:

- `preprocessing.legacy_equivalence` helpers that localize the first diverging
  operation;
- careful raw-TXT → `prepare_legacy_tracking` chaining that drops the trailing
  summary row only once;
- `preprocessing.legacy_characterization` synthetic cases and
  `analysis_utils_reference_steps` for cumulative → filter → vigor → bout
  metric → bout detection, plus synchronize/interpolate checks;
- stimulus annotation, trial segmentation, block assignment, and scaled-vigor
  characterization helpers;
- compact `figure-legacy-review` QC figure from `samples_legacy-v1.parquet`;
- `compare-legacy-pickle` for local gzip pickle vs Parquet (JSON report; no
  table dumps). Historical pickles must stay on disk and are never sent to an
  LLM.

Local pickle files now exist for `20221115_04` and `20221116_12`. End-to-end
reports are written under Quality checks after local comparison.


## Prohibited work

Do not in this step:

- implement the manuscript vigor formula;
- add spatial smoothing;
- activate the unused second threshold;
- correct frame-loss logic;
- change interpolation;
- change baseline windows or scaling;
- change exclusion criteria;
- change output cohorts.

Those belong to versioned scientific-correction work after equivalence.
