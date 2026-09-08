# Step 09 — Trial Outcomes, Alignment, and Temporal Profiles

**Status:** In progress - prototype/fixture-only  
**Change class:** Behavior-preserving legacy branch plus **candidate** corrected outcomes  
**Prototype scope:** Two-fish corrected temporal/trial artifacts and metric comparison  
**Paper-scope dependencies:** Step 08, gate C1, and scientific gate O approved
before Step 07 biological selection  
**Unlocks:** Statistics, learner classification, and paper panel data
**Done (fixtures):** corrected temporal/trial + corrected metric comparison on
both local fish. **Open:** cohort-scale canonicalization / Gate O.

## Objective

Create one canonical analytical layer that maps processed frame activity to
fish-aware trial summaries and temporal profiles without embedding plotting or
statistical model logic.

## Scientific gate O implementation

Reuse the definitions approved before Step 07 and implement them at full
paper-scope scale:

- primary and secondary outcomes;
- baseline, CS, trace, expected-US, and post-US windows;
- CS and US alignment conventions;
- total activity summary;
- movement probability/fraction definition;
- conditional intensity definition;
- bout outcomes;
- analytical scaling;
- role of legacy normalized vigor;
- coverage thresholds;
- time bins and boundary conventions;
- within-fish aggregation.

## Work packages

### 09.1 Apply canonical trial map

Annotate every eligible row with:

```text
recording_id
trial_id
alignment
event_id
trial_number
phase
block_5_id
block_10_id
catch status
expected_us_time_s
```

Reject rows with missing or multiply matched trial definitions.

### 09.2 Build aligned samples

Preserve:

- absolute timestamp;
- frame/sample identity;
- trial-relative time;
- CS-relative time;
- US-relative time where applicable;
- activity metric and movement detector identities;
- validity and coverage.

Avoid creating inconsistent independent CS and US source tables when one
canonical table plus alignment views is sufficient.

### 09.3 Define outcome functions

Implement pure functions for:

#### Total activity

Continuous metric including valid rest frames. Define mean, median, integral,
or another summary explicitly.

#### Movement probability and fraction moving

Summarize the metric-qualified movement-state artifact. Rest contributes zero
movement rather than missingness.

#### Conditional intensity

Activity among frames/trials classified as moving. Always report its movement
coverage and sample count.

#### Bout outcomes

- bout count;
- initiation rate;
- duration;
- interbout interval;
- optional bout-level peak/integral.

#### Legacy outcomes

Reproduce current:

- out-of-bout missing vigor;
- scaled vigor;
- normalized vigor response/baseline ratio;
- current baseline and response masks.

Legacy fields are explicitly suffixed or metadata-qualified.

### 09.4 Build trial summary artifact

Candidate fields:

```text
experiment_id
recording_id
fish_id
condition_id
cohort_id
trial_id
alignment
event_id
trial_number
phase
block_5_id
block_10_id
is_catch
metric_id
detector_id
baseline_total_activity
response_total_activity
movement_probability
fraction_time_moving
conditional_intensity
bout_count
bout_rate
mean_bout_duration
legacy_scaled_vigor
legacy_normalized_vigor
valid_baseline_fraction
valid_response_fraction
```

The authoritative trial identity is
`(recording_id, trial_id, alignment)`. `experiment_id`, `fish_id`,
`trial_number`, and `event_id` are required consistency and provenance fields.
The primary key also includes metric/detector identity where multiple candidates
are retained.

### 09.5 Build temporal bins

Aggregate in this order:

```text
frames -> fish × trial × time bin
trials -> fish × block/phase × time bin
fish -> population summaries later
```

Never aggregate thousands of frames or trials directly across fish.

Record:

- bin edges and labels;
- left/right closure;
- contributing frames;
- contributing trials;
- contributing fish;
- valid coverage.

### 09.6 Build catch-trial outputs

Use explicit trial-map catch status, not a hard-coded list in plotting code.

Provide:

- per-catch-trial profiles;
- pooled catch profiles where scientifically specified;
- expected-US timing windows;
- fish-level suppression/timing measures.

### 09.7 Add coverage and sample-flow outputs

For every condition, phase, block, time bin, metric, and outcome:

- available fish;
- included fish;
- contributing fish;
- trial count;
- valid-frame fraction;
- missingness reason distribution.

Build an explicit fish-by-required-block completeness table. Any rule requiring
minimum trials in every prespecified block must evaluate every required block
for each fish; passing one block must not retain a fish with insufficient data
in another required block.

### 09.8 Preserve legacy stage equivalence

Compare migrated equivalents of current grouping, scaled-vigor, and
normalized-vigor data before accepting corrected outcome changes.

## Required tests

- Every trial maps exactly once
- Catch status and expected-US timing
- CS/US alignment boundaries
- Baseline/response window inclusivity
- Rest represented correctly
- Tracking invalidity remains missing, not rest
- Conditional intensity uses moving frames only
- Fish-first aggregation
- Equal fish weighting
- Input row order invariance
- Excluded fish cannot affect output
- Zero-baseline behavior
- Empty movement windows
- Time-bin edge cases
- Coverage counts reconcile to source
- Minimum-trial eligibility requires all prespecified blocks, not any one block

## Deliverables

- Canonical aligned-sample views
- Trial outcome artifact
- Temporal-bin artifact
- Catch-trial artifacts
- Coverage and sample-flow tables
- Legacy-equivalence report for stages 3-5
- Outcome data dictionary and methods specification

## Exit gate

Every paper outcome has one named function, unit, schema field, metric/detector
identity, window definition, cohort, reference test, and coverage report.

## Downstream invalidation

- Trial-map change: alignment/outcomes onward
- Window or outcome formula change: named outcomes onward
- Time-bin change: temporal profiles onward
- Cohort change: all cohort-dependent outcomes onward
- Figure style change: no outcome invalidation

## Pilot progress

**Done (fixtures):** corrected temporal/trial outcomes and corrected metric
comparison on both `20221115_04` and `20221116_12`. Early single-fish
candidate artifact (`788602f`, `candidate_temporal_outcomes-v2.parquet`) remains
useful characterization evidence.

This two-fish candidate corrected path does not satisfy cohort-scale, Gate O,
population, metric-selection, or confirmatory parts of the exit gate.
