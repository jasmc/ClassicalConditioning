# Step 07 — Movement Detection, Metric Validation, and Selection

**Status:** In progress  
**Done:** Development + corrected movement/temporal/trial + two-fish corrected compare/runner.  
**Open:** Gate T1 selection/partitions; annotations/video review.  
**Change class:** Scientific candidate validation and decision  
**Depends on:** Step 06; scientific gate O before biological selection  
**Unlocks:** Frozen corrected preprocessing recipe

## Objective

Evaluate the candidate activity metrics technically and biologically, select a
primary activity metric and companion outcomes without confirmation-set
leakage, and freeze all required preprocessing and movement-state parameters.

## Validation data partitions

Before comparing learning effects, assign data to:

### Technical development

Used for:

- formula debugging;
- smoothing and outlier handling;
- movement/rest threshold calibration;
- video-annotation agreement;
- basic positive controls.

### Metric selection

Used to apply the prespecified scorecard and choose a metric. The partition must
be independent enough that technical tuning does not overfit every observation.

### Confirmation

Untouched until the metric, parameters, detector, outcomes, windows, and
statistical test are frozen.

Preferred split levels, subject to the Step 00 feasibility/power rule:

1. held-out experiments;
2. held-out acquisition days/rig batches;
3. held-out fish assigned before development;
4. nested fish-level cross-fitting when a permanent holdout is underpowered.

Do not randomly split frames from the same fish between development and
confirmation.

## Scientific gate O for selection

Before Work packages 07.5, 07.7, and 07.8, freeze the shared:

- trial map and catch-trial definitions;
- baseline, CS, trace, expected-US, and post-US windows;
- total activity, movement probability, conditional intensity, and bout
  outcome functions;
- time-bin definition;
- fish-aware aggregation order;
- coverage rules.

These versioned functions and settings are reused unchanged by Step 09. Step 07
must not create private selection-only versions of the outcomes.

## Work packages

### 07.1 Define metric-specific movement state

Each metric has different units and noise. Define:

```text
metric_id
detector_id
detector_version
threshold method
threshold value or calibration artifact
minimum bout duration in ms
minimum interbout interval in ms
edge policy
validity requirement
```

Convert durations to samples after resolving recording timing.

Possible development detectors:

- deterministic noise-calibrated threshold;
- dual threshold/hysteresis;
- probabilistic two-state model as an exploratory alternative.

The existing threshold is preserved only for the legacy metric recipe.

### 07.2 Build movement-state artifact

```text
recording_id
fish_id
frame_id
metric_id
detector_id
detector_version
moving
bout_id
bout_start
bout_end
threshold
valid
quality_flags
```

### 07.3 Validate synthetic behavior

Use Step 06 patterns to verify:

- rest remains inactive;
- stationary bend is not movement;
- known motion is detected;
- bout boundaries obey declared rules;
- missing points and long gaps do not create false bouts;
- point-density changes do not materially redefine activity;
- spike handling does not erase genuine strong movement.

### 07.4 Perform blinded video validation

Create a balanced, blinded sample of:

- rest;
- weak/strong bouts;
- C- and S-bends;
- rhythmic swimming;
- struggle-like movement;
- ambiguous movement;
- tracking failures.

Observers annotate:

- movement/rest;
- onset and offset;
- relative strength;
- rhythmic/irregular coordination;
- unusable tracking.

Report inter-rater reliability and candidate agreement without using group or
learner labels during annotation.

### 07.5 Test positive and negative controls

Positive control:

- known US-evoked movement onset, peak, and magnitude.

Negative/stability controls:

- quiet pre-CS periods;
- synthetic stationary postures;
- known tracking-only displacement where available.

### 07.6 Run robustness grid

Prespecify reasonable values for:

- temporal smoothing;
- curvature spatial smoothing;
- valid-tail fraction;
- included proximal/distal points;
- final-point inclusion;
- outlier policy;
- movement threshold;
- time-bin width;
- measured versus reconstructed representation, if both valid.

This is a bounded sensitivity grid, not unrestricted optimization.

### 07.7 Build candidate outcomes for selection

For each candidate:

- total activity including rest;
- movement probability;
- fraction time moving;
- conditional intensity;
- bout count/rate;
- bout duration;
- valid-frame and valid-tail coverage.

Use fish-aware aggregation.

Use the gate O outcome implementation that Step 09 will apply to the full
corrected dataset.

### 07.8 Apply frozen selection scorecard

Score:

1. mathematical/synthetic validity;
2. blinded video agreement;
3. movement/rest classification;
4. baseline stability;
5. fish-level repeatability;
6. US-response detection;
7. tracking-artifact resistance;
8. point-density and smoothing stability;
9. usable-frame fraction;
10. direction and consistency of anticipatory effects;
11. interpretability;
12. computational and downstream usability.

Weights and tie-breaking must be fixed before inspecting the selection result.
P-value magnitude cannot dominate.

### 07.9 Freeze selection

Record:

- selected primary activity metric;
- required companion movement-probability outcome;
- conditional-intensity and bout outcomes;
- tail representation;
- timing and smoothing;
- tail-length weighting;
- detector and thresholds;
- validity limits;
- selection evidence;
- confirmation partition and analysis.
- validation mode (`permanent_holdout` or `nested_fish_cross_fit`);
- primary confirmatory result source;

### 07.10 Run confirmation

For `permanent_holdout`, apply the frozen recipe once to held-out confirmation
data. For `nested_fish_cross_fit`, select inside each training fold and evaluate
only in the corresponding outer test fold. Report all candidate confirmation
results where practical, but do not retune a frozen recipe from confirmation
outcomes.

If confirmation fails, label the recipe failed and create a new development
cycle/version. Do not retrospectively alter the frozen run.

## Core types

```text
MovementStateConfig
MovementStateDetector
MetricSelectionSpec
MetricSelectionResult
ValidationPartition
MetricScorecard
```

## Deliverables

- Versioned detector configurations
- Movement-state and bout artifacts
- Video annotation protocol and de-identified annotation outputs
- Synthetic and positive-control reports
- Robustness report
- Candidate scorecard
- Frozen metric-selection decision record
- Held-out confirmation report
- Validation-partition/cross-fit manifest
- Selected corrected preprocessing recipe

## Exit gate

Scientific gate T1 is approved: the primary activity metric, representation,
filtering, timing, weighting, movement detector, companion outcomes, units, and
confirmation evidence are frozen. Gate O outcome functions used for selection
are versioned for direct reuse in Step 09.

## Failure conditions

Do not select a metric if:

- X/Y or angle semantics remain unresolved for that metric;
- results depend strongly on an unprincipled smoothing/threshold choice;
- confirmation data influenced tuning;
- video and synthetic validation fail;
- usable coverage differs by condition in an unexplained way;
- the selection is based primarily on statistical significance.

## Pilot calibration progress

Implemented:

```text
369bf43 feat(analysis): add exploratory movement-state calibration
788602f feat(analysis): add candidate movement outcome profiles
9212634 feat(validation): add movement smoothing sensitivity
4c027cc feat(trace): add balanced detector review
```

Candidate detector:

- 10 ms rolling median resolved to 7 measured frames;
- exactly the quietest 20% of 100 ms windows selected with deterministic tie
  handling;
- low/high thresholds use quiet-window 0.99/0.999 quantiles;
- strict exceedance hysteresis;
- measured-time 40-frame-equivalent minimum duration and 10-frame-equivalent
  maximum interbout gap;
- invalid/gap/coverage samples remain explicitly invalid rather than rest;
- all 79 US events have complete evaluated positive-control windows.

The retired five-metric local-fixture tables and smoothing sensitivity are
preserved in [the archived pilot-evidence note](./Archive/RETIRED_FIVE_METRIC_PILOT_EVIDENCE.md).
They demonstrate detector sensitivity rather than accuracy and do not select a
metric or smoothing strength. Only the three active metrics in `DECISIONS.md`
may appear in current candidate artifacts or scorecards.

Trace-review generation now produces six complete, non-overlapping windows for
quiet behavior, strong movement, detector disagreement, and early/middle/late
US events. The local PNG, self-contained HTML, lossless trace table, and
annotation CSV include exact smoothed detector inputs, thresholds, movement
states, bout IDs, and verified artifact lineage. The pilot annotation status is
still `unreviewed`.

The second local fish reproduces material detector sensitivity across 0/10/20
ms smoothing; no smoothing or detector setting is approved. Its retired
five-metric values are preserved in the same archived pilot-evidence note.

Remaining (implementation; not two-fish scientific approval):

- reviewed trace/video annotation workflow;
- threshold, curvature-spatial-smoothing, and remaining bounded robustness
  report dimensions;
- false-positive and repeatability assessment plumbing;
- frozen selection/confirmation partition artifacts;
- Gate T1 approval (requires partitions beyond fixture debugging).
