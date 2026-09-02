# Learner-Stratified Vigor Analysis Plan

## Purpose

Add an explicit learner classification to the fish-level analysis metadata, propagate that classification safely into trial-level and time-series datasets, and analyze conditioned-response dynamics separately for:

- Conditioned learners
- Conditioned non-learners
- Reference/control fish
- Fish that could not be classified

The resulting analysis should describe the temporal profile of behavior:

- In individual fish
- In individual trials
- In selected catch trials
- Across grouped trials
- Across five-trial blocks
- Across experimental phases
- Relative to CS onset
- Relative to US onset

This plan intentionally separates **classification**, **data annotation**, **temporal aggregation**, **visualization**, and **statistical inference**.

## Deferred end-to-end methodology review

Canonical learner classification is intentionally deferred until the rest of
the analysis refactor is fully implemented and the corrected preprocessing,
cohort, outcomes, and primary statistics are frozen. At that point, begin with
an end-to-end brainstorming and scientific design review rather than choosing
among the existing scripts by default.

The review should consider whether the learner construct is best represented
as a binary or multiclass label, a continuous learning score, a response
probability, a latent trajectory, or no separate learner construct at all.
Potential tools and alternatives include:

- PCA or related component analyses to understand feature covariance and
  dimensional structure, without treating components as validated classes;
- power analysis and simulation to quantify detectable separation, validation
  sample requirements, label uncertainty, and false-positive control;
- longitudinal or hierarchical trajectory models;
- change-point, mixture, clustering, supervised, semi-supervised, and
  continuous-score approaches;
- held-out, cross-fitted, or independent validation designs;
- sensitivity to activity metric, detector, scaling, feature window, cohort,
  missingness, and threshold choices.

This is an exploratory decision phase, not a commitment to PCA or any other
method. Record candidate approaches, assumptions, estimands, validation
requirements, rejection reasons, and the rationale for whether learner
classification should proceed.

## Important scientific caveat: avoid circular analysis

The current learner classifier uses acquisition and extinction-related normalized-vigor features derived from the same behavioral experiment. If those labels are then used to compare learners and non-learners on the same trials and response windows, the analysis is circular:

```text
behavioral response
    -> learner label
    -> split fish by learner label
    -> test the same behavioral response
```

Learners will differ from non-learners in the data used to define the groups by construction.

Therefore:

- Plots using the same data for classification and visualization are valid as **descriptive classifier characterization**.
- P-values comparing learner and non-learner groups on the classification data are not independent confirmation of learning.
- Confirmatory inference requires held-out trials, held-out phases, cross-fitting, or an independent experiment.
- Every figure and table must state whether it is descriptive, held-out, cross-validated, or independently validated.

## Prerequisites

Before using learner labels in paper-level claims:

1. Resolve the relevant analytical issues documented in `../docs/analysis/ANALYSIS_ISSUES.md`.
2. Complete the non-learner refactor and freeze corrected inputs and primary
   outcomes.
3. Conduct the deferred end-to-end methodology review above.
4. Select one canonical learner-classification implementation or explicitly
   decide not to classify.
5. Freeze the classifier, if selected:
   - Feature definitions
   - Feature directions
   - Minimum trial requirements
   - Point versus conservative rule
   - Voting threshold
   - Mahalanobis-distance threshold
   - Control/reference definition
   - Bootstrap settings
   - Random seed
6. Validate that fish exclusions are applied consistently.
7. Recalculate classification after corrected preprocessing if vigor, bout detection, scaling, or missing-data handling changes.
8. Ensure the classification export is enabled and versioned.

The current nominal implementation is:

```text
6_LearnersQuantification.py
```

The repository also contains `new`, `improved`, and `WIP` variants. These must not be mixed within one analysis.

## Core design

Do not edit or overwrite the original per-fish or condition-level datasets.

Create:

1. A canonical fish-level classification manifest
2. Derived labeled time-series datasets
3. Derived labeled per-trial summary datasets
4. Explicit analysis subsets for each learner stratum

```text
Learner classifier
        |
        v
Fish-level classification manifest
        |
        +------------------------------+
        |                              |
        v                              v
Condition-level time series     Per-trial summary table
        |                              |
        +--------------+---------------+
                       |
                       v
               Validated many-to-one join
                       |
                       v
             Labeled derived datasets
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Learners    Non-learners   Reference/control
          |            |             |
          +------------+-------------+
                       |
                       v
        Single-trial, block, phase, and
             temporal-profile analyses
```

## 1. Define the canonical fish key

### Problem

`Fish_ID` alone may not remain unique when datasets from different experiments are combined. A day-and-fish identifier can also collide if naming conventions change or if the same fish appears in multiple derived files.

### Required composite key

Use the most specific available key:

```text
Experiment_ID
Condition
Fish_ID
```

If `Fish_ID` already contains day and fish number, retain those components separately for auditing:

```text
Day
Fish_No
Fish_ID
```

If one fish can appear under multiple analysis alignments, do not include alignment in the biological fish key. Store alignment as analysis metadata:

```text
Alignment = CS | US
```

### Validation

The classification manifest must contain no duplicate rows for:

```text
Experiment_ID + Condition + Fish_ID + Classifier_Run_ID
```

Any duplicate should stop the join instead of being silently resolved.

## 2. Create a canonical classification manifest

### Source

The current classifier can export:

```text
Fish_Learner_Classification_CS.csv
```

with fields including:

- `Fish_ID`
- `Condition`
- `Mahalanobis_Distance`
- `Vote_Point`
- `Vote_Conservative`
- `Is_Outlier`
- `Is_Learner_Point`
- `Is_Learner_Conservative`
- Feature values
- Feature standard errors

Do not use only a single Boolean column. Preserve the classifier evidence and provenance.

### Proposed schema

```text
Experiment_ID
Condition
Fish_ID
Cohort_Role
Classification_Eligible
Classification_Ineligible_Reason
Learner_Point
Learner_Conservative
Learner_Status
Analysis_Stratum
Mahalanobis_Distance
Distance_Threshold
Distance_Threshold_CI_Low
Distance_Threshold_CI_High
Vote_Point
Vote_Conservative
Feature_Acquisition
Feature_Acquisition_SE
Feature_Extinction
Feature_Extinction_SE
Classification_Alignment
Classifier_Name
Classifier_Version
Classifier_Run_ID
Classifier_Config_Hash
Cohort_Manifest_Hash
Code_Commit
Generated_At
```

### Recommended categorical fields

#### `Cohort_Role`

```text
Reference
Conditioned
Other
```

#### `Learner_Status`

Use the selected primary classification rule:

```text
Learner
Non-learner
Unclassified
```

`Unclassified` means the classifier could not assign a valid label, for example because required features or trials were missing. It must not be silently converted to non-learner.

#### `Analysis_Stratum`

Use a paper-facing group:

```text
Reference
Conditioned learner
Conditioned non-learner
Conditioned unclassified
Other learner
Other non-learner
Other unclassified
```

This avoids calling a control fish a “conditioned non-learner.” Control fish can still retain their classifier outputs for false-positive diagnostics, but their analysis stratum remains `Reference`.

### Primary rule

Select one primary label before examining learner-stratified plots:

```text
Learner_Status = Learner when Learner_Conservative is true
Learner_Status = Non-learner when eligible and Learner_Conservative is false
Learner_Status = Unclassified when not eligible
```

The point-estimate rule should be retained for sensitivity analysis rather than mixed with the conservative rule in one plot.

## 3. Add labels to the table containing all fish

### Do not mutate the source table

Create a derived labeled table:

```text
<original stem>_with_learner_status.pkl
```

or preferably an explicit analysis-run directory:

```text
build\learner_stratified\<run_id>\
```

### Join behavior

For pandas:

```python
labeled = all_fish.merge(
    classification_manifest[
        [
            "Experiment_ID",
            "Condition",
            "Fish_ID",
            "Cohort_Role",
            "Learner_Status",
            "Analysis_Stratum",
            "Learner_Point",
            "Learner_Conservative",
            "Mahalanobis_Distance",
            "Classifier_Run_ID",
        ]
    ],
    on=["Experiment_ID", "Condition", "Fish_ID"],
    how="left",
    validate="many_to_one",
    indicator=True,
)
```

The `many_to_one` validation is essential: there are many time points or trials per fish but exactly one classification row per fish.

### Missing matches

Rows without a classification match must become:

```text
Learner_Status = Unclassified
Analysis_Stratum = Unmatched
```

They must also be reported. Do not silently drop them.

### Required join report

Generate:

```text
Number of source fish
Number of manifest fish
Number matched
Number unmatched in source
Number present only in manifest
Duplicate key count
Fish count by condition
Fish count by learner status
Fish count by analysis stratum
```

Save the unmatched IDs to a CSV for inspection.

### Tables to label

Apply the same manifest to:

1. Condition-level CS-aligned time-series tables
2. Condition-level US-aligned time-series tables
3. Pooled per-trial normalized-vigor tables
4. Panel-ready fish-level summary tables

Do not independently recreate learner labels inside each plotting script.

## 4. Define analysis populations explicitly

Create named, reusable filters.

### Reference population

```python
reference = labeled[labeled["Analysis_Stratum"] == "Reference"]
```

### Conditioned learners

```python
conditioned_learners = labeled[
    labeled["Analysis_Stratum"] == "Conditioned learner"
]
```

### Conditioned non-learners

```python
conditioned_nonlearners = labeled[
    labeled["Analysis_Stratum"] == "Conditioned non-learner"
]
```

### Unclassified fish

```python
conditioned_unclassified = labeled[
    labeled["Analysis_Stratum"] == "Conditioned unclassified"
]
```

Always report unclassified fish. The primary learner/non-learner panels may omit them visually, but captions and sample-flow tables must state how many were omitted and why.

### Recommended comparisons

Descriptive:

- Conditioned learners versus conditioned non-learners
- Conditioned learners versus reference
- Conditioned non-learners versus reference
- All conditioned fish versus reference

Sensitivity:

- Conservative labels
- Point-estimate labels
- Eligible fish only
- All technically valid fish with unclassified shown separately

## 5. Preserve multiple behavioral outcomes

Because vigor outside bouts is currently represented as missing, the learner-stratified analysis should not rely on conditional vigor alone.

For every time window, trial, block, and phase, derive:

### Movement occurrence

```text
Any movement in window: yes/no
```

### Fraction of time moving

```text
number of samples classified as bout / total valid samples
```

### Bout count

```text
number of detected bouts in window
```

### Conditional bout vigor

```text
mean or median vigor among samples inside bouts
```

### Integrated movement

If scientifically justified:

```text
area under a zero-inclusive vigor curve
```

This incorporates both movement occurrence and intensity, but it must be clearly distinguished from conditional vigor.

### Scaled vigor

Use only after the canonical scaling definition and baseline interval are corrected and documented.

### Normalized vigor

Retain as a trial summary, but do not use it as the only learner-stratified outcome.

## 6. Define temporal resolutions

The analysis should support the following levels.

## A. Individual fish, individual trial

Purpose:

- Inspect the raw temporal pattern
- Confirm stimulus alignment
- Identify whether group results reflect representative fish
- Detect missingness or isolated bouts

For each selected fish and trial, plot:

- Raw or smoothed tail angle
- Bout indicator
- Conditional vigor
- Zero-inclusive movement signal if defined
- CS interval
- Expected and actual US times
- Baseline and conditioned-response windows
- Learner status and classifier evidence

Use the same axis limits for learner and non-learner examples where comparison is intended.

## B. Single trial across fish

Purpose:

- Compare temporal profiles at a specific trial number
- Show the development and extinction of the response

Candidate trials:

- Late pre-training trial
- Early training trial
- Training catch trials
- Late training trial
- Early test trial
- Late test trial

For each `Analysis_Stratum`:

1. Bin time within each fish and trial.
2. Retain one value per fish per time bin.
3. Calculate the group median or another prespecified summary across fish.
4. Compute fish-level bootstrap confidence intervals.

Do not treat time samples or fish-trial rows as independent animals.

## C. Training catch trials

The delay configuration identifies training catch trials such as:

```text
25, 39, 53, 59
```

Verify the exact numbering after all offsets and preprocessing conventions.

Plot each catch trial separately:

- One row or column per catch trial
- Reference, learner, and non-learner profiles
- Shared x and y scales
- Fish count for each panel
- Pre-CS baseline
- CS interval
- Expected US time, even though the US is omitted

Catch trials are useful because they reduce contamination by the unconditioned response.

However, if catch-trial behavior contributes to classifier features, the circularity caveat still applies.

## D. Grouped trials

Aggregate neighboring trials into prespecified groups:

- Five-trial blocks
- Ten-trial blocks where historically used
- Early and late parts of each phase
- Catch-trial groups when appropriate

Recommended calculation:

1. Bin the temporal signal.
2. Aggregate trials within each fish first.
3. Aggregate the resulting fish-level curves across fish second.

This prevents fish with more valid trials from receiving greater weight.

For example:

```text
fish + block + time bin
    -> median across trials
analysis stratum + time bin
    -> median across fish
```

## E. Phase-level profiles

Create profiles for:

- Pre-training
- Early training
- Late training
- Early test
- Late test
- Re-training or retention phases where present

Use phase definitions from the experiment configuration, not hard-coded plot-specific trial lists.

Phase plots should display:

- Temporal median
- Fish-level confidence interval
- Number of fish
- Median valid trials per fish
- Movement probability or coverage beneath the vigor trace

## F. Trial-by-trial scalar summaries

Alongside temporal curves, plot by trial:

- Normalized vigor
- Baseline vigor
- Response-window vigor
- Movement probability
- Fraction of time moving
- Bout count

Display:

- Individual fish trajectories with low opacity
- Group median or estimated marginal mean
- Fish-level uncertainty
- Phase boundaries
- Catch-trial markers

## 7. Time binning and within-fish aggregation

### Time bins

Use explicit bins in seconds, for example:

```text
0.5 s for detailed profiles
1.0 s for compact major-figure panels
```

The bin width must be stored in the analysis configuration and output filename/provenance.

### Required aggregation order

For grouped-trial profiles:

```text
frame/sample
    -> time-bin value for each fish and trial
    -> trial-group value for each fish and time bin
    -> group summary across fish and time bin
```

Never pool all frames from all fish directly. That would overweight:

- Fish with more bouts
- Fish with more valid samples
- Fish with more valid trials

### Uncertainty

Use:

- Fish-level bootstrap for a single selected trial
- Hierarchical bootstrap for grouped trials:
  - Resample fish
  - Resample trials within each sampled fish

Use at least 1,000 bootstrap samples for development and preferably 5,000-10,000 for final manuscript intervals.

Store the bootstrap seed and method.

## 8. Proposed temporal-profile outputs

## Output 1: Classification and cohort table

One row per fish:

```text
Condition
Fish_ID
Classification eligibility
Learner status
Analysis stratum
Point and conservative labels
Distance
Votes
Features and uncertainty
Number of valid trials per phase
Exclusion status
```

## Output 2: Learner-status sample flow

For every condition:

```text
Raw fish
Technically valid fish
Behaviorally included fish
Classification eligible fish
Learners
Non-learners
Unclassified
```

## Output 3: Individual-fish atlas

For every classified conditioned fish:

- Trial-by-time heatmap
- Bout/movement-probability heatmap
- Normalized-vigor trajectory
- Learner label
- Classification distance and votes

Save learners and non-learners in separate directories and also generate a combined ordered grid.

## Output 4: Single-trial temporal profiles

For prespecified trials:

- Reference
- Conditioned learner
- Conditioned non-learner

Include vigor and movement occurrence.

## Output 5: Catch-trial profiles

Separate panels for every training catch trial and selected test trials.

## Output 6: Five-trial block profiles

For key blocks:

- Late pre-training
- Early training
- Late training
- Early test
- Late test

## Output 7: Phase-level temporal profiles

One panel per phase or a heatmap with:

- Rows: phases or trial blocks
- Columns: time relative to CS/US
- Separate facets: learner status

## Output 8: Trial-by-trial trajectories

Plot scalar behavioral outcomes over trial number for each analysis stratum.

## Output 9: Difference curves

Exploratory difference profiles:

```text
Conditioned learner - Reference
Conditioned non-learner - Reference
Conditioned learner - Conditioned non-learner
```

Use simultaneous or appropriately corrected uncertainty bands if making time-localized inferential claims.

## 9. Plot design

### Recommended color semantics

Keep experimental condition and learner status visually separable.

Option A:

- Color represents condition.
- Line style or saturation represents learner status.

Option B:

- Facet by learner status.
- Retain the same condition color in every facet.

Option B is safer for the paper because condition colors already have established meanings.

Recommended facets:

```text
Reference | Conditioned learner | Conditioned non-learner
```

Do not assign the reference group a learner/non-learner visual identity.

### Required plot annotations

- CS onset and duration
- Expected US onset
- Actual US interval when present
- Baseline interval
- Conditioned-response interval
- Trial or block name
- Number of fish
- Number of trials contributing
- Learner-rule name in the caption

### Coverage display

Temporal vigor curves can look stable even when few fish contribute at some times. Add:

- A coverage strip
- A small contributing-fish panel
- Or opacity/masking below a prespecified coverage threshold

For each time bin report:

```text
number of contributing fish
fraction of stratum contributing
```

## 10. Statistical analysis strategy

## Descriptive analysis using the classification data

Allowed:

- Plotting the temporal profiles used to understand what the classifier selected
- Reporting effect sizes descriptively
- Showing classifier features and trajectories
- Comparing point and conservative classifications

Required wording:

```text
These analyses characterize groups defined from the same behavioral data and are not independent tests of learner-group differences.
```

## Confirmatory option A: Held-out phases

Classify using training data and evaluate on held-out test trials, or classify using acquisition and evaluate a distinct retention/retraining phase.

The held-out outcome must not contribute to the classifier.

## Confirmatory option B: Held-out trials

Predefine:

- Classification trial set
- Evaluation trial set

For example:

```text
Classification: pre-training and selected training trials
Evaluation: held-out catch trials and test trials
```

This design must account for whether the classifier still measures the intended acquisition phenotype.

## Confirmatory option C: Cross-fitting

For each fish or trial fold:

1. Fit classifier thresholds and reference statistics without the evaluation fold.
2. Assign the label using training-fold information.
3. Evaluate behavior in the held-out fold.
4. Combine held-out predictions and outcomes.

Avoid using a fish's evaluation response to generate its label for that same evaluation.

## Confirmatory option D: Independent experiment

Freeze the classifier on the current dataset, then apply it unchanged to a new cohort.

This is the strongest validation.

## Suggested models

For time-resolved behavior, avoid separate uncorrected tests at every time bin.

Consider:

- Functional data summaries over prespecified windows
- Generalized additive mixed models
- Mixed-effects models with time basis functions
- Cluster-based permutation procedures
- Prespecified area-under-curve summaries

Always include fish as the biological repeated-measures unit.

For movement occurrence:

- Binomial mixed model

For bout count:

- Poisson or negative-binomial mixed model

For conditional vigor:

- Appropriate transformed Gaussian or robust mixed model

For combined behavior:

- Two-part/hurdle model

## 11. Proposed modules

Suggested files:

```text
figures/
`-- panels/
    `-- learner_temporal_profiles.py

learner_analysis/
|-- __init__.py
|-- classification_manifest.py
|-- label_join.py
|-- temporal_profiles.py
|-- trial_groups.py
|-- statistics.py
|-- validation.py
`-- export.py
```

### `classification_manifest.py`

- Convert canonical classifier output into the stable manifest
- Create categorical statuses
- Store provenance
- Validate uniqueness

### `label_join.py`

- Attach one fish-level classification to many time-series/trial rows
- Enforce many-to-one cardinality
- Report unmatched and duplicate IDs
- Save derived labeled datasets

### `trial_groups.py`

- Resolve trial lists from `experiment_configuration.py`
- Define pre-training, training, catch, test, and phase groups
- Avoid hard-coded offsets scattered across plotting scripts

### `temporal_profiles.py`

- Time binning
- Within-fish trial aggregation
- Across-fish summaries
- Contribution counts
- Fish-level and hierarchical bootstrap

### `statistics.py`

- Descriptive versus held-out analysis modes
- Movement occurrence model
- Conditional-vigor model
- Multiple-comparison handling
- Effect-size tables

### `validation.py`

- Key uniqueness
- Join coverage
- Expected fish counts
- Expected trials
- Status counts
- No unclassified-to-non-learner coercion
- No classification/evaluation leakage in confirmatory mode

## 12. Configuration

Create an immutable configuration:

```python
@dataclass(frozen=True)
class LearnerStratifiedAnalysisConfig:
    experiment: str
    alignment: str
    classifier_run_id: str
    primary_rule: str = "conservative"
    analysis_mode: str = "descriptive"
    time_bin_s: float = 0.5
    bootstrap_iterations: int = 5000
    bootstrap_seed: int = 10
    include_unclassified: bool = True
    single_trials: tuple[int, ...] = ()
    catch_trials: tuple[int, ...] = ()
    trial_groups: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = (
        "movement_probability",
        "fraction_time_moving",
        "conditional_vigor",
    )
```

Allowed `analysis_mode` values:

```text
descriptive
held_out_trials
held_out_phase
cross_fitted
independent_validation
```

The selected mode must be included in every output filename and provenance file.

## 13. Output structure

```text
Processed data/
`-- Learner stratified analysis/
    `-- <classifier_run_id>/
        |-- manifests/
        |   |-- fish_classification_manifest.csv
        |   |-- join_report.csv
        |   `-- unmatched_fish.csv
        |-- labeled_data/
        |   |-- all_fish_CS_labeled.pkl
        |   |-- all_fish_US_labeled.pkl
        |   `-- normalized_vigor_trials_labeled.pkl
        |-- panel_data/
        |-- statistics/
        |-- figures/
        |   |-- individual_fish/
        |   |-- single_trials/
        |   |-- catch_trials/
        |   |-- five_trial_blocks/
        |   |-- phases/
        |   `-- trial_trajectories/
        `-- provenance/
```

## 14. Validation checklist

Before plotting:

- [ ] One canonical classifier implementation is selected.
- [ ] Classifier configuration is frozen and hashed.
- [ ] Every manifest key is unique.
- [ ] Every labeled row has at most one classification.
- [ ] Unmatched fish are reported.
- [ ] Unclassified fish are not converted to non-learners.
- [ ] Fish counts match the cohort manifest.
- [ ] Trial groups match experiment configuration.
- [ ] Catch-trial numbering is verified.
- [ ] CS and US alignments are correct.
- [ ] Time-bin widths and boundaries are explicit.
- [ ] Within-fish aggregation occurs before across-fish aggregation.
- [ ] Bootstrap resampling uses fish as the biological unit.
- [ ] Coverage is recorded for every temporal bin.
- [ ] Descriptive and confirmatory modes are clearly distinguished.
- [ ] Held-out analyses contain no label/outcome leakage.

Before manuscript use:

- [ ] Learner rule is stated in the caption.
- [ ] Classification eligibility is reported.
- [ ] Learner, non-learner, reference, and unclassified counts are shown.
- [ ] Point-rule sensitivity is available.
- [ ] Movement occurrence is analyzed alongside conditional vigor.
- [ ] No circular learner/non-learner p-value is presented as independent validation.
- [ ] Figure provenance identifies classifier run and input data.

## 15. Implementation phases

## Phase 1: Freeze and validate classification

1. Select the canonical learner script.
2. Resolve classifier-related analysis issues.
3. Export classification results for every eligible fish.
4. Create the canonical classification manifest.
5. Add control/reference and unclassified statuses.
6. Generate a classification sample-flow table.

### Exit criterion

Every fish has exactly one auditable status or an explicit reason for being unclassified.

## Phase 2: Add labels to derived datasets

1. Add stable experiment and condition identifiers to source tables.
2. Join the classification manifest to CS time-series data.
3. Join it to US time-series data.
4. Join it to normalized-vigor trial summaries.
5. Validate many-to-one cardinality.
6. Save labeled derived copies.

### Exit criterion

All matches, mismatches, duplicates, and status counts are reported and reproducible.

## Phase 3: Build descriptive temporal profiles

1. Implement time binning.
2. Implement within-fish aggregation.
3. Implement fish-level and hierarchical bootstrap.
4. Produce individual-fish atlases.
5. Produce selected single-trial profiles.
6. Produce catch-trial profiles.
7. Produce five-trial-block profiles.
8. Produce phase-level profiles.
9. Produce trial-by-trial outcome trajectories.

### Exit criterion

Learner, non-learner, and reference temporal profiles regenerate from explicit labeled data and show contribution counts.

## Phase 4: Add complementary movement outcomes

1. Movement occurrence
2. Fraction of time moving
3. Bout count
4. Conditional vigor
5. Optional integrated zero-inclusive movement

### Exit criterion

The interpretation no longer depends solely on vigor conditional on movement.

## Phase 5: Add non-circular validation

Choose one:

- Held-out catch trials
- Held-out test phase
- Cross-fitting
- Independent cohort

Implement leakage checks and save held-out predictions.

### Exit criterion

At least one learner-stratified result is evaluated on behavior that did not define the corresponding label.

## Phase 6: Build paper panels

1. Integrate the analysis with `SCIENTIFIC_FIGURE_PIPELINE_PLAN.md`.
2. Create reusable learner-profile panel renderers.
3. Compose:
   - Classification flow
   - Feature space
   - Individual examples
   - Single-trial temporal profiles
   - Block/phase profiles
   - Held-out validation
4. Export SVG, PDF, and raster previews.
5. Run scientific and accessibility QC.

## 16. Recommended first pilot

Use the delay-conditioning experiment and CS-aligned data.

### Pilot groups

- Delay learners
- Delay non-learners
- Delay unclassified
- Matched unpaired controls as reference

### Pilot panels

1. Classification sample flow
2. Classifier feature space
3. One learner and one non-learner individual heatmap
4. Late pre-training temporal profile
5. Late-training catch-trial temporal profile
6. Early-test temporal profile
7. Late-test temporal profile
8. Trial-by-trial movement probability
9. Trial-by-trial conditional vigor

### Pilot purpose

Determine:

- Whether labels join correctly
- Whether the selected groups have adequate sample size
- Whether differences reflect movement probability, bout vigor, or both
- Whether temporal suppression appears before the expected US
- Whether apparent differences persist outside the classifier-defining summaries

## 17. Definition of done

The learner-stratified analysis is complete when:

- [ ] The post-refactor end-to-end methodology review is recorded.
- [ ] Continuous, longitudinal, component-based, and class-based alternatives
      have been considered.
- [ ] Power or simulation evidence documents feasible validation and
      uncertainty, or explains why it is not applicable.
- [ ] One canonical classification implementation is used.
- [ ] A versioned fish-level classification manifest exists.
- [ ] Every fish has a learner, non-learner, reference, or unclassified status.
- [ ] Labels are joined with validated many-to-one cardinality.
- [ ] Original datasets remain unchanged.
- [ ] Labeled CS, US, and trial-summary datasets are saved as derived artifacts.
- [ ] Individual-trial profiles are available.
- [ ] Selected single-trial group profiles are available.
- [ ] Catch-trial profiles are available.
- [ ] Five-trial-block profiles are available.
- [ ] Phase-level profiles are available.
- [ ] Trial-by-trial trajectories are available.
- [ ] Movement occurrence and conditional vigor are both analyzed.
- [ ] Temporal coverage and contributing-fish counts are shown.
- [ ] Fish-level or hierarchical bootstrap intervals are used.
- [ ] Descriptive circular analyses are labeled as such.
- [ ] At least one held-out or independent validation is implemented before confirmatory claims.
- [ ] Every figure records classifier run, cohort, inputs, and analysis settings.

## Immediate next steps

1. Complete the non-learner refactor and freeze corrected inputs, cohort,
   outcomes, and primary inference.
2. Hold the end-to-end learner-methodology workshop.
3. Define the learner estimand and whether classification is scientifically
   necessary.
4. Design power/simulation and independent-validation requirements before
   comparing candidate methods.
5. Evaluate inherited classifiers alongside PCA-informed diagnostics,
   continuous scores, and longitudinal/model-based alternatives.
6. Select and freeze a canonical method only if the evidence supports one.
7. Enable and inspect a versioned evidence-rich classification export.
8. Build and validate the fish-level classification manifest.
9. Join labels using `validate="many_to_one"` and generate sample-flow reports.
10. Produce validation-mode-specific temporal profiles and inference.
