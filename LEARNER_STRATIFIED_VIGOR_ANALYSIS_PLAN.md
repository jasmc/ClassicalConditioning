# Learner-Stratified Vigor Analysis Plan

## Documentation baseline

This plan describes:

- `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
- Source feature commits `505e29a` and `88fe4fc`
- Cherry-picked equivalents `4d6939f` and `f372101`
- Working-tree LogMedian compatibility adaptations dated 2026-08-28

Unlike the earlier plan, this baseline contains one LogMedian learner
classifier and an implemented initial learner-stratified pipeline.

## Objective

Label each fish as:

- Reference
- Conditioned learner
- Conditioned non-learner
- Conditioned unclassified

Then characterize the temporal profile of:

- Conditional vigor
- Movement probability
- Eventually fraction time moving and bout count

at:

- Individual fish/trial level
- Selected single trials
- Catch trials
- 10-trial blocks
- 5-trial blocks
- Experimental phases
- CS and US alignments

## Current implementation

`learner_stratified_vigor_pipeline.py` currently implements:

- Explicit control and `3sTrace` input paths
- Per-condition checkpoints
- Median baseline and CR trial summaries
- Log-space subtraction normalized vigor
- Condition-qualified classifier fish keys
- LogMedian learner classifier invocation
- Source fish-ID restoration
- Fish-level classification manifest
- Reference/learner/non-learner/unclassified strata
- Many-to-one manifest join
- Fish-first block profiles
- Fish-first catch-trial profiles
- Conditional vigor
- Movement probability
- Catch-trial timing windows
- Near-US strengthening and anticipatory-suppression metrics
- Pooled catch-trial suppression plots
- Coverage tables
- SVG and PNG figures
- JSON provenance
- Synthetic tests

## Scientific interpretation

The current mode is:

```text
descriptive_classifier_characterization
```

The same experiment contributes to:

1. Learner classification
2. Learner-stratified temporal profiles

Therefore learner/non-learner differences are expected by construction and are
not independent evidence that the classification generalizes.

## Canonical label source

Use only:

```text
6_LearnersQuantification_LogMedian.py
```

with:

- Experiment `all3sTrace`
- Conditions `control` and `3sTrace`
- Median log-vigor differences
- Directional joint statistic
- Approved classifier alpha

Do not import or recreate labels from removed WIP/standard classifier files.

## Fish identity

The current pipeline protects against identical source fish IDs appearing in
different conditions by creating:

```text
<condition>::<fish_id>
```

for classification, then restoring the source ID through a validated key map.

The stable manifest key is:

```text
Condition + Fish_ID
```

For future multi-experiment analysis, extend it to:

```text
Experiment_ID + Condition + Fish_ID
```

## Classification manifest

Current key fields:

- `Condition`
- `Fish_ID`
- `Classification_Eligible`
- `Learner_Primary`
- `Learner_Status`
- `Analysis_Stratum`
- `Cohort_Role`
- `Classifier_Name`
- `Classifier_Run_ID`
- `Classifier_Alpha`
- `Classification_Alignment`

Retain classifier evidence:

- BLUP features
- Feature SE
- Directional z-scores
- Joint statistic
- Empirical p-value
- Votes

Do not convert unclassified fish to non-learners.

## Temporal analysis order

For grouped profiles:

```text
sample
  -> fish/trial/time-bin median or movement probability
  -> median across trials within fish
  -> median and bootstrap CI across fish
```

This gives each fish equal weight.

## Existing outputs

### Manifests

- Raw classifier output
- Fish-key map
- Fish classification manifest
- Classification sample flow
- Unmatched fish report on failure

### Panel data

- Conditional vigor by 10-trial block
- Movement probability by 10-trial block
- Conditional vigor by catch trial
- Movement probability by catch trial
- Coverage tables
- Per-fish catch timing metrics
- Catch timing summaries
- Pooled anticipatory-suppression summaries

### Figures

- Block profiles
- Catch-trial profiles
- Catch timing/suppression figures

## Required extensions

## 1. Individual fish atlas

For every classified conditioned fish, generate:

- Trial-by-time LogMedian heatmap
- Movement-probability heatmap
- Trial summary trajectory
- Learner status
- Classifier features and uncertainty

Create separate learner, non-learner, and unclassified indexes.

## 2. Single-trial profiles

Predefine representative trials:

- Late pre-training
- Early training
- Each training catch trial
- Late training
- Early test
- Late test

Use the same axes across strata.

## 3. Five-trial block profiles

The classifier already derives 5-trial feature blocks. Export the same mapping
as a canonical trial map and use it for temporal profiles.

Do not independently reconstruct block names in plotting code.

## 4. Phase profiles

Add:

- Pre-training
- Early training
- Late training
- Early test
- Late test

Aggregate trials within fish before the group summary.

## 5. US-aligned profiles

Add explicit US-aligned control and `3sTrace` inputs.

Verify:

- Expected versus actual US timing
- Catch trials have no actual US
- CR and trace windows are alignment-specific

## 6. Complementary outcomes

Add:

- Fraction time moving
- Bout count
- Optional zero-inclusive integrated movement

Keep conditional vigor separate from occurrence.

## 7. Coverage policy

For every profile bin, report:

- Contributing fish
- Fraction of stratum
- Median contributing trials per fish

Predefine whether low-coverage bins are masked, faded, or omitted.

## 8. Classification stability

Evaluate:

- Observed control learner rate
- Label stability under fish bootstrap
- Alpha sensitivity
- Feature-window sensitivity
- Per-fish SE versus control-SD scoring

## Non-circular validation

Before confirmatory paper claims, implement one:

### Held-out trials

Classify on one trial set and evaluate prespecified unused catch/test trials.

### Held-out phase

Classify acquisition without using a later retention/retraining outcome, then
evaluate that phase.

### Cross-fitting

Generate labels and evaluate outcomes in separate folds.

### Independent cohort

Freeze the classifier and apply it unchanged to new fish.

## Statistical models

Avoid uncorrected tests at every time bin.

Recommended outcome-specific models:

| Outcome | Candidate model |
| --- | --- |
| Movement occurrence | Binomial mixed model |
| Bout count | Negative-binomial mixed model |
| Conditional log-vigor | Robust/Gaussian mixed model after diagnostics |
| Combined occurrence/intensity | Two-part or hurdle model |

Use fish as the repeated-measures unit.

## Artifact improvements

Current implementation uses pickle checkpoints and CSV tables.

Migrate to:

- Parquet for temporal and trial tables
- Parquet plus CSV for manifests
- JSON for configuration and provenance

Add trusted input hashes. Current size/mtime checkpoint signatures are not
strong content identity.

## Figure set for the paper

Recommended learner-characterization figure:

1. Classification sample flow
2. Feature-space classifier panel
3. Learner and non-learner example heatmaps
4. Late pre-training profile
5. Catch-trial profiles
6. Early-test profile
7. Late-test profile
8. Movement probability across trials
9. Conditional vigor across trials
10. Held-out validation panel

The caption must state:

- Classifier
- Alpha
- Features
- Cohort
- Number of fish
- Whether evaluation is descriptive or held out

## Implementation phases

### Phase 1: validate integrated pipeline

- Run full tests in the pinned environment.
- Run on explicit paper-scope LogMedian inputs.
- Verify fish counts and conditions.
- Inspect classifier diagnostics.

### Phase 2: canonical artifacts

- Add experiment ID.
- Add schema versions.
- Add content hashes.
- Convert checkpoints to Parquet.

### Phase 3: profile expansion

- Individual atlas
- Single trials
- Five-trial blocks
- Phases
- US alignment
- Complementary outcomes

### Phase 4: statistical validation

- Label stability
- Sensitivity analyses
- Held-out or cross-fitted analysis

### Phase 5: paper figures

- Reusable renderers
- Major-figure composition
- Provenance
- Accessibility and scientific review

## Definition of done

- [ ] One classifier version is frozen.
- [ ] Every fish has a reference, learner, non-learner, or unclassified status.
- [ ] Joins are validated many-to-one.
- [ ] Individual, single-trial, block, phase, and alignment profiles exist.
- [ ] Movement occurrence and conditional vigor are both reported.
- [ ] Coverage is visible.
- [ ] Fish-level/hierarchical bootstrap is used.
- [ ] Descriptive analyses are labeled as circular.
- [ ] A held-out or independent validation exists for confirmatory claims.
- [ ] Artifacts and figures record provenance.

