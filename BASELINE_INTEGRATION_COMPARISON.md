# Baseline and Learner-Branch Integration Comparison

## Documentation baseline

This comparison describes:

- LogMedian baseline:
  `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
- Current learner-stratified source branch:
  `experiment/learner-stratified-vigor` at
  `88fe4fccfed1c21d05a59772a16fd5534fb0683f`
- Learner feature commits:
  `505e29adead4a31e79fe37e986f3d3cbad731a8b` and
  `88fe4fccfed1c21d05a59772a16fd5534fb0683f`
- Integrated branch:
  `agents/bf46-learner-stratified`
- Cherry-picked equivalents:
  `4d6939f` and `f372101`
- Working-tree LogMedian compatibility adaptations dated 2026-08-28

## Branch relationship

The histories diverged:

```text
b0dfcb3
    |
    +--> 505e29a --> 88fe4fc  learner-stratified source branch

separate LogMedian line
    |
    +--> ... --> bf46bf7
```

`bf46bf7` is not an ancestor of the source learner-stratified branch. The two
feature commits were therefore transplanted onto a new branch created exactly
at `bf46bf7`.

## Clean worktree

Worktree:

```text
C:\Users\Public\More projects\ClassicalConditioning.worktrees\bf46-learner-stratified
```

Branch:

```text
agents/bf46-learner-stratified
```

Initial worktree commit:

```text
bf46bf7b02baaf6d8138d9772f255881f86c3c78
```

The original worktrees were not modified to create this branch.

## What `bf46bf7` changes relative to `b0dfcb3`

The LogMedian line:

- Renames step 3 to `3_FishGrouping_LogMedian.py`.
- Renames step 4 to `4_ScaledVigorPlotting_LogMedian.py`.
- Renames step 5 to `5_NormalizedVigorPlotting_LogMedian.py`.
- Consolidates learner analysis into
  `6_LearnersQuantification_LogMedian.py`.
- Removes the unsuffixed, `new`, `improved`, and `WIP` learner scripts.
- Removes the legacy `my_*` module family.
- Adds `pipeline_utils.py`.
- Adds `heatmap_utils.py`.
- Adds `open_scaled_vigor_df.py`.
- Adds `0_Pipeline_Data_Flow_Description.txt`.
- Adds `test_discarded_heatmap_grid.py`.
- Updates steps 1 and 2 and shared utilities.

The analysis changes from a standard mean/ratio family toward:

- Rolling median
- Log vigor
- Baseline median subtraction
- Median CR and baseline summaries
- Log-space difference normalized vigor
- Zero-centered figures and learner features

## What the learner-stratified commits add

The two source commits add:

| File | Purpose |
| --- | --- |
| `learner_stratified_vigor_pipeline.py` | Classification manifest and temporal/catch analysis |
| `run_learner_stratified_vigor.ps1` | Pinned Windows runner |
| `requirements-learner-vigor.txt` | Pinned scientific dependencies |
| `test_learner_stratified_vigor_pipeline.py` | Synthetic and integration tests |
| `.gitignore` update | Ignore derived learner build/environment outputs |

The four added feature files were byte-identical on the source and integrated
branches immediately after cherry-picking.

## Why clean cherry-picks were not sufficient

The source feature commits were written against `b0dfcb3` and assumed:

```text
6_LearnersQuantification_WIP.py
Mean baseline
Mean CR
Normalized vigor = CR / baseline
condition ID = trace
```

The LogMedian baseline provides:

```text
6_LearnersQuantification_LogMedian.py
Median baseline log-vigor
Median CR log-vigor
Normalized vigor = Median CR - Median baseline
condition ID = 3sTrace
```

Git reported no textual conflict because the feature files were new. The
integration conflict was semantic and only became visible through code
inspection and tests.

## Compatibility adaptations

The current working tree adapts the transplanted pipeline to:

1. Import `6_LearnersQuantification_LogMedian.py`.
2. Build `Median 15 s before`.
3. Build `Median CR`.
4. Calculate normalized vigor by subtraction.
5. Use `3sTrace` in classifier inputs and manifests.
6. Use LogMedian classifier names and provenance.
7. Use a default `3sTrace_CS_new_logmedian.pkl` path in the runner.
8. Correct `RESPONSE_COLUMN_NAME` in the standalone LogMedian classifier from
   `Mean CR` to `Median CR`.
9. Update synthetic and classifier-integration tests.
10. Update `0_Pipeline_Data_Flow_Description.txt` to label the integrated
    baseline, describe conditioned suppression, correct the LogMedian
    subtraction direction, and include step 6 plus the optional integrated
    stage.

## Preserved learner-stratified behavior

The integration preserves:

- Explicit source paths
- Condition checkpoints
- Pinned environment
- Legacy pandas pickle compatibility
- Condition-qualified fish keys
- Many-to-one classification joins
- Reference/learner/non-learner/unclassified strata
- Fish-first temporal aggregation
- Fish-level bootstrap intervals
- Conditional vigor
- Movement probability
- Catch-trial profiles
- Catch timing metrics
- SVG/PNG output
- CSV manifest/panel output
- JSON provenance
- Circularity warning

## Validation

In the pinned environment:

```text
15 learner-stratified tests passed
3 experiment-configuration tests passed
```

The classifier integration test ran rather than being skipped.

The tests validate code integration and synthetic behavior. They do not
validate conclusions on experimental data.

## Remaining differences between branches

The integrated branch contains the LogMedian pipeline from `bf46bf7`; the
source learner-stratified branch still contains the older standard pipeline.

Therefore the full branch diff remains large and intentionally includes:

- LogMedian file renames
- Removal of competing classifier files
- Removal of legacy modules
- Shared utility additions
- Updated step-1 and step-2 code

The goal is not to merge every source-branch file. The goal is:

```text
bf46bf7 LogMedian baseline
+ learner-stratified feature behavior
+ compatibility adaptations
```

## Documentation consequence

Documentation written against `b0dfcb3` must not be reused unchanged.

The integrated documents:

- Treat LogMedian step 3-6 files as current.
- Treat the learner pipeline as an orchestrator, not another classifier.
- Remove the obsolete four-classifier choice.
- Retain only architecture recommendations that still apply.
- Label the exact commit baseline.
