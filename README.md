# Classical Conditioning Analysis

Python analysis code for a head-fixed larval zebrafish classical-conditioning
study. The pipeline synchronizes high-speed tail tracking with conditioned
stimulus (CS) and unconditioned stimulus (US) events, detects movement bouts,
quantifies vigor, compares experimental conditions, classifies individual
learners, and generates scientific figures.

## Documentation baseline

This document describes:

- LogMedian baseline commit:
  `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
- Learner-stratified source commits:
  `505e29adead4a31e79fe37e986f3d3cbad731a8b` and
  `88fe4fccfed1c21d05a59772a16fd5534fb0683f`
- Their cherry-picked commits on branch `agents/bf46-learner-stratified`:
  `4d6939f` and `f372101`
- Working-tree compatibility adaptations made on 2026-08-28:
  - Use `6_LearnersQuantification_LogMedian.py`, not the removed WIP file.
  - Build median log-vigor differences, not mean vigor ratios.
  - Use the configured condition ID `3sTrace`, not `trace`.

It does not describe the older `b0dfcb3` standard/ratio pipeline.

## Scientific overview

The assay tests whether larval zebrafish learn the temporal relationship
between a visual CS and an aversive optovin-mediated US.

Principal paradigms include:

- Delay conditioning: CS and US overlap.
- Trace conditioning: a stimulus-free interval separates CS and US.
- Unpaired control: exposure is matched without a stable CS-US contingency.

Learning is evaluated as predictive suppression of tail-movement vigor. The
current paper focuses on delay and short-trace conditioning, including a
3-second trace interval.

## End-to-end pipeline

```text
Experiment configuration
        |
Raw camera, tracking, and stimulus logs
        |
        v
1. Preprocess each fish
        |
        +--> QC plots and protocol plots
        +--> discard/exclusion review
        |
        v
Per-fish processed pickle files
        |
        +--> 2. Optional example-fish figures
        |
        v
3. Group fish with LogMedian processing
        |
        v
Condition-level CS/US LogMedian files
        |
        +--> 4. Scaled/log-vigor temporal figures
        |
        +--> 5. Median trial summaries and population statistics
                      |
                      v
             6. LogMedian learner classification
                      |
                      v
       Learner-stratified 3-second-trace profiles
```

See `0_Pipeline_Data_Flow_Description.txt` for the tracked baseline description
and `ANALYSIS_FILES_INDEX.md` for the integrated file-by-file index.

## Analysis stages

### Step 1: preprocessing and quality control

File:

```text
1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py
```

Sub-stages:

- `run_preprocess()`: read, synchronize, filter, calculate vigor, detect bouts,
  identify trials, and save one processed table per fish.
- `run_plot_individual_trials()`: per-fish QC plots.
- `run_plot_protocols()`: verify stimulus timing.
- `run_discard()`: apply inclusion criteria and move/record excluded fish.

Control flags:

```python
RUN_PREPROCESS = False
RUN_PLOT_INDIVIDUALS = True
RUN_PLOT_PROTOCOLS = False
RUN_DISCARD = False
```

Run the sub-stages deliberately. In particular, review QC before running the
discard workflow.

### Step 2: example-fish figures

File:

```text
2_ExampleFishPlotting.py
```

Produces selected:

- Tail-angle and vigor traces
- Individual-trial heatmaps
- Bout close-ups
- Tail-trajectory plots

This is an optional publication-figure branch. Later analytical stages do not
depend on its outputs.

### Step 3: LogMedian fish grouping

File:

```text
3_FishGrouping_LogMedian.py
```

For each fish and trial:

1. Mask vigor outside detected bouts.
2. Apply a trailing rolling median.
3. Downsample.
4. Log-transform positive vigor.
5. Subtract a per-trial baseline median.
6. Concatenate fish by condition and alignment.

Outputs use names such as:

```text
<condition>_CS_new_logmedian.pkl
<condition>_US_new_logmedian.pkl
```

### Step 4: LogMedian temporal profiles

File:

```text
4_ScaledVigorPlotting_LogMedian.py
```

Builds and renders:

- Contribution/count heatmaps
- LogMedian vigor heatmaps
- Selected catch-trial profiles
- Pooled catch-trial profiles
- Trial-block profiles

The `Scaled vigor (AU)` column in this branch is a baseline-centered log-vigor
quantity. It is not the percentile-scaled ratio used by the older standard
pipeline.

### Step 5: LogMedian normalized-vigor analysis

File:

```text
5_NormalizedVigorPlotting_LogMedian.py
```

For each fish and trial:

```text
Normalized vigor =
    median log-vigor in CR window
    - median log-vigor in baseline window
```

Because subtraction occurs in log space, this corresponds to a log ratio of
raw medians. Zero is the no-change reference.

Outputs include:

- Per-fish, per-trial summary data
- Block line plots and boxplots
- Phase summaries
- Trial-by-trial trajectories
- Nonparametric tests and mixed-effects analyses

### Step 6: LogMedian learner classification

File:

```text
6_LearnersQuantification_LogMedian.py
```

The repository contains one active learner-classification implementation at
this baseline. Earlier unsuffixed, `new`, `improved`, and `WIP` files were
removed on the LogMedian line.

The classifier:

- Uses median log-vigor differences.
- Extracts acquisition and recovery features with mixed models.
- Computes directional feature scores.
- Uses a one-sided joint quadratic statistic.
- Calibrates a threshold from controls.
- Produces one primary `Is_Learner` label.

### Integrated learner-stratified analysis

Files:

```text
learner_stratified_vigor_pipeline.py
run_learner_stratified_vigor.ps1
requirements-learner-vigor.txt
test_learner_stratified_vigor_pipeline.py
```

This optional pipeline is specific to the full 3-second-trace dataset. It:

- Reads explicit control and `3sTrace` CS-aligned LogMedian files.
- Builds median per-trial classifier input.
- Runs the LogMedian classifier.
- Creates a fish-level classification manifest.
- Keeps controls in a `Reference` stratum.
- Labels conditioned fish as learner, non-learner, or unclassified.
- Builds fish-first temporal profiles by 10-trial block and catch trial.
- Analyzes conditional vigor and movement probability.
- Calculates catch-trial timing metrics.
- Exports figures, panel data, manifests, checkpoints, and provenance.

These plots characterize groups defined from the same experiment. They are
descriptive and are not independent confirmation of learner-group differences.

## Configuration

Current experiment definitions are in:

```text
experiment_configuration.py
```

General processing defaults are in:

```text
general_configuration.py
```

Most numbered scripts also contain top-level `RUN_*`, experiment, plotting, and
statistical settings. Verify them before execution.

Important configured condition IDs include:

```text
control
delay
3sTrace
10sTrace
incTrace
```

Use condition IDs from `ExperimentConfig.cond_types`; do not infer them from
display names or raw filename substrings.

## Current storage formats

The tracked pipeline currently relies heavily on pandas pickle files and CSV
exports. The integrated learner pipeline also writes JSON provenance and
checkpoint metadata.

For the planned migration:

- Prefer Parquet for canonical tables.
- Prefer JSON for configuration and provenance.
- Keep CSV as a human-review export.
- Use NPZ for moderate dense numerical arrays.
- Use Zarr only for genuinely large chunked multidimensional arrays.
- Treat existing pickle files as legacy migration inputs.
- Use SVG and PDF for vector figures and PNG for previews.

No experimental data is included in this repository documentation. Raw and
processed data should remain outside source control.

## Running the learner-stratified pipeline

The PowerShell wrapper creates a pinned virtual environment and runs the
pipeline:

```powershell
.\run_learner_stratified_vigor.ps1 `
  -TracePath "D:\path\3sTrace_CS_new_logmedian.pkl" `
  -ControlPath "D:\path\control_CS_new_logmedian.pkl" `
  -OutputPath "D:\path\derived-output"
```

Use `-Force` to rebuild condition checkpoints.

The wrapper installs versions pinned in:

```text
requirements-learner-vigor.txt
```

## Tests

Run the learner-stratified tests in the pinned environment:

```powershell
.\.venv-learner-vigor\Scripts\python.exe -m unittest -v `
  test_learner_stratified_vigor_pipeline.py
```

The repository also contains:

```text
test_discarded_heatmap_grid.py
```

That file is primarily a discarded/included heatmap-grid utility script, not a
comprehensive automated test suite.

## Known limitations

Before using results as final paper evidence, review `ANALYSIS_ISSUES.md`.
Important open concerns include:

- The preprocessing vigor definition does not match the manuscript definition.
- Spatial filtering and the secondary bout threshold are incomplete.
- Excluded files can re-enter grouping through recursive file discovery.
- Stage-3 and stage-5 baseline windows are not the same.
- Vigor is conditional on movement because non-bout samples are missing.
- Missingness filtering is disabled by default in step 5.
- Some statistical bootstrap settings differ between scripts.
- Pickle compatibility and artifact selection remain fragile.
- The learner-stratified analysis is descriptive and circular by design.

## Related documentation

- `ANALYSIS_FILES_INDEX.md`
- `ANALYSIS_ISSUES.md`
- `BASELINE_INTEGRATION_COMPARISON.md`
- `SCIENTIFIC_FIGURE_PIPELINE_PLAN.md`
- `LEARNER_STRATIFIED_VIGOR_ANALYSIS_PLAN.md`
- `CODEBASE_MIGRATION_PLAN.md`
