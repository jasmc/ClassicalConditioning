# Analysis Files Index

## Documentation baseline

This index describes:

- `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
- Source feature commits `505e29a` and `88fe4fc`
- Cherry-picked equivalents `4d6939f` and `f372101`
- Working-tree LogMedian compatibility adaptations dated 2026-08-28

It does not describe the older `b0dfcb3` standard/ratio tree.

## Execution order

| Order | File | Required? | Main input | Main output |
| --- | --- | --- | --- | --- |
| 0 | `experiment_configuration.py` | Yes | Experiment selection | Protocol and condition configuration |
| 0 | `general_configuration.py` | Yes | Shared settings | Processing windows and labels |
| 1A | `1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py` | Yes | Raw camera/tracking/protocol logs | Per-fish processed pickle |
| 1B | Same step-1 file | QC | Per-fish data and raw protocols | QC and protocol figures |
| 1C | Same step-1 file | Cohort decision | QC-reviewed fish | Discard lists and moved files |
| 2 | `2_ExampleFishPlotting.py` | Optional | Per-fish pickle | Example-fish figures |
| 3 | `3_FishGrouping_LogMedian.py` | Yes | Per-fish pickles | Condition/alignment LogMedian tables |
| 4A | `4_ScaledVigorPlotting_LogMedian.py` | Yes for temporal figures | Step-3 tables | Binned temporal artifacts |
| 4B | Same step-4 file | Optional by panel | Binned artifacts | Heatmaps and line plots |
| 5A | `5_NormalizedVigorPlotting_LogMedian.py` | Yes for trial statistics | Step-3 tables | Per-fish/per-trial median summary |
| 5B | Same step-5 file | Optional by analysis | Trial summary | Block, phase, and LME outputs |
| 6 | `6_LearnersQuantification_LogMedian.py` | Optional | Step-5 trial summary | Fish classification table |
| 7 | `learner_stratified_vigor_pipeline.py` | Optional | Explicit step-3 control and 3sTrace files | Learner-stratified temporal analysis |

## Pipeline diagram

```text
configuration
    |
raw acquisition files
    |
    v
step 1 preprocessing
    |
    +--> step 1 QC/protocol review --> discard decision
    |
    +--> step 2 example figures
    |
    v
per-fish processed files
    |
    v
step 3 LogMedian grouping
    |
    +--> step 4 time-resolved figures
    |
    +--> step 5 trial summaries and statistics
              |
              v
       step 6 learner labels
              |
              +-------------------+
                                  |
step-3 temporal data -------------+
                                  v
             step 7 learner-stratified profiles
```

## Numbered analysis files

## `1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py`

Current stage-1 entry point.

Functions:

- `run_preprocess()`
- `run_plot_individual_trials()`
- `run_plot_protocols()`
- `run_discard()`

Responsibilities:

- Raw file discovery
- Camera/frame validation
- Tail-tracking input
- Synchronization and interpolation
- Stimulus integration
- Tail-angle processing
- Vigor and bout detection
- Trial and block assignment
- Per-fish serialization
- QC plots
- Exclusion workflow

This file is multifunctional. Its internal order is:

```text
preprocess -> inspect plots/protocol -> decide exclusion
```

## `2_ExampleFishPlotting.py`

Optional publication-figure entry point for selected fish.

Capabilities:

- Trial traces
- Individual-trial heatmaps
- Bout zoom
- Tail trajectories

Difference from step 1:

- Step 1 plots are primarily QC-oriented.
- Step 2 plots are selected and publication-oriented.

## `3_FishGrouping_LogMedian.py`

Current stage-3 grouping implementation.

Distinctive LogMedian operations:

- Bout masking
- Trailing rolling median
- Downsampling
- `log(vigor)` for positive values
- Per-trial baseline median subtraction

Output column:

```text
Scaled vigor (AU)
```

means baseline-centered log-vigor in this branch.

The file is configured independently through its top-level `EXPERIMENT` value.

## `4_ScaledVigorPlotting_LogMedian.py`

Current stage-4 temporal-analysis implementation.

Sub-stages:

- Build pooled outputs
- Count heatmap
- scaled/log-vigor heatmap
- individual catch-trial profiles
- pooled catch-trial profiles
- per-block profiles

All `RUN_*` values are currently enabled at the tracked baseline. Review them
before running the file.

## `5_NormalizedVigorPlotting_LogMedian.py`

Current stage-5 trial-summary and population-statistics implementation.

Primary trial metric:

```text
Median CR - Median baseline
```

in log-vigor space.

Sub-stages:

- Data aggregation
- Block line plots
- Block boxplots
- Phase summary
- Trial-by-trial mixed-effects analysis

## `6_LearnersQuantification_LogMedian.py`

Sole active learner-classification implementation on the LogMedian baseline.

Key characteristics:

- Median-based input columns
- Difference rather than ratio normalized vigor
- No additional log transform
- Control-anchored BLUP features
- Directional scores
- One-sided joint statistic
- Empirical control threshold
- One `Is_Learner` label

The older files:

```text
6_LearnersQuantification.py
6_LearnersQuantification_new.py
6_LearnersQuantification_improved.py
6_LearnersQuantification_WIP.py
```

are absent from `bf46bf7`. They should not be documented as simultaneous
choices for this baseline.

## Integrated learner-stratified files

## `learner_stratified_vigor_pipeline.py`

Added by source commits `505e29a` and `88fe4fc`.

Purpose:

- Build compact trial and temporal checkpoints from explicit `control` and
  `3sTrace` LogMedian condition files.
- Invoke `6_LearnersQuantification_LogMedian.py`.
- Restore source fish IDs after condition-qualified classifier keys.
- Create a fish-level manifest.
- Attach learner strata to temporal data with a many-to-one join.
- Produce block, catch-trial, and timing analyses.

Working-tree integration adaptations:

- Replaced missing WIP classifier import with the LogMedian classifier.
- Replaced mean/ratio trial metrics with median log-space subtraction.
- Replaced `trace` analysis ID with `3sTrace`.

## `run_learner_stratified_vigor.ps1`

Windows runner that:

- Creates `.venv-learner-vigor`
- Installs pinned dependencies
- Runs with the Agg backend
- Preserves an execution log
- Fails if the Python command fails

## `requirements-learner-vigor.txt`

Pinned environment for the optional learner-stratified pipeline.

It is not yet a dependency manifest for the entire repository.

## `test_learner_stratified_vigor_pipeline.py`

Automated `unittest` coverage for:

- LogMedian trial summaries
- Temporal binning
- Fish-first aggregation
- Classification manifest behavior
- Condition-qualified fish keys
- Catch-trial selection
- Catch timing metrics
- Checkpoint signatures
- Legacy pandas pickle compatibility
- Optional classifier integration

## Shared modules

| File | Responsibility |
| --- | --- |
| `analysis_utils.py` | Synchronization, filtering, vigor, bouts, trial mapping, tail geometry, figure annotations |
| `data_io.py` | Camera, protocol, sync, and tracking readers |
| `experiment_configuration.py` | `ExperimentType`, `ExperimentConfig`, experiment factory |
| `general_configuration.py` | Shared validation, filtering, bout, timing, and plotting defaults |
| `file_utils.py` | Output paths, fish IDs, discard lists |
| `figure_saving.py` | Filename-safe figure export |
| `plotting_style.py` | Shared Matplotlib/Seaborn style |
| `pipeline_utils.py` | Shared discard, file-loading, heatmap scaling, and trial-metric helpers |
| `heatmap_utils.py` | Heatmap matrices, axes scaffolding, and phase-block rendering |

## Utility and diagnostic files

| File | Role |
| --- | --- |
| `0_Pipeline_Data_Flow_Description.txt` | Tracked LogMedian pipeline description from `bf46bf7` |
| `open_scaled_vigor_df.py` | Loads and rewrites LogMedian condition pickles after condition-label normalization |
| `test_discarded_heatmap_grid.py` | Generates discarded/included heatmap-grid diagnostics; not a conventional unit-test module |
| `ClassicalConditioning.code-workspace` | VS Code workspace |
| `jasmc.code-profile` | Editor profile artifact |

`open_scaled_vigor_df.py` modifies files in place through `to_pickle(path)`;
use it only with an intentional backup and explicit input review.

## Versions and overlapping implementations

## Standard versus LogMedian pipeline

### Older standard tree (`b0dfcb3`)

Used:

- `3_FishGrouping.py`
- `4_ScaledVigorPlotting.py`
- `5_NormalizedVigorPlotting.py`
- Four learner-classifier files
- Legacy `my_*` modules

Typical metric:

- Mean-based smoothing and/or mean trial metrics
- Ratio normalized vigor
- Percentile scaling

### Current LogMedian baseline (`bf46bf7`)

Uses:

- `3_FishGrouping_LogMedian.py`
- `4_ScaledVigorPlotting_LogMedian.py`
- `5_NormalizedVigorPlotting_LogMedian.py`
- `6_LearnersQuantification_LogMedian.py`
- `pipeline_utils.py`
- `heatmap_utils.py`

Typical metric:

- Rolling median
- Log vigor
- Median summaries
- Subtraction in log space
- Zero-centered reference

These pipelines are not interchangeable. Their processed values, reference
levels, figures, and classifier inputs differ.

## Scaled-vigor variants within the current tree

Even on the LogMedian branch, different transformations coexist:

1. Step 1 computes a percentile-scaled per-fish value for QC.
2. `pipeline_utils.compute_scaled_vigor_for_heatmap()` performs bout-mean
   replacement and percentile display scaling for step-1/2 heatmaps.
3. Step 3 creates baseline-centered log-vigor for population analysis.
4. Step 4 can subtract an immediate pre-stimulus median again for line display.

These should be named separately:

- QC heatmap normalization
- Analytical LogMedian vigor
- Display centering
- Display clipping

## Normalized-vigor variants

`pipeline_utils.compute_normalized_vigor_per_trial()` supports step-1/2
individual plotting.

`5_NormalizedVigorPlotting_LogMedian.py` creates the downstream population
trial table.

The latter is the source for step 6, but both should eventually call one
canonical tested metric function.

## Learner classifier versus learner-stratified orchestrator

These are complementary, not competing versions:

- `6_LearnersQuantification_LogMedian.py` defines classification.
- `learner_stratified_vigor_pipeline.py` prepares explicit full-dataset inputs,
  calls the classifier, creates a manifest, and analyzes temporal profiles.

## Exclusion mechanisms

Exclusion currently appears in several forms:

- Step 1 moves files and writes text lists.
- Step 3 recursively discovers per-fish pickles.
- Steps 4-6 have `APPLY_FISH_DISCARD` switches.
- The learner-stratified pipeline disables classifier-side discard filtering
  and assumes explicit input files define its source cohort.

This is not one authoritative cohort system.

## Changes from `bf46bf7` to the integrated branch

The two transplanted feature commits add only:

```text
.gitignore entry
learner_stratified_vigor_pipeline.py
requirements-learner-vigor.txt
run_learner_stratified_vigor.ps1
test_learner_stratified_vigor_pipeline.py
```

They do not modify the LogMedian stage-1 through stage-6 files in their original
form.

The transplanted files are byte-identical to the source learner-stratified
branch before the working-tree compatibility adaptations documented above.

## Canonical status

| Area | Canonical for this baseline | Status |
| --- | --- | --- |
| Preprocessing | Step-1 script | Current, but scientific issues remain |
| Grouping | `3_FishGrouping_LogMedian.py` | Current |
| Temporal population plots | `4_ScaledVigorPlotting_LogMedian.py` | Current |
| Trial summaries/statistics | `5_NormalizedVigorPlotting_LogMedian.py` | Current |
| Learner classification | `6_LearnersQuantification_LogMedian.py` | Current |
| Learner-stratified orchestration | `learner_stratified_vigor_pipeline.py` | Integrated optional stage |
| Scientific figures | Existing scripts | Migration planned |
| Artifact format | Pickle/CSV | Current legacy format; migration planned |
| Cohort definition | File movement and flags | Not canonicalized |

