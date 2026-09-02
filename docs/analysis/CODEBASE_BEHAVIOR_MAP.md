# Preexisting Codebase Behavior Map

## Purpose

This document records verified executable behavior that must be understood
before moving or replacing the preexisting analysis code. It complements the
more detailed `ANALYSIS_FILES_INDEX.md` and `ANALYSIS_ISSUES.md`.

## Scale and topology

The root contains:

- five required numbered pipeline scripts;
- one optional example-fish script;
- four incompatible learner scripts;
- seven modern shared support modules;
- three older `my_*` modules;
- more than 20,000 lines of preexisting Python.

The numbered scripts import shared root modules directly. No modern script
currently imports the three `my_*` modules, but historical branches may depend
on them.

## Verified current `main` flow

### Stage 1 — preprocessing

Inputs:

```text
*_cam.txt
*_mp tail tracking.txt
*_stim control.txt
```

Verified operations:

1. Discover tracking files and infer companion names.
2. Parse fish metadata from underscore-separated filenames.
3. Read camera timing after skipping a configured initial frame count.
4. Estimate frame rate/reference frame and a legacy accumulated-drift
   frame-loss flag.
5. Read only selected `angleN` columns from tracking; measured `xN/yN` fields
   are currently discarded.
6. Drop the final tracking row unconditionally.
7. Filter camera/tracking to the reference frame, inner-merge on `FrameID`, and
   interpolate to a uniform expected-rate grid with linear extrapolation.
8. Read protocol events and annotate stimuli.
9. Recompute `AbsoluteTime` using a regular step based on predicted frame rate.
10. Cumulatively sum angles across tail points.
11. Apply centered temporal rolling mean; the current function does not apply
    the intended spatial smoothing.
12. Calculate absolute frame-to-frame speed of the distal cumulative angle.
13. Calculate a rolling maximum-minus-minimum bout metric.
14. Detect bouts using the primary threshold, gap merge, and minimum duration;
    the configured secondary threshold is not used.
15. Identify trials and blocks.
16. Calculate per-trial percentile-scaled vigor using samples earlier than the
    immediate 15-second pre-stimulus interval.
17. Save a gzip pickle and optional QC/protocol/individual plots.
18. The discard workflow can move processed and raw files.

The refactor must first reproduce this route under a legacy recipe, except that
new code never mutates raw files or writes canonical pickle.

### Stage 2 — example fish

Optional publication branch:

- selected-fish angle/vigor traces;
- per-trial heatmaps;
- normalized-vigor views;
- bout close-ups;
- reconstructed tail-position histograms.

It reads stage-1 fish artifacts and does not feed required population stages.

### Stage 3 — grouping

Verified standard-main route:

1. Load per-fish pickle.
2. restore/derive fish identity and block labels;
3. split CS and US trials;
4. set vigor outside bouts to missing;
5. calculate P10/P90 scaling from `time < -baseline_window`;
6. apply right-aligned rolling mean within fish/trial;
7. downsample every tenth row;
8. concatenate by condition;
9. save condition/alignment pickle.

The discard list is loaded for heatmap grids but is not applied to remove rows
before condition concatenation.

### Historical LogMedian grouping route

The separate historical branch describes:

```text
bout mask
-> right-aligned rolling median
-> downsample
-> log positive vigor
-> subtract baseline log median
```

This is a distinct recipe, not the current `main` baseline.

The `historical-logmedian-v1` preservation route additionally reproduces the
historical float32 vigor coercion and rolling output, constructs `Fish` from
`Day` and `Fish no.` when needed, and retains CS and US as separate grouping
domains. Its bounded-memory writer processes only complete contiguous trials
and rejects group-order or non-strict trial-time violations rather than
changing historical semantics.

Pilot `20221115_04` produced 1,083,772 authenticated rows across 94 CS and 78
US trials. Independent execution of the historical operation sequence matched
the first CS and final US trial outputs exactly. No local historical stage-1
pickle was recoverable; two multi-gigabyte pooled CS LogMedian pickles lack
authenticated lineage and are retained only as read-only historical artifacts.

### Stage 4 — scaled-vigor analysis

Verified responsibilities:

- discover top-level condition/alignment pickle files by filename suffix,
  including `_CS`, `_CS_selectedFish`, and `_CS_allFish` variants;
- optionally filter exact `Fish` identifiers from the configured discard list;
- convert time to seconds;
- mask non-bout scaled vigor as missing;
- first aggregate the median and non-missing count across fish at each exact
  time/trial/block, then average those medians and sum those counts within time
  bins;
- divide binned sample counts by the total number of fish, producing average
  contributing samples per fish rather than a bounded fish fraction;
- create count heatmap tables;
- apply an additional per-trial P10/P90 normalization and clipping for SV
- heatmaps, using all negative-time bins;
- use fixed catch-trial lists, with two additional retraining trials when the
  experiment name contains `long`;
- create lineplot/catch/block outputs from separately saved pooled artifacts;
- render figures.

Pooled output filenames retain pickle storage and encode condition lists using
Python list formatting. Rendering discovers matching files independently; some
lineplot routes use only the first matching pooled artifact. The script mixes
artifact building and rendering behind independent run flags and catches broad
exceptions around every enabled top-level route.

### Stage 5 — normalized-vigor analysis

Verified standard-main route:

- default exclusion application is false;
- discover per-condition pooled pickle files by underscore-delimited filename
  fields and take the first match for each condition;
- save a per-fish/per-trial gzip-pickle artifact whose name embeds Python
  representations of the response window and condition list;
- use inclusive `Series.between` masks for CS and US baseline/response windows,
  so CS time zero belongs to both means and the US-shifted negative endpoint
  belongs to both means;
- use arithmetic window means of bout-masked vigor and calculate response
  divided by baseline without a separate zero-baseline guard;
- calculate per-window non-missing fractions, but leave invalidation disabled by
  default; when enabled, either window exceeding 90% missing invalidates all
  three trial metrics;
- downstream plot loaders are inconsistent: most take the first filesystem
  match, while the trial-by-trial loader explicitly selects the newest matching
  artifact by modification time;
- retain a fish if it reaches the minimum trial count in any block, rather than
  requiring the minimum in every required block;
- create block, phase, boxplot, trial, nonparametric, and mixed-model results;
- use Mann-Whitney tests for within- and between-condition block comparisons,
  with Holm-family corrections in the plotting routes;
- use 100 seaborn bootstrap resamples with a fixed seed for trial trajectories,
  without declaring fish as the cluster-resampling unit;
- fit global condition-by-block, block-local condition-by-centered-trial, and
  per-trial condition mixed models on `log(value + 1)`, adjusting response for
  log baseline and grouping repeated observations by fish;
- apply Benjamini-Hochberg FDR separately to block mean/slope tests and across
  per-trial condition tests;
- return mixed-model failures as strings and continue plotting/reporting rather
  than failing the stage.

The script has separate `first` and `latest` artifact loaders, so different
rendering/statistical routes can consume different files when multiple matching
artifacts exist.

### Stage 6 — learners

Four large scripts encode incompatible:

- input precedence: newest root pickle versus CSV-first legacy-folder searches;
- feature definitions: the original uses two late-training blocks while the
  other variants use six, changing both estimates and eligibility;
- mixed-model targets: three model response with baseline adjustment, while WIP
  models log normalized-vigor ratio without a baseline covariate;
- covariance/decision methods: Mahalanobis outlier gates versus a one-sided
  shrinkage-whitened joint statistic;
- thresholds and voting: conservative confidence-limit votes, 0.60 Gaussian
  probability votes, or unanimous directional votes at empirical 90th/95th
  control percentiles;
- uncertainty and reference handling, including full-control versus leave-one-
  out control means;
- output schemas and active export/plot defaults.

No implementation is canonical. Historical commits `4d6939f` and `f372101`
contain useful learner-stratified orchestration and catch-timing work, but
depend on a divergent LogMedian branch and will be ported selectively later.
The complete comparison is recorded in
`docs/analysis/LEARNER_VARIANT_BEHAVIOR_MATRIX.md`.

## Shared-module observations

### `data_io.py`

- Broad parse fallbacks often return `None`.
- Tracking ingestion assumes selected non-frame fields are angles.
- It ignores measured XY fields in the supplied raw recording.
- It drops the final tracking row even when that row is a valid frame.

### `analysis_utils.py`

Combines:

- frame/timing validation;
- protocol summaries;
- synchronization/interpolation;
- filtering and vigor;
- bout detection;
- trial/block annotation;
- figure and SVG layout helpers.

It must be split operation by operation, not moved wholesale.

### Configuration

`general_configuration.py` exposes a mutable module singleton with derived
fields. `experiment_configuration.py` returns mutable experiment objects with
machine-specific paths and a large conditional factory. Script-level constants
override or duplicate both.

### Figures

`figure_saving.py` currently sanitizes paths and delegates to `savefig`.
`plotting_style.py` keeps SVG text as text but current figures do not assign a
complete semantic artist registry or embed their generating code invocation.

## Preservation rule

For every replacement:

```text
executable code
-> current-behavior contract
-> characterization test
-> package implementation
-> legacy/new comparison
-> compatibility wrapper
-> only then move old implementation
```

Documentation or handwritten descriptions never substitute for executable
characterization.
