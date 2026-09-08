# Step 00B — Preexisting Codebase Characterization

**Status:** Complete
**Change class:** Observation and behavior preservation only  
**Depends on:** Step 00A intake and existing source inventory  
**Unlocks:** Safe replacement and organization of legacy executable files

## Objective

Understand and test the entire preexisting codebase as executable behavior
before moving or replacing scientific logic.

The executable code is the primary source for legacy behavior. Handwritten
descriptions, manuscript methods, and comments are compared with it and any
disagreement is recorded.

## Scope

### Required scripts

```text
1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py
3_FishGrouping.py
4_ScaledVigorPlotting.py
5_NormalizedVigorPlotting.py
```

### Optional branches

```text
2_ExampleFishPlotting.py
6_LearnersQuantification.py
6_LearnersQuantification_new.py
6_LearnersQuantification_improved.py
6_LearnersQuantification_WIP.py
historical LogMedian branch
historical learner-stratified commits
```

### Shared and legacy modules

```text
analysis_utils.py
data_io.py
experiment_configuration.py
general_configuration.py
file_utils.py
figure_saving.py
plotting_style.py
my_general_variables.py
my_experiment_specific_variables.py
my_functions.py
```

## Work packages

### 00B.1 Static inventory

For every file record:

- lines and top-level symbols;
- imports and consumers;
- global configuration and run flags;
- input discovery;
- read/write formats;
- generated artifacts;
- broad failure handling;
- mutation and file-moving behavior;
- corresponding historical variants.

### 00B.2 Stage-1 executable characterization

Test current:

- tracking field selection and final-row removal;
- frame-rate/reference-frame behavior;
- synchronization and interpolation;
- cumulative-angle transformation;
- absence/effect of spatial filtering;
- temporal filtering and edge removal;
- point-15 vigor;
- rolling bout metric;
- ignored secondary threshold;
- trial/block annotation;
- percentile scaling;
- raw and processed file-moving behavior.

Do not create new canonical pickle files.

### 00B.3 Stage-3 characterization

Test:

- CS/US split;
- bout-to-missing conversion;
- early-baseline P10/P90 scaling;
- rolling mean alignment;
- downsampling;
- fish identity and condition concatenation;
- actual use of discard lists;
- output schema.

### 00B.4 Stage-4 characterization

Separate and test:

- pooled artifact discovery;
- fish filtering;
- time-bin construction;
- median/count aggregation hierarchy;
- count normalization;
- additional heatmap P10/P90 scaling;
- lineplot baseline transforms;
- catch/block selections;
- build versus render outputs.

### 00B.5 Stage-5 characterization

Test:

- first-match artifact selection;
- CS/US window masks and endpoint inclusion;
- mean baseline/response aggregation;
- ratio calculation;
- missing-fraction behavior;
- all-block trial eligibility;
- bootstrap unit/count;
- nonparametric families;
- mixed-model formulas, failures, and diagnostics.

### 00B.6 Learner variant characterization

Create one comparison matrix for:

- input artifact;
- response variable;
- features;
- minimum data requirements;
- model;
- covariance/distance statistic;
- threshold and voting;
- uncertainty;
- outputs;
- active defaults;
- validation mode.

Do not select a canonical classifier in this step.

### 00B.7 Historical LogMedian route

Compare executable branch code with the handwritten description. Record exact
differences from the standard `main` route and preserve it as a separate recipe.

### 00B.8 Repository organization gate

Use `REPOSITORY_MIGRATION_MAP.md`. No executable source moves until the
corresponding characterization and compatibility conditions pass.

## Progress evidence

Completed characterization commits:

```text
af032e2 test(legacy): characterize stage one behavior
e034726 test(legacy): characterize downstream transformations
3709776 feat(preprocessing): add frozen legacy recipe comparison
```

Verified by tests:

- current tracking reader selects angles, discards measured XY, and removes the
  final source row;
- spatial-window changes do not affect current filtering output;
- current vigor is the absolute derivative of one signal;
- the secondary bout threshold does not affect output;
- frame-ID gaps do not trigger the current timing-based loss flag;
- stage 3 uses early-baseline P10/P90 scaling and masks rest as missing;
- stage 4 forces a zero bin edge and includes zero in its display baseline;
- stage 4 discovers pooled inputs by filename suffix and applies exact fish-ID
  exclusions when enabled;
- stage 4 first pools fish by exact time/trial/block, then averages those
  medians within bins and divides summed non-missing sample counts by fish
  count;
- stage 4 applies a second per-trial P10/P90 scaling across all negative-time
  bins for heatmaps and uses fixed catch-trial selections;
- stage 4 build and rendering routes are independently flag-controlled, retain
  pickle intermediates, and broadly catch top-level failures;
- stage 5 retains a fish when only one block reaches the minimum count.
- stage 5 uses inclusive CS and US baseline/response masks, causing the boundary
  sample to contribute to both window means;
- stage 5 computes response divided by baseline from arithmetic means of
  bout-masked vigor, while its 90% missing-window invalidation is disabled by
  default;
- stage 5 condition aggregation takes the first matching source artifact;
  downstream loaders inconsistently choose either the first or newest matching
  trial artifact;
- stage 5 block comparisons use Mann-Whitney tests and Holm-family correction;
  its trial trajectories use 100 fixed-seed seaborn bootstrap resamples without
  fish-level cluster resampling;
- stage 5 mixed models use log-response ANCOVA formulas for global
  condition-by-block, block-local condition-by-centered-trial, and per-trial
  condition effects, with fish grouping and Benjamini-Hochberg corrections;
- stage 5 converts model exceptions to reported error strings and continues.
- all four learner variants require complete acquisition/extinction features
  and at least ten retained fish, but disagree on artifact precedence, epoch
  membership, response quantity, control reference, multivariate statistic,
  threshold, voting rule, false-positive target, and export defaults;
- the original/new routes use Mahalanobis outlier gates with conservative or
  probabilistic votes, while improved/WIP use leave-one-out control references
  and one-sided joint statistics;
- a package `legacy-paper-v1` route reproduces stage-1 operations from the
  lossless intake artifacts and records verified source hashes and limitations;
- the complete pilot produced 10,836,172 trial-aligned rows with 94 CS and
  78 US trials.
- the historical LogMedian transform has an executable reference-equivalence
  contract, including the historical float32 rolling-median calculation and
  `Day`/`Fish no.` identifier construction;
- bounded-memory publication rejects non-contiguous trial groups,
  non-strictly-increasing trial time, and source groups outside the required
  global sort order;
- pilot `20221115_04` produced 1,083,772 LogMedian rows from 172 contiguous
  trial groups in 32.4 seconds; the 111,425,573-byte Zstandard Parquet,
  summary, and completion marker passed hash authentication;
- the first CS and final US pilot groups exactly matched the independent
  executable historical reference, including row identity, dtypes, values,
  and missingness.

Historical-artifact limitation:

- no recoverable historical stage-1 pickle was found locally, so a direct
  full-output stage-1 pickle comparison cannot be performed;
- the only recovered scientific pickles are
  `control_CS_new_logmedian.pkl` (5,531,425,570 bytes) and
  `trace_CS_new_logmedian.pkl` (4,770,233,312 bytes);
- their names and pickle prefixes identify pooled pandas LogMedian artifacts,
  but they contain no external provenance sidecar and cannot be inspected with
  bounded memory; they are retained read-only as unauthenticated historical
  artifacts and are not accepted as pilot or stage-1 equivalence evidence.

Package-owned trial mapping and measured-time alignment are now available, but
they remain separate from the frozen legacy characterization.

Candidate improvements are implemented separately in Step 06 and do not close
remaining legacy-characterization work.

Final automated total: 106 tests across intake, artifact integrity,
characterization, frozen legacy preprocessing, candidate metrics, measured-time
profiles, comparison reports, semantic figure export, movement sensitivity, and
trace review.

## Deliverables

- `docs/analysis/CODEBASE_BEHAVIOR_MAP.md`
- Static dependency and artifact inventory
- Characterization tests for every scientific stage
- Standard-versus-LogMedian behavior matrix
- Learner-version behavior matrix
- File-by-file migration map
- List of unresolved code/comment/manuscript disagreements

## Exit gate

Every preexisting executable file is classified as active, optional,
historical, duplicated, or superseded; every scientific transformation has a
current-behavior contract and characterization evidence; file moves can occur
without losing behavior or hiding scientific changes.

**Exit-gate result:** Passed, with the unavailable historical stage-1 pickle
comparison recorded above as an evidence limitation rather than silently
inferred equivalence.
