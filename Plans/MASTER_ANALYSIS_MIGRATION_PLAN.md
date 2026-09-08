# Master Analysis Migration and Paper Reproduction Plan

## 1. Authority and purpose

This is the authoritative implementation map for the ClassicalConditioning
analysis migration. It combines:

- exact preservation of the current executable analysis;
- replacement of pickle with safer canonical artifacts;
- a modular Python package, CLI, and notebook-friendly API;
- explicit scientific correction of known analysis issues;
- versioned comparison of tail-dynamics and vigor measures;
- later integration of optional imaging without changing behavior semantics;
- cohort, statistical, learner, and figure reproducibility;
- immutable legacy-equivalent and corrected paper releases.

Detailed scientific source plans remain authoritative within their own subject
areas. This master plan determines their implementation order, dependencies,
versioning, and release gates.

### 1.1 Decision hierarchy

Use this order when documents differ:

1. approved scientific analysis specification and signed decision records;
2. this master implementation plan and its step exit gates;
3. approved domain plans for tail dynamics, learners, and figures;
4. audit findings and analysis-issue notes;
5. descriptions of current/legacy behavior.

The analysis-issues audit identifies credible risks and recommended
investigations. It does not quantify their effect on the data and cannot by
itself select a corrected formula, cohort, or model. Every issue is tracked in
[the analysis issue traceability register](./ANALYSIS_ISSUES_TRACEABILITY.md).

## 2. Non-negotiable outcome

The migration has two different correctness targets:

### 2.1 Legacy-equivalent target

The first target reproduces the current pipeline's scientific tables and
outputs under a pinned environment. It preserves current behavior even where
that behavior is known or suspected to be scientifically wrong.

It exists to prove that engineering changes did not silently alter results.

### 2.2 Corrected scientific target

The second target implements approved mathematical and scientific
specifications. Corrected output is compared with the legacy target, but it is
not required to match it.

It exists to support the scientific paper.

No artifact may be described as both legacy-equivalent and scientifically
corrected. Mixed changes require separation before acceptance.

## 3. Implementation form

The implementation will be a Python package under:

```text
src/classical_conditioning/
```

The same stage functions will be callable from:

1. Python code;
2. a command-line interface;
3. optional review and exploration notebooks;
4. temporary compatibility wrappers in the current numbered scripts.

Notebooks will not contain unique canonical calculations. They may inspect,
filter, review, and display package outputs.

### 3.1 Local execution boundary

All acquisition reading, conversion, processing, statistics, model fitting,
figure rendering, and artifact storage run locally.

- No Databricks workspace, DBFS, Unity Catalog, SQL warehouse, cluster, job, or
  Databricks SDK dependency.
- No upload or synchronization of scientific artifacts to Databricks.
- LLM calls may contain code, plans, schemas, and concise diagnostics, but not
  bulk raw or processed files.
- Interactive HTML embeds its JavaScript locally and does not require a CDN.
- Raw acquisition files are immutable and are opened read-only.

### 3.2 Git implementation protocol

Implementation begins on a new branch created from a user-confirmed base. The
repository currently has no documented branch convention, so branch name and
base must be confirmed before creation.

Use atomic semantic commits:

```text
feat(ingestion): add lossless camera reader
test(ingestion): cover missing and duplicate frames
refactor(preprocessing): preserve legacy point-15 vigor
feat(metrics): add segment angular-speed sum
docs(plans): archive completed ingestion step
```

Do not mix behavior-preserving and scientific-correction work in one commit.
Do not push unless explicitly requested.

### 3.3 Migration method (absorbed from former CODEBASE plan)

Use an incremental strangler migration, not a big-bang rewrite:

- Frozen dataclasses for identities, configurations, artifacts, and result
  bundles; enums for finite scientific categories.
- DataFrames and NumPy arrays for observations; pure functions for scientific
  transforms; small orchestrators for stage execution.
- Explicit adapters for file I/O; compatibility wrappers while numbered scripts
  retire.
- Characterization tests freeze current/legacy behavior first; scientific
  reference tests prove approved corrections separately. Never let a
  characterization fixture silently become a scientific approval.
- Package-first path wins. The former Lane-1 “script-side correction” shortcut
  is retired: corrected behavior is implemented and tested in
  `src/classical_conditioning`, not as another undocumented root-script path.
- File organization and script-move rules live in
  [REPOSITORY_MIGRATION_MAP.md](./REPOSITORY_MIGRATION_MAP.md). Historical
  detail of the old dual-lane plan is archived at
  [Archive/CODEBASE_MIGRATION_PLAN.md](./Archive/CODEBASE_MIGRATION_PLAN.md).

## 4. Integrated analysis architecture

Canonical behavior and optional imaging architecture diagrams, route matrix,
and figure-mode rules live in
[ANALYSIS_ARCHITECTURE.md](./ANALYSIS_ARCHITECTURE.md). This master plan keeps
only the invariants:

1. behavior hashes must remain invariant when imaging is enabled or disabled;
2. legacy and corrected behavior routes share the same artifact contracts;
3. maintained figure modes are publication SVG/PDF and static PNG; interactive
   local HTML is frozen (Gate F in [DECISIONS.md](./DECISIONS.md)).

The behavior branch is authoritative and does not depend on imaging
availability. The optional imaging and multimodal stages are deferred and
specified in
[the behavior and imaging integration plan](./BEHAVIOR_IMAGING_INTEGRATION_PLAN.md).

Mechanistic analyses branch from the common tail representation:

```text
common tail representation
    +-> PCA and eigen-tail dynamics
    +-> rhythmicity and spectral power
    +-> curvature kymographs and traveling-wave analysis
```

These remain exploratory until separately specified and validated.

## 5. Fundamental data boundaries

### 5.1 Raw source inventory

One immutable record per source file:

```text
source_id
recording_id
experiment_id
fish_id
source_type
path
size_bytes
modified_time
content_hash
inventory_version
```

Raw source files stay outside Git. Their inventory and hashes may be versioned.

### 5.2 Raw scientific tables

- Camera frames and timestamps
- Tracking points, angles, coordinates, and confidence where available
- Protocol/stimulus events
- Fish and recording metadata

### 5.3 Tail representation

The common representation carries, where scientifically supportable:

```text
frame_id
timestamp_ms
point_id
position_along_tail
local_angle_rad
orientation_rad
x_body_normalized
y_body_normalized
curvature
tail_length_weight
valid
quality_flags
```

The representation must state whether coordinates are measured or
reconstructed. Those are different scientific representations, not minor
versions of one another.

### 5.4 Frame activity

Each frame-level measure carries:

```text
metric_id
metric_implementation_version
metric_parameter_set_id
value
unit
valid_fraction
quality_flags
```

### 5.5 Movement state and bouts

Movement state is qualified by both the metric and detector:

```text
metric_id
movement_detector_id
movement_detector_version
moving
bout_id
bout_start
bout_end
threshold
threshold_source
```

There is no unqualified canonical `Bout` column shared by incompatible metrics.

### 5.6 Trial outcomes

Canonical trial and window summaries include:

- total activity, including rest;
- movement probability;
- fraction of time moving;
- bout count, initiation rate, duration, and interbout interval;
- conditional movement intensity;
- legacy scaled and normalized vigor where needed for equivalence;
- candidate corrected outcomes (after Gate O);
- valid-sample and valid-tail coverage.

### 5.7 Results and figures

Statistical result artifacts contain formulas, estimands, input IDs, sample
counts, diagnostics, coefficients, uncertainty, contrasts, multiplicity, and
software details. Figures consume saved panel-data artifacts rather than
recalculating scientific outcomes while drawing.

## 6. Canonical file formats

| Artifact | Canonical format | Compatibility/export format |
| --- | --- | --- |
| Tabular source mirrors, processed samples, trial, cohort, and result data | Parquet with lossless Zstandard compression | CSV for review where useful |
| Resolved configuration | JSON | Human-readable YAML/TOML may be an input |
| Provenance and artifact manifests | JSON | None |
| Large dense multidimensional tail arrays | HDF5 may be selected after a local benchmark | NPZ or Zarr comparison candidates |
| Statistical model summaries | Parquet plus JSON metadata | CSV/text summaries |
| Publication figures | SVG and PDF | None |
| Static review figures | PNG | None |
| Interactive figures | Self-contained local HTML (frozen; no further investment) | Optional notebook view |
| Existing pandas artifacts | Read-only legacy pickle | Conversion/comparison input only |

New code does not write pickle. During legacy equivalence, the new in-memory
result and Parquet output are compared with frozen existing pickle artifacts or
the current legacy function output.

All scientific-data compression is lossless. Ingestion does not downcast,
quantize, round, or otherwise reduce source numeric precision. Any later dtype
change is an explicit, tested transformation rather than a storage shortcut.

Parquet remains primary for tables. HDF5 is reconsidered only when tail
dynamics creates a genuinely dense, homogeneous frame × point × coordinate
array and local benchmarks show materially better slicing, size, or write
performance. If selected, it uses lossless compression, checksums, a documented
chunk shape, and an explicit schema.

Logical scientific content is the cross-environment artifact identity. Its hash
is calculated from the schema version, canonical column and row order,
normalized dtypes, explicit missingness, and values. A separate file-byte hash
may verify a stored file, but it is not expected to survive legitimate
re-serialization by a different Parquet library or compression version.

## 7. Equivalence policy

### 7.1 Exact comparisons

Require exact agreement for:

- biological and technical identifiers;
- condition, alignment, trial, block, phase, and catch assignments;
- included fish;
- table keys, row counts, and column order after canonical normalization;
- missing-value masks;
- Boolean movement/bout masks for legacy equivalence in the pinned reference
  environment;
- integer counts and sample-size tables;
- statistical formulas and reference categories;
- panel-data membership and figure labels.

### 7.2 Numerical comparisons

Deterministic transforms should initially be compared exactly after dtype and
ordering normalization. If exact agreement is impossible, define a
quantity-specific tolerance before accepting the migration.

Do not use one broad tolerance for every output.

The accepted numerical baseline must be regenerated in the pinned Step 01
environment before Step 05 equivalence is approved. A threshold-mask exception
is permitted only when the underlying continuous value is within a
predeclared epsilon of the threshold; every such differing sample must be
enumerated rather than hidden in an aggregate tolerance.

### 7.3 Statistical comparisons

In the pinned environment, compare:

- model input rows and fish;
- formula and random-effects structure;
- convergence status;
- coefficients and covariance;
- standard errors and intervals;
- raw and adjusted p-values;
- bootstrap seed, resampling unit, and interval method.

Warnings and failed diagnostics are part of the result. A legacy-equivalent
model with poor diagnostics is not scientifically approved.

### 7.4 Figure comparisons

Compare panel data before visual output. Then compare dimensions, axes, labels,
units, limits, ticks, colors, legends, and rasterized previews within a pinned
rendering environment.

## 8. Versioning model

### 8.1 Metric identity versus implementation version

Different quantities receive different IDs:

```text
legacy_distal_angular_speed
segment_absolute_angular_speed_sum
all_point_angular_rms
whole_tail_xy_rms_speed
whole_tail_xy_mean_speed
curvature_rate_rms
```

An implementation version changes how one defined quantity is calculated:

```text
whole_tail_xy_rms_speed implementation 1.0.0
whole_tail_xy_rms_speed implementation 1.0.1
```

Do not describe RMS and mean speed as versions of the same metric.
The manuscript-described sum of absolute segment angular speeds is retained as
a required reconciliation benchmark even if the scorecard ultimately selects
another primary metric.

### 8.2 Parameter-set version

Every scientifically meaningful parameter combination receives an ID and
content hash:

```text
angular-rms-default-v1
angular-rms-low-smoothing-v1
curvature-valid-tail-80-v1
```

### 8.3 Representation version

Examples:

```text
legacy-cumulative-angle-v1
measured-body-centered-xy-v1
reconstructed-body-centered-xy-v1
local-curvature-v1
```

Measured and reconstructed XY remain distinct representation identities.

### 8.4 Detector version

Movement-state and bout algorithms are independently versioned:

```text
legacy-bout-detector-v1
noise-calibrated-threshold-v1
two-state-probabilistic-v1
```

### 8.5 Schema version

Schema versions describe storage contracts, not scientific equations.

### 8.6 Analysis recipe

An analysis recipe pins the complete set of stage implementations and
parameters:

```text
legacy-paper-v1
tail-candidate-development-v1
tail-candidate-confirmation-v1
corrected-paper-v2
```

### 8.7 Scientific status

Every analysis recipe and artifact has one status:

```text
legacy_reproduction
candidate_development
candidate_confirmation
exploratory
corrected_candidate
paper_approved
```

## 9. Configuration and invalidation

Configuration resolves once:

```text
package defaults
  -> experiment definition
  -> named analysis recipe
  -> explicit command override
  -> validation
  -> frozen resolved configuration
```

Exploratory overrides write explicitly versioned artifacts or temporary local
outputs and cannot mutate paper-approved artifacts.

Each stage hashes only relevant settings. Examples:

| Change | Earliest invalidated stage |
| --- | --- |
| Source columns or file parser | Ingestion |
| Frame-loss or timestamp policy | Raw validation |
| Angle semantics or segment lengths | Tail representation |
| Body-axis alignment | Body-centred representation |
| Metric equation or smoothing | Named activity metric |
| Movement threshold | Named detector |
| Cohort decision | Cohort application |
| Trial map | Trial annotation |
| Baseline or response window | Outcomes |
| Time-bin width | Temporal profiles |
| Bootstrap or model | Statistics |
| Learner threshold/features | Classification |
| Color or layout | Figure rendering |

The CLI must support an explanation of reuse and invalidation before execution.

## 10. Main domain and orchestration types

### 10.1 Immutable definitions

```text
FishKey
FishMetadata
Recording
TimeWindow
ConditionSpec
TrialSpec
ExperimentSpec
ArtifactRef
ArtifactMetadata
AnalysisManifest
StageExecution
StageResult
CohortManifest
ClassificationManifest
TailRepresentationSpec
TailRepresentationRef
ActivityMetricConfig
MovementStateConfig
MetricSelectionSpec
PanelSpec
MajorFigureSpec
```

### 10.2 Finite categories

```text
Alignment
Phase
Paradigm
ConditionRole
ArtifactFormat
ScientificStatus
ActivityOutcome
```

### 10.3 Small interfaces

```text
ArtifactRepository
RecordingRepository
TailRepresentationBuilder
ActivityMetric
MovementStateDetector
LearnerClassifier
```

### 10.4 Orchestrators

```text
PreprocessingPipeline
QualityControlPipeline
CohortPipeline
TrialAnalysisPipeline
MetricValidationPipeline
PopulationStatisticsPipeline
LearnerClassificationPipeline
FigureBuildPipeline
ReproductionPipeline
```

Orchestrators coordinate explicit inputs, pure functions, validation, saving,
and provenance. Scientific equations remain pure functions over NumPy arrays
and DataFrames.

## 11. Stage execution contract

Every stage declares:

```text
stage_id
implementation_version
accepted input artifact types and schema versions
relevant configuration fields
output artifact types and schema versions
validation checks
upstream and downstream dependencies
cache key
failure policy
```

A stage never discovers the "latest" compatible file. It receives explicit
artifact references or the project artifact manifest resolves one active,
versioned artifact.

## 12. Intended user interfaces

### 12.1 Python

```python
analysis = load_analysis("corrected-paper-v2")
samples = load_processed_samples(analysis, fish_id="20230307_12")
trial = select_trial(samples, trial_number=25)
profile = build_temporal_profile(...)
```

### 12.2 CLI

```powershell
python -m classical_conditioning plan --recipe legacy-paper-v1
python -m classical_conditioning run preprocess --recipe legacy-paper-v1 --fish 20230307_12
python -m classical_conditioning run activity-metrics --recipe tail-candidate-development-v1 --metric-set tail-candidates-v1
python -m classical_conditioning run statistics --recipe corrected-paper-v2
python -m classical_conditioning reproduce --release paper-corrected-v2
```

Required selectors:

```text
--experiment
--condition
--fish
--only
--from
--until
--dry-run
--force
--explain
```

### 12.3 Figure modes

Maintained modes consume the same saved panel data:

```text
publication  -> SVG + PDF
static       -> PNG
```

Interactive self-contained local HTML is implemented and frozen (Gate F); it
receives no further investment. The mode changes presentation only. It cannot
change cohorts, windows, statistics, binning, or scientific values.
Full-resolution exploration reads local Parquet/HDF5 on demand; large
recordings are not embedded wholesale in HTML.

Operational figure workstreams, QC, and CLI exposure:
[12_FIGURES_CLI_AND_NOTEBOOKS.md](./12_FIGURES_CLI_AND_NOTEBOOKS.md).
Historical detail:
[Archive/SCIENTIFIC_FIGURE_PIPELINE_PLAN.md](./Archive/SCIENTIFIC_FIGURE_PIPELINE_PLAN.md).

### 12.4 Notebooks

Recommended notebooks:

```text
01_preprocessing_qc.ipynb
02_cohort_review.ipynb
03_metric_validation.ipynb
04_population_exploration.ipynb
05_learner_diagnostics.ipynb
06_figure_review.ipynb
```

They are clients of the package and do not own canonical calculations.

## 13. Local artifact structure

Use one stable project tree rather than one directory per invocation:

```text
Paper data/
|-- Raw single fish data/      # immutable source
|-- Processed data/
|   `-- <fish-id>/
|-- Quality checks/
|   `-- <fish-id>/
|-- Tables/
|-- Models/
|-- Figures/
|   |-- Publication/
|   |-- PNG/
|   `-- Interactive/           # frozen HTML outputs only
`-- Metadata/
    |-- source-manifest.json
    |-- analysis-config.json
    |-- artifact-manifest.json
    `-- execution-log.jsonl
```

Scientific changes create artifact-level versions, not duplicate trees for
ordinary executions.

## 14. Scientific correction rules

Every correction requires:

1. a decision record;
2. a mathematical specification;
3. synthetic reference tests;
4. before/after comparison;
5. cohort and coverage comparison;
6. effect-size and uncertainty comparison;
7. figure comparison;
8. manuscript impact assessment.

Examples include vigor, filtering, frame loss, bout detection, missingness,
cohort construction, scaling, normalized vigor, bootstrap, longitudinal
models, and learner classification.

## 15. Candidate metric selection safeguards

Selecting the metric that produces the smallest p-value would invalidate
confirmatory interpretation. Candidate development therefore uses:

1. a technical development subset for formulas, noise, smoothing, and video
   calibration;
2. a prespecified selection dataset or nested, fish-level validation procedure;
3. a scorecard frozen before confirmation;
4. held-out fish, days, or experiments for confirmation when an a priori power
   check supports a permanent holdout;
5. transparent reporting of all candidates.

The scorecard includes:

- synthetic correctness;
- agreement with blinded video annotations;
- movement/rest discrimination;
- baseline stability;
- fish-level repeatability;
- known US-response detection;
- tracking-artifact resistance;
- stability across point density and smoothing;
- usable-frame coverage;
- consistency of the anticipatory effect;
- interpretability;
- downstream usability.

P-values cannot dominate the scorecard.

Before candidate biological validation, gate O freezes the outcome functions,
trial mapping, windows, binning, and aggregation used by every candidate.
Step 09 reuses those implementations on the full corrected dataset; it does
not create a second selection-specific outcome implementation.

The confirmatory result must come from data not used to select the metric. If
a permanent holdout would be underpowered under a prespecified minimum-fish
rule, the default is nested fish-level cross-fitting: metric selection occurs
inside each training fold and evaluation occurs only in its outer test fold.
A naive model fitted to the full cohort after effect-informed metric selection
is reported as descriptive estimation, not as the primary confirmatory
p-value.

## 16. Cohort and missing-data policy

Technical validity and behavioral engagement are separate fields.

The primary cohort must not exclude a fish merely because it expresses the
hypothesized outcome by becoming immobile in the response window. Behavioral
engagement rules may define sensitivity populations but cannot silently define
the primary intention-to-analyze population.

Rest is a valid behavioral state, not generic missing data. Tracking absence,
insufficient valid tail length, unavailable frames, and invalid protocol
coverage remain explicit missingness/validity states.

## 17. Statistical policy

Fish is the biological replication unit. Frames and trials are repeated
observations.

The final specification must define:

- primary population;
- primary outcome and estimand;
- longitudinal time representation;
- model family and link;
- random-effects structure;
- planned contrasts;
- multiple-comparison families;
- bootstrap unit and interval method;
- convergence and singularity gates;
- missing-data assumptions;
- sensitivity analyses.

The preferred outcome decomposition is:

1. total activity;
2. movement probability or fraction time moving;
3. conditional intensity when active;
4. bout count/rate and duration.

Ratios with unstable or near-zero baselines are not default primary outcomes.

## 18. Learner-analysis policy

Learner labels record their input metric, classifier, features, cohort, and
validation mode. A classifier trained from distal-point vigor is not
automatically valid for whole-tail activity.

Using the same behavior to define labels and test learner/non-learner
differences is descriptive, not confirmatory. Confirmatory learner analysis
requires held-out trials/phases, cross-fitting, or independent data.

## 19. Paper release contents

The final release contains:

```text
release.json
source-inventory.json
resolved-config.json
environment lock and environment report
artifact-manifest.json
scientific decision records
cohort and sample-flow tables
canonical processed and summary artifacts
model inputs, outputs, and diagnostics
panel-data tables
SVG/PDF figures and PNG previews
legacy-versus-corrected comparison
methods parameter table
claim-to-artifact index
reproduction log
```

The release must be reconstructible from immutable raw sources or verified
upstream artifacts plus the recorded code and environment.

## 20. Implementation sequence

Follow the detailed documents in [the implementation step
index](./IMPLEMENTATION_STEP_INDEX.md):

1. Governance and baseline
2. Package and environment
3. Artifacts, schemas, and provenance
4. Domain and configuration
5. Ingestion and raw validation
6. Legacy preprocessing equivalence
7. Tail representation and candidate metrics
8. Metric validation and selection
9. Full reprocessing, QC, and cohort
10. Outcomes and temporal profiles
11. Statistics and sensitivity
12. Learner classification
13. Figures, CLI, and notebooks
14. Releases and legacy retirement

## 21. Definition of complete

The programme is complete only when:

- the current pipeline has an accepted legacy-equivalent release;
- no canonical table depends on pickle;
- every paper-scope raw recording is accounted for;
- candidate metric selection is validated without confirmation leakage;
- the corrected full dataset is reprocessed under one frozen recipe;
- every downstream table proves its cohort and upstream provenance;
- primary outcomes and inference are prespecified and diagnosed;
- learner claims state and satisfy their validation mode;
- paper figures regenerate from saved panel data;
- one documented command reproduces the approved paper release;
- legacy scripts are wrappers or explicitly archived.
