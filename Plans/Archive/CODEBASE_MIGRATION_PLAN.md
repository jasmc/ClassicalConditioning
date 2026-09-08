# Codebase Migration Plan

> **Archived 2026-09-08.** This detailed strangler plan is historical. Living
> authorities are [MASTER_ANALYSIS_MIGRATION_PLAN.md](../MASTER_ANALYSIS_MIGRATION_PLAN.md)
> (§3.3 migration method) and
> [REPOSITORY_MIGRATION_MAP.md](../REPOSITORY_MIGRATION_MAP.md) (file layout and
> moves). Lane-1 script-side corrections are retired; package-first wins.
> Do not treat this file as an active implementation queue.

## Purpose

Migrate the ClassicalConditioning analysis repository from large, stateful research scripts into a tested, reproducible scientific analysis package without losing the ability to work interactively in VS Code or notebooks.

The target design is:

```text
Immutable domain definitions and configurations
                    +
        validated tabular scientific data
                    |
                    v
          pure transformation functions
                    |
                    v
       small workflow/orchestrator classes
                    |
                    v
      versioned artifacts and provenance
                    |
                    v
      reusable panels and figure builders
```

The migration must preserve current behavior where preservation is intended, isolate deliberate scientific corrections, and make every meaningful difference measurable.

## Migration lanes

Run two coordinated lanes instead of forcing urgent scientific corrections to wait for the complete architecture.

### Lane 1: scientific-correction fast path

Purpose:

- Resolve the critical analysis issues needed for the paper.
- Reprocess the paper dataset with corrected, tested calculations.
- Quantify whether conclusions change.

Constraints:

- It may use the current script structure where necessary.
- It must still use explicit configuration, synthetic reference tests, a frozen cohort manifest, and recorded provenance.
- It must not introduce another undocumented implementation that later becomes permanent.
- Corrected functions should be written so they can move into the package without redesign.

Priority order:

1. Canonical vigor
2. Filtering and bout detection
3. Frame-loss detection
4. Full raw-data reprocessing
5. Cohort definition
6. Missing-data and movement-outcome handling
7. Scaling and normalized vigor
8. Fish-level bootstrap and primary statistics

### Lane 2: structural package migration

Purpose:

- Build the durable package, schemas, artifact model, API, CLI, and figure system.

Lane 2 follows the phased strangler migration below. Lane 1 outputs become validated reference artifacts and scientific specifications for Lane 2.

### Coordination rule

Lane 1 decides approved scientific behavior. Lane 2 must not independently redefine it.

## Related documents

- `../README.md`: project overview and current pipeline
- `../docs/analysis/ANALYSIS_FILES_INDEX.md`: current file inventory, execution order, and version map
- `../docs/analysis/ANALYSIS_ISSUES.md`: scientific, statistical, and reproducibility issues
- `SCIENTIFIC_FIGURE_PIPELINE_PLAN.md`: scientific figure architecture
- `LEARNER_STRATIFIED_VIGOR_ANALYSIS_PLAN.md`: learner/non-learner analysis design

## Executive decision

Refactor the repository, but do not perform a big-bang object-oriented rewrite.

Use:

- Frozen dataclasses for identities, configurations, protocol definitions, artifacts, and result bundles
- Enums for finite scientific categories
- DataFrames and NumPy arrays for observations
- Pure functions for scientific transformations
- A small number of pipeline classes for orchestration
- Explicit repositories/adapters for file I/O
- Declarative figure specifications and reusable panel functions
- Compatibility wrappers while numbered scripts are retired incrementally

Avoid:

- One mutable `Experiment` god object
- One Python object per frame
- Deep inheritance trees for experiment types
- Hidden state transitions such as `fish.normalize().filter().plot()`
- Figure-specific subclasses such as `Figure2`
- Simultaneous refactoring and untracked scientific corrections
- Replacing the complete working pipeline before validating intermediate stages

## Goals

### Scientific goals

1. Give every scientific quantity one canonical definition.
2. Make units, windows, thresholds, alignments, and cohorts explicit.
3. Prevent different analysis stages from silently using different fish.
4. Make classification and statistical methods versioned and auditable.
5. Trace every result and figure to code, data, configuration, and cohort.
6. Provide reference tests for scientifically important calculations.

### Engineering goals

1. Remove analytical dependence on module-level mutable globals.
2. Separate file I/O, computation, plotting, and export.
3. Replace ambiguous artifact discovery with explicit references.
4. Define and validate DataFrame schemas.
5. Make stages callable from scripts, tests, notebooks, and a CLI.
6. Preserve interactive use and inspectable intermediate data.
7. Support incremental migration with compatibility wrappers.
8. Make failures explicit and actionable.

### Non-goals

The migration does not initially aim to:

- Introduce a database
- Create a web application
- Replace pandas with a custom object store
- Create one object for every trial or frame
- Reimplement stable scientific libraries
- Automatically choose a learner classifier
- Correct every analysis issue in one release
- Reformat all manuscript figures before foundational analysis is validated

## Success criteria

The migration is successful when:

1. A small fixture runs end to end through the new package.
2. Current behavior is reproduced for behavior-preserving stages within declared tolerances.
3. Deliberate scientific corrections are isolated, tested, and documented.
4. Each table passes a named schema contract.
5. Each artifact records provenance.
6. Cohort membership is defined once and reused everywhere.
7. Trial/block/phase mapping is defined once and reused everywhere.
8. Numbered scripts are thin wrappers or retired.
9. At least one major paper figure regenerates from explicit inputs.
10. Tests cover core calculations, integration boundaries, statistics, and figures.
11. Researchers can run and inspect individual stages without understanding internal framework machinery.

For scientifically corrected stages, success means agreement with the approved scientific reference specification, not agreement with legacy numerical output.

## Architectural principles

## 1. Functional core, imperative shell

The scientific core should consist of functions that:

- Receive explicit inputs
- Return explicit outputs
- Avoid reading global variables
- Avoid reading or writing files
- Avoid plotting
- Avoid mutating caller-owned data where practical
- Validate assumptions
- Preserve units and keys

The imperative shell should:

- Resolve configuration
- Locate explicit input artifacts
- Call scientific functions
- Save outputs
- Record provenance
- Report progress

## 2. Composition over inheritance

Represent experiment differences as data:

```python
ExperimentSpec(
    paradigm=Paradigm.TRACE,
    trace_interval_s=3.0,
    cs_duration_s=10.0,
    ...
)
```

Do not create:

```python
class ThreeSecondTraceExperiment(TraceExperiment):
    ...
```

## 3. Immutable definitions, immutable run settings

Use frozen dataclasses for definitions and resolved settings.

```python
@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float
```

Never change a run configuration after processing begins.

`frozen=True` prevents attribute rebinding but does not make a contained DataFrame immutable. For table-bearing result objects, integrity is enforced by:

- Copy-on-entry or read-only convention
- Canonical serialization
- Stored content hash
- Hash validation on load
- No in-place mutation in scientific functions

## 4. Tables for observations

Use tabular structures for:

- Frame/sample observations
- Stimulus events
- Trial-aligned samples
- Trial summaries
- Fish summaries
- Cohort manifests
- Statistical results

Do not build a deeply nested object graph containing every observation.

## 5. Explicit artifacts

Each stage receives explicit input artifact references and returns explicit output artifact references.

Avoid:

```text
load the first matching file
load the latest matching file
infer the cohort from the filename
```

## 6. Scientific names and display names are different

Use stable machine identifiers:

```text
trial_number
fish_id
condition_id
scaled_vigor
```

Use display labels only in the figure layer:

```text
Trial number
Normalized vigor (AU)
```

## 7. Units are part of the contract

Column names, schemas, or metadata must distinguish:

- Frames
- Milliseconds
- Seconds
- Degrees
- Degrees per millisecond
- Ratios
- Dimensionless display normalization

## 8. Fail closed

The new pipeline should stop on:

- Duplicate fish keys
- Ambiguous artifact selection
- Invalid trial mapping
- Missing required columns
- Cohort mismatch
- Unknown condition
- Missing provenance
- Failed model convergence when inference depends on the model

Do not continue with success-shaped empty outputs.

## Proposed package structure

```text
ClassicalConditioning/
|-- pyproject.toml
|-- src/
|   `-- classical_conditioning/
|       |-- __init__.py
|       |-- domain/
|       |   |-- experiment.py
|       |   |-- fish.py
|       |   |-- trials.py
|       |   |-- cohort.py
|       |   |-- artifacts.py
|       |   `-- provenance.py
|       |-- config/
|       |   |-- general.py
|       |   |-- experiments.py
|       |   |-- preprocessing.py
|       |   |-- population.py
|       |   |-- statistics.py
|       |   `-- figures.py
|       |-- schemas/
|       |   |-- raw.py
|       |   |-- samples.py
|       |   |-- trials.py
|       |   |-- summaries.py
|       |   |-- cohorts.py
|       |   `-- classification.py
|       |-- io/
|       |   |-- camera.py
|       |   |-- tracking.py
|       |   |-- protocol.py
|       |   |-- artifacts.py
|       |   `-- hashing.py
|       |-- preprocessing/
|       |   |-- synchronization.py
|       |   |-- filtering.py
|       |   |-- vigor.py
|       |   |-- bouts.py
|       |   |-- stimuli.py
|       |   `-- segmentation.py
|       |-- analysis/
|       |   |-- cohorts.py
|       |   |-- alignment.py
|       |   |-- blocks.py
|       |   |-- scaling.py
|       |   |-- binning.py
|       |   |-- trial_metrics.py
|       |   |-- population.py
|       |   |-- bootstrap.py
|       |   |-- longitudinal.py
|       |   `-- learners/
|       |       |-- base.py
|       |       |-- features.py
|       |       |-- classifier.py
|       |       `-- manifest.py
|       |-- figures/
|       |   |-- theme.py
|       |   |-- panels/
|       |   |-- specs/
|       |   |-- compose.py
|       |   |-- export.py
|       |   `-- qc.py
|       |-- pipelines/
|       |   |-- preprocess.py
|       |   |-- quality_control.py
|       |   |-- group.py
|       |   |-- scaled_vigor.py
|       |   |-- normalized_vigor.py
|       |   |-- learners.py
|       |   `-- figures.py
|       |-- cli/
|       |   |-- main.py
|       |   `-- commands/
|       `-- exceptions.py
|-- tests/
|   |-- unit/
|   |-- contracts/
|   |-- integration/
|   |-- regression/
|   |-- statistics/
|   |-- figures/
|   |-- end_to_end/
|   `-- fixtures/
|-- scripts/
|   `-- compatibility/
|-- docs/
|-- build/
`-- legacy/
```

The final directory structure may be introduced gradually. Creating empty architecture in advance is not useful; add modules only when migrating real behavior.

## Domain model

## `ExperimentSpec`

Represents a protocol definition, not loaded observations.

```python
@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paradigm: Paradigm
    conditions: tuple[ConditionSpec, ...]
    cs_duration_s: float
    response_window: TimeWindow
    trial_map: tuple[TrialSpec, ...]
```

Responsibilities:

- Validate condition uniqueness
- Validate trial-map completeness
- Expose condition and trial lookup

Not responsible for:

- Reading files
- Storing all fish data
- Running analysis
- Drawing figures

## `FishKey` and `FishMetadata`

```python
@dataclass(frozen=True, order=True)
class FishKey:
    experiment_id: str
    day: str
    fish_number: str


@dataclass(frozen=True)
class FishMetadata:
    key: FishKey
    condition_id: str
    strain: str
    age_dpf: int
    rig_id: str
```

The key must be stable and unique across combined experiments.

## `Recording`

Represents one acquisition and its source artifacts.

```python
@dataclass(frozen=True)
class Recording:
    fish: FishMetadata
    camera_log: ArtifactRef
    tracking_log: ArtifactRef
    protocol_log: ArtifactRef
```

It does not load all data into object attributes by default.

## `TrialSpec`

```python
@dataclass(frozen=True)
class TrialSpec:
    trial_number: int
    alignment: Alignment
    phase: Phase
    block_5_id: str
    block_10_id: str
    is_catch: bool
    expected_us_time_s: float | None
```

## `Phase`

Use an enum:

```python
class Phase(str, Enum):
    PRIMING = "priming"
    PRE_TRAINING = "pre_training"
    TRAINING = "training"
    TESTING = "testing"
    RETRAINING = "retraining"
    RETENTION = "retention"
```

Do not call phases `Session` unless separate acquisition sessions are scientifically and technically distinct.

## `CohortManifest`

Represents one explicit analysis cohort.

```python
@dataclass(frozen=True)
class CohortManifest:
    cohort_id: str
    table: pd.DataFrame
    content_hash: str
```

Minimum columns:

```text
experiment_id
fish_id
condition_id
technical_valid
included
exclusion_reason
cohort_role
```

The DataFrame is mutable at the Python level. `content_hash` is the integrity control; code must validate it when loading and create a new manifest after any change.

## `ArtifactRef`

```python
@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    path: Path
    content_hash: str
    schema_version: str
```

## `AnalysisRun`

```python
@dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    experiment_id: str
    cohort_id: str
    config_hash: str
    code_commit: str
    random_seeds: Mapping[str, int]
    software_environment_hash: str
    created_at: datetime
```

The environment record should include:

- Python version
- Package versions
- NumPy/SciPy/statsmodels versions
- BLAS/LAPACK implementation
- Thread-count settings
- Matplotlib backend and font inventory for figure builds

## `MajorFigureSpec`

Use a declarative specification rather than a figure subclass.

```python
@dataclass(frozen=True)
class MajorFigureSpec:
    figure_id: str
    width_mm: float
    height_mm: float
    mosaic: tuple[tuple[str, ...], ...]
    panels: Mapping[str, PanelSpec]
```

## Data contracts

Define contracts before moving transformations.

## Raw camera table

Required fields:

```text
frame_id
elapsed_time_ms
absolute_time_ms
```

Checks:

- Frame ID integer
- Monotonic timestamps
- Unique frame IDs
- Explicit missing-frame report

## Raw tracking table

Required fields:

```text
frame_id
angle_0_deg
...
angle_n_deg
```

Checks:

- One row per frame
- Known number of segments
- Finite values or explicit missing values
- Units recorded

## Stimulus event table

Required fields:

```text
event_type
start_time_ms
end_time_ms
source_event_id
```

Checks:

- Start before end
- Known event types
- Stable ordering

## Processed sample table

Candidate fields:

```text
experiment_id
fish_id
condition_id
alignment
trial_number
trial_time_s
absolute_time_ms
tail_angle_deg
vigor_deg_per_ms
bout
cs_active
us_active
phase
block_5_id
block_10_id
```

Primary key:

```text
experiment_id + fish_id + alignment + trial_number + sample_index
```

## Trial summary table

Candidate fields:

```text
experiment_id
fish_id
condition_id
alignment
trial_number
phase
block_5_id
block_10_id
baseline_vigor
response_vigor
normalized_vigor
movement_probability
fraction_time_moving
bout_count
valid_baseline_fraction
valid_response_fraction
```

Primary key:

```text
experiment_id + fish_id + alignment + trial_number
```

## Classification manifest

Use the schema described in `LEARNER_STRATIFIED_VIGOR_ANALYSIS_PLAN.md`.

## Configuration design

## Configuration layers

1. `ExperimentSpec`: scientific protocol
2. `PreprocessingConfig`: signal processing
3. `CohortConfig`: inclusion policy
4. `PopulationAnalysisConfig`: scaling, binning, aggregation
5. `StatisticsConfig`: models, resampling, corrections
6. `LearnerClassifierConfig`: one versioned classifier
7. `FigureTheme`: visual design
8. `MajorFigureSpec`: figure layout

## Resolution rules

Configuration should resolve in one place:

```text
defaults
  -> experiment-specific values
  -> explicit run overrides
  -> validation
  -> frozen resolved configuration
```

The resolved configuration is saved with every run.

Do not allow:

- A dataclass default and top-level constant to disagree
- Plotting code to redefine analysis windows
- Filename suffixes to be the only record of configuration

## Serialization

All frozen configurations should support stable serialization to JSON-compatible data.

Hash:

- Resolved configuration
- Cohort manifest
- Input artifacts
- Trial map

## Canonical artifact formats

Use:

- **Parquet** for canonical tabular artifacts where supported
- **JSON** for configuration, provenance, schemas, and manifests
- **CSV** only for human-readable exports and interoperability
- **NumPy NPZ** for dense numerical arrays when Parquet is unsuitable
- **SVG/PDF** for vector figures

Treat pickle as a legacy compatibility format. New canonical artifacts should not depend on arbitrary Python object deserialization.

### Recommended format by artifact

| Artifact | Recommended format | Reason |
| --- | --- | --- |
| Processed sample/time-series tables | Parquet | Typed columns, compression, column selection, language interoperability |
| Per-trial and per-fish summaries | Parquet | Stable tabular schema and efficient analysis |
| Cohort and classification manifests | Parquet plus CSV export | Typed canonical source plus human-readable copy |
| Small configuration and provenance records | JSON | Portable, reviewable, and easy to validate |
| Large dense numerical arrays | NPZ | Compact NumPy-native storage without arbitrary object deserialization |
| Large chunked multidimensional arrays | Zarr, only if needed | Chunked access and scalable array storage |
| Human review tables | CSV | Universally readable, but not the canonical typed source |
| Vector figures | SVG and PDF | Editable and publication-ready vector output |
| Raster previews | PNG | Portable preview format |
| Legacy compatibility | Gzip-compressed pickle | Read-only migration input; do not use for new canonical outputs |

### Why Parquet is preferred over pickle for tables

Parquet provides:

- Explicit column types
- Efficient compression
- Selective column reading
- Better compatibility outside Python
- Safer loading because it does not deserialize arbitrary Python objects
- Easier schema validation

Potential limitations:

- pandas categorical metadata must be tested across versions.
- Sparse pandas columns may need conversion to dense or a documented representation.
- DataFrame indexes should be reset into explicit columns before saving.
- Units and scientific meaning still require schema metadata; Parquet alone does not provide them.

### Migration from existing pickle files

1. Treat existing pickle files as immutable legacy inputs.
2. Load them only in the controlled migration environment.
3. Validate columns, dtypes, keys, categories, and row counts.
4. Normalize index and sparse-column representations.
5. Write a schema-versioned Parquet artifact.
6. Compare scientific summaries before accepting the conversion.
7. Record the source pickle hash in the new artifact's provenance.

Stable table hashes should use:

- Canonical column order
- Canonical row order by declared keys
- Normalized dtypes
- Explicit missing-value representation
- Schema version

## Pipeline classes

Use pipeline classes only for orchestration.

## `PreprocessingPipeline`

Responsibilities:

1. Resolve one recording
2. Read raw artifacts through adapters
3. Validate frame sequence
4. Synchronize streams
5. Apply canonical filtering
6. Calculate canonical vigor
7. Detect bouts
8. Annotate stimuli
9. Segment trials
10. Validate processed table
11. Save artifact and provenance

Scientific calculations remain in pure functions.

## `QualityControlPipeline`

Responsibilities:

- Calculate QC metrics
- Generate QC tables
- Generate review figures
- Propose technical exclusion reasons
- Write a draft cohort manifest

It should not silently move raw files.

## `GroupingPipeline`

Responsibilities:

- Apply an explicit cohort manifest
- Split alignments
- Validate trial mapping
- Smooth/downsample only when explicitly configured
- Save condition-level or pooled artifacts

## `PopulationAnalysisPipeline`

Responsibilities:

- Build temporal profiles
- Build trial summaries
- Aggregate within fish before across fish
- Save coverage and sample-size tables

## `LearnerClassificationPipeline`

Responsibilities:

- Load explicit trial summary and cohort artifacts
- Apply one named classifier version
- Export one versioned classification manifest
- Record model diagnostics

The classifier algorithm should conform to a small protocol/interface.

## `FigureBuildPipeline`

Responsibilities:

- Load explicit panel data
- Render reusable panels
- Compose major figures
- Export formats
- Save provenance
- Run figure QC

## Scientific correction policy

## Track A: behavior-preserving migration

Purpose:

```text
Change structure without changing scientific output.
```

Permitted changes:

- Move code
- Introduce dataclasses
- Replace globals with arguments
- Add schemas
- Separate I/O and computation
- Add explicit return values
- Add wrappers

Validation:

- Characterization tests
- Regression fixtures
- Table comparisons
- Figure structural comparisons

Track A equivalence work should focus on:

- File parsing
- Identity handling
- Schema conversion
- Artifact plumbing
- Pure structural extraction

For scientific stages already scheduled for correction, capture legacy output as a comparison baseline but do not spend substantial effort certifying it as the desired result.

## Track B: deliberate scientific correction

Purpose:

```text
Change scientific behavior to match the approved specification.
```

Examples:

- Correct vigor definition
- Implement spatial smoothing
- Implement secondary bout threshold
- Correct frame-loss detection
- Correct baseline window
- Unify scaled vigor
- Apply cohort consistently
- Change missing-data treatment
- Use fish-level bootstrap
- Change statistical model

Each correction requires:

1. Scientific decision record
2. Mathematical specification
3. Synthetic reference tests
4. Before/after result comparison
5. Sample-size comparison
6. Effect-size comparison
7. Figure comparison
8. Manuscript impact assessment

## Track labels

Every migration pull request or commit should identify:

```text
behavior-preserving
scientific-correction
mixed - not allowed unless explicitly approved
```

Prefer separate branches or pull requests for the two tracks.

## Test strategy

## Test infrastructure

Use the repository's chosen test runner after adding a dependency manifest. Do not invent multiple testing frameworks.

Tests should be deterministic, small, and runnable without access to private full datasets.

## Fixture policy

Fixtures must:

- Contain no sensitive or unnecessary raw data
- Be small enough for source control when permitted
- Have documented expected behavior
- Include edge cases
- Use stable IDs

Fixture types:

1. Fully synthetic numerical fixtures
2. Synthetic end-to-end recording fixture
3. Small de-identified real-data regression fixture only if separately approved and legally/ethically permitted
4. Saved expected tables and summaries

### Experimental-data exclusion

This migration plan does not include experimental data, embed data in documentation, or require data to be committed to Git.

Default policy:

- Raw and processed experimental data remain outside the source repository.
- Documentation contains schemas, examples, and paths only, not observations.
- Automated tests use synthetic fixtures by default.
- Real-data regression tests are optional and require explicit approval.
- If real-data fixtures are approved, store only the minimum de-identified subset in an approved protected location; reference it through configuration rather than embedding it in the plan.
- Never commit raw camera, tracking, stimulus, per-fish, pooled, or classification data merely to support the migration.
- Git ignore rules and pre-commit/CI checks should prevent accidental addition of known data formats and result directories.

The artifact and provenance system records paths, identifiers, schemas, and hashes. A hash identifies content but does not include or reconstruct that content.

## Unit tests

### Vigor

Test:

- Constant angles
- One moving segment
- Multiple segments moving together
- Opposing segment movement
- Known angular speed
- Missing values
- Frame-rate scaling

### Filtering

Test:

- Constant signal
- Impulse
- Step
- Sinusoid
- Edges
- Missing samples
- Spatial window
- Temporal window

### Bout detection

Test:

- Valid bout
- Too-short bout
- Short inter-bout gap
- Valid separate bouts
- Primary threshold
- Secondary threshold
- Boundary bout
- No movement

### Synchronization

Test:

- Perfect frame sequence
- One missing frame
- Multiple missing frames
- Jitter
- Duplicate frame
- Protocol between frames
- Protocol outside acquisition

### Trial segmentation

Test:

- Exact onset
- Exact offset
- Negative and positive trial bounds
- Catch trial
- CS and US alignment
- Overlapping windows

### Scaling

Test:

- Known baseline and response
- Zero baseline
- No baseline movement
- Quantile display normalization
- Clipping
- Analytical versus display transforms

### Cohort

Test:

- Included fish retained
- Excluded fish removed
- Unknown fish rejected or explicitly unmatched
- Duplicate manifest row rejected
- Cohort hash stable

### Blocks

Test:

- Every trial maps once
- Catch status correct
- Phase boundaries correct
- Five- and ten-trial mapping consistent

## Contract tests

For every schema:

- Required columns
- Column types
- Units
- Key uniqueness
- Valid categories
- Monotonic time
- Allowed missingness
- Alignment validity
- Condition validity

## Characterization tests

Before migrating each current stage, capture:

- Input inventory
- Output columns
- Dtypes
- Row counts
- Fish counts
- Trial counts
- Bout counts
- Time range
- Selected numerical summaries
- Output names
- Figure size and axes inventory

Characterization tests preserve current behavior; they do not certify correctness.

## Integration tests

Build a compact synthetic experiment:

```text
2 conditions
2-3 fish per condition
pre-training, training, and test phases
several CS and US events
one catch trial
known bouts
one missing-frame case
one excluded fish
```

Test:

```text
raw tables
  -> synchronization
  -> filtering
  -> vigor
  -> bouts
  -> trial segmentation
  -> cohort application
  -> grouping
  -> trial metrics
  -> temporal profiles
```

## Regression tests

Use selected representative real fish if permitted:

- Control
- Apparent learner
- Apparent non-learner
- Sparse movement
- Tracking edge case
- Catch-trial response

Compare:

- Canonicalized tables
- Scientific summaries
- Cohort counts
- Model inputs

Do not compare raw pickle bytes.

## Statistical tests

For a fixed dataset, assert:

- Formula
- Grouping variable
- Reference category
- Number of observations
- Number of fish
- Coefficients
- Standard errors
- Confidence intervals
- p-values within tolerance
- Convergence
- Random-effect structure
- Correction method

Pin the numerical environment for exact regression suites:

- Package versions
- BLAS/LAPACK implementation
- Thread counts
- Random seeds

Before the primary model is approved, favor synthetic invariants, input-shape checks, convergence checks, and broad effect-direction checks over brittle exact p-value baselines.

## Property tests

Important invariants:

- Input row order does not change grouped output.
- Each fish receives equal weight in fish-level aggregation.
- Re-running with the same seed produces the same bootstrap result.
- Adding an excluded fish does not change selected-cohort results.
- One trial maps to exactly one phase and block at each resolution.
- No-movement data produces no bouts.
- Constant signals remain constant after documented edge handling.
- Classification joins are many-to-one.

## Figure tests

Structural:

- Canvas size
- Axes count
- Labels and units
- Limits and ticks
- Legend entries
- Condition colors
- Panel labels
- Colorbar ranges
- Output formats

Visual regression:

- Selected representative panels
- Tolerance-based image comparison
- Separate references for intentional scientific corrections

Use the non-interactive Agg backend, pinned Matplotlib and font versions, and fixed figure dimensions.

Do not approve or commit final visual-regression baselines for a scientific panel until the corresponding scientific corrections are approved. Legacy images may be retained as clearly labeled comparison artifacts, not as correctness references.

## End-to-end acceptance test

One command:

```powershell
python -m classical_conditioning build-fixture
```

should generate:

- Processed samples
- QC report
- Cohort manifest
- Condition-level data
- Trial summary
- Population result
- One figure
- Provenance JSON

## Migration phases

## Phase 0: Governance and baseline freeze

### Objectives

- Prevent loss of current behavior and outputs
- Establish decision authority

### Tasks

1. Identify scientific owner and code owner.
2. Identify current manuscript figures and their source scripts.
3. Freeze current outputs in a dated snapshot.
4. Record Git commit and environment.
5. Record known data locations without committing private data.
6. Select representative regression fish.
7. Decide target Python version.
8. Decide dependency-management approach.
9. Decide test runner.
10. Define change labels for behavior-preserving versus correction work.
11. Create one consolidated, versioned `SCIENTIFIC_ANALYSIS_SPECIFICATION.md`.
12. Link individual scientific decision records from that specification.

### Deliverables

- Baseline snapshot manifest
- Ownership matrix
- Migration decision log
- Approved fixture list
- Initial scientific analysis specification

### Exit gate

No structural migration begins until current important outputs are identifiable.

## Phase 1: Package and test foundation

### Objectives

- Make code importable
- Establish repeatable tests

### Tasks

1. Add `pyproject.toml`.
2. Record current runtime dependencies and versions.
3. Create `src/classical_conditioning/`.
4. Create `tests/`.
5. Configure one test command.
6. Add CI or a documented local validation command.
7. Add a package version.
8. Add custom exception types.

### Deliverables

- Installable local package
- Test runner
- Empty but valid package
- Dependency manifest

### Exit gate

- Package imports successfully.
- Tests run from a clean environment.
- Current scripts still run unchanged.

## Phase 2: Characterization and scientific reference tests

### Objectives

- Build the safety net before moving behavior

### Tasks

1. Create synthetic camera, tracking, and protocol fixtures.
2. Add characterization tests for stage 1.
3. Add reference tests for vigor.
4. Add reference tests for filtering.
5. Add reference tests for bout detection.
6. Add reference tests for synchronization.
7. Add reference tests for trial mapping.
8. Add reference tests for scaling and normalized vigor.
9. Capture stage-3, stage-4, and stage-5 output contracts.

### Deliverables

- Fixture library
- Core scientific reference tests
- Current-behavior characterization reports

### Exit gate

Core calculations have tests that can detect both structural drift and known scientific discrepancies.

## Scientific decision gate A: preprocessing definitions

Before Phase 6 begins, the scientific owner must approve and record:

- Canonical vigor definition
- Spatial filtering definition
- Temporal filtering definition
- Bout-detection algorithm and thresholds
- Frame-loss policy
- Trial-alignment convention

These decisions must be consolidated into `SCIENTIFIC_ANALYSIS_SPECIFICATION.md`.

## Phase 3: Domain definitions and configuration

### Objectives

- Replace implicit string/global concepts with explicit definitions

### Tasks

1. Add `TimeWindow`, `Alignment`, `Phase`, and `Paradigm`.
2. Add `FishKey` and `FishMetadata`.
3. Add `ConditionSpec`.
4. Add `TrialSpec` and trial-map validation.
5. Add `ExperimentSpec`.
6. Port `general_configuration.py` into frozen config dataclasses.
7. Port one pilot experiment from `experiment_configuration.py`.
8. Add stable serialization and hashes.
9. Add compatibility adapters from current config objects.

### Deliverables

- Domain module
- Frozen resolved configuration
- Validated pilot experiment

### Exit gate

The pilot experiment can be represented without importing legacy global configuration.

## Phase 4: Schema and artifact layer

### Objectives

- Make tables and files explicit

### Tasks

1. Define schemas for raw and processed tables.
2. Add schema validation helpers.
3. Add `ArtifactRef`.
4. Add content hashing.
5. Add artifact metadata/provenance format.
6. Add explicit artifact resolver.
7. Implement safe output directories keyed by run ID.
8. Prevent silent overwrite.

### Deliverables

- Schema contracts
- Artifact API
- Provenance JSON schema

### Exit gate

A saved pilot artifact can be loaded only after identity, schema, and hash validation.

## Phase 5: Extract file I/O

### Objectives

- Separate parsing from scientific computation

### Migration order

1. Camera reader
2. Tracking reader
3. Protocol reader
4. Fish filename parser
5. Artifact writer

### Tasks

- Move one function at a time from `data_io.py` and `file_utils.py`.
- Preserve compatibility wrappers.
- Add malformed-file tests.
- Replace broad catches with specific exceptions.
- Make parsing formats explicit.

### Exit gate

Current stage-1 script can use the new readers through adapters and produce equivalent parsed tables.

## Phase 6: Extract preprocessing core

### Objectives

- Isolate and test the most scientifically consequential transformations

### Migration order

1. Frame validation
2. Synchronization
3. Interpolation
4. Tail-angle transformation
5. Spatial filtering
6. Temporal filtering
7. Vigor
8. Bout metric
9. Bout detection
10. Stimulus annotation
11. Trial segmentation

### Process for each operation

1. Write mathematical contract.
2. Add characterization test.
3. Extract behavior-preserving implementation.
4. Compare outputs.
5. If correction is required, create a separate correction change.
6. Update method documentation.

### Exit gate

The pilot fish can be processed using the new core with an explicit report of preserved and deliberately changed values.

## Phase 6B: Full corrected batch reprocessing

### Objectives

- Apply the approved preprocessing core to every recording required for the paper
- Produce the complete corrected dataset from which cohort decisions will be made

### Preconditions

- Scientific decision gate A approved
- Core synthetic tests passing
- Pilot comparisons reviewed
- Input recording inventory frozen

### Tasks

1. Process every paper-scope recording from raw camera, tracking, and protocol files.
2. Record successes, failures, warnings, and reasons.
3. Produce one processed sample artifact per fish.
4. Produce a complete coverage report.
5. Produce before/after summaries relative to the legacy pipeline.
6. Investigate all unexpected failures or large numerical differences.
7. Freeze the corrected processed-artifact inventory.

### Deliverables

- Full corrected processed dataset
- Recording-level processing report
- Coverage and failure report
- Legacy-versus-corrected difference report
- Artifact and environment provenance

### Exit gate

- Every paper-scope recording is accounted for.
- Failed recordings have an explicit disposition.
- Corrected processed artifacts are approved as the sole input to cohort construction.

No cohort may be frozen from legacy or partially reprocessed samples.

## Scientific decision gate B: cohort and missingness

Before Phase 7 begins, the scientific owner must approve and record:

- Technical validity criteria
- Behavioral inclusion policy
- Treatment of response-dependent criteria
- Missing-data semantics
- Required movement-coverage thresholds
- Primary analysis population
- Sensitivity-analysis populations

## Phase 7: Cohort and QC migration

### Objectives

- Replace mutable file-moving exclusions with explicit cohort data

### Tasks

1. Define technical QC metrics.
2. Define behavioral inclusion criteria.
3. Separate technical validity from behavioral engagement.
4. Generate a draft cohort manifest.
5. Support human review and reason editing.
6. Freeze the approved cohort manifest.
7. Apply cohort by validated join.
8. Stop moving raw files as part of analysis.
9. Retain legacy exclusion-file import.

All QC metrics must be recalculated from the full corrected Phase-6B artifacts.

### Exit gate

Every downstream dataset can prove which fish it contains and why.

## Phase 8: Grouping, alignment, and block mapping

### Objectives

- Replace stage-specific trial mappings and grouping behavior

### Tasks

1. Create canonical trial map.
2. Implement CS/US alignment functions.
3. Implement five-trial, ten-trial, and phase mapping.
4. Identify catch trials explicitly.
5. Implement fish-first aggregation.
6. Extract smoothing/downsampling as explicit optional transforms.
7. Save condition-level artifacts with provenance.

### Exit gate

Stage-3 outputs for the pilot are reproducible from explicit cohort and trial maps.

## Phase 9: Behavioral metrics and temporal profiles

### Objectives

- Canonicalize overlapping stage-1, stage-3, stage-4, and stage-5 calculations

### Tasks

1. Define conditional bout vigor.
2. Define movement probability.
3. Define fraction time moving.
4. Define bout count.
5. Decide analytical scaled-vigor formula.
6. Separate analytical scaling from display normalization.
7. Define normalized vigor.
8. Implement baseline and response windows.
9. Implement time binning.
10. Implement coverage reporting.
11. Implement fish-level and hierarchical bootstrap.

### Exit gate

Every metric has one canonical function, unit, schema field, and reference test.

## Scientific decision gate C: population inference

Before Phase 10 begins, the scientific owner must approve and record:

- Primary behavioral outcomes
- Analytical scaled-vigor definition
- Normalized-vigor definition
- Bootstrap resampling unit
- Bootstrap iterations and seed policy
- Primary longitudinal model
- Planned contrasts
- Multiple-comparison families

## Phase 10: Population statistics

### Objectives

- Centralize inference and diagnostics

### Tasks

1. Extract nonparametric comparisons.
2. Define comparison families.
3. Extract multiple-comparison correction.
4. Define one primary longitudinal model.
5. Validate random-effects structure.
6. Add convergence/singularity checks.
7. Add effect-size tables.
8. Save model inputs and results.
9. Add sensitivity analyses.

### Exit gate

Statistical results can be regenerated independently from a versioned trial-summary artifact.

## Phase 11: Learner classifier selection and migration

### Objectives

- Resolve four incompatible classifier versions

### Tasks

1. Define classifier evaluation criteria.
2. Compare versions using the same frozen input.
3. Compare control false-positive rate.
4. Compare label stability under resampling.
5. Compare sensitivity to transforms and thresholds.
6. Review circularity and validation strategy.
7. Select one canonical classifier.
8. Assign classifier version.
9. Implement it behind a stable interface.
10. Export canonical classification manifest.
11. Archive alternative implementations with documentation.

### Exit gate

One classifier version, configuration, and output schema are approved.

## Scientific decision gate D: learner classification

Before any learner-stratified confirmatory analysis begins, the scientific owner must approve and record:

- Canonical classifier implementation and version
- Response variable and transformation
- Feature definitions
- Reference/control definition
- Threshold calibration
- Primary learner label
- Validation mode
- Output schema

The approved classifier configuration and input artifact must be frozen and recorded in the classification manifest.

## Phase 12: Figure migration

### Objectives

- Implement `SCIENTIFIC_FIGURE_PIPELINE_PLAN.md`

### Tasks

1. Extend immutable figure theme.
2. Separate panel data preparation from drawing.
3. Migrate one trace panel.
4. Migrate one heatmap.
5. Migrate one time-course panel.
6. Migrate one block/statistics panel.
7. Compose one pilot major figure.
8. Add SVG/PDF/PNG export.
9. Add structural and visual tests.
10. Add provenance and QC.

### Exit gate

One paper figure regenerates from explicit input artifacts and passes QC.

## Phase 13: CLI and compatibility wrappers

### Objectives

- Provide stable commands while preserving researcher workflows

### Proposed commands

```powershell
python -m classical_conditioning preprocess --experiment allDelay --run-id <id>
python -m classical_conditioning qc --run-id <id>
python -m classical_conditioning cohort approve --run-id <id>
python -m classical_conditioning group --run-id <id>
python -m classical_conditioning analyze temporal --run-id <id>
python -m classical_conditioning analyze normalized-vigor --run-id <id>
python -m classical_conditioning classify-learners --run-id <id>
python -m classical_conditioning figure --figure Figure_2 --run-id <id>
```

### Compatibility

Numbered scripts should become thin wrappers that:

- Translate existing top-level settings into new config
- Emit deprecation warnings
- Call the new pipeline
- Preserve familiar outputs during transition

### Exit gate

The same stage can be invoked from CLI, Python, and notebook code.

## Phase 14: Legacy retirement

### Objectives

- Remove ambiguity without destroying historical reference

### Tasks

1. Confirm no active imports of `my_*`.
2. Move legacy modules under `legacy/`.
3. Add deprecation documentation.
4. Keep compatibility import shims temporarily if needed.
5. Archive noncanonical learner scripts.
6. Remove ambiguous first/latest artifact loading.
7. Remove duplicated scientific functions.
8. Remove obsolete `RUN_*` paths after migration.

### Exit gate

Every active workflow uses the package; legacy files are clearly non-executable reference or archived.

## Phase 15: Full validation and paper snapshot

### Objectives

- Demonstrate readiness for scientific use

### Tasks

1. Run full unit and integration suite.
2. Run all pilot regression comparisons.
3. Reprocess every raw recording in the defined paper scope, or verify by artifact hash that it was already reprocessed under the approved canonical pipeline.
4. Compare cohorts.
5. Compare metrics.
6. Compare statistical conclusions.
7. Regenerate manuscript figures.
8. Review methods text.
9. Create release notes.
10. Create immutable paper-analysis snapshot.

### Exit gate

Scientific owner approves the corrected results and paper artifacts.

The paper scope must be explicit. Experiments outside the current paper may migrate later, but no paper result may depend on a partially migrated experiment.

## Compatibility strategy

## Dual-run period

For migrated stages:

1. Run legacy path.
2. Run new path.
3. Compare normalized outputs.
4. Produce a difference report.
5. Approve equivalence or document correction.

## Output comparison

Compare:

- Fish set
- Trial set
- Row counts
- Column values
- Missingness
- Summary statistics
- Model input
- Model results
- Figure structure

## Tolerances

Define tolerances per quantity:

- Exact: IDs, categories, trial assignments
- Integer exact: counts
- Numeric tight: deterministic transforms
- Numeric moderate: optimization/model fits
- Visual tolerance: raster comparison

## Rollback

Each phase should leave the previous path callable until the new path passes its exit gate.

Do not remove legacy implementation in the same change that introduces its replacement.

## Delivery units

Prefer small vertical slices.

Example:

```text
FishKey + filename parser + schema + tests + compatibility wrapper
```

Avoid large horizontal changes such as:

```text
Create all domain classes before migrating any real behavior
```

## Process scaling

The gates in this plan protect scientific results, but the administrative process should remain proportionate to a small research team.

Use:

- One concise decision log rather than multiple approval systems
- One scientific owner and one implementation owner, who may be the same person
- Combined phase reviews where risks are low
- Lightweight checklists for structural utilities
- Full gates only for scientific transformations, cohorts, statistics, classifiers, and paper figures

Phases may run in parallel when dependencies permit. They are dependency stages, not necessarily calendar sprints.

## Suggested change sequence

1. Package/test foundation
2. Fish identity and file parsing
3. Camera/tracking/protocol readers
4. Frame validation
5. Synchronization
6. Vigor and bout detection
7. Trial map
8. Cohort manifest
9. Grouping
10. Trial metrics
11. Population statistics
12. Learner classifier
13. Figures
14. CLI
15. Legacy retirement

## Decision records required

Create short decision records for:

1. Canonical vigor definition
2. Filtering windows
3. Bout thresholds
4. Frame-loss policy
5. Missing-data semantics
6. Scaled-vigor formula
7. Cohort criteria
8. Bootstrap unit
9. Longitudinal statistical model
10. Learner classifier
11. Figure dimensions and theme
12. Artifact storage and versioning

### Decision-to-phase binding

| Decision | Must be approved before |
| --- | --- |
| Vigor, filtering, bout detection, frame loss | Phase 6 |
| Trial alignment and segmentation | Phase 6 |
| Cohort and inclusion policy | Phase 7 |
| Missing-data semantics | Phase 7 |
| Scaled and normalized vigor | Phase 9 completion |
| Bootstrap unit and seed policy | Phase 10 |
| Longitudinal model and planned contrasts | Phase 10 |
| Learner classifier | Phase 11 completion and any learner-stratified inference |
| Figure dimensions and theme | Phase 12 |
| Artifact format and versioning | Phase 4 |

All records roll up into `SCIENTIFIC_ANALYSIS_SPECIFICATION.md`, the authoritative human-readable specification.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Big-bang rewrite | Untraceable result differences | Incremental strangler migration |
| God-object design | Hidden state and tight coupling | Pure functions plus small orchestrators |
| Object-per-frame model | Memory and performance regression | DataFrames/arrays for observations |
| Refactor hides correction | Cannot explain changed results | Separate change tracks |
| Weak characterization | Regression goes unnoticed | Freeze fixtures and summaries first |
| Legacy behavior is wrong | Tests preserve an error | Separate characterization from scientific reference tests |
| Cohort drift | Figures use different fish | One hashed cohort manifest |
| File auto-discovery | Wrong artifact loaded | Explicit `ArtifactRef` |
| Config duplication | Active value unclear | One frozen resolved config |
| Model failure hidden | Invalid inference | Convergence and singularity gates |
| Research workflow becomes cumbersome | Users bypass package | Preserve interactive functions and wrappers |
| Framework overengineering | Migration stalls | Add architecture only with migrated behavior |
| Process overengineering | Work stalls in approvals | Scale gates to scientific risk and combine low-risk reviews |
| Dependency/version drift | Results change | Lock environment for paper release |
| Learner versions mixed | Invalid labels | One versioned classification manifest |
| Figure changes alter science | Misleading panels | Separate panel data, rendering, and layout |

## Performance plan

Before and after each data-heavy migration, measure:

- Peak memory
- Wall time
- Output size
- Row count

Use:

- Vectorized NumPy/pandas
- Efficient dtypes
- Chunked reading only when required
- Optional Numba only behind tested functions

Do not optimize by changing scientific behavior.

## Researcher usability plan

Provide a notebook-friendly API:

```python
run = load_analysis_run("...")
samples = load_processed_samples(run, fish_id="...")
trial = select_trial(samples, trial_number=25)
profile = build_temporal_profile(...)
draw_temporal_profile(ax, profile, theme)
```

Researchers should be able to:

- Inspect tables
- Run one fish
- Run one condition
- Override presentation settings
- Save exploratory outputs separately

Exploratory overrides must not silently become paper-analysis defaults.

## Documentation plan

Maintain:

- Architecture overview
- Data dictionary
- Configuration reference
- Pipeline command reference
- Experiment registry
- Scientific decision records
- Artifact/provenance specification
- Migration status matrix
- Legacy-to-new mapping
- Paper reproduction instructions

## Migration status matrix

Track each area:

| Area | Characterized | Reference tested | Extracted | Dual-run compared | Canonical | Legacy retired |
| --- | --- | --- | --- | --- | --- | --- |
| File parsing |  |  |  |  |  |  |
| Synchronization |  |  |  |  |  |  |
| Filtering |  |  |  |  |  |  |
| Vigor |  |  |  |  |  |  |
| Bout detection |  |  |  |  |  |  |
| Trial segmentation |  |  |  |  |  |  |
| Cohort |  |  |  |  |  |  |
| Grouping |  |  |  |  |  |  |
| Scaled vigor |  |  |  |  |  |  |
| Normalized vigor |  |  |  |  |  |  |
| Statistics |  |  |  |  |  |  |
| Learner classifier |  |  |  |  |  |  |
| Figures |  |  |  |  |  |  |

## Definition of done for a migrated component

A component is migrated only when:

- [ ] Scientific contract is documented.
- [ ] Inputs and outputs are typed or schema-validated.
- [ ] Units are explicit.
- [ ] Pure computational logic is separated from I/O.
- [ ] Characterization tests exist where replacing current behavior.
- [ ] Scientific reference tests exist where mathematically meaningful.
- [ ] Edge cases are tested.
- [ ] Errors are explicit.
- [ ] Provenance is recorded.
- [ ] Legacy and new outputs are compared.
- [ ] Differences are classified as regression or approved correction.
- [ ] Compatibility wrapper exists if the old entry point remains supported.
- [ ] Documentation is updated.
- [ ] Exit gate is approved.

## Definition of done for the full migration

- [ ] All active experiments use the package configuration model.
- [ ] All active data artifacts use versioned schemas.
- [ ] Raw I/O is separated from scientific transformations.
- [ ] Core metrics have synthetic reference tests.
- [ ] One cohort manifest controls every downstream stage.
- [ ] Trial/block/phase mapping is centralized.
- [ ] Scaled and normalized vigor have one canonical analytical definition each.
- [ ] Statistical models are versioned and diagnosed.
- [ ] One learner classifier is canonical.
- [ ] Figure panels are reusable and reproducible.
- [ ] A CLI and Python API are documented.
- [ ] Numbered scripts are retired or thin wrappers.
- [ ] Legacy modules are archived and cannot be imported accidentally.
- [ ] The end-to-end fixture passes.
- [ ] Paper figures regenerate from explicit inputs.
- [ ] A paper-analysis release records code, environment, data hashes, cohort, configuration, statistics, and figures.

## Immediate first milestone

The first milestone should be deliberately small:

### Scope

```text
Package foundation
+ FishKey/FishMetadata
+ TimeWindow/Alignment
+ one pilot ExperimentSpec
+ camera/tracking/protocol schemas
+ file parsers
+ synthetic fixtures
+ parser and identity tests
```

### Out of scope

- Vigor corrections
- Cohort redesign
- Population statistics
- Learner classification
- Figure refactoring

### Milestone acceptance

1. Existing scripts still run.
2. New package imports.
3. One fish's raw files parse through the new adapters.
4. Parsed tables match current readers.
5. Fish identity is stable and validated.
6. Tests run from one documented command.

## Next milestone

After the foundation:

```text
frame validation
  -> synchronization
  -> interpolation
  -> explicit comparison report
```

Do not begin migrating downstream statistics until the sample-level foundation is stable.

## Plan review checklist

Before beginning implementation, review this plan against:

- [ ] Incremental migration rather than rewrite
- [ ] Clear boundary between refactor and scientific correction
- [ ] Compatibility with pandas/NumPy/statsmodels
- [ ] Interactive researcher workflow
- [ ] Data privacy constraints
- [ ] Test-data availability
- [ ] Current manuscript deadlines
- [ ] Ownership and approval gates
- [ ] Learner-classifier decision dependency
- [ ] Figure-pipeline dependency
- [ ] Rollback at every phase
- [ ] Measurable exit criteria

## Independent plan review

This plan was independently reviewed after its first draft.

### Blocking findings addressed

1. **Full corrected reprocessing was not required before cohort freeze.**
   - Added Phase 6B.
   - Made corrected paper-scope artifacts a prerequisite for cohort construction.

2. **Scientific decisions were not bound to consuming phases.**
   - Added scientific decision gates A-D.
   - Added a decision-to-phase binding table.
   - Added a consolidated scientific analysis specification.

### Additional findings addressed

- Added a scientific-correction fast path.
- Scoped behavior-preserving equivalence work to appropriate stages.
- Added random seeds and numerical-environment provenance.
- Clarified that frozen dataclasses do not make DataFrames immutable.
- Defined Parquet/JSON as canonical artifact formats and pickle as legacy.
- Added environment controls for statistical and visual regression.
- Prohibited final visual baselines before scientific correction approval.
- Scaled governance to a small research team.
- Reconciled full paper-scope reprocessing requirements.

### Review conclusion

After these revisions, the migration is suitably incremental and scientifically gated. The highest residual risk is execution discipline: downstream cohort, statistical, classifier, or figure work must not begin before the relevant scientific decision and upstream artifact gates are satisfied.
