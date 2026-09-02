# Codebase Migration Plan

## Documentation baseline

This migration plan is based on:

- `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
- Source feature commits `505e29a` and `88fe4fc`
- Cherry-picked equivalents `4d6939f` and `f372101`
- Working-tree LogMedian compatibility adaptations dated 2026-08-28

It supersedes the earlier plan written against `b0dfcb3`.

## Decision

Refactor incrementally using:

```text
immutable definitions and configuration
              +
validated DataFrames/arrays
              |
              v
pure scientific functions
              |
              v
small pipeline orchestrators
              |
              v
versioned artifacts and provenance
```

Do not build a deeply nested, mutable object hierarchy.

## What changed in the baseline

Compared with `b0dfcb3`, `bf46bf7` already completed useful consolidation:

- Standard step-3 through step-6 files became one LogMedian analysis line.
- Competing learner-classifier files were removed.
- Legacy `my_*` modules were removed.
- `pipeline_utils.py` and `heatmap_utils.py` were added.
- A tracked data-flow description was added.
- Output path handling was improved.

The migration should build on those changes, not recreate removed alternatives.

The later learner-stratified work adds:

- Explicit source paths
- Pinned optional environment
- Checkpoints
- Fish-key validation
- Classification manifest
- Fish-first temporal aggregation
- Movement probability
- Catch-trial timing
- Provenance
- Synthetic tests

These are patterns to retain.

## Goals

### Scientific

- One canonical definition for each metric
- Explicit units, windows, alignments, and trial mappings
- One cohort manifest
- One learner classifier
- Saved model inputs and diagnostics
- Traceability from figure to source artifact

### Engineering

- Remove scientific dependence on mutable globals
- Separate I/O, transformation, statistics, rendering, and saving
- Validate DataFrame contracts
- Replace implicit file discovery with explicit artifacts
- Make stages callable from Python, CLI, tests, and notebooks
- Retire numbered scripts gradually through compatibility wrappers

### Non-goals

- No object per frame
- No object per trial stored across the dataset
- No deep experiment inheritance
- No `Figure1`/`Figure2` subclasses
- No database requirement
- No big-bang rewrite
- No experimental data committed with this migration

## Domain model

Use frozen dataclasses for definitions and identity.

### `ExperimentSpec`

Represents a protocol, not loaded observations.

```python
@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paradigm: Paradigm
    conditions: tuple[ConditionSpec, ...]
    cs_duration_s: float
    response_window: TimeWindow
    trials: tuple[TrialSpec, ...]
```

### `FishKey`

```python
@dataclass(frozen=True, order=True)
class FishKey:
    experiment_id: str
    day: str
    fish_number: str
```

### `FishMetadata`

```python
@dataclass(frozen=True)
class FishMetadata:
    key: FishKey
    condition_id: str
    strain: str
    age_dpf: int
    rig_id: str
```

### `TimeWindow`

```python
@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float
```

Validate that start precedes end and define boundary inclusion explicitly.

### `TrialSpec`

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

### `Recording`

References one fish's raw artifacts. It should not eagerly contain every row.

### `CohortManifest`

Wraps a validated fish-level table and content hash.

A frozen dataclass does not make its DataFrame immutable. Integrity comes from:

- No in-place mutation
- Canonical serialization
- Content hash
- Validation on load

### `ArtifactRef`

```python
@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    path: Path
    content_hash: str
    schema_version: str
```

### `AnalysisRun`

Records:

- Run ID
- Code commit
- Experiment
- Cohort hash
- Configuration hash
- Input artifact hashes
- Random seeds
- Python/package/BLAS versions
- Creation time

### `MajorFigureSpec`

Declarative layout and panel mapping, not a figure-specific subclass.

## Observation model

Keep observations in DataFrames or arrays.

Canonical tables:

### Raw camera

```text
frame_id
elapsed_time_ms
absolute_time_ms
```

### Raw tracking

```text
frame_id
angle_0_deg ... angle_n_deg
```

### Stimulus event

```text
event_id
event_type
start_time_ms
end_time_ms
```

### Processed sample

```text
experiment_id
condition_id
fish_id
alignment
trial_number
sample_index
trial_time_s
vigor_deg_per_ms
log_vigor
baseline_centered_log_vigor
bout
phase
block_5_id
block_10_id
is_catch
```

### Trial summary

```text
experiment_id
condition_id
fish_id
alignment
trial_number
median_baseline_log_vigor
median_cr_log_vigor
normalized_vigor_log_difference
movement_probability
fraction_time_moving
bout_count
valid_baseline_fraction
valid_cr_fraction
```

Do not use the ambiguous generic name `Scaled vigor (AU)` in new schemas.

## Canonical file formats

| Artifact | Format |
| --- | --- |
| Tabular samples and summaries | Parquet |
| Cohort/classification manifests | Parquet plus CSV review copy |
| Configuration/provenance | JSON |
| Dense arrays | NPZ |
| Large chunked multidimensional arrays | Zarr only when needed |
| Human review tables | CSV |
| Vector figures | SVG and PDF |
| Preview figures | PNG |
| Existing artifacts | Pickle as read-only legacy input |

Parquet is preferred because it preserves typed columns, compresses well,
supports selective reads, is interoperable, and avoids arbitrary Python object
deserialization.

Before converting a pickle:

1. Validate columns, dtypes, index, categories, and keys.
2. Normalize sparse and index representations.
3. Write schema-versioned Parquet.
4. Compare scientific summaries.
5. Record the source pickle hash.
6. Keep the source immutable until migration is accepted.

No raw or processed experimental observations are included in this plan or
required in Git. Tests use synthetic data by default.

## Package target

```text
src/classical_conditioning/
|-- domain/
|-- config/
|-- schemas/
|-- io/
|-- preprocessing/
|-- analysis/
|   `-- learners/
|-- figures/
|-- pipelines/
|-- cli/
`-- exceptions.py

tests/
|-- unit/
|-- contracts/
|-- integration/
|-- regression/
|-- statistics/
|-- figures/
|-- end_to_end/
`-- fixtures/
```

Add modules only when migrating real behavior; do not create an empty framework.

## Classes versus functions

| Need | Preferred construct |
| --- | --- |
| Identity/configuration | Frozen dataclass |
| Finite scientific category | Enum |
| Large observations | DataFrame/NumPy |
| Scientific calculation | Pure function |
| Table validation | Schema/validation function |
| File parsing/storage | Adapter or small class |
| Multi-step workflow | Small pipeline class |
| Panel drawing | Function accepting `Axes` |
| Figure layout | Dataclass specification |

Pipeline classes orchestrate. They do not contain the mathematical
implementation.

## Scientific correction policy

### Track A: structural refactor

Must preserve approved behavior:

- File parsing
- Identity
- Schema conversion
- Artifact plumbing
- Function extraction

Validate with characterization and regression comparisons.

### Track B: scientific correction

Expected to change results:

- Vigor definition
- Spatial filtering
- Secondary bout threshold
- Frame-loss policy
- Exclusion/cohort policy
- Baseline window
- Missing-data semantics
- Bootstrap unit
- Statistical model

Each correction requires:

- Scientific decision
- Mathematical specification
- Synthetic tests
- Before/after cohort and result comparison
- Manuscript impact assessment

Do not hide Track B changes inside a refactor.

## Scientific decisions and gates

| Decision | Required before |
| --- | --- |
| Vigor/filter/bout/frame-loss definitions | Preprocessing migration |
| Trial alignment and boundaries | Segmentation migration |
| Technical and behavioral inclusion | Cohort freeze |
| Missing-data semantics | Trial metrics |
| Analytical LogMedian baseline | Grouping and temporal analysis |
| Normalized-vigor definition | Trial-summary migration |
| Bootstrap unit and iterations | Population inference |
| Primary model and contrasts | Statistical migration |
| Learner classifier settings | Learner-stratified inference |
| Figure size/theme | Paper figure migration |

Consolidate decisions in one versioned
`SCIENTIFIC_ANALYSIS_SPECIFICATION.md`.

## Test strategy

### Characterization tests

Capture current:

- Columns and dtypes
- Row/fish/trial counts
- Time ranges
- Bout counts
- Missingness
- Numerical summaries
- Output inventory

These describe current behavior; they do not prove correctness.

### Scientific unit tests

#### Vigor

- No movement
- One segment
- Multiple aligned segments
- Opposing segments
- Known angular speed
- Missing samples

#### Filtering

- Constant
- Impulse
- Step
- Sinusoid
- Edges
- Spatial and temporal windows

#### Bouts

- Valid bout
- Too short
- Short gap merge
- Secondary-threshold failure
- Recording boundary

#### Synchronization

- Perfect sequence
- One missing frame
- Multiple missing frames
- Duplicate frame
- Jitter
- Out-of-range stimulus

#### Trial mapping

- Onset/offset boundaries
- CS/US alignment
- Catch trials
- Phase and block membership

#### LogMedian metrics

- Positive vigor log transform
- Non-positive handling
- Baseline median
- Baseline subtraction
- Median CR
- CR minus baseline

### Contract tests

Check:

- Required fields
- Dtypes
- Units
- Keys
- Categories
- Monotonic time
- One condition per fish
- One trial mapping per trial
- Many-to-one classification joins

### Integration fixture

Synthetic:

- Two conditions
- Several fish
- Pre-training/training/test
- Catch trial
- Known bouts
- Missing frame
- Excluded fish

Run from raw tables to trial summary and one figure.

### Statistical tests

Pin:

- Package versions
- BLAS/LAPACK
- Thread count
- Seed

Assert:

- Model formula and reference
- Fish and observation counts
- Convergence
- Coefficients and uncertainty within tolerance
- Correction method

### Figure tests

Structural:

- Dimensions
- Axes
- Labels/units
- Limits/ticks
- Legends/colors
- Colorbars

Visual regression:

- Agg backend
- Pinned Matplotlib and fonts
- Approved scientific baseline only

## Migration phases

## Phase 0: freeze and inventory

- Record current paper figures and source artifacts.
- Record environment and commits.
- Inventory paper-scope raw recordings.
- Select synthetic and approved regression fixtures.
- Create scientific analysis specification.

Gate: important current outputs are identifiable.

## Phase 1: package and test foundation

- Add `pyproject.toml`.
- Add package and tests directories.
- Record complete dependencies.
- Configure one test command.
- Keep current scripts operational.

Gate: clean environment imports package and runs tests.

## Phase 2: identities and experiment definitions

- Add enums, `TimeWindow`, `FishKey`, `FishMetadata`, `TrialSpec`, and
  `ExperimentSpec`.
- Port `all3sTrace` first.
- Add serialization and validation.
- Adapt existing `ExperimentConfig`.

Gate: `all3sTrace` resolves without hidden global changes.

## Phase 3: schema and artifact layer

- Define table schemas.
- Add Parquet/JSON artifact I/O.
- Add hashes and run IDs.
- Reject ambiguous file selection and overwrite.
- Add legacy pickle adapters.

Gate: one converted artifact round-trips and matches scientific summaries.

## Phase 4: raw I/O

Extract:

1. Fish filename parser
2. Camera reader
3. Tracking reader
4. Protocol reader

Gate: parsed tables match approved current behavior.

## Phase 5: preprocessing core

After scientific approval, extract and correct:

1. Frame validation
2. Synchronization
3. Interpolation
4. Tail geometry
5. Spatial filter
6. Temporal filter
7. Vigor
8. Bout metric
9. Bout detection
10. Stimulus annotation
11. Trial segmentation

Gate: synthetic references pass and pilot fish differences are explained.

## Phase 6: full paper-scope reprocessing

- Process every paper-scope recording.
- Record success/failure reasons.
- Produce coverage and legacy comparison reports.
- Freeze corrected sample artifacts.

Gate: every paper-scope recording is accounted for.

## Phase 7: QC and cohort

- Separate technical validity from behavioral engagement.
- Generate a reviewable cohort manifest.
- Stop moving raw files.
- Apply cohort by validated join.
- Preserve legacy exclusion import for transition.

Gate: every downstream fish is traceable to one cohort decision.

## Phase 8: trial map and LogMedian grouping

- Centralize CS/US alignment.
- Centralize 5-/10-trial blocks and phases.
- Mark catch trials.
- Define one analytical baseline window.
- Implement rolling median, log transform, and baseline subtraction.

Gate: step-3 pilot output is reproduced or deliberately corrected.

## Phase 9: trial and temporal metrics

- Conditional vigor
- Movement probability
- Fraction time moving
- Bout count
- Median baseline and CR
- Log difference normalized vigor
- Time bins and coverage
- Fish-first and hierarchical bootstrap

Gate: each metric has one name, unit, function, and reference test.

## Phase 10: statistics

- Define primary longitudinal model.
- Extract planned nonparametric contrasts.
- Define comparison families.
- Require convergence diagnostics.
- Save model inputs and results.

Gate: results regenerate from a versioned trial-summary artifact.

## Phase 11: learner analysis

- Keep the LogMedian classifier as the sole starting implementation.
- Evaluate control false-positive rate and label stability.
- Version classifier settings.
- Export classification manifest.
- Add held-out/cross-fitted validation before confirmatory claims.

Gate: one approved classifier configuration and schema.

## Phase 12: figures

Implement `SCIENTIFIC_FIGURE_PIPELINE_PLAN.md`.

Gate: one paper figure regenerates from explicit artifacts and passes QC.

## Phase 13: CLI and wrappers

Provide:

```powershell
python -m classical_conditioning preprocess ...
python -m classical_conditioning group ...
python -m classical_conditioning analyze ...
python -m classical_conditioning classify-learners ...
python -m classical_conditioning figure ...
```

Numbered scripts become thin compatibility wrappers.

## Phase 14: legacy retirement

- Remove duplicate functions after dual-run validation.
- Archive obsolete scripts.
- Remove first/latest artifact selection.
- Retire pickle outputs.
- Prevent imports from legacy locations.

## Phase 15: paper release

- Run full tests.
- Verify full paper-scope reprocessing hashes.
- Compare cohorts and conclusions.
- Regenerate figures.
- Reconcile manuscript methods.
- Create immutable release manifest.

## Migration lanes

Urgent scientific corrections may proceed in the current script structure while
the package foundation is built, provided they:

- Have explicit tests.
- Follow the approved scientific specification.
- Produce versioned artifacts.
- Are written for later extraction.

The structural migration must consume, not redefine, those decisions.

## Compatibility and rollback

For each migrated stage:

1. Run current and new paths on the same fixture.
2. Compare keys, rows, values, missingness, and summaries.
3. Classify differences as regression or approved correction.
4. Keep old path callable until the gate passes.
5. Remove old implementation in a later change.

## Process scale

This is a small research project. Use:

- One scientific owner
- One implementation owner, possibly the same person
- One concise decision log
- Lightweight review for low-risk utilities
- Full review for scientific calculations, cohorts, models, classifiers, and
  paper figures

Phases are dependency gates, not mandatory calendar sprints.

## Immediate milestone

Scope:

```text
pyproject and test command
+ FishKey, TimeWindow, Alignment
+ all3sTrace ExperimentSpec
+ raw schemas
+ existing reader adapters
+ synthetic fixtures
+ identity and parser tests
```

Do not include vigor corrections or figure refactoring in this first milestone.

## Definition of done

- [ ] Every scientific metric has one approved definition.
- [ ] Core calculations have synthetic tests.
- [ ] Tables have versioned schemas.
- [ ] Paper-scope recordings are reprocessed.
- [ ] One cohort manifest controls all stages.
- [ ] Trial/block/phase mapping is centralized.
- [ ] LogMedian baseline semantics are consistent.
- [ ] Statistical models are diagnosed and versioned.
- [ ] Learner classifier and manifest are versioned.
- [ ] Figures regenerate from explicit artifacts.
- [ ] Numbered scripts are wrappers or archived.
- [ ] Canonical tables use Parquet rather than pickle.
- [ ] End-to-end fixture passes.
- [ ] Paper release records code, environment, inputs, cohort, configuration,
      results, and figures.

## Plan review

This revision was checked against the integrated LogMedian tree.

Changes from the older migration plan:

- Removed the four-classifier selection premise.
- Recognized `6_LearnersQuantification_LogMedian.py` as the only current
  classifier.
- Added the implemented learner-stratified pipeline as a migration input.
- Preserved its fish-key, manifest, fish-first aggregation, tests, and
  circularity patterns.
- Added the discovered mean/ratio versus median/subtraction integration
  correction.
- Focused scientific corrections on issues still present at `bf46bf7`.
- Kept Parquet/JSON recommendations and explicit no-data policy.

Residual execution risk:

- Downstream cohort, classifier, and figure migration must not proceed before
  preprocessing and baseline decisions are approved and paper-scope data is
  reprocessed.

