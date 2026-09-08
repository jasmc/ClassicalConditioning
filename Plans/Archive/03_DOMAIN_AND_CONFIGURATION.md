# Step 03 — Domain Definitions and Resolved Configuration

**Status:** Complete for `allDelay` / local fixture scope  
**Change class:** Behavior-preserving first; corrections later  
**Depends on:** Steps 00-01; coordinate with Step 02 artifact types  
**Unlocks:** Explicit experiment definitions and stage-specific invalidation

**Completed (scoped):** 2026-08-31  
**Archive note:** Detailed plan archived after `allDelay` local-fixture exit.
Additional paper experiments remain deferred until required by scope.

Progress note: `allDelay` experiment spec, `legacy-paper-v1` resolved recipe,
stage-scoped hashing, FishKey identity, trial-map export, source-trace report,
and `resolve-config` CLI are **done**. Additional paper experiments remain
deferred until required by scope.

## Objective

Replace implicit strings, globals, duplicated constants, and mutable run
settings with validated immutable definitions while reproducing current active
values exactly.

## Core domain types

```text
FishKey
FishMetadata
Recording
TimeWindow
ConditionSpec
TrialSpec
ExperimentSpec
Alignment
Phase
Paradigm
ConditionRole
ScientificStatus
```

Observation data remains in DataFrames and NumPy arrays.

## Configuration layers

```text
ExperimentSpec
RawValidationConfig
SynchronizationConfig
TailRepresentationConfig
FilteringConfig
ActivityMetricConfig
MovementStateConfig
CohortConfig
OutcomeConfig
BootstrapConfig
StatisticsConfig
LearnerClassifierConfig
FigureTheme
MajorFigureSpec
ResolvedAnalysisConfig
```

## Work packages

### 03.1 Audit configuration sources

Create a field-level map from:

- current general configuration;
- each experiment definition;
- top-level constants in numbered scripts;
- learner script variants;
- environment variables;
- implicit path and filename conventions;
- plot-library defaults that affect output.

Record conflicting active values rather than selecting one during legacy
reproduction.

### 03.2 Define stable identity

Validate whether `experiment_id + fish_id` is globally unique. If source fish
IDs can collide, use:

```text
experiment_id
day
fish_number
```

Condition is metadata unless the identity audit proves it is needed to
disambiguate source identifiers. Alignment is never part of the biological fish
key.

### 03.3 Define canonical trial map

For every active experiment:

```text
recording_id
trial_id
trial_number
alignment
event_id
phase
block_5_id
block_10_id
catch status
expected US time
stimulus properties
```

`trial_id` is a stable technical identity within `recording_id`.
`trial_number` is the experiment-facing ordinal. `event_id` identifies the
canonical source/alignment event and must not be inferred independently by a
downstream modality.

Validate:

- unique `(recording_id, trial_id, alignment)` mapping;
- unique and consistent source event assignment;
- complete expected range;
- consistent block nesting;
- known catch trials;
- phase boundaries;
- CS/US alignment availability.

### 03.4 Preserve legacy settings as recipes

Example:

```text
legacy-paper-v1
```

This recipe records current behavior, including suspicious values such as:

- point-15 vigor;
- current temporal filtering;
- absent/effectively absent spatial filtering;
- ignored second bout threshold;
- current baseline masks;
- stage-specific exclusion settings;
- current bootstrap counts;
- current statistical formulas.

The configuration may label known issues but must not fix them.

### 03.5 Define corrected and candidate recipe inputs

Create schemas for, but do not approve values for:

```text
tail-candidate-development-v1
tail-candidate-confirmation-v1
corrected-paper-v2
```

Values requiring scientific approval remain blocked or explicitly provisional.

### 03.6 Implement stable serialization

Resolved configuration must serialize to JSON-compatible data with:

- explicit units;
- enum values rather than display labels;
- no unserializable callables;
- stable ordering;
- content hash;
- source-layer trace showing where each value was resolved.

### 03.7 Implement stage-scoped configuration hashing

Each stage declares the subset that affects it. For example:

```text
activity metric:
    tail representation ID
    metric implementation
    smoothing
    weighting
    time policy
```

A figure-color change must not invalidate preprocessing.

## Required tests

- Invalid windows rejected
- Duplicate conditions rejected
- Duplicate/incomplete trial mappings rejected
- Unknown phase/alignment/condition rejected
- Stable serialization and hashing
- Same effective config gives same hash
- Relevant setting change changes affected stage hash
- Irrelevant setting change does not change unaffected stage hash
- Legacy configuration adapter reproduces every audited active constant

## Deliverables

- Domain modules
- One validated pilot experiment
- Legacy configuration adapters
- Named recipe schema
- Resolved configuration JSON
- Configuration source/override report
- Trial-map artifacts

## Exit gate

A pilot experiment and the legacy baseline can be represented without relying
on mutable module-level globals, and the resolved legacy recipe reproduces the
active settings of each current stage.

## Failure conditions

Do not proceed if:

- identity uniqueness is unresolved;
- trial mapping is incomplete or ambiguous;
- two configuration layers silently disagree;
- units cannot be determined;
- configuration serialization loses information.
