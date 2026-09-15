# Pipeline and package guide

This document is the companion to [README.md](README.md). It explains the
supported candidate-only workflow in `src/classical_conditioning/pipeline.py`
and records the purpose of every Python source file in
`src/classical_conditioning/`.

The word **called** below has a precise meaning: a function is reached by a
normal call to `run_pipeline`, either directly or through a downstream stage.
An imported module whose functions are never invoked is identified separately.
`__pycache__/` directories contain generated Python bytecode and are not source
files or part of the design.

## The short version

`pipeline.py` is a coordinator. It does not compute tail metrics itself. It:

1. selects recordings;
2. optionally inventories raw files;
3. intakes complete raw triplets into immutable, validated derived artifacts;
4. runs one frozen candidate recipe family for each eligible recording;
5. compares the successful recordings at cohort level;
6. optionally renders figures from that comparison; and
7. writes a run ledger.

The supported package route is `candidate`. The old executable legacy route is
not part of the installable package; its source is preserved, without an active
import path, under `Archive/package/`.

```text
PipelineRunConfig JSON
        |
        v
pipeline.py: select recordings, optional inventory, intake
        |
        v
candidate_runner.py: one recording at a time
        |
        +-- corrected_frame_preprocessing.py        (corrected runner only)
        +-- candidate_metric_kernel.py                shared metric definitions
        +-- candidate_metrics_from_corrected_frames.py  normal activity metrics
        +-- benchmarks/candidate_metrics_from_intake.py  direct-intake benchmark
        +-- movement_state.py                        movement and bouts
        +-- temporal_profiles.py                     event-aligned profiles
        +-- trial_outcomes.py                        one row per trial/outcome
        |
        v
metric_comparison.py: recording and cohort summaries
        |
        +-- figures/metric_comparison.py             (only if requested)
        |
        v
Metadata/<analysis_id>_pipeline_run.json             run ledger
```

## `pipeline.py`, step by step

### Before `run_pipeline`: validate and resolve configuration

The CLI `run-pipeline` command loads JSON through `run_config.py`. It validates
paths, the experiment name, route, IDs, batch size, and recipe choice. It
rejects retired legacy fields and a `legacy` route with a migration error.
`pipeline.py` receives the resulting immutable `PipelineRunConfig`; it does
not parse JSON itself.

### 1. Select the recording cohort

`resolve_pipeline_recording_ids()` uses `recording_ids` exactly when supplied.
Otherwise it asks `intake.discover_recordings()` for complete camera, tracking,
and protocol triplets under `raw_dir`, preserving discovery order. Optional
`keep_conditions` is a filename-condition filter, not a scientific exclusion
rule. The latter is intentionally left to an explicit cohort policy.

### 2. Create the output root and start progress reporting

The pipeline creates `save_dir`, never writes into `raw_dir`, and uses
`progress.py` to report named stages. It initializes status fields that later
become the ledger: completed/skipped/failed intake IDs, the candidate manifest,
and generated PNG paths.

### 3. Optional raw-file inventory

When `run_inventory` is true, `inventory.write_recording_inventory()` writes
`Metadata/recording_inventory.json`. This is a read-only QC/provenance scan:
it discovers triplets and hashes source files. It is useful evidence, but the
pipeline does not require a prior inventory because intake independently
validates its own inputs.

### 4. Intake raw acquisition triplets

When `run_intake` is true, `intake.intake_recordings()` validates and converts
the selected triplets to lossless Parquet plus QC/provenance artifacts beneath
`save_dir`. A recording can be:

* **completed** — intake ran successfully this time;
* **skipped** — a compatible, existing intake artifact was accepted; or
* **failed** — intake did not produce a usable artifact.

Completed and skipped recordings remain eligible downstream. If
`continue_on_error` is false, any failure stops the run; otherwise the pipeline
continues with the eligible subset. If no recording is eligible, it stops
before analysis.

### 5. Run the candidate recipe family per recording

When `routes` includes `candidate` (the only supported route), `pipeline.py`
maps the selected runner recipe to its matching metric source, then calls
`analysis.candidate_runner.run_candidate_development_pipeline()`.

The runner freezes a compatible recipe family and records cryptographic lineage
between its artifacts. It never silently mixes a development metric family with
a corrected temporal, movement, or comparison artifact family.

For each recording, the runner performs the following stages in order:

| Stage | Module | What it produces and why it exists |
| --- | --- | --- |
| Corrected preprocessing | `preprocessing/corrected_frame_preprocessing.py` | Only for `candidate-corrected-runner-v1`. Builds corrected frame-level input and validates frame/protocol alignment before metrics are calculated. |
| Activity metrics | `preprocessing/candidate_metrics_from_corrected_frames.py` or `preprocessing/benchmarks/candidate_metrics_from_intake.py` | Produces the three exploratory whole-tail candidate metrics. The corrected implementation consumes corrected frames; the development implementation consumes intake artifacts directly. |
| Movement state and bouts | `analysis/movement_state.py` | Calibrates a movement indicator and applies the shared historical-envelope bout detector. It produces a frame-level movement/bout state artifact. |
| Temporal profiles | `analysis/temporal_profiles.py` | Aligns metrics and movement state to CS/US protocol events, bins time, and writes event-aligned profiles. |
| Per-trial outcomes | `analysis/trial_outcomes.py` | Collapses frames within prespecified windows into one outcome row per trial, including activity, movement, and bout outcomes. |

The normal default is `candidate-corrected-runner-v1`. The still-supported
`candidate-development-runner-v1` is an active benchmark family, not archived
legacy execution: it skips corrected preprocessing and derives its metric input
directly from intake artifacts. Both routes remain explicitly versioned and
label their outputs as non-paper-approved candidate analyses.

Each stage either reuses a compatible completed artifact or recomputes it when
`overwrite` is true. After each stage the runner checks recipe ID, recording ID,
hashes, completion marker, and upstream hash linkage. This prevents using
stale downstream files after an upstream input changes.

## The preprocessing folder: one metric implementation, two input routes

`src/classical_conditioning/preprocessing/` contains four active implementation
files plus a benchmark entry point. It is
easy to mistake their names for four alternative analyses, but the important
division is this:

```text
intake camera/tracking/protocol Parquet
        |
        +-----------------------------------------------+
        |                                               |
        v                                               v
benchmarks/candidate_metrics_from_intake.py   corrected_frame_preprocessing.py
tail-candidate-development-v1                 corrected-preprocess-v1
        |                                               |
        |                                               v
candidate_metric_kernel.py <---------------- candidate_metrics_from_corrected_frames.py
same shared formulas and schema                          |
        |                                                |
        +----------------- same three metric columns ---+
                                                        |
                                                        v
                                      movement, temporal, trial, cohort stages
```

### `corrected_frame_preprocessing.py`: corrected frame preparation

This is not an activity-metric implementation. It builds the intermediate
`frame_preprocessed_corrected-v1.parquet` artifact used by the normal corrected
runner. Its frozen policy is intentionally conservative:

* preserves measured camera timestamps rather than rebuilding a uniform
  700 Hz timebase;
* rejects duplicate or non-monotonic frame/timestamp order;
* marks derivatives invalid across frame gaps or intervals longer than the
  configured maximum rather than interpolating across them;
* leaves temporal and spatial filtering disabled; and
* subtracts the tail base for translation correction but applies no body
  rotation without an independent body-axis measurement.

It writes validity and timing fields alongside measured tracking coordinates,
plus a summary and completion marker. Its purpose is to make data-quality and
timebase decisions explicit before any metric is calculated.

### `candidate_metric_kernel.py`: shared metric science

`calculate_candidate_metrics()` is the shared numerical implementation
of the three frame-level metric columns:

1. `tail_length_weighted_angular_l1_rad_per_ms`;
2. `whole_tail_xy_mean_speed_tail_lengths_per_ms`; and
3. `legacy_distal_angular_speed_rad_per_ms`, an active historical benchmark.

It also contains geometry agreement checks, frame-order checks, coordinate
extraction, weighting, and the immutable `CandidateMetricConfig`. Downstream
movement, temporal, trial-outcome, and figure code imports the shared column
list from here, so the six-column schema has one authoritative definition.

### `benchmarks/candidate_metrics_from_intake.py`: development benchmark

This is the deliberately separate entry point for the writer that calls the
shared kernel on raw intake. `build_candidate_activity_metrics()` writes
`tail-candidate-development-v1`. It reads authenticated intake camera and
tracking Parquet directly and writes
`frame_activity_candidates-v1.parquet`. This is an active development
benchmark route. It is not the default corrected workflow, but it must remain
available for controlled development-versus-corrected comparisons.

### `candidate_metrics_from_corrected_frames.py`: corrected-source adapter

This file does **not** define a second set of activity metrics. It imports the
shared calculation and schema from `candidate_metric_kernel.py`, verifies the completed
`corrected-preprocess-v1` artifact, runs the same calculation on its measured
coordinates, and intersects the resulting derivative-valid mask with the
corrected frame-validity mask. It then writes the separate,
provenance-distinct `frame_activity_candidates-corrected-v1.parquet` artifact.

The separate file and recipe are intentional. They prevent a metric table made
from direct intake input from being mistaken for one made from corrected,
gap-aware input. The copied streaming/output mechanics are a maintenance cost,
but not a scientific duplicate: the data lineage and invalid-frame policy are
different, while the formula is shared by import. The adapter test suite checks
that this relationship remains true.

### `__init__.py`: lazy public interface

This file performs no preprocessing. It exposes the three builders and their
result/config dataclasses through lazy imports, keeping package import cheap and
avoiding loading concrete recipe modules until a caller needs one.

### Which file should I use?

| Your task | Use | Do not use |
| --- | --- | --- |
| Run the normal analysis | `run-pipeline` with its default `candidate-corrected-runner-v1` recipe. It calls the two corrected files in the right order. | Do not call preprocessing modules manually unless debugging a stage. |
| Inspect or rerun corrected frame preparation | `corrected_frame_preprocessing.py` | Do not use the intake-metric module; it does not create corrected frames. |
| Inspect or rerun normal corrected metrics after corrected frames exist | `candidate_metrics_from_corrected_frames.py` | Do not substitute the direct-intake module; it lacks the corrected validity lineage. |
| Reproduce the active development benchmark for a controlled comparison | `benchmarks/candidate_metrics_from_intake.py` | Do not label its output as corrected or mix it with corrected downstream artifacts. |
| Reproduce retired historical execution | None of these active files. Read `Archive/package/` only as source history. | Do not run archived code through the supported package. |

### Are any versions redundant?

No active file is safe to delete as a simple duplicate:

| Apparent overlap | Actual distinction | Keep? |
| --- | --- | --- |
| `benchmarks/candidate_metrics_from_intake.py` vs `candidate_metrics_from_corrected_frames.py` | Same metric formula, but direct-intake versus corrected/gap-aware source lineage and distinct output/recipe identities. | Yes. |
| `candidate_metric_kernel.py` vs either metric writer | Shared numerical formula/schema versus provenance-specific artifact writer. | Yes. |
| `corrected_frame_preprocessing.py` vs either candidate file | Frame preparation and validity policy versus activity-metric calculation. | Yes. |
| `__init__.py` vs concrete modules | Public lazy-import facade versus implementation. | Yes. |

The main technical concern is shared scientific code rather than duplicated
formulae: `candidate_metrics_from_corrected_frames.py` imports the metric
kernel. A change to shared extraction, geometry, or metric code therefore
affects both routes and should be accompanied by tests for both
artifact families. If the two writers ever become burdensome to maintain, the
safe refactor is to keep the kernel stable—not to merge their recipes or
discard their distinct provenance.

### 6. Cohort metric comparison

Once at least one recording completed all per-recording stages,
`analysis.metric_comparison.build_candidate_metric_comparison()` consumes their
event-aligned profiles. It writes recording-level and cohort-level Parquet
summaries for the same five-metric comparison recipe family. This is a
descriptive comparison across candidate metrics, not an approved confirmatory
statistical analysis. The runner verifies that each comparison input is the
exact temporal-profile artifact it expects.

### 7. Optional figures

When `run_figures` is true, `pipeline.py` calls
`figures.metric_comparison.build_metric_comparison_figure()` once per requested
outcome. It selects the comparison recipe paired to the runner (corrected or
development), reads its cohort summary, and writes the PNG/SVG figure outputs.
Figure generation occurs after the comparison, never directly from raw frames.
If the expected comparison file is absent, the pipeline gives a configuration
error explaining that analysis may be incomplete or the analysis ID/recipe may
not match.

### 8. Write the pipeline ledger

Regardless of optional stages, successful completion writes
`Metadata/<analysis_id>_pipeline_run.json` through `artifacts.write_json_atomic`.
It records the resolved configuration, final eligible IDs, intake states,
candidate-runner manifest name, and figure paths. It is a run ledger, not a
scientific result: scientific data and QC/provenance live in the individual
stage directories.

## Package inventory and pipeline reachability

### Package root

| File | Called by `pipeline.py`? | Purpose and reason |
| --- | --- | --- |
| `__init__.py` | Import-time only | Defines the public package surface and re-exports selected intake/errors. Python loads it when the package is imported; it is not an analysis stage. |
| `__main__.py` | No | Makes `python -m classical_conditioning` invoke the CLI. The CLI may invoke `pipeline.py` when the user chooses `run-pipeline`. |
| `artifacts.py` | Directly and transitively | Atomic JSON/Parquet publication, hashes, staging, manifests, and verification. `pipeline.py` uses it for the ledger; nearly every producing stage uses it for integrity. |
| `cli.py` | No; caller of it | Defines all command-line commands. `run-pipeline` is one entry point; many commands intentionally expose individual stages, QC, figures, or downstream analysis separately. |
| `cohort.py` | No | Explicit cohort manifest freezing/application. It is kept outside the pipeline because scientific inclusion/exclusion requires a reviewed policy rather than implicit discovery. |
| `comparison.py` | No | Generic command-line comparison of two row-aligned Parquet artifacts. It is a diagnostic utility, not the scientific cohort metric comparison. |
| `environment.py` | No | Checks supported runtime/dependency/font environment and can write an environment report. This is called by the CLI before commands, not by `pipeline.py`. |
| `exceptions.py` | Directly and transitively | Named configuration, schema, and artifact-integrity errors used to fail safely with actionable messages. |
| `intake.py` | Directly | Discovers source triplets and turns them into validated, lossless intake Parquet and QC reports. This is the raw-to-derived boundary. |
| `inventory.py` | Directly, optional | Read-only discovery, source hashing, and tracking-header inventory. Only reached when `run_inventory` is enabled. |
| `paths.py` | Directly and transitively | Path-safety checks, source/derived separation, recording-condition parsing, and naming helpers. |
| `pipeline.py` | Entry point | Config-driven orchestration described above. It coordinates stages but contains no metric/detector implementation. |
| `progress.py` | Directly and transitively | Structured console progress, timings, and per-item status. It has no scientific effect. |
| `run_config.py` | Directly | Defines `PipelineRunConfig`, loads pipeline JSON, validates supported candidate settings, and serializes resolved settings into provenance. |

### `analysis/`

| File | Called by `pipeline.py`? | Purpose and reason |
| --- | --- | --- |
| `__init__.py` | Import-time convenience only | Lazy public exports for standalone analysis commands. It does not schedule analysis stages. |
| `benchmarks/__init__.py` | Import-time only | Marks optional sensitivity work as benchmarks rather than automatic analysis. |
| `candidate_runner.py` | Directly | The per-recording and cohort orchestrator for the candidate family. It freezes compatible recipe choices, performs lineage verification, and writes its own manifest. |
| `benchmarks/movement_sensitivity.py` | No | Explicit entry point for detector-parameter sensitivity checks. Its calculation remains beside the detector because both share the same state logic. |
| `inference/fish_bootstrap.py` | No | Post-pipeline fish-level bootstrap uncertainty analysis from authenticated model input. It is deliberately not automatic because resampling choices are statistical decisions. |
| `inference/fish_permutation.py` | No | Post-pipeline fish-level permutation testing. Like bootstrap, it is a separate inferential workflow, not a default descriptive pipeline step. |
| `inference/__init__.py` | Import-time only | Documents the optional inference namespace; it does not select a model or run statistics. |
| `metric_comparison.py` | Transitively | Builds the recording/cohort summaries after temporal profiles. This is the pipeline's cohort comparison stage. |
| `inference/mixed_effects.py` | No | Fits candidate mixed-effects models using the model-input artifact. It is optional downstream inference and needs its own model specification/dependency. |
| `inference/model_input.py` | No | Converts authenticated per-trial outcomes into a model-ready data set with coverage information. It supports downstream statistics but the core pipeline stops at descriptive cohort comparison. |
| `movement_state.py` | Directly and transitively | Maps runner recipe to metric family, calibrates frame-level movement, and detects bouts using the active shared historical-envelope detector. |
| `temporal_profiles.py` | Transitively | Aligns frame-level candidate/movement artifacts to experiment events and creates time-binned profiles for analysis and figures. |
| `review/trace_review.py` | No | Creates balanced trace windows and review figures for human detector assessment. It is a QC/review command because human annotation cannot be silently automated in the pipeline. |
| `review/__init__.py` | Import-time only | Documents the review namespace; it does not create review figures itself. |
| `trial_outcomes.py` | Transitively | Builds and verifies trial-level outcomes from candidate frames, movement states, and experiment windows. |

### `config/`

| File | Called by `pipeline.py`? | Purpose and reason |
| --- | --- | --- |
| `__init__.py` | Import-time/transitive support | Re-exports experiment, identity, and trial-map definitions. Temporal and trial outcome code uses this stable public configuration surface. |
| `domain.py` | Transitively | Immutable scientific domain types: experiment, trial, condition, alignment, phase, time window, and fish identity. |
| `experiments.py` | Transitively | Defines supported experiment specifications and their trial schedule. `run_config.py`, temporal profiling, trial outcomes, comparisons, and figures use it to interpret timing/conditions. |
| `export.py` | No | CLI-only `resolve-config` output: exports candidate recipe IDs, frozen stage settings, experiment spec, and trial map for review. It does not run automatically because it is an inspection artifact. |
| `identity.py` | No | Parses/constructs fish identity keys from recording IDs, primarily for cohort and downstream fish-level analysis. |
| `trial_map.py` | Imported as configuration support, not invoked by the normal pipeline | Builds an explicit trial/block map for config export and other consumers. The normal temporal stage uses the experiment block lookup from `experiments.py`; this file remains available for transparent review. |

### `figures/`

| File | Called by `pipeline.py`? | Purpose and reason |
| --- | --- | --- |
| `__init__.py` | Import-time convenience only | Lazy public exports for figure CLI commands. |
| `export.py` | Directly and transitively | Figure modes and robust export: writes image files, embeds provenance, and validates SVG semantic IDs. `pipeline.py` uses `FigureMode`; figure builders use the export function. |
| `metric_comparison.py` | Directly, optional | Renders pipeline-requested cohort metric-comparison figures after comparison summaries are available. |
| `temporal_profiles.py` | Transitively for labels/support; figure builder itself is not called | Defines per-recording candidate profile/heatmap figures. Its metric labels support metric-comparison figures; its own renderer is exposed through a separate CLI command for targeted review. |
| `theme.py` | Transitively | Central typography, colour, axes, stimulus-window, and layout definitions used by figure modules and environment checks. |

### `ingestion/`

| File | Called by `pipeline.py`? | Purpose and reason |
| --- | --- | --- |
| `__init__.py` | Import-time convenience only | Re-exports raw-reading and audit helpers for dedicated ingestion commands. |
| `frame_sequence.py` | No | Validates camera frame ordering/continuity for the standalone `validate-raw` command. Intake has its own streaming validation path. |
| `readers.py` | No | Typed readers for standalone raw-triplet validation. Kept separate from high-volume intake so validation can inspect source files without producing artifacts. |
| `schemas.py` | Transitively | Column normalization and schema validation used by intake and inventory. It protects the acquisition contract before downstream computation. |
| `tracking_audit.py` | No | Classifies and summarizes tracking columns for the standalone `audit-tracking` QC command. |
| `validate_raw.py` | No | Combines readers, frame-sequence validation, and tracking audit into a standalone preflight report. It is optional QC, not required by normal intake. |

### `preprocessing/`

| File | Called by `pipeline.py`? | Purpose and reason |
| --- | --- | --- |
| `__init__.py` | Import-time convenience only | Lazy exports for current corrected preprocessing and candidate metrics. |
| `candidate_metric_kernel.py` | Transitively | The single numerical definition of the three candidate metrics, their configuration, geometry checks, and shared extraction helpers. It is reused by both artifact writers. |
| `candidate_metrics_from_corrected_frames.py` | Transitively for corrected runner | Reads corrected frames, applies the corrected validity mask, and computes the corrected-family activity metrics. |
| `benchmarks/__init__.py` | Import-time only | Identifies runnable direct-intake comparisons as active benchmarks rather than archived legacy code. |
| `benchmarks/candidate_metrics_from_intake.py` | Transitively for the development benchmark only | Calls the shared kernel to compute development-family metrics directly from intake artifacts. The normal corrected route does not import this benchmark module. |
| `corrected_frame_preprocessing.py` | Transitively for corrected runner | Performs corrected frame preprocessing: derives geometry/protocol-aligned inputs, validates order/timing, and produces the artifact consumed by corrected metrics. |

### `operations/`

| File | Called by `pipeline.py`? | Purpose and reason |
| --- | --- | --- |
| `__init__.py` | Import-time only | Documents the operations namespace. |
| `batch_work.py` | No | Plans and executes resumable batch work from a separate manifest. It is separate because scheduling is an operational choice, not an analytical stage. |

## Files intentionally outside the automatic pipeline

The core pipeline ends at audited descriptive cohort summaries and optional
figures. The following files are active but intentionally manual/CLI-driven:

* `cohort.py` — cohort inclusion policy;
* `analysis/review/trace_review.py` — human detector review;
* `analysis/inference/model_input.py`, `mixed_effects.py`, `fish_permutation.py`, and
  `fish_bootstrap.py` — downstream inferential choices;
* `analysis/benchmarks/movement_sensitivity.py` — optional detector sensitivity;
* `operations/batch_work.py` — operations/scheduling;
* `ingestion/validate_raw.py` and `tracking_audit.py` — optional raw QC;
* `config/export.py` — reviewable configuration export; and
* figure modules for targeted per-recording review.

Keeping these explicit avoids a pipeline run quietly making scientific cohort,
annotation, or statistical-model decisions on the user's behalf.

## What to read when editing analysis logic

For a normal corrected candidate run, read these files in this order:

1. `pipeline.py` — cohort selection and high-level control flow;
2. `analysis/candidate_runner.py` — exact stage order and lineage checks;
3. `preprocessing/corrected_frame_preprocessing.py` and
   `preprocessing/candidate_metrics_from_corrected_frames.py` — corrected frame and metric
   computation;
4. `preprocessing/candidate_metric_kernel.py` — the shared three metric definitions;
5. `analysis/movement_state.py` — smoothing, calibration, and bout detection;
6. `analysis/temporal_profiles.py` and `analysis/trial_outcomes.py` — event
   alignment and outcome aggregation;
7. `analysis/metric_comparison.py` — recording/cohort aggregation; and
8. `config/experiments.py` plus `config/domain.py` — the protocol timing and
   scientific definitions applied throughout.

For historical implementation context, inspect `Archive/package/` as read-only
source history. It is intentionally not callable from the active package.
