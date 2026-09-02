# Step 12 — Figure System, CLI, and Notebook Workflows

**Status:** In progress
**Change class:** Behavior-preserving rendering first; presentation changes separately  
**Depends on:** Step 02 and each scientific stage whose output is exposed  
**Unlocks:** Practical researcher use and reproducible paper builds

## Objective

Expose the same package implementation through CLI, Python, notebooks, and
compatibility scripts, and generate publication figures from saved panel data
without embedding hidden analysis.

## Part A — CLI and stage orchestration

### Required commands

```powershell
python -m classical_conditioning plan --recipe <recipe>
python -m classical_conditioning run ingest --recipe <recipe>
python -m classical_conditioning run preprocess --recipe <recipe>
python -m classical_conditioning run acquisition-check --recipe <recipe>
python -m classical_conditioning run cohort --recipe <recipe>
python -m classical_conditioning run activity-metrics --recipe <recipe>
python -m classical_conditioning run outcomes --recipe <recipe>
python -m classical_conditioning run statistics --recipe <recipe>
python -m classical_conditioning run classify --recipe <recipe>
python -m classical_conditioning run figures --recipe <recipe> --figure-mode <mode>
python -m classical_conditioning reproduce --release <release>
```

### Required selectors

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

### Plan/explain output

Before execution, report:

```text
selected recipe and scientific status
resolved inputs and hashes
resolved stage versions
cached/current/stale/missing status
reason each stale stage must rerun
reused upstream artifacts
planned outputs
estimated scope
```

### Failure behavior

- Nonzero exit on stage failure
- Recording-level batch failures summarized without hiding them
- No success-shaped empty output
- Atomic writes
- Resumable completed recording tasks
- `--force` creates/replaces only explicitly targeted non-release work under
  defined rules; immutable releases are never overwritten

## Part B — Python API

Provide simple functions:

```python
load_analysis(...)
list_artifacts(...)
load_artifact(...)
run_stage(...)
select_fish(...)
select_trial(...)
build_temporal_profile(...)
```

Researchers should not need to understand the internal artifact repository to
inspect one fish.

## Part C — Notebooks

Recommended notebooks:

```text
01_preprocessing_qc.ipynb
02_cohort_review.ipynb
03_metric_validation.ipynb
04_population_exploration.ipynb
05_learner_diagnostics.ipynb
06_figure_review.ipynb
```

Rules:

- notebooks import package functions;
- notebooks identify the recipe and artifact IDs at the top;
- exploratory overrides write to explicit temporary or versioned outputs;
- no unique canonical equation lives only in a notebook;
- rerunning top to bottom is supported;
- notebook output is not the sole paper artifact.

## Part D — Figure architecture

Separate:

```text
scientific outcome artifact
-> panel-data builder
-> reusable panel renderer
-> major-figure composition
-> publication/static/interactive rendering
-> figure QC
```

### Figure types

- `FigureTheme`
- `FigureMode`
- `PanelSpec`
- `MajorFigureSpec`
- `FigureBuildPipeline`

Do not create figure-number subclasses.

### Figure modes

| Mode | Output | Use |
| --- | --- | --- |
| `publication` | SVG and PDF | Manuscript composition and final archive |
| `static` | PNG | Fast review, QC, presentations, and visual regression |
| `interactive` | Self-contained local HTML or local notebook | Pan, zoom, hover, trace selection, and local exploration |

All modes consume identical panel data. Interactive HTML embeds required
JavaScript locally and does not use a CDN. Very large recordings remain in
local Parquet/HDF5 and are aggregated or queried by range instead of being
embedded wholesale.

### Semantic publication SVG

For `publication` mode:

- assign stable Matplotlib `gid` values to panels, axes, axis titles, tick
  labels, spines, series, confidence intervals, stimulus marks, legends, and
  annotations;
- embed an inert compact JSON metadata record containing the analysis recipe,
  source commit/file/symbol/hash, exact reproduction call, input artifact IDs,
  cohort hash, and sidecar hash;
- write a complete `.figure.json` sidecar with the artist registry and
  data-field mappings;
- validate unique IDs and SVG-to-sidecar references;
- keep raw observations and executable JavaScript out of SVG metadata.

The sidecar is authoritative because Matplotlib's internal SVG grouping can
change between renderer versions.

### Panel-data artifacts

Each panel-data table records:

- source result/outcome IDs;
- cohort hash;
- metric and detector;
- aggregation;
- sample sizes;
- display transformation;
- intended panel ID.

Rendering code cannot redefine analysis windows, cohorts, bootstrap, or
statistics.

### Figure versioning

Separate:

- scientific panel-data version;
- panel-renderer version;
- figure-layout version;
- theme version.

A color change reruns rendering only. A changed outcome reruns panel data and
dependent figures.

### Figure QC

Structural:

- dimensions;
- axes and panel count;
- labels and units;
- limits and ticks;
- stimulus annotations;
- condition colors;
- legends;
- panel labels;
- sample sizes;
- output formats.

Visual:

- selected raster regression comparisons;
- pinned backend, Matplotlib, and fonts;
- separate references for approved scientific corrections.

## Part E — Compatibility wrappers

The numbered scripts progressively become thin wrappers that:

- translate current constants into a named legacy recipe;
- issue a deprecation notice;
- call one package stage;
- preserve familiar legacy outputs during transition;
- do not duplicate scientific functions.

## Required tests

- Same stage result through CLI and direct Python call
- One-fish and one-condition selection
- `--dry-run` performs no writes
- `--explain` correctly localizes config invalidation
- Resume uses only matching hashes
- Figure panel data unchanged by rendering
- Structural figure tests
- Semantic SVG ID, metadata, and sidecar-contract tests
- Compatibility wrapper regression
- Notebook smoke execution where practical

## Exit gate

Every supported stage can be run from CLI and Python; researchers can inspect
intermediates in notebooks; one major paper figure regenerates from explicit
panel-data artifacts and passes structural/visual QC. Scientific gate F is
approved for any figure designated final.

## Pilot progress

Implemented:

```text
b43aee9 feat(figures): add semantic publication and interactive modes
```

Candidate CS-profile outputs are saved under:

```text
Paper data/Figures/PNG/<fish-id>/
Paper data/Figures/Interactive/<fish-id>/
Paper data/Figures/Publication/<fish-id>/
```

Publication verification:

- clean source commit embedded;
- semantic SVG and PDF generated;
- sidecar hash embedded in SVG and verified;
- PDF hash recorded and verified;
- 250 semantic registry entries;
- five heatmap artists mapped to x, y, value, and coverage fields;
- expected-sample coverage below 90% masked from display.
- total activity, movement probability, fraction time moving, conditional
  intensity, and bout-rate routes available from the same v2 panel table;
- static and interactive outcome scales use the same fixed or metric-specific
  99th-percentile limits;
- shared-unit outcomes use outcome-specific colorbar labels.

Remaining:

- apply semantic export to legacy and final paper panels;
- add publication dimensions/theme decision;
- add figure-level visual regression baselines after scientific approval;
- add notebooks and broader reusable panel library.
