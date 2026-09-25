# Figures and Reproducible Reporting

**Status:** In progress  
**Done:** Routine two-process PNG scheduling, figure-status ledger, four per-fish
profiles per alignment, five comparison outcomes per alignment, five reviewed-
cohort figure families, a matched frozen-cohort population heatmap with fish
coverage, a focused `figure-example-traces` command for Figure 1 C/D trial
traces, semantic SVG/PDF export and proposed Figure 1–4 registry.
Interactive HTML is frozen (Gate F).
**Open:** Approval of the population heatmap signal and matched cohorts, approved Figure 1 examples,
frozen learner representation, verified Figure 4 protocol timing and signed data,
publication dimensions, and visual regression.
**Change class:** Behavior-preserving rendering first; presentation changes separately  
**Depends on:** Implemented artifact-integrity contract and each scientific
stage whose output is exposed; final figure identity is recorded through the
frozen release manifest and existing source/output hashes
**Unlocks:** Practical researcher use and reproducible paper builds  
**Absorbed detail:** Earlier scientific figure and paper-automation plans are
preserved in Git history; their current requirements are below.

## Objective

Expose the same package implementation through CLI, Python, notebooks, and
compatibility scripts, and generate publication figures from saved panel data
without embedding hidden analysis.

## Part A — CLI and stage orchestration

### Future command surface (not the current CLI)

The current supported entry point is `classical-conditioning run-pipeline
--config configs/example-run.json`. It always inventories, verifies intake,
runs corrected three-metric analysis, and schedules every ready routine figure.
Its summary records each figure as completed, failed, or blocked. The command
list below is a design sketch, not implemented syntax.

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
- `--force` creates/replaces only explicitly targeted working outputs under
  defined rules; the final analysis release is never overwritten

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
- exploratory overrides write to explicit temporary outputs, never an
  unlabelled paper release;
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
-> publication/static rendering (interactive HTML frozen)
-> figure QC
```

### Maintained workstreams (from archived figure plan)

| Workstream | Status | Notes |
| --- | --- | --- |
| A Scientific foundations | Open with Gates | Cohort/hash and analysis correctness before final figures |
| B Design system / theme | In progress | Shared theme tokens |
| C Separate analysis, rendering, saving | In progress | Panel data vs draw vs export |
| D Major-figure composition | Open | Manuscript layouts |
| E Interactive adjustment | **Frozen** | Gate F: no further investment |
| F Safe SVG layer | In progress | Semantic gid + sidecar |
| G Export and provenance | In progress | PNG + SVG/PDF |
| H Figure QC | Open | Structural + visual regression |
| I Automation | Open | CLI/batch figure builds |

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
| `publication` | SVG and PDF | Manuscript composition and final archive — **maintained** |
| `static` | PNG | Fast review, QC, presentations, and visual regression — **maintained** |
| `interactive` | Self-contained local HTML | **Frozen** (Gate F): already implemented; no further polish |

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

### Stable paths and authenticated replacement

Derived recipe, analysis, and figure paths are stable: they do not acquire a
run-number suffix. Cohort manifests are immutable. For replaceable derived
artifacts, a changed upstream hash or setting rebuilds the affected data and
atomically replaces the same path; a presentation-only change rerenders the
figure from authenticated panel data. The sidecar and run ledger carry cohort,
metric, settings, source, and code hashes so replacement is auditable.

The proposed paper layout registry is
[`configs/paper-figures/behavior-paper.json`](../configs/paper-figures/behavior-paper.json).
Figure 1 follows the paper scaffold; Figure 4 follows the later user-approved
three-assay learner-stratified block/catch layout. Figure 2 heatmaps require a
frozen fish cohort and explicit coverage; Figure 3 and paper Figure 4 data
remain gated on the learner manifest and reviewed cohorts.
See the [panel-by-panel draft comparison](../docs/analysis/figures/05_DRAFT_COMPARISON.md).

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

## Part F — Paper figure registry and build

The paper-specific automation plan is merged here so infrastructure and paper
panels cannot drift into separate active plans. For Figure 1 and Figure 4,
the paper scaffold and figure list define the intended roles; the supplied PNG
drafts are comparison evidence, not target layouts. Use this precedence:

1. frozen paper release manifest;
2. approved paper scaffold and figure list;
3. executable figure registry;
4. candidate prose and milestone notes.

The proposed registry now exists at
`configs/paper-figures/behavior-paper.json` with stable panel IDs, source
mappings, export rules, and explicit blocked reasons. Before release, expand
each approved panel declaration to include its paper,
figure, panel, renderer and panel-data recipe; exact input artifact IDs; cohort
ID and hash; ordered conditions; fish/trial selectors; metric, shared detector,
outcome and alignment; named windows; aggregation order; uncertainty method and
seed; statistical result IDs; display transform; coverage rule; axes,
annotations and dimensions; and required output modes.

Scientific fields resolve to named, authenticated recipes. Figure configuration may
change display-only fields but cannot silently change cohorts, exclusions,
windows, aggregation, detector, statistics, or learner definitions.

### Planned main figures

| Figure | Purpose | Gate |
| --- | --- | --- |
| 1 | Experimental setup, protocol, metric definition, and representative fish | Approved source artwork and representative-recording rule |
| 2 | Population evidence for delay and trace conditioning | Frozen cohort, outcomes, and population inference |
| 3 | Learner representation, uncertainty, heterogeneity, and validation | Active learner-plan exit gate |
| 4 | Descriptive signed CR profiles for Delay, 3sTrace and 10sTrace, grouped by the approved learner representation | Approved learner manifest, reviewed cohorts and verified expected-US timing; renderer adaptation if Gate L selects a noncategorical representation |

Figure 3 is required because learner analysis is important to the paper. Its
final grammar depends on the learner decision: continuous/probabilistic outputs
must be supported even if a hard categorical threshold is rejected.

Supplementary families cover US validation, detailed protocol/baseline,
no-optovin control, paired/unpaired US alignment, coverage and heatmap QC,
individual trajectories, timing/catch-trial extensions, and visual-CS cohort
sensitivity. Their exact numbering remains provisional until the registry is
approved.

### Panel-data and rendering boundary

```text
analysis artifacts + cohort/model/classifier manifests
    -> resolved paper figure specification
    -> authenticated panel-data builders (Parquet + JSON)
    -> reusable display-only renderers
    -> major-figure composer
    -> review PNG + publication SVG/PDF + sidecar
    -> structural QC + visual regression + release manifest
```

Statistics and classification are loaded from approved result artifacts, never
recomputed by renderers. A scientist must be able to inspect every plotted
number without reverse-engineering a Matplotlib object.

### Commands

```text
classical-conditioning paper-figures plan --spec ...
classical-conditioning paper-figures explain --spec ... --panel ...
classical-conditioning paper-figures build --spec ... [--figure ...]
classical-conditioning paper-figures validate --spec ...
classical-conditioning paper-figures manifest --spec ...
```

`plan` and `explain` are read-only. `build` targets a paper, figure, or panel.
`validate` checks scientific inputs, structure, labels, dimensions, provenance,
and visual baselines. `manifest` freezes the complete figure release. Syncing
publication PDFs to the paper repository is a separate explicit action.

### Implementation sequence

1. Freeze paper scope, figure naming, primary metric/outcomes, detector,
   cohorts, windows, and biological replicate.
2. Implement the typed registry, validation, `plan`, and `explain`.
3. Build reusable panel-data builders: temporal heatmap and coverage,
   fish-level trajectory, paired phase change, condition/model estimate, raw
   trace, protocol timeline, learner diagnostics, and timing estimates.
4. Build Figure 1 preparation, protocol, single-fish traces, and heatmaps with
   its supporting assay/QC outputs.
5. Build Figure 2 matched-cohort heatmaps, block and trial results, and their
   supporting coverage, trajectories, and diagnostics.
6. Build Figure 3 from approved learner artifacts and its validation outputs;
   then build Figure 4 signed learner-stratified block and catch profiles.
   Keep independently evaluated response timing in the
   [supplementary plan](./10_SUPPLEMENTARY_FIGURES.md).
7. Validate the figures from a clean analysis commit and review Figure 4's
   pooled-learner CR result. The
   [tail mechanistic analyses](./09_TAIL_MECHANISTIC_ANALYSES.md) then follow.
8. Freeze the final analysis release manifest, sync outputs explicitly, and
   compile the paper.

### Known blockers

- Short-trace protocol identity and labels require reconciliation.
- The 10-s trace, no-optovin, and red-CS-only cohorts need explicit package
  specifications or frozen cohort definitions.
- Catch-trial and expected-US identities must be emitted by canonical outcome
  artifacts.
- Learner representation and validation are governed by the active learner
  plan, not inferred inside figures.
- Manual setup/tracking assets need an approved source and stable location.
- Final publication dimensions, fonts, and visual baselines remain open.

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
  intensity, and bout-rate routes available from the same authenticated panel table;
- static and interactive outcome scales use the same fixed or metric-specific
  99th-percentile limits;
- shared-unit outcomes use outcome-specific colorbar labels.

Remaining:

- apply semantic export to legacy and final paper panels;
- add publication dimensions/theme decision;
- add figure-level visual regression baselines after scientific approval;
- add notebooks and broader reusable panel library.
