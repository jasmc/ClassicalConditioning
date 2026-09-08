# Repository Migration and Organization Map

## Rule

Preexisting executable files remain in place until their replacement:

1. has a written current-behavior contract;
2. has characterization tests;
3. reproduces the required legacy output;
4. has all consumers migrated or a compatibility wrapper;
5. passes the relevant scientific gate.

Then preserve history with a Git move and leave a thin wrapper at the familiar
numbered entry point when needed.

## Target structure

Target layout (add modules only when migrating real behavior):

```text
ClassicalConditioning/
|-- pyproject.toml
|-- src/classical_conditioning/
|   |-- config/                 # experiments, recipes, trial map, identity
|   |-- ingestion/              # readers, schemas, validation, audits
|   |-- preprocessing/          # legacy and corrected recipes
|   |-- analysis/               # metrics, movement, outcomes, stats runners
|   |-- figures/                # theme, panels, export
|   |-- artifacts.py            # hashing, transactional publish
|   |-- cohort.py
|   |-- pipeline.py
|   |-- cli.py
|   `-- ...                     # paths, intake, inventory, progress, etc.
|-- scripts/
|   `-- compatibility/          # thin wrappers after numbered scripts move
|-- legacy/
|   |-- scripts/                # full numbered implementations after wrappers
|   |-- modules/                # archived my_* helpers
|   `-- variants/
|       `-- logmedian/
|-- tests/
|   `-- characterization/
|-- docs/
|   `-- analysis/
|-- Plans/
`-- configs/
```

Longer aspirational trees from the archived CODEBASE plan (separate
`domain/`, `schemas/`, `io/`, `pipelines/`, `statistics/`, `learners/`
packages) remain optional future splits; do not create empty architecture
ahead of migrated behavior. Current code already uses flat modules plus the
folders above.

Numbered root scripts stay until replacements have characterization tests,
equivalence evidence, migrated consumers or wrappers, and the relevant
scientific gate. Then Git-move implementations under `legacy/scripts/` and
leave thin wrappers at the familiar entry points when needed.

## Numbered scripts

| Current file | Current responsibility | Replacement | Move only after |
| --- | --- | --- | --- |
| `1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py` | Raw parsing, synchronization, filtering, vigor, bouts, trials, QC plots, exclusion and file moves | `io/`, `preprocessing/`, `quality/`, `pipelines/preprocess.py` | Exact stage-1 characterization; corrected decisions separate; raw file moving disabled; compatibility wrapper tested |
| `2_ExampleFishPlotting.py` | Selected-fish traces, heatmaps, bout zooms, tail trajectories | `figures/panels/` and figure specs | Panel-data/render split reproduces selected figures |
| `3_FishGrouping.py` | Bout masking, percentile scaling, rolling mean, downsampling, condition concatenation | `analysis/cohorts.py`, `analysis/scaling.py`, `analysis/aggregation.py` | Standard-main route characterized and Parquet consumers work |
| `4_ScaledVigorPlotting.py` | Pooled time bins, count/SV tables, heatmaps, line plots | `analysis/temporal.py`, `figures/panels/` | All four current transformations and artifacts are separately reproduced |
| `5_NormalizedVigorPlotting.py` | Trial ratio, block/phase summaries, tests and LME | `analysis/trial_metrics.py`, `statistics/`, `figures/panels/` | Trial metrics and models reproduce frozen inputs/results with diagnostics |
| `6_LearnersQuantification*.py` | Four incompatible learner classifiers and figures | `learners/` adapters and selected classifier | Every variant characterized; canonical decision and validation mode approved |

The root numbered files may remain thin wrappers during the transition. Their
full legacy implementations move under `legacy/scripts/` only after wrappers
are in place.

## Shared modules

| Current file | Treatment |
| --- | --- |
| `data_io.py` | Extract readers one at a time into `io/`; preserve adapters until stage 1 is migrated |
| `analysis_utils.py` | Split by behavior only after each function has direct tests; do not move as one bulk refactor |
| `experiment_configuration.py` | Convert one experiment at a time to immutable specs; keep compatibility factory |
| `general_configuration.py` | Replace mutable singleton with resolved immutable config; preserve legacy adapter |
| `file_utils.py` | Replace tuple-return folder creation with named local paths; keep filename parser compatibility |
| `figure_saving.py` | Extend into versioned figure export with semantic SVG metadata; migrate callers gradually |
| `plotting_style.py` | Convert style constants into immutable themes after figure characterization |

## Legacy modules

Direct search found no active modern imports of:

```text
legacy/modules/my_general_variables.py
legacy/modules/my_experiment_specific_variables.py
legacy/modules/my_functions.py
```

Important historical experiments, the LogMedian branch, and learner variants
have now been checked, with evidence in
`Archive/HISTORICAL_LOGMEDIAN_PIPELINE.md` and
`../docs/analysis/LEARNER_VARIANT_BEHAVIOR_MATRIX.md`. The modules were moved
together to `legacy/modules/`; an executable repository-organization test
prevents new active imports.

## Historical LogMedian variant

The LogMedian branch is a distinct scientific route, not a cleanup patch.
Relevant commits and files are reviewed during the corresponding stages and
ported under `legacy/variants/logmedian/` or versioned package implementations.
Do not merge its deletions/renames wholesale.

## Non-analysis files

| File | Planned treatment |
| --- | --- |
| `ClassicalConditioning.code-workspace` | Keep at root unless a standard `.vscode/` setup replaces it |
| `jasmc.code-profile` | Removed after confirming it was a user-specific exported editor profile with no analysis consumer |
| `tmp_axis_title_spine_anchor.png` | Removed after confirming it was an unreferenced temporary image, not a test fixture |
| `build/` | Keep ignored local outputs only; scientific outputs live under `Paper data` |
| `.venv-learner-vigor/` | Local ignored environment; replace with project environment after dependency migration |

## Final organization and stale-file gate

After the refactor is fully implemented, perform a complete tracked and ignored
file inventory before release:

1. classify every root file as active wrapper, current support module, archived
   legacy source, documentation, configuration, or removable generated output;
2. move numbered implementations only after replacement equivalence and wrapper
   tests pass;
3. delete obsolete wrappers only after all documented commands and consumers
   have migrated;
4. remove stale build, cache, editor, temporary figure, and superseded generated
   artifacts without touching raw or scientific project data;
5. search imports, documentation links, CLI references, and artifact resolvers
   after every move;
6. run the frozen full suite and clean-environment reproduction;
7. record every retained legacy file and its reason in the release manifest.

## Documentation

All implementation plans live under `Plans/`. Analysis audits and generated
behavior maps may move under `docs/analysis/` after link updates. The root
README remains the user entry point.
