# Classical Conditioning in Larval Zebrafish

Python analysis for a head-fixed larval zebrafish classical-conditioning assay.
The installable package under `src/classical_conditioning` is the supported way
to run new analysis. Numbered root scripts remain for legacy reproduction only.

## Install

Supported runtime: **CPython ≥ 3.12** on 64-bit Windows. Dependencies are pinned
in `uv.lock`.

### Prerequisites

1. Clone this repository.
2. Install [uv](https://docs.astral.sh/uv/) **or** use **Python 3.12 or 3.13** with `pip`.
   Python 3.14 is not supported yet: `pyarrow` has no compatible wheel and fails
   importing `pyarrow.lib`. Use the project venv (`.venv`) created by `uv sync`.
3. Choose a writable **save directory** with enough free disk space for Parquet
   outputs (order of ~1–2 GB compressed per fish after intake; scale up for
   full cohorts).

### Recommended install (uv)

```powershell
cd "C:\path\to\ClassicalConditioning"
uv sync --frozen --all-extras
uv run python -m unittest discover -s tests
```

`--all-extras` installs the full scientific stack used by tests and optional
figure modes (seaborn, statsmodels, plotly, etc.). The base package alone is
enough for intake and Parquet analysis, but not for the full test suite or
interactive HTML figures.

### Alternative install (pip)

```powershell
python -m pip install -e ".[legacy,interactive]"
python -m classical_conditioning --help
```

Record the environment on each machine:

```powershell
uv run classical-conditioning environment-report `
  --output "<SAVE-DIR>\Metadata\environment.json"
```

## Run from a config file

All relocatable batch work is driven by a JSON file. You set **where raw data
live**, **where to save outputs**, **which experiment**, and **which analysis
routes** to run — then invoke one command.

```powershell
uv run classical-conditioning run-pipeline --config configs\example-run.json
```

Copy [configs/example-run.json](configs/example-run.json) and edit the paths.

### Required config fields

| Field | Meaning |
| --- | --- |
| `raw_dir` | Folder containing immutable camera / tracking / protocol triplets. Never written. |
| `save_dir` | Writable project tree (`Processed data`, `Quality checks`, `Metadata`, `Figures`). |
| `experiment` | Package experiment id (see table below). |
| `analysis_id` | Label for this run; used in output paths and manifests. |

### Common optional fields

| Field | Default | Meaning |
| --- | --- | --- |
| `routes` | `["candidate"]` | Which analysis pipelines to run. Add `"legacy"` only to reproduce the frozen historical route. |
| `keep_conditions` | all complete triplets | Filename condition tokens to keep (lowercased). |
| `recording_ids` | null | Explicit fish list; omit to auto-discover under `raw_dir`. |
| `run_inventory` | false | Write `Metadata/recording_inventory.json` before intake. |
| `run_intake` | true | Convert raw triplets to lossless Parquet under `save_dir`. |
| `run_figures` | false | Write cohort metric-comparison PNGs after candidate route. |
| `overwrite` | false | Replace existing derived artifacts. |
| `continue_on_error` | true | Keep going when one fish fails (intake / candidate). |
| `candidate_runner_recipe` | `candidate-corrected-runner-v1` | Six-metric corrected route. |
| `legacy_alignment` | `CS` | Legacy statistics alignment. |
| `legacy_run_statistics` | true | Run legacy cohort inference after scaled/normalized vigor. |

Legacy and candidate cohort outputs use `{analysis_id}-legacy` and
`{analysis_id}-candidate` unless you override `legacy_analysis_id` /
`candidate_analysis_id`.

### What `run-pipeline` does

```text
optional inventory
  -> intake-batch (all matching triplets)
  -> [candidate route, default] candidate-runner
         corrected-preprocess-v1 -> six activity metrics -> movement state
         -> temporal outcomes -> trial outcomes -> cohort comparison
  -> [legacy route, opt-in] legacy-paper-v1 preprocess per fish
                  -> legacy-runner (scaled/normalized vigor + statistics)
  -> [optional] cohort metric-comparison figures
  -> Metadata/<analysis_id>_pipeline_run.json summary
```

`run_figures` writes only the **cohort metric-comparison** figure. Per-recording
heatmaps are a separate command (see below).

While a run is in progress, status goes to **stderr** so normal result paths can
still be captured from stdout:

- `==> Stage name` banners for major phases (intake, candidate runner, figures)
- `tqdm` progress bars when several recordings or outcomes are processed
- `> step-name: running / done in …s` flags inside each fish for the five
  candidate stages (preprocess, metrics, movement, temporal profiles, trial outcomes)
- `[i/n] recording-id: completed` per-fish summaries

Use `--quiet` on `run-pipeline`, `candidate-runner`, or `execute-batch` to
suppress this. In JSON configs, set `"show_progress": false` for `run-pipeline`.

Package experiments currently available:

| `experiment` | Conditions | CR window |
| --- | --- | --- |
| `allDelay` | control, delay | 0–9 s |
| `fixedVsIncreasingTrace` | control, fixedtrace | 0–13 s |

### Raw file rules

Intake and inventory recursively search `raw_dir` for complete triplets:

| Kind | Filename suffix |
| --- | --- |
| Camera | `_cam.txt` |
| Tracking | `_mp tail tracking.txt` |
| Protocol | `_stim control.txt` |

- Recording ID = first two `_` fields (`YYYYMMDD_NN`).
- Condition = third `_` field, lowercased.
- `save_dir` must not equal `raw_dir`. If nested inside raw, name it
  `Paper data`.

### Save-directory layout

```text
<save_dir>/
|-- Processed data/<recording-id>/
|-- Processed data/Analyses/<analysis-id>/
|-- Quality checks/<recording-id>/
|-- Metadata/
`-- Figures/
```

## Analysis pipeline architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F4F4F6', 'primaryTextColor': '#22242A', 'primaryBorderColor': '#B8BCC4', 'lineColor': '#6E7480', 'fontSize': '13px', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph TD
    RAW[/"Immutable raw triplets<br/>cam + tail tracking + stim control"/]:::raw

    RAW --> INTAKE["<b>intake-batch</b><br/>lossless Parquet, SHA-256 recorded<br/>raw_dir never written"]

    INTAKE --> PRE["<b>corrected-preprocess-v1</b><br/>measured timestamps, frame gaps,<br/>per-point validity masks"]

    PRE --> MET["<b>tail-candidate-corrected-v1</b><br/>6 activity metrics per frame"]:::key
    PRE --> DET["<b>movement-candidate-corrected-v2</b><br/>ONE shared bout detector<br/>legacy envelope on distal speed<br/><i>metric-independent</i>"]:::fix

    MET --> PROF
    DET --> PROF["<b>candidate-temporal-outcomes-corrected-v3</b><br/>align to CS/US onset, −45..+45 s, 0.5 s bins<br/>per-metric intensity + shared bout outcomes"]

    PROF --> TRIAL["<b>candidate-trial-outcomes</b><br/>per-trial baseline vs response"]
    TRIAL --> COHORT["<b>cohort metric comparison</b><br/>standardized difference per fish"]

    PROF --> FIG1["<b>FIG 1</b> total activity, raw<br/>6 rows = 6 metrics"]:::fig
    PROF --> FIG2["<b>FIG 2</b> total activity, scaled<br/>6 rows = 6 metrics"]:::fig
    PROF --> FIG3["<b>FIG 3</b> conditional intensity, raw<br/>6 rows = 6 metrics"]:::fig
    PROF --> FIG4["<b>FIG 4</b> bout outcomes<br/>3 rows, metric-free"]:::fig
    COHORT --> FIGC["<b>cohort comparison</b><br/>6 groups = 6 metrics"]:::fig

    INTAKE -.-> LEG["<b>legacy route</b> (opt-in)<br/>legacy-paper-v1 -> legacy-runner<br/>frozen historical reproduction"]:::legacy

    classDef raw fill:#FFFFFF,stroke:#6E7480,stroke-width:1px,color:#22242A
    classDef key fill:#1F6FEB,stroke:#1A5FCC,stroke-width:1px,color:#FFFFFF
    classDef fix fill:#2DA44E,stroke:#1F7A3A,stroke-width:1px,color:#FFFFFF
    classDef fig fill:#F4F4F6,stroke:#6E7480,stroke-width:1px,color:#22242A
    classDef legacy fill:#EDEDF0,stroke:#B8BCC4,stroke-width:1px,color:#6E7480
```

Note that metrics and the detector are **siblings**, not a chain: the detector
does not consume the six metrics. It runs once on the distal cumulative-angle
speed — the signal the historical pipeline used — and every metric inherits its
segmentation.

Every stage writes a completion marker containing the SHA-256 of its outputs,
and each stage re-verifies the marker of the stage before it. A stage refuses to
run on stale or edited upstream artifacts rather than silently producing
mismatched results.

### The shared bout detector

Bout detection is a property of the animal's behavior, not of the metric used to
describe it, so exactly one detector runs per recording. It reproduces the
historical four-step rule from `legacy/modules/my_functions.py`:

1. Build an envelope: centered rolling **max minus rolling min** of the smoothed
   distal cumulative-angle speed (windows 28.6 ms and 571.4 ms).
2. Threshold the envelope at **4 deg/ms**.
3. Merge bouts separated by less than **14.3 ms**, then drop bouts shorter than
   **57.1 ms**.
4. Drop bouts whose peak instantaneous angular speed never reaches
   **1 deg/ms**.

Historical values were frame counts at an interpolated 700 FPS (20, 400, 10, 40
frames) and degrees; they are applied here as milliseconds of *measured* time
and converted to rad/ms, so the detector no longer assumes a fixed frame rate.
Windows never span a tracking discontinuity.

## Command reference

All analysis commands take `--project-dir` (= your `save_dir`). Run
`uv run classical-conditioning <command> --help` for the full flag list.

### Everyday use

| Command | Purpose |
| --- | --- |
| `run-pipeline` | Run everything from one JSON config. **Start here.** |
| `inventory` | Discover and hash raw triplets; report completeness. |
| `intake-batch` | Convert every complete triplet to Parquet. |
| `candidate-runner` | All six candidate stages per fish + cohort comparison. |
| `figure-candidate-profiles` | Per-fish metric heatmaps. |
| `figure-metric-comparison` | Cohort metric comparison bars. |
| `environment-report` | Record pinned versions for reproducibility. |

### Individual candidate stages

Useful for debugging or partial reruns, in dependency order:

| Command | Purpose |
| --- | --- |
| `preprocess --recipe corrected-preprocess-v1` | Measured-time frames and validity masks. |
| `activity-metrics --recipe tail-candidate-corrected-v1` | The six per-frame metrics. |
| `movement-state` | Threshold calibration and bout detection. |
| `temporal-profiles` | Trial-aligned 0.5 s bins. |
| `candidate-trial-outcomes` | Per-trial baseline and response. |
| `compare-candidate-metrics` | Cohort metric comparison table. |

### Quality control and diagnostics

| Command | Purpose |
| --- | --- |
| `validate-raw` | Check one triplet before intake. |
| `audit-tracking` | Inventory tracking columns without assuming validity. |
| `compare` / `compare-legacy-pickle` | Diff Parquet artifacts, or Parquet against a historical pickle. |
| `compare-routes` | Summarize legacy vs candidate preprocessing behavior. |
| `movement-sensitivity` | Vary detector parameters and report outcome sensitivity. |
| `trace-review` | Balanced trace windows for human detector review. |
| `resolve-config` | Dump resolved recipe JSON and trial map. |

### Cohort, statistics, and batch control

| Command | Purpose |
| --- | --- |
| `freeze-cohort` / `apply-cohort` | Freeze a reviewed fish list, then filter to it. |
| `plan-batch` / `execute-batch` | Deterministic per-recording work plan, then run pending or failed. |
| `candidate-mixed-effects` | Mixed-effects model over candidate outcomes. |
| `candidate-fish-permutation` / `candidate-fish-bootstrap` | Fish-level permutation and bootstrap. |
| `candidate-model-input` | Export the model input table. |

### Legacy reproduction only

`legacy-runner` runs the frozen stage-3/4/5 chain; the individual steps
(`legacy-logmedian`, `legacy-standard-main`, `legacy-scaled-vigor`,
`legacy-normalized-vigor`, `legacy-statistics`), plus `figure-legacy-review` and
`legacy-candidate-outcome-comparison`, exist to reproduce and audit historical
numbers. Do not use them for new analysis.

## Figures

Two figure families are built from the candidate route. Both take `--mode`:
`static` (PNG), `publication` (SVG/PDF with provenance sidecar), and for
profiles also `interactive` (self-contained HTML with hover values).

### Per-fish heatmaps

Four figures, selected with `--figure`:

| `--figure` | Rows | Cell value |
| --- | --- | --- |
| `total-activity-raw` | 6 metrics | Mean metric per bin, native units |
| `total-activity-scaled` | 6 metrics | Same, two-layer scaled to 0-1 |
| `conditional-intensity-raw` | 6 metrics | Mean metric inside bouts, native units |
| `bout-outcomes` | 3 outcomes | Movement probability / fraction time moving / bout rate |

```powershell
uv run classical-conditioning figure-candidate-profiles `
  --project-dir "<SAVE>" --recording-id 20221115_04 `
  --trial-type CS --figure total-activity-raw --mode static `
  --recipe candidate-temporal-outcomes-corrected-v3
```

All four for one fish:

```powershell
foreach ($fig in "total-activity-raw","total-activity-scaled","conditional-intensity-raw","bout-outcomes") {
  uv run classical-conditioning figure-candidate-profiles `
    --project-dir "<SAVE>" --recording-id 20221115_04 `
    --trial-type CS --figure $fig --mode static `
    --recipe candidate-temporal-outcomes-corrected-v3
}
```

### Cohort metric comparison

```powershell
uv run classical-conditioning figure-metric-comparison `
  --project-dir "<SAVE>" --analysis-id <analysis-id>-candidate `
  --trial-type CS --outcome movement-probability --mode static
```

Outputs go to `Figures/PNG/<recording-id>/` and
`Figures/PNG/Analyses/<analysis-id>/` (or `Figures/Publication/...`,
`Figures/Interactive/...`).

## What the heatmap figures actually show

### Axes, shared by all four figures

- **x-axis: time from stimulus onset**, −45 s to +45 s, in 0.5 s bins
  (180 bins). Zero is CS or US onset depending on `--trial-type`.
- **y-axis: trial number** of that trial type. Each pixel row is one trial, so
  you read learning across the session by scanning upward.
- A **shaded vertical band** marks the stimulus window.
- **Blank cells are masked, not zero.** A bin is painted only if at least
  **90 %** of its *expected* frames are usable, where expected frames = bin
  width / median frame interval. That gate catches dropped frames and invalid
  tracking alike, so sparse data never masquerades as low activity.

What changes between figures is what a **row** means.

### Figures 1-3: per-metric intensity (6 rows)

Rows are the six metrics. These figures are legitimately per-metric, because
each row is a genuinely different measurement of tail motion.

| Figure | Cell value |
| --- | --- |
| 1, `total-activity-raw` | Mean of the metric over valid frames in the bin, in native units |
| 2, `total-activity-scaled` | The same quantity after the historical two-layer per-trial scaling |
| 3, `conditional-intensity-raw` | Mean of the metric over frames **inside a detected bout**, native units |

**Conditional intensity** exists because total activity conflates two things:
how *often* the animal moved and how *hard* it moved. A bin with one violent
flick and a bin of constant weak wiggling can average the same. Conditional
intensity averages only over frames inside a bout, so it answers "given that it
was moving, how vigorous was the movement?" Roughly,
`total activity ≈ fraction of time moving × conditional intensity`.

**Two-layer scaling** in Figure 2 reproduces the pre-refactor transform exactly:

1. **Layer 1, on frames, per trial.** `(v − P10) / (P90 − P10)`, where the
   quantiles come from samples earlier than **−15 s** in that trial.
2. **Layer 2, after binning, per trial.** A second `(v − P10) / (P90 − P10)`
   over every **pre-onset** bin, then clipped to `[0, 1]`.

A trial with no usable pre-baseline window yields NaN rather than being
rescaled against itself.

### Figure 4: bout-detection outcomes (3 rows, metric-free)

Rows are three different outcomes, all derived from the single shared detector:

| Row | Cell value |
| --- | --- |
| Movement probability | Moving frames / detector-valid frames |
| Fraction time moving | Moving *time* / detector-valid *time*, weighted by each frame's `DeltaTimeMs` |
| Bout rate | Bout **onsets** x 60000 / valid time = initiations per minute (counts starts, not duration) |

There is no metric dimension here. One detector produced one segmentation, so
these three numbers are properties of the animal's behavior. They are written
identically onto every metric row of the profile table, and the figure reads a
single metric's rows to avoid drawing the same data six times.

### Colour scaling

- **Figure 2** uses a single shared colorbar fixed to `[0, 1]`: every row is
  already on the same scale, so one colorbar is honest and cross-row comparison
  is meaningful.
- **Figures 1, 3, and 4** get **one colorbar per row**, because rows carry
  different units (rad/ms vs px/ms vs bouts/min) or different natural ranges.
  Within Figure 4, the two proportions are fixed to `[0, 1]` and bout rate uses
  its own 99th-percentile limit.

The consequence: in Figures 1 and 3, **colour is not comparable between rows**,
by construction — each row is in its own units with its own colorbar. Use those
rows to read temporal and across-trial structure within a metric; use the cohort
figure for magnitude comparisons.

Every exported figure carries a provenance record with the input artifact hash,
the source-file hash, the exact reproduction command, and the per-artist field
mapping (column, coverage field, threshold, display scale, and whether the row
came from the shared detector).

### What the cohort figure shows

`figure-metric-comparison` collapses each fish to a single number per metric:
the **standardized difference** `(response − baseline) / baseline SD`, computed
per fish so every fish contributes equally regardless of trial count. Bars are
grouped by metric and coloured by condition (control / delay / trace). This is
the figure to use when asking whether an effect is consistent across fish, and
whether the choice of metric changes that answer. It is descriptive only — no
metric is paper-approved and no inferential claim is attached.

## Scientific status

- Package outputs are **exploratory** until scientific gates pass.
- Legacy route preserves known old behavior for comparison.
- Candidate route compares six metrics descriptively; no metric is
  paper-approved. The sixth (`legacy_distal_angular_speed`) is a
  measured-time benchmark of the historical formula, not a reproduction of the
  historical pipeline.
- Bout detection is **shared and metric-independent**: one legacy-style
  envelope detector runs on the distal cumulative-angle speed, and every metric
  inherits its segmentation. Bout-derived outcomes therefore describe behavior,
  not detector calibration. The thresholds are the historical constants, not
  values validated against video labels on this data, so blinded manual
  validation is still required.
- Cohort figures use fish-equal **standardized difference**
  `(response − baseline) / baseline SD`.

See [docs/analysis/CURRENT_IMPLEMENTATION_STATUS.md](docs/analysis/CURRENT_IMPLEMENTATION_STATUS.md)
for implementation detail and known limits.

## Legacy numbered scripts

Root scripts `1_…` through `6_…` and `experiment_configuration.py` are the
historical pipeline (machine-specific paths, in-script `RUN_*` flags). Prefer
`run-pipeline` or the package CLI for new work.

## Related documents

| Document | Contents |
| --- | --- |
| [docs/analysis/CURRENT_IMPLEMENTATION_STATUS.md](docs/analysis/CURRENT_IMPLEMENTATION_STATUS.md) | Implementation status |
| [docs/analysis/LEGACY_VS_CORRECTED_WORKFLOW.md](docs/analysis/LEGACY_VS_CORRECTED_WORKFLOW.md) | Route pairing |
| [Plans/DECISIONS.md](Plans/DECISIONS.md) | Locked decisions |

Manuscript: `C:\Users\Public\More projects\Paper\Learning paper`

## Repository folders and cleanup guidance

The repository contains source code, migration documentation, legacy
implementations, tests, and local development files. It does not contain the
raw scientific dataset or the normal `Paper data` output tree; those are
configured outside this repository.

### Project folders

| Folder | Contents | Cleanup guidance |
| --- | --- | --- |
| `.git/` | Git history, branches, remotes, hooks, and object storage | Never delete manually |
| `.pytest_cache/` | Temporary pytest cache | Safe to delete; regenerates automatically |
| `.venv/` | Local Python environment and installed dependencies | Safe to recreate with `uv sync --frozen --all-extras` |
| `.vscode/` | Formatting, linting, and editor settings | Keep if these editor conventions are useful |
| `configs/` | Example pipeline configuration | Keep; edit a copy for real runs |
| `docs/analysis/` | Architecture, behavior, implementation, and scientific audits | Keep |
| `legacy/modules/` | Archived historical helper modules | Keep for reproducibility until legacy retirement |
| `Plans/` | Active migration plans, decisions, notes, and completed-plan archive | Keep while migration is active |
| `scripts/` | Small standalone inspection utilities | Keep if legacy artifact inspection is needed |
| `src/` | Supported installable `classical_conditioning` package | Keep |
| `tests/` | Unit, integration, characterization, and legacy-equivalence tests | Keep |

Generated `__pycache__/` folders and `src/classical_conditioning.egg-info/`
are disposable. `push.log` is also an ignored local log. The root numbered
scripts and helper modules remain for historical reproduction; do not remove
them until their documented replacement, consumers, and equivalence tests have
passed the relevant scientific gates.

Two root files need a human decision rather than automatic deletion:
`README2.md` appears to be an old analysis transcript, and
`MY___PLANS. we need to add some kinda loading ba` contains scratch notes and
pasted runtime output. Review their history and references before archiving or
deleting them.

