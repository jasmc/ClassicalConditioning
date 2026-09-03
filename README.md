# Classical Conditioning in Larval Zebrafish

Python analysis for a head-fixed larval zebrafish classical-conditioning assay.
The installable package under `src/classical_conditioning` is the supported way
to run new analysis. Numbered root scripts remain for legacy reproduction only.

## Install

Supported runtime: **CPython 3.12 or 3.13** on 64-bit Windows. Dependencies are
pinned in `uv.lock`. Python 3.14 is not supported because the locked `pyarrow`
release does not provide a compatible native extension for it.

### Prerequisites

1. Clone this repository.
2. Install [uv](https://docs.astral.sh/uv/) **or** use Python 3.12+ with `pip`.
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
| `routes` | `["legacy", "candidate"]` | Which analysis pipelines to run. |
| `keep_conditions` | all complete triplets | Filename condition tokens to keep (lowercased). |
| `recording_ids` | null | Explicit fish list; omit to auto-discover under `raw_dir`. |
| `run_inventory` | false | Write `Metadata/recording_inventory.json` before intake. |
| `run_intake` | true | Convert raw triplets to lossless Parquet under `save_dir`. |
| `run_figures` | false | Write cohort metric-comparison PNGs after candidate route. |
| `overwrite` | false | Replace existing derived artifacts. |
| `continue_on_error` | true | Keep going when one fish fails (intake / candidate). |
| `candidate_runner_recipe` | `candidate-corrected-runner-v1` | Five-metric corrected route. |
| `legacy_alignment` | `CS` | Legacy statistics alignment. |
| `legacy_run_statistics` | true | Run legacy cohort inference after scaled/normalized vigor. |

Legacy and candidate cohort outputs use `{analysis_id}-legacy` and
`{analysis_id}-candidate` unless you override `legacy_analysis_id` /
`candidate_analysis_id`.

### What `run-pipeline` does

```text
optional inventory
  -> intake-batch (all matching triplets)
  -> [legacy route] legacy-paper-v1 preprocess per fish
                  -> legacy-runner (scaled/normalized vigor + statistics)
  -> [candidate route] candidate-runner (corrected five-metric pipeline
                                         + cohort comparison)
  -> [optional] cohort metric-comparison figures
  -> Metadata/<analysis_id>_pipeline_run.json summary
```

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

## Single-step commands

Individual stages remain available for debugging or partial reruns. All take
`--project-dir` (= your `save_dir`) after intake:

```powershell
uv run classical-conditioning intake-batch --input-dir "<RAW>" --project-dir "<SAVE>"
uv run classical-conditioning legacy-runner --project-dir "<SAVE>" --recording-id ... --analysis-id ...
uv run classical-conditioning candidate-runner --project-dir "<SAVE>" --recording-id ... --analysis-id ...
uv run classical-conditioning figure-metric-comparison --project-dir "<SAVE>" --analysis-id ...
```

Use `uv run classical-conditioning --help` for the full command list.

## Scientific status

- Package outputs are **exploratory** until scientific gates pass.
- Legacy route preserves known old behavior for comparison.
- Candidate route compares five metrics descriptively; no metric is
  paper-approved.
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
