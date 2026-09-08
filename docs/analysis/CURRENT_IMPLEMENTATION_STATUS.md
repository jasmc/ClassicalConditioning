# Current Implementation Status

Status of the installable `classical_conditioning` package and local runs.
How to invoke commands: [README.md](../../README.md).

Last updated: 2026-09-01.

## Summary

| Area | Status |
| --- | --- |
| Environment (`uv.lock`, CPython ≥ 3.12) | Implemented |
| Lossless intake + inventory | Implemented; relocatable `--input-dir` / `--project-dir` |
| Config-driven `run-pipeline` | Implemented (`configs/example-run.json`) |
| Experiments in package config | `allDelay`, `fixedVsIncreasingTrace` |
| Legacy stage-1 + historical LogMedian | Implemented (reproducibility benchmark) |
| Corrected preprocess (no Gate P interp/filter) | Implemented |
| Corrected five-metric candidate route + runner | Implemented |
| Cohort metric comparison (condition-aware) | Implemented |
| Single-fish profile figures (dev + corrected recipes) | Implemented |
| Cohort standardized-difference figures | Implemented |
| Windows artifact ACL staging fix | Implemented in code |
| Paper-approved metric / detector / statistics | **Not done** |
| Full 3 s Trace cohort (12 or 39 fish) on D: | **Not done** |
| 4-fish corrected plumbing run figures on disk | **Blocked by ACL lock on prior artifacts** |

## Runnable locally

```text
immutable TXT acquisition
  -> inventory (optional)
  -> lossless Parquet intake / intake-batch
  -> acquisition integrity report
  -> resolved config + trial map (allDelay | fixedVsIncreasingTrace)
  -> tracking-field audit / validate-raw
  -> frozen legacy stage-1 preprocessing
  -> corrected measured-time frame preprocessing
  -> candidate metrics (development or corrected pairing)
  -> movement-state calibration and bouts
  -> temporal profiles + trial outcomes
  -> five-metric recording/cohort comparison
  -> candidate-runner orchestration
  -> profile figures (static / interactive / publication)
  -> cohort metric-comparison figures (standardized difference)
  -> freeze-cohort / apply-cohort
  -> legacy scaled/normalized vigor + statistics runner
```

## Path contract

- **Raw:** `raw_dir` in run config, or `--input-dir` on single commands.
- **Save:** `save_dir` in run config, or `--project-dir` on single commands.
- Batch entry point: `uv run classical-conditioning run-pipeline --config <json>`.
- Nested save inside raw is allowed only as a folder named exactly `Paper data`.
- Condition token comes from the third underscore field of the recording stem,
  lowercased (`fixedTrace` → `fixedtrace`).

## Experiments

| `experiment_id` | Conditions in package | CR window | Notes |
| --- | --- | --- | --- |
| `allDelay` | control, delay | 0–9 s | Default on many CLI commands |
| `fixedVsIncreasingTrace` | control, fixedtrace | 0–13 s | Increasing Trace excluded from package config |

Keep-conditions for the 3 s Trace plumbing run: `--keep-condition control`
and `--keep-condition fixedtrace`.

## Corrected candidate route (recipe pairing)

Frozen in `CandidateMetricSource` as `tail-candidate-corrected-v1`:

| Stage | Recipe / artifact |
| --- | --- |
| Preprocess | `corrected-preprocess-v1` |
| Metrics | `tail-candidate-corrected-v1` |
| Movement | `movement-candidate-corrected-v2` (one shared detector) |
| Temporal | `candidate-temporal-outcomes-corrected-v3` |
| Trial outcomes | `candidate-trial-outcomes-corrected-v1` |
| Comparison | `candidate-metric-comparison-corrected-v1` |
| Runner | `candidate-corrected-runner-v1` |
| `scientific_status` | `candidate_corrected` |
| Paper-approved | **No** |

Development (intake-sourced) pairing remains available and is still the default
on several single-step CLIs. Pass corrected recipes explicitly when needed.
`figure-candidate-profiles` defaults to the development temporal recipe; use
`--recipe candidate-temporal-outcomes-corrected-v3` for corrected fish.
`figure-metric-comparison` defaults to the corrected comparison recipe.

## Recent engineering (2026-08-31 → 2026-09-01)

### Relocatable 3 s Trace support

- `paths.py`: nested `Paper data`, reserved derived dirs, condition parsing.
- `intake-batch` with `--keep-condition` / repeated `--recording-id`.
- Metric comparison reads experiment CR window and groups cohort summaries by
  `Condition ID` (`all`, `control`, `fixedtrace`).

### Figures

- `figure-candidate-profiles` resolves temporal artifacts via
  `resolve_candidate_metric_source` (development or corrected).
- `figure-metric-comparison` plots fish points and equal-recording mean bars of
  **standardized difference** for the five candidate metrics.

### Artifact ACL fix (Windows)

`tempfile.mkdtemp` staging produced non-inheritable DACLs; `os.replace` carried
them onto published files so later sessions got `Access denied` while still
seeing names and sizes.

Fix: `artifact_staging()` in `artifacts.py`, used by intake and all publish
call sites. New publishes inherit the destination ACL. **Already-written**
locked files under existing project trees are not repaired automatically.

To unlock a previously written project tree (elevated PowerShell):

```powershell
takeown /F "<SAVE-DIR>" /R /D Y
icacls "<SAVE-DIR>" /reset /T /C /Q
```

Then regenerate figures.

## Local runs

### Pilot fixtures (unchanged scientific meaning)

Root used historically:

```text
C:\Users\Public\More projects\Paper data
```

| Recording | Role |
| --- | --- |
| `20221115_04` | Primary fixture; legacy + candidate development + corrected debug |
| `20221116_12` | Second fixture; intake REVIEW (boundary frame-range mismatch) |

Two-fish development runner `candidate-development-twofish-v1` exercised
multi-recording plumbing only. Not paper-approved.

### 4-fish corrected Trace plumbing run

| Field | Value |
| --- | --- |
| Save tree | `C:\Users\Public\More projects\Paper data` |
| Analysis id | `c-copy-4fish-cohort-v1` |
| Recipe | `candidate-corrected-runner-v1` |
| Experiment | `fixedVsIncreasingTrace` |
| Fish | `20230315_05`, `20230316_11` (control); `20230315_06`, `20230316_03` (fixedtrace) |
| Raw source for this trial | C: copy under `Paper data\Raw single fish data` (subset also on D:) |
| Cohort Parquet | Present under `Processed data\Analyses\c-copy-4fish-cohort-v1\` |
| Figures | **Not written** — read/write blocked by ACL on 31/08 artifacts |
| Caveat | `curvature_change_rms` was all zeros on this slice |
| Interpretation | Plumbing check only (N = 4). Not a cohort result. |

Intended full Trace inventory (not yet processed in-package at scale):

| Tree | Notes |
| --- | --- |
| `D:\2023 02-03_Fixed vs increasing trace (3 s)` | Primary raw root discussed for full run |
| Keep | control + fixedTrace (~39 fish); exclude increasing Trace |
| C: subset | ~12 keep fish among Mar 15–16 copies |

This Cursor session could not write to `D:`. Prefer a writable save location with
enough free space (C: Paper data was already space-constrained after the
4-fish trial).

## Scientific status by recipe family

- `legacy-paper-v1`: reproducibility benchmark; known limitations retained.
- `corrected-preprocess-v1`: measured-time frames + validity masks; **Gate P
  open** (no approved interp/filter).
- Candidate development and corrected pairings: exploratory prototypes.
- `candidate-metric-comparison-*`: descriptive only; no inferential claim;
  no metric is paper-approved.
- Gates still open for paper claims include P, T1, C0, and S (see Plans).

## Not implemented / blocked next

- Unlock ACL on existing Paper data tree, then write the 4-fish figures.
- Full Control + fixed Trace run from D: into a spacious save directory.
- Gate P interpolation / temporal–spatial filter freeze.
- Detector / smoothing scientific approval.
- Paper-scale cohort QC and exclusions.
- Confirmatory mixed-effects and Gate S statistics freeze.
- Canonical learner classifier.
- Optional imaging integration (deferred; must not alter behavior artifacts).
- Final manuscript figures from approved routes.

## Active plan pointers

| Doc | Use |
| --- | --- |
| [Plans/DECISIONS.md](../../Plans/DECISIONS.md) | Locked decisions |
| [Plans/IMPLEMENTATION_STEP_INDEX.md](../../Plans/IMPLEMENTATION_STEP_INDEX.md) | Step index |
| [Plans/HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md](../../Plans/HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md) | Two-fish candidate handoff |
| [Plans/Archive/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md](../../Plans/Archive/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md) | Pickle timebase classification |
| [docs/analysis/LEGACY_VS_CORRECTED_WORKFLOW.md](LEGACY_VS_CORRECTED_WORKFLOW.md) | Route pairing |

Engineering focus for Trace work: unlock or re-run into a clean save tree,
generate standardized-difference cohort figures, then scale fish count — not
interpreting N = 4 as a scientific result.
