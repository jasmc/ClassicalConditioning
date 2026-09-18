# allDelay analysis — Mac to Windows handover

Prepared: 2026-09-18

## Objective

Run the refactored `allDelay` analysis for **every complete recording** in the
raw-data folder, from lossless intake through per-fish figures, cohort figures,
the frozen cohort, and the final learning-onset mixed-effects-model (LME)
figures.

**No data loss is permitted.** Do not delete raw files or any derived artifact
to make room. The new computer therefore needs enough free storage to retain
the full raw-data copy *and* all derived outputs at the same time.

## Current state

This repository is the authoritative codebase. The relevant fixes and the
resumable 10-fish smoke-test runner are already committed:

| Commit | What it contains |
| --- | --- |
| `f2fcfe5` | Reads legacy `ID` camera headers correctly during lossless intake. |
| `1948cf9` | Corrected figure export/publication behavior and LME layout. |
| `76b5e6d` | Resumable 10-fish end-to-end smoke-test script. |

The 10-fish smoke test has completed at this original output location:

```text
/Volumes/JOAQUIM/Digested Data/allDelay-refactor-smoke-10fish
```

It included five delay and five control fish. Its completed learning-onset
figures are:

```text
Figures/PNG/Analyses/allDelay-refactor-smoke-learning-onset-v1/learning-onset.png
Figures/PNG/Analyses/allDelay-refactor-smoke-learning-onset-v1/learning-diagnostics.png
```

The smoke analysis did **not** localize an onset: the simultaneous lower
confidence band did not stay above `delta_min = 0` for three consecutive
trials. That is a result of the small smoke cohort, not a claim about the full
experiment.

The full-cohort project on this PC is
`F:\Digested Data\allDelay-full-v1`. It already contains artifacts from a
partial 10-fish run. **Preserve and resume this directory in place**: do not
delete, replace, or overwrite any existing artifact. The pipeline recognizes
completed work; resume with the same project directory and without
`--overwrite` so that only remaining work is added.

## Transfer checklist

1. Copy or clone this entire repository, including `uv.lock`, `configs/`,
   `scripts/`, `src/`, `tests/`, and this file. Use commit `76b5e6d` or a later
   descendant.
2. Copy the immutable raw-data tree in full. Its Mac source is:

   ```text
   /Volumes/JOAQUIM/Raw Data/allDelay
   ```

   On Windows, place it somewhere like:

   ```text
   J:\Raw Data\allDelay
   ```

3. Optionally copy the completed 10-fish smoke-test output as a reference.
   It is not required for the full run.
4. On the Windows computer, use 64-bit Python **3.12 or 3.13**. Python 3.14 is
   currently unsupported by the Parquet stack.
5. Reserve at least **500 GB free space** on the drive holding the new output
   directory. The raw `allDelay` tree was about 241 GB on the Mac; the full
   lossless derived project is expected to require well over 250 GB. More free
   space is preferable because processing uses temporary staging files.

## Install and verify on Windows

In PowerShell:

```powershell
cd "C:\path\to\ClassicalConditioning"
uv sync --frozen --all-extras
uv run python -m unittest discover -s tests
```

If `uv` is not installed, install it from <https://docs.astral.sh/uv/>. The
project lockfile pins the Python packages, so use `uv sync --frozen` rather
than installing ad-hoc versions.

Before processing, make a lossless inventory and inspect the reported count:

```powershell
$raw = "J:\Raw Data\allDelay"
$out = "F:\Digested Data\allDelay-full-v1"
New-Item -ItemType Directory -Force -Path $out | Out-Null
uv run classical-conditioning inventory --input-dir $raw --output "$out\Metadata\recording_inventory.json" --inspect-tracking-headers
```

On the Mac inventory, there were **57 complete recordings**: **28 control**
and **29 delay**, with no incomplete triplets. Reconfirm this on the copy;
the inventory hashes document exactly what was processed.

Write the environment record too:

```powershell
uv run classical-conditioning environment-report --output "$out\Metadata\environment.json"
```

## Full lossless candidate pipeline

Create `configs\allDelay-full-windows.json` from the following template, editing
only the two Windows paths if yours differ:

```json
{
  "raw_dir": "J:\\Raw Data\\allDelay",
  "save_dir": "F:\\Digested Data\\allDelay-full-v1",
  "experiment": "allDelay",
  "analysis_id": "allDelay-full-v1",
  "routes": ["candidate"],
  "keep_conditions": ["control", "delay"],
  "candidate_runner_recipe": "candidate-corrected-runner-v1",
  "run_inventory": true,
  "run_intake": true,
  "run_figures": true,
  "figure_outcomes": [
    "total-activity",
    "movement-probability",
    "fraction-time-moving",
    "conditional-intensity",
    "bout-rate"
  ],
  "overwrite": false,
  "continue_on_error": true
}
```

Then start the resumable pipeline:

```powershell
uv run classical-conditioning run-pipeline --config configs\allDelay-full-windows.json
```

If the inventory above has already been written to
`<project>\Metadata\recording_inventory.json`, set `"run_inventory": false`
in the run configuration before invoking `run-pipeline`. The pipeline otherwise
correctly refuses to replace that provenance record when `overwrite` is false.

Do not use `--overwrite` for a resume. The pipeline recognizes completed
artifacts and proceeds with work that remains. If a recording fails, retain
all output and inspect its run manifest/log before rerunning that one fish.

This command writes the complete lossless intake, all corrected candidate
intermediates, trial outcomes, quality-control artifacts, and the five cohort
metric-comparison figures. It does not write the four profile figures per
fish, the frozen full cohort, or the LME figures; those are the next steps.

## Per-fish figures

After the pipeline succeeds, run this PowerShell loop. It reads the inventory
produced above, selects every complete control/delay recording, and creates
four static profile figures per fish without modifying raw data.

```powershell
$project = "F:\Digested Data\allDelay-full-v1"
$inventory = Get-Content "$project\Metadata\recording_inventory.json" -Raw | ConvertFrom-Json
$fish = $inventory.records |
  Where-Object { $_.status -eq "COMPLETE" -and $_.condition_id -in @("control", "delay") } |
  ForEach-Object { $_.recording_id } |
  Sort-Object -Unique

foreach ($id in $fish) {
  foreach ($figure in "total-activity-raw", "total-activity-scaled", "conditional-intensity-raw", "bout-outcomes") {
    uv run classical-conditioning figure-candidate-profiles `
      --project-dir $project --recording-id $id --trial-type CS `
      --figure $figure --mode static
  }
}
```

Expected figure location per fish:

```text
<project>\Figures\PNG\Fish\<fish-id>\...
```

## Freeze the all-fish cohort before inference

The pipeline discovers complete files; it does **not** decide scientific
inclusion. Before inferential statistics, create and review one cohort CSV.
For the requested technical all-complete analysis, the cohort should have 57
rows (28 control, 29 delay) and make the inclusion policy explicit. Do not
describe it as a final publication cohort until quality-control review is
complete.

Use `configs\allDelay-refactor-smoke-10fish-cohort.csv` as the column-format
example. Give the new file a name such as:

```text
configs\allDelay-full-v1-cohort.csv
```

Every row must state the fish ID, condition, technical validity, whether it is
primarily included, review status, reviewer, and review timestamp. Preserve
excluded fish as rows with an explicit exclusion reason; never silently remove
them.

After a reviewer approves the CSV, freeze it once:

```powershell
uv run classical-conditioning freeze-cohort `
  --project-dir "F:\Digested Data\allDelay-full-v1" `
  --input configs\allDelay-full-v1-cohort.csv `
  --cohort-id allDelay-full-v1 `
  --policy-id all-complete-triplets-v1
```

## Full-cohort LME and final figures

These commands consume the frozen cohort and retained trial-outcome data. They
do not need to reread raw video/tracking files.

```powershell
$project = "F:\Digested Data\allDelay-full-v1"
$cohort = "allDelay-full-v1"
$analysis = "allDelay-full-learning-onset-v1"

uv run classical-conditioning build-cohort-trial-outcomes `
  --project-dir $project --cohort-id $cohort `
  --metric-recipe tail-candidate-corrected-v1

uv run classical-conditioning learning-onset `
  --project-dir $project --cohort-id $cohort --analysis-id $analysis `
  --metric tail_length_weighted_angular_l1 --outcome total-activity `
  --test-condition delay --delta-min 0 `
  --bootstrap 499 --permutations 9999

uv run classical-conditioning figure-learning-onset `
  --project-dir $project --analysis-id $analysis --mode static

uv run classical-conditioning figure-learning-diagnostics `
  --project-dir $project --analysis-id $analysis --mode static
```

The final two PNGs will be under:

```text
<project>\Figures\PNG\Analyses\allDelay-full-learning-onset-v1\learning-onset.png
<project>\Figures\PNG\Analyses\allDelay-full-learning-onset-v1\learning-diagnostics.png
```

## How to read the two final LME figures

`learning-onset.png` has three panels:

* **A — Fish trajectories.** Thin lines are individual fish and thick lines
  summarize their condition. The response is compared with each trial's own
  baseline; lower values here represent stronger suppression. Blue is control,
  magenta is delay.
* **B — Learning-onset contrast.** This is the adjusted delay-minus-control
  contrast over trial number, relative to pretraining. The grey envelope is a
  simultaneous 95% fish-level bootstrap band. Learning is established only at
  the first trial where its *lower* boundary is above `delta_min` for three
  consecutive trials. Here `delta_min = 0`, so it requires evidence of a
  reliably positive adjusted delay effect, not merely a point estimate above
  zero.
* **C — Planned blocks.** Point estimates and intervals for the prespecified
  training/test blocks. This is supporting evidence; panel B supplies the
  formal onset decision.

`learning-diagnostics.png` checks whether the LME is an adequate summary. The
residual-versus-fitted panel should show a roughly horizontal cloud with
similar spread; the Q-Q panel should follow its diagonal. Tail departures can
occur in biological activity data, but strong patterns, funnels, or curvature
should prompt model review before interpreting the onset claim.

## Statistical definitions carried into the full run

* `delta_min`: the smallest delay-versus-control effect that counts as
  scientifically meaningful. It is set to `0` for the present technical run,
  which means the threshold is simply a positive adjusted difference. If the
  study needs a non-zero biological effect threshold, choose and document it
  before looking at the full result.
* Resampling/bootstrapping: the program resamples **fish**, not individual
  frames or trials. Each resample refits the longitudinal model. This respects
  repeated measurements within fish and produces uncertainty bands for the
  contrast trajectory.
* The 3-trial persistence rule guards against calling a one-trial fluctuation
  “learning.” The output records both the raw crossing and the sustained onset
  decision.

## Do not lose or overwrite

* Raw data are immutable. Do not run any command whose output is inside the
  raw directory.
* Keep the raw-file inventory and environment record with the results.
* Resume `F:\Digested Data\allDelay-full-v1` in place. It contains the
  partial 10-fish run; do not delete or overwrite it.
* Keep all intermediate Parquet, QC, manifests, profile figures, cohort
  figures, frozen cohort, LME tables, and LME figures. The purpose of moving to
  the PC is to avoid deleting any of these.
* Back up the completed output directory before changing a frozen cohort or
  rerunning inference with `--overwrite`.

## First things to inspect if a command stops

1. Confirm the raw-drive and output-drive letters did not change.
2. Confirm free disk space remains comfortably above the size of the next
   recording's outputs.
3. Look in `<project>\Metadata` for the pipeline/run manifest and in the
   affected fish's output directories for stage completion records.
4. Resume the same command without `--overwrite`. Do not delete partial work.
5. If a single fish consistently fails, save its error text and raw inventory
   record; process the remaining fish, then diagnose that fish separately.

