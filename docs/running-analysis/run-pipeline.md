# Candidate workflow guide

**Scope:** Technical reference for supported commands and implementation/comparison routes. Some recipes below are historical or exploratory. Accepted paper methods are defined by [Current figures](../paper/figures/README.md) and the exact frozen selections; this manual does not supersede them.

This is the practical guide for the supported, exploratory candidate pipeline.
It complements the [root README](../../README.md), which contains installation
instructions and the full configuration reference. Historical code under
The original source under `legacy/` is outside this workflow.

## Before the first run

1. Create the project environment with the supported CPython 3.12 or 3.13
   runtime, as described in the README.
2. Identify a read-only `raw_dir` containing complete recording triplets:
   `*_cam.txt`, `*_mp tail tracking.txt`, and `*_stim control.txt`.
3. Choose an empty or dedicated writable `save_dir`. It must not be `raw_dir`.
   If it is nested inside `raw_dir`, its first directory must be `Paper data`.
4. Copy [example-run.json](../../configs/example-run.json), set `raw_dir`,
   `save_dir`, `experiment`, and a unique `analysis_id`.
5. Treat all candidate outputs as exploratory. A successful run is evidence of
   computational completion, not paper approval or a scientific exclusion rule.

## Routine command

```powershell
uv run classical-conditioning run-pipeline --config <YOUR-CONFIG>.json
```

The JSON config is the run record. Inventory, verified intake, corrected
three-metric analysis, and every figure with ready inputs always run. There are
no inventory, intake, route, or figure switches in routine JSON.

## What success looks like

The command reports named stages on stderr. A normal candidate run performs:

```text
full source inventory and hashed status ledger
  -> intake (one raw triplet per recording)
  -> corrected preprocessing
  -> candidate activity metrics
  -> shared movement/bout state
  -> temporal profiles
  -> trial outcomes
  -> technical and exploratory discarding assessment for every inventoried recording
  -> cohort metric comparison
  -> per-fish, comparison, and ready frozen-cohort figures in two render workers
  -> Metadata/<analysis-id>_pipeline_run.json
```

Inspect these outputs in order:

| Question | First artifact to inspect |
| --- | --- |
| Did every raw file form a complete triplet? | `Metadata/recording_inventory.json` and `Metadata/intake_status.json` |
| Did intake accept the acquisition? | `Quality checks/<recording-id>/acquisition_report.html` and `acquisition_summary.json` |
| Which recordings actually reached analysis? | `Metadata/<analysis-id>_pipeline_run.json` and its `selection_assessment` link to `Processed data/Discarding/` |
| Did a stage produce valid lineage? | Corresponding `Metadata/*_complete.json` marker |
| What did the cohort comparison calculate? | `Processed data/Analyses/<analysis-id>/` plus its QC summary |

See [output and provenance](outputs-and-provenance.md) for the exact purpose of
these directories and markers.

## Automatic protocol verification

Every routine candidate run audits authenticated stimulus events before analysis,
including resumed runs and the direct-intake benchmark. Standalone `preprocess`
also audits; supply `--experiment allDelay`, `all3sTrace` or `all10sTrace` for
explicit context. Without it, only unambiguous filename labels resolve an assay.

The audit compares filename aliases, configured condition and recorded timing,
with inclusive ±200 ms tolerance for the 10 s CS and paired-US onset/gap.
Delay/3sTrace/10sTrace latencies are 9/13/20 s from CS onset. Controls may have
accidental coincidences; a fixed paired schedule conflicts with a control label.
Unpaired timing alone cannot identify which assay a control belongs to or prove
randomization. Pulse lengths are compared descriptively with nominal 50/100/500 ms;
the ±200 ms tolerance is not applied to pulse lengths.

Findings **warn and continue**; they never relabel recordings, exclude fish or
change analysis windows. Existing schema, identity and provenance guards remain.
Evidence and a readable view live at `Quality checks/<recording-id>/protocol_verification.json`
and `.html`, with source hashes, settings, event identities and per-trial findings.
Acquisition reports, runner manifests and routine run summaries expose the audit.
To audit existing intake without numerical processing:

```sh
.venv/bin/python -m classical_conditioning verify-protocol --project-dir <SAVE> --recording-id <ID> --experiment all3sTrace
```

`validate-raw --input-dir <RAW> --experiment all3sTrace` performs the same checks
before conversion. Raw filenames remain unchanged; `3sTrace`/`10sTrace` and
supported historical aliases resolve centrally to the existing stored condition IDs.

The first US is retained as US 1. There are 18 USs before training; paired
training US IDs are **19–64**, with blocks 19–28, 29–37, 38–46, 47–55 and 56–64.
Every recording must contain **exactly 94 CSs and 79 USs**: 78 regular USs plus the separate final 500 ms check. Both totals are always compared and reported; missing and additional events are discrepancies, never an acceptable minimum.
Temporal/trial summaries bind the current trial-map hash. Old or missing mappings
are rejected on active reuse and rebuilt by the runner, cascading through dependent
outputs; corrected frames and metrics are unaffected. Frozen results and legacy
reproduction definitions are preserved, and are not regenerated by this change.

## Safe resume and recovery

- With `overwrite: false` (the default), a stage may reuse only a completed
  artifact whose marker and hashes verify. It does not treat mere file presence
  as success.
- With `continue_on_error: true`, failures are recorded per recording and the
  descriptive candidate comparison uses recordings that completed every needed
  stage. This is not a reviewed or frozen fish cohort.
- An unchanged failed intake is skipped with its reason. After fixing the
  problem, run `retry-intake --input-dir <RAW> --project-dir <SAVE>
  --recording-id <ID>` explicitly (or change the source triplet).
- Use `overwrite: true` only when deliberately rebuilding the affected outputs;
  it replaces derived artifacts and their markers, never raw input.
- Start diagnostics with `validate-raw`, `audit-tracking`, `compare`,
  `movement-sensitivity`, or `trace-review` rather than editing Parquet or JSON
  outputs by hand.

## Stimulus timing plots

Routine runs automatically generate descriptive stimulus diagnostics after the
candidate stage, including resumed runs. Tracking/behavior failures do not
prevent independent raw-log diagnostics. This stage does not exclude fish,
change behavior or test whether controls are random.

For current assays, use the approved protocol adapter:

```sh
.venv/bin/python -m classical_conditioning verify-protocol --project-dir <SAVE> --raw-dir <RAW> --experiment allDelay --analysis-id timing-review
```

Repeat `--recording-id <ID>` to select fish; omit it for all discovered fish.
`--raw-dir` supplies authentic logs and permits operation before intake. Without
it, source paths resolve from intake manifests. Unavailable or changed sources
remain failures, with no historical fallback. The fixed-assay verifier also
runs for available intake unless `--timing-only` is specified. Without an
experiment/configuration, expectations remain unassigned.

For future protocols, copy and edit
[`protocol-timing.example.json`](../../configs/protocol-timing.example.json):

```sh
.venv/bin/python -m classical_conditioning verify-protocol --project-dir <SAVE> --raw-dir <RAW> --timing-config <TIMING-JSON> --analysis-id future-timing
```

This uses the configurable timing engine without the fixed-assay verifier. It
does not extend the behavioral pipeline's three assay recipes. An existing-assay
pipeline config can set `protocol_timing_config` to a JSON path with a matching
`experiment_id`.

The plot set contains recording timelines, dual CS/US-aligned rasters, ordered
timing measurements and companion processed-timing residuals; condition/phase/
US-role fish-by-lag heatmaps with equal-fish means, half-bin sensitivity, day/batch
strata and observable-time coverage; and separate CS/US schedule-similarity
matrices with residuals for up to three closest pairs per phase.

Defaults are ±30 s for rasters, ±30 s/1 s bins and ±300 s/5 s bins for lag views.
All CS–US pairs contribute within each horizon; exact target coincidences are
computed independently of bins. Training, maintenance and viability pulses stay
separate. A configured catch slot does not establish absence of a nearby US.
Half-bin sensitivity retains equal-width bins and omits incomplete edge bins.
Phase-specific lag views select CS and US within the same assigned phase;
cross-boundary neighbors remain visible in raw rasters. Their window clipping
and acquisition/gap coverage are recorded explicitly.
Counts per anchor are not density-normalized or rate-corrected. Schedule
comparison removes only a common time offset; drift remains, while unequal
counts or fewer than two events are uncheckable.

```text
Quality checks/<ID>/stimulus-timing/<analysis-id>/
    index.html, evidence.json, timeline.png, trials.png, timing.png, alignment.png
Figures/Stimulus timing/<analysis-id>/
    index.html, evidence.json, lags-*.png, schedules-*.png
Metadata/<analysis-id>_stimulus_timing.json
```

The pipeline report records the stage and plots; collection HTML links recording
views. Evidence retains raw timestamp text, row identities, discrepancies,
coverage and coincidences. Transactional publication binds input/output hashes
and one presentation-contract snapshot, publishing the summary last. Re-running
refreshes diagnostics without recomputing behavior. Manual changes and concurrent
input/output edits stop replacement. No files are automatically pruned and no
page is automatically opened.

### Timing configuration and provenance

- `event_types` maps CS/US to raw labels. `phases` declares names, counts, US
  roles and optional nominal `cs_duration_ms`/`us_duration_ms` and block sizes. `cs_block_counts` and
  `us_block_counts` optionally partition phases into unequal blocks.
- `conditions` supplies aliases, role, catch slots and explicit `{cs, us,
  latency_s}` pairings, including variable latencies. Slots start at one,
  independently for CS and US, across the entire session.
- `recordings` may supply condition, day, `batch_id` with `batch_provenance`,
  and `cs_schedule_id`/`us_schedule_id` with `schedule_provenance`. Only explicit
  intended schedule IDs enter the additional equal-schedule curve; its known
  subset is labeled. Logged similarity does not assign batches or schedule IDs.
- Ordinal correspondence requires complete, usable, unique counts. Missing,
  extra or damaged events leave slots unresolved; raw rasters and unresolved
  whole-recording lag views remain. Equal totals cannot rule out compensating
  missing/extra events. `event_annotations` can independently resolve a
  `source_row` using optional `slot`, `phase`, `role`, `block` and required
  `provenance`; these are separate from observed event numbers.
- `phase_bounds_s` optionally maps phase names to acquisition-relative
  `[start, end]` seconds. A matching `phase_bounds_provenance` entry has
  `{"kind": "recorded", "source": "..."}` or `kind: configured`. Only recorded
  boundaries constrain observable phase exposure; configured boundaries are
  labeled guides. Observed frame-ID gaps are shown; camera clock reversals make
  continuous acquisition coverage unavailable.
- Optional `processed_markers` contains an absolute JSON `path`, its `sha256`
  and `raw_protocol_sha256`. List rows require `type` (CS/US), `source_row` or
  `event_number`, and absolute `beg_ms`/optional `end_ms`. Otherwise the current
  authenticated corrected trial-outcome family is used when available. Unmatched
  events, stale trial-map fingerprints and phase-label disagreement are separate
  findings. Outcome-anchor equality does not verify camera registration; these
  tables do not provide processed offsets. With unresolved correspondence,
  explicit raw source-row identity is required for processed matching; ordinal
  processed labels remain unmatched.

Malformed/nonfinite/fractional timestamps, nonpositive durations and scientific
notation remain visible. Missing observations are not zeros. No inferential
null or automatic randomness/learnability verdict is implemented.

## Manual stages: when they are appropriate

Run individual stages only for a bounded diagnostic, controlled comparison, or
partial rebuild. Their required order is:

1. `preprocess --recipe corrected-preprocess`
2. `activity-metrics --recipe tail-candidate-corrected`
3. `movement-state`
4. `temporal-profiles`
5. `candidate-trial-outcomes`
6. `compare-candidate-metrics`

The benchmark direct-intake metrics route is active but comparison-only; do not
mix its outputs with the normal corrected route. The pipeline prevents many
incompatible combinations through recipe/marker checks, but a human review
should also compare recipe identifiers before interpreting tables.

## Where to continue reading

- [Current pipeline guide](pipeline.md): source-module call graph.
- [Output and provenance](outputs-and-provenance.md): data tree and hash markers.
- [Troubleshooting](run-pipeline.md): common safe recovery paths.
- [Metrics and bouts](../paper/methods/activity-and-bouts.md): scientific meaning and current limits.
- [Figure guide](rendering.md): render commands and interpretation.


## Troubleshooting the candidate pipeline

**Scope:** Technical reference for supported commands and implementation/comparison routes. Some recipes below are historical or exploratory. Accepted paper methods are defined by [Current figures](../paper/figures/README.md) and the exact frozen selections; this manual does not supersede them.

This guide lists safe diagnostics. Do not edit raw text files, Parquet tables,
or `*_complete.json` markers merely to make a stage continue.

| Symptom | Likely boundary that failed | Safe first action | Recovery |
| --- | --- | --- | --- |
| Python/`pyarrow` import error | Unsupported or incomplete runtime | `classical-conditioning environment-report` | Use the project CPython 3.12/3.13 environment and reinstall/sync dependencies. |
| No recordings selected | No complete triplets or filter mismatch | `inventory` or inspect `recording_ids`/`keep_conditions` | Correct the config or raw triplet naming; do not create dummy files. |
| Intake failed | Raw schema, timing, or protocol inconsistency | `validate-raw --input-dir <RAW-DIR>` | Review its report and the per-fish acquisition report; correct source/config only with provenance. |
| Tracking columns unclear | Assumption about tail-point columns | `audit-tracking --input <TRACKING-FILE>` | Use the audit to investigate raw schema before preprocessing. |
| Existing output is rejected | Marker/hash/upstream lineage mismatch | Read the corresponding `Metadata/*_complete.json` and QC summary | Preserve evidence, then rebuild the affected stage with deliberate `overwrite`. |
| A stage is slow or needs re-run selection | Large or interrupted cohort work | `plan-batch`, then `execute-batch` | Resume only pending/failed work from the batch manifest. |
| Detector outcome is surprising | Smoothing/threshold/trace behaviour | `movement-sensitivity` and `trace-review` | Treat output as an exploratory diagnostic; do not tune routine results silently. |
| Two artifacts disagree | Schema/value/route difference | `compare` | Compare matching recipe/source artifacts and document why they differ. |

## What to include in a bug report or review request

Provide the exact command/config, Python environment report, recording and
analysis IDs, relevant run ledger, completion marker, and QC summary. Do not
send only a screenshot of a figure: it lacks the artifact and lineage context
needed to reproduce the issue.

## Explicit non-recoveries

Never resolve an error by deleting raw input, changing source filenames after
intake without recording it, editing output hashes, or copying a completion
marker between recordings. These actions destroy the provenance checks designed
to make recovery safe.
