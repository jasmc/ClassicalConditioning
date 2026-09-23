# Candidate workflow guide

This is the practical guide for the supported, exploratory candidate pipeline.
It complements the [root README](../../README.md), which contains installation
instructions and the full configuration reference. Historical code under
`Archive/` is not part of this workflow.

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

See [output and provenance](OUTPUT_AND_PROVENANCE.md) for the exact purpose of
these directories and markers.

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

- [Current pipeline guide](CURRENT_PIPELINE_GUIDE.md): source-module call graph.
- [Output and provenance](OUTPUT_AND_PROVENANCE.md): data tree and hash markers.
- [Troubleshooting](TROUBLESHOOTING.md): common safe recovery paths.
- [Metrics and bouts](METRICS_AND_BOUTS.md): scientific meaning and current limits.
- [Figure guide](FIGURE_GUIDE.md): render commands and interpretation.
