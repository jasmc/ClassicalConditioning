# Output and provenance guide

`save_dir` is a derived-data project, not a second raw-data folder. The active
pipeline separates machine-readable data, human review evidence, provenance,
and rendered figures so that each can be reviewed and regenerated safely.

## Directory contract

| Directory | Contents | Review rule |
| --- | --- | --- |
| `Processed data/` | Parquet tables consumed by later stages | Never edit in place; rebuild instead. |
| `Quality checks/` | Acquisition reports, coverage summaries, diagnostics, QC images | Review here before interpreting results. |
| `Metadata/` | Raw-source manifests, resolved settings, markers, hashes, run ledgers | Retain alongside every exported data set. |
| `Figures/` | PNG, publication, or interactive renderings | Regenerate from verified source artifacts. |

## Per-recording flow

For `<recording-id>`, intake writes lossless `camera.parquet`, `tracking.parquet`,
and `stimulus_events.parquet` below `Processed data/<recording-id>/`. The normal
corrected route then creates a corrected frame table, candidate metrics, shared
movement/bout state, temporal profiles, and trial outcomes in the same folder.

Every stage also writes two independent forms of evidence:

- a human-facing summary in `Quality checks/<recording-id>/`; and
- a machine-verifiable completion marker in `Metadata/`.

The summary explains coverage, gaps, counts, or diagnostic outcomes. The marker
records artifact paths/hashes and upstream lineage. They answer different
questions and should be retained together.

## Cohort, inference, and operational outputs

- `Processed data/Analyses/<analysis-id>/` contains tables pooled across
  successful recordings, such as metric comparison, model input, permutation,
  bootstrap, mixed-effects, and learning-onset outputs.
- `Processed data/Cohorts/<cohort-id>/` contains frozen reviewed cohort data;
  it is created only by explicit cohort commands, not by filename filtering.
- `Processed data/Batches/<batch-id>/` records resumable operational state for
  `plan-batch` and `execute-batch`.
- Matching analysis/cohort QC reports live below `Quality checks/Analyses/` or
  `Quality checks/Cohorts/`; matching completion markers live under `Metadata/`.

## How artifact verification governs reuse

Completed files are eligible for reuse only when their completion marker
authenticates the expected outputs and upstream source. This protects against:

- a manually edited Parquet table;
- an output copied from another recording or recipe;
- an incomplete prior publish; and
- a downstream artifact built from stale input.

`overwrite: false` therefore means “verify and reuse only compatible complete
artifacts,” not “skip whenever a filename exists.” When verification fails,
preserve the evidence, diagnose the mismatch, and rebuild deliberately with
`overwrite: true` if that is the intended recovery.

Trial-outcome summaries record the selected experiment and resolved
`config.response_window_s`. Verification compares that saved window with the
current experiment definition. Older Trace outcomes made with a 9 s response
window are rejected by per-fish and cohort loaders; rerun the corrected route
before rebuilding downstream cohort, LME, and ratio-figure artifacts.

## Output-review checklist

1. Read the pipeline run ledger in `Metadata/` to see config, selected IDs,
   skips, failures, and generated figures.
2. Read per-recording acquisition/coverage summaries in `Quality checks/`.
3. Confirm the recipe IDs and marker lineage before joining or comparing tables.
4. Keep `Metadata/` when copying a result set to another machine or collaborator.
5. Treat figures as views; use the Parquet table and QC summary as the review
   source when a visual result is surprising.

For filename examples and the directory table, see the [README output locations](../../README.md#output-locations). The exact recipe-bearing suffix
is intentionally variable and should be read from the marker rather than
guessed from this guide.
