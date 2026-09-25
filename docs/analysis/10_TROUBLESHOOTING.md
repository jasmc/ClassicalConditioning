# Troubleshooting the candidate pipeline

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
