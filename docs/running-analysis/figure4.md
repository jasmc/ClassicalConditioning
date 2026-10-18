# Figure 4 learner-stratified CR profiles

**Scope:** Technical reference for supported commands and implementation/comparison routes. Some recipes below are historical or exploratory. Accepted paper methods are defined by [Current figures](../paper/figures/README.md) and the exact frozen selections; this manual does not supersede them.

Figure 4A–C are descriptive Delay, 3sTrace and 10sTrace profiles. Each figure has the nine CS ten-trial blocks from `ExperimentSpec` and one pooled-catch row. The five individual catch trials, movement probability and signed-signal fish coverage are supplementary figures for both main and individual-catch groups. The older condition-wide 0–1 scaled activity plots remain distinct outputs.

## Required inputs

Obtain these from the approved workstreams; do not create substitute labels or change hashes to make a run pass:

1. One selected metric: `tail_length_weighted_angular_l1`, `whole_tail_xy_mean_speed_normalized`, or `legacy_distal_angular_speed`.
2. A reviewed cohort ID for each of `allDelay`, `all3sTrace`, and `all10sTrace`, plus the project directory holding each cohort. Each cohort needs authenticated corrected candidate metrics, movement states, temporal profiles, stimulus events, source manifests, and cohort trial outcomes for its fish.
3. The frozen Gate L fish-classification CSV/Parquet table and its same-stem `.manifest.json` sidecar. Its metric and cohort hashes must match the selection above. It must classify every fish in the three cohorts exactly once, including controls and any `Unclassified` fish with reasons.
4. The three `assessment-summary.json` files named in that sidecar, with all five associated assessment Parquet files still present and hash-matching. The selection assessments must use `tail-candidate-corrected` and the selected metric.
5. A writable output project directory and figure output directory. Keep the input project directories available throughout analysis.

## Frozen classifier input

`figure4-analyze` consumes a CSV or Parquet fish table plus a JSON file with the same stem and suffix `.manifest.json`. It does not fit or select a classifier. The table must contain one row per `(experiment_id, condition_id, fish_id)` across every requested cohort.

For an explicitly exploratory partial-assay review, the command also accepts a nonempty subset of the three assays. In that case, the classifier cohort and assessment maps must contain exactly the supplied assays. The saved analysis records `partial_assay_review`; this is not a full three-assay paper result. The 3sTrace use is documented in [the fixed-trace exploratory run](../../records/figure-history/shared/trace3-exploratory-run-2026-09-24.md).

The fish table columns are:

```text
experiment_id, condition_id, fish_id, classifier_label,
classification_eligible, ineligible_reason, input_metric_id
```

`classifier_label` is `Learner`, `Non-learner` or `Unclassified`; controls receive the same classifier rule while retaining their reference cohort role. `Unclassified` requires an ineligibility reason. Optional `recording_id` and `cohort_role` columns are checked against the reviewed cohort. The JSON contains `table_sha256`, `input_metric_id`, `classifier_execution_id`, `validation_mode`, `cohort_hashes` and `selection_assessments` keyed by exactly the requested experiment IDs (all three for a full paper run). Each selection assessment entry has `path`, `sha256`, and `assessment_hash` for the authenticated `assessment-summary.json` from `assess-discarding`. The command verifies its metric, experiment, corrected metric recipe and all five constituent artifact hashes. The selected metric must match the classifier's metric.

```json
{
  "table_sha256": "<SHA-256 of fish-classification.parquet>",
  "input_metric_id": "tail_length_weighted_angular_l1",
  "classifier_execution_id": "<frozen Gate L run ID>",
  "validation_mode": "<Gate L validation mode>",
  "cohort_hashes": {"allDelay": "<hash>", "all3sTrace": "<hash>", "all10sTrace": "<hash>"},
  "selection_assessments": {
    "allDelay": {"path": "/path/to/assessment-summary.json", "sha256": "<file hash>", "assessment_hash": "<assessment identity>"},
    "all3sTrace": {"path": "/path/to/assessment-summary.json", "sha256": "<file hash>", "assessment_hash": "<assessment identity>"},
    "all10sTrace": {"path": "/path/to/assessment-summary.json", "sha256": "<file hash>", "assessment_hash": "<assessment identity>"}
  }
}
```

For setup and transfer instructions, use [the current Figure 4 handover](../project-maintenance/figure-04-transfer.md).

## Run order

One project directory can hold all three cohorts. When experiments are in separate project directories, supply the corresponding `--*-project-dir` overrides. `--project-dir` also selects where the saved Figure 4 analysis goes.

```bash
classical-conditioning figure4-analyze \
  --project-dir /path/to/output-project --analysis-id figure4-v1 \
  --metric tail_length_weighted_angular_l1 \
  --delay-cohort-id DELAY_ID --trace3-cohort-id TRACE3_ID \
  --trace10-cohort-id TRACE10_ID \
  --learner-manifest /path/to/fish-classification.parquet \
  --delay-project-dir /path/to/delay-project \
  --trace3-project-dir /path/to/trace3-project \
  --trace10-project-dir /path/to/trace10-project

classical-conditioning figure4-render \
  --analysis-summary '/path/to/output-project/Processed data/Analyses/figure4-v1/figure4/analysis.json' \
  --output-dir /path/to/figure4-review --mode static
```

Use either of the other supported metric IDs in place of the example, with its matching frozen classifier manifest. `render-paper-panels --figure-set figure4` runs these two stages in the same order using the same Figure 4 flags and records a paper-panel run summary. Its `--output-dir` basename becomes the Figure 4 analysis ID. `--plan` lists the commands without running them.

## Signal and provenance

The analysis computes the shared signed bout-log-vigor value at 0.5-second bins from −20 to +20 seconds, using each trial's −20 to 0 second baseline. It masks bins with less than 0.9 valid expected-frame coverage. No-bout signed bins stay missing; their movement probability is retained separately. Trial medians are formed within each fish before equal-fish group medians and fish IQRs. Saved trial, fish, group and sample-flow Parquet tables precede rendering. The group table contains per-bin contributing fish and trial counts for both outcomes.

Expected-US guides are derived from authenticated paired-training `Reinforcer` events and compared with `ExperimentSpec`. The command stops if they disagree. The fixed 3sTrace source-protocol audit supports the declared 13-s latency; every Figure 4 fish remains subject to the event check. Catch and non-US rows show the verified paired-training expectation, not an observed US. Same-data learner curves carry no independent group-inference claim or p-values. Independent response timing is specified in the [supplementary plan](../../Plans/09_SUPPLEMENTARY_FIGURES.md).

## Saved outputs and timing checks

The command writes `trial-bins.parquet`, `fish-bins.parquet`, `group-bins.parquet`, `sample-flow.parquet`, `analysis.json`, and `complete.json` under:

```text
<OUTPUT_PROJECT>/Processed data/Analyses/figure4-v1/figure4/
```

It computes signed bout-log-vigor in 0.5 s bins from −20 to +20 s with each trial's −20 to 0 s baseline and a 0.9 coverage threshold. Missing no-bout signed values stay missing; movement probability is separate. Trials are pooled within fish before the equal-fish median and IQR. Trial 65 belongs to both Test 1 and the pooled catch group. The pooled catch set is 25, 39, 53, 59, 65. Group tables include fish and trial counts per bin, including explicit zero-fish groups.

**Timing gate:** The fixed 3sTrace protocol audit in `records/scientific-evidence/audits/trace-preflight.json` finds paired-US events around 13 s, and `ExperimentSpec` now declares 13 s. The analyzer still checks recorded `Reinforcer` events for each conditioned fish and stops on disagreement, missing/extra paired US events, or US events in non-US trials. Do not shift a guide or disable this check to force a figure.

For a complete three-assay analysis, this produces **18 review PNG figures**: per experiment, one ten-row main figure (nine CS blocks plus pooled catches) and five supplementary figures covering individual signed catches, main and individual movement probability, and main and individual contributing-fish coverage. Four classifier groups, their fish counts, fish-IQR bands where at least two fish contribute, CS interval, and verified paired-training expected-US guides are drawn from saved tables. The x-axis is −20…+20 s; a 20 s US sits at the right boundary.

For SVG/PDF publication files, rerun `figure4-render` with `--mode publication` and a separate clean output directory. The publication exporter requires a clean Git working tree and embeds provenance. Do not use `--overwrite` unless intentionally replacing an output with the same analysis ID and recipe.


## Acceptance checks and delivery

1. Inspect `analysis.json`: selected metric, three cohort IDs/hashes, classifier execution/validation identity, expected-US evidence for all assays, and four table hashes. Confirm `complete.json` binds that summary.
2. Inspect `sample-flow.parquet` for every cohort fish, including unclassified reasons and controls retaining the `reference` role. Check all four plotted strata and zero-fish counts.
3. Inspect the ten main rows and five individual catches per experiment; confirm all 18 PNGs, their `.figure.json` sidecars, and the ±20 s axes. For publication mode, confirm each figure has SVG and PDF outputs and a provenance sidecar.
4. Confirm the three main figures are Delay (4A), 3sTrace (4B), and 10sTrace (4C). Treat the same-data learner curves as descriptive; add no learner-versus-nonlearner significance marks.
5. Save the exact source commit, selected metric, cohort IDs/hashes, classifier manifest and hash, selection assessment hashes, `analysis.json`, `complete.json`, all four tables, `paper-panel-run.json` if used, and figure files/sidecars together with the review record.

If authentication fails, keep the diagnostic and repair the upstream artifact through its owning pipeline. If rendering fails, preserve the saved analysis tables; the renderer reads them and performs no classification or signal recalculation.
