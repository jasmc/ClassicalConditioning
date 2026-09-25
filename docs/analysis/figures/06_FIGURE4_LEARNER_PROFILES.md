# Figure 4 learner-stratified CR profiles

Figure 4A–C are descriptive Delay, 3sTrace and 10sTrace profiles. Each figure has the nine CS ten-trial blocks from `ExperimentSpec` and one pooled-catch row. The five individual catch trials, movement probability and signed-signal fish coverage are supplementary figures for both main and individual-catch groups. The older condition-wide 0–1 scaled activity plots remain distinct outputs.

## Frozen classifier input

`figure4-analyze` consumes a CSV or Parquet fish table plus a JSON file with the same stem and suffix `.manifest.json`. It does not fit or select a classifier. The table must contain one row per `(experiment_id, condition_id, fish_id)` across all three reviewed cohorts, with:

```text
experiment_id, condition_id, fish_id, classifier_label,
classification_eligible, ineligible_reason, input_metric_id
```

`classifier_label` is `Learner`, `Non-learner` or `Unclassified`; controls receive the same classifier rule while retaining their reference cohort role. `Unclassified` requires an ineligibility reason. Optional `recording_id` and `cohort_role` columns are checked against the reviewed cohort. The JSON contains `table_sha256`, `input_metric_id`, `classifier_execution_id`, `validation_mode`, `cohort_hashes` keyed by all three experiment IDs, and `selection_assessments` keyed the same way. Each selection assessment entry has `path`, `sha256`, and `assessment_hash` for the authenticated `assessment-summary.json` from `assess-discarding`. The command verifies its metric, experiment, corrected metric recipe and all five constituent artifact hashes. The selected metric must match the classifier's metric.

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

Expected-US guides are derived from authenticated paired-training `Reinforcer` events and compared with `ExperimentSpec`. The command stops if they disagree, including a 3sTrace 9-versus-13-second discrepancy. Catch and non-US rows show the verified paired-training expectation, not an observed US. Same-data learner curves carry no independent group-inference claim or p-values. Independent response timing is specified in the [supplementary plan](../../../Plans/10_SUPPLEMENTARY_FIGURES.md).
