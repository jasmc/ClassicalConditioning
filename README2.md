The repository trace is complete enough to follow the command without running the full data-heavy pipeline. I’m now reading the detailed call-chain report and will cross-check the key cohort boundary against the implementation and tests.

Read 

Searched for regex `build_legacy_scaled_vigor_cohort|build_legacy_statistics|build_candidate_metric_comparison|run_pipeline` (`ClassicalConditioning` · `src/**/*.py`), 24 results

Searched for regex `run_inventory|run_intake|routes|legacy_run_statistics|figure_outcomes` (`ClassicalConditioning` · `configs/example-run.json`), 5 results

Searched for regex `run-pipeline|cohort|scaled vigor|metric comparison` (`ClassicalConditioning` · `tests/**/*.py`), 89 results

The key boundary is confirmed: `run-pipeline` is a two-route workflow, and cohort pooling happens only after all eligible per-recording stages for a route finish. I’m checking the exact failure/resume behavior and whether the standalone cohort-manifest machinery is involved, since that distinction matters for interpreting “cohort-level.”

Read 

Read 

Read 

Read 

Read 

Read 

One important operational detail surfaced: with `overwrite: false`, an existing `recording_inventory.json` can stop the run before intake, rather than silently refreshing it. I’m verifying that guard and the configured condition counts so the explanation distinguishes the intended pipeline from what this exact current output directory will do.

Searched for regex `def write_recording_inventory|overwrite` (`ClassicalConditioning` · `src/classical_conditioning/inventory.py`), 3 results

Read 

Read , lines 430 to 606

Ran terminal command: $inventory = Get-Content 'F:\Results_refactored\Metadata\recording_inventory.json' -Raw | ConvertFrom-Json; $inventory.records | Group-Object condition_id | Sort-Object Name | Select-Object Name,Count; $inventory.records | Where-Object { $_.condition_id -in @('control','delay') } | Measure-Object

Ran terminal command: uv run pytest tests/test_legacy_runner.py tests/test_candidate_runner.py tests/test_run_pipeline.py

The supplied inventory contains 23 complete `control`/`delay` recordings (11 control, 12 delay); the 34 `trace`/`delayNoMK801` records are outside this example config’s filter. The targeted test command could not start because `pytest` is not installed/available in the active `uv` environment, so I’m relying on the test source and static implementation checks rather than claiming a test run.

Searched for regex `def intake_recording|def preprocess_legacy_recording|def build_corrected_preprocessing|def calculate_candidate_metrics|def aggregate_event_profiles|def aggregate_trial_outcomes` (`ClassicalConditioning` · `src/classical_conditioning/**/*.py`), 7 results

Read 

Read 

Read 

Completed: *Trace pipeline command and config* (1/5)

## Important for this exact command

With `example-run.json`:

- Raw data: `J:\Raw Data\allDelay`
- Output: `Results_refactored`
- Conditions retained: `control`, `delay`
- Routes: `legacy`, then `candidate`
- Inventory enabled
- `overwrite: false`

Your existing `recording_inventory.json` means the command will currently stop at the inventory step with `FileExistsError`. The inventory writer refuses to replace an existing inventory unless overwrite is enabled.

The inventory contains 71 records, but this configuration selects only:

- 11 `control`
- 12 `delay`
- **23 recordings total**

The `trace` and `delayNoMK801` records are not selected.

## Complete pipeline order

### 1. CLI and configuration loading

`classical-conditioning` invokes:

- `__main__.py`
- `classical_conditioning.cli.main()`

The CLI:

1. Parses `run-pipeline`.
2. Reads `example-run.json`.
3. Validates paths, experiment, routes, recipes, alignment, and options.
4. Calls `pipeline.run_pipeline()`.

Resolved analysis IDs:

- Legacy: `delay-cohort-v1-legacy`
- Candidate: `delay-cohort-v1-candidate`

### 2. Recording discovery

Because `recording_ids` is not specified, the pipeline recursively searches `J:\Raw Data\allDelay` for complete triplets:

- `*_cam.txt`
- `*_mp tail tracking.txt`
- `*_stim control.txt`

For each recording it derives:

- Recording ID from the first two underscore-separated fields.
- Condition from the third field.

It then keeps only `control` and `delay`.

### 3. Inventory generation

Because `run_inventory: true`, it attempts to write:

```text
F:\Results_refactored\Metadata\recording_inventory.json
```

The inventory scan itself records all discovered source records; it does not apply the `keep_conditions` filter. The filter is applied when selecting recordings for processing.

With `overwrite: false`, an existing inventory causes the run to stop here.

### 4. Batch intake

Because `run_intake: true`, the selected recordings are converted from raw text to Parquet.

For each recording:

1. Read camera data.
2. Read tracking data.
3. Read stimulus protocol data.
4. Validate schemas and timestamps.
5. Calculate acquisition statistics.
6. Check frame and timestamp integrity.
7. Write lossless Parquet files.
8. Generate acquisition QC plots.
9. Write source metadata and manifests.

Typical outputs:

```text
Processed data/<recording-id>/camera.parquet
Processed data/<recording-id>/tracking.parquet
Processed data/<recording-id>/stimulus_events.parquet
Quality checks/<recording-id>/acquisition_summary.json
Quality checks/<recording-id>/acquisition_report.html
Quality checks/<recording-id>/figures/*.png
Metadata/<recording-id>_source_manifest.json
```

With `continue_on_error: true`, failed recordings are recorded and excluded from later stages.

---

# Legacy route

The legacy route runs first.

## 5. Legacy preprocessing per recording

Function:

```text
preprocess_legacy_recording()
```

For each active recording, it performs the historical preprocessing sequence:

1. Discard the first 13,999 camera rows.
2. Select the legacy tracking angle columns.
3. Drop the final tracking row.
4. Convert angles from radians to degrees.
5. Validate and rename protocol columns.
6. Estimate frame rate.
7. Determine the reference frame.
8. Apply the historical frame-loss rejection rule.
9. Remove rows before the reference frame.
10. Check tracking-angle validity.
11. Synchronize camera and tracking data by `FrameID`.
12. Interpolate onto the nominal 700 Hz grid.
13. Check protocol acquisition coverage.
14. Annotate CS and US stimulus periods.
15. Reconstruct `AbsoluteTime`.
16. Convert local angles to cumulative angles.
17. Apply the legacy rolling temporal filter.
18. Calculate distal cumulative-angle derivative vigor.
19. Calculate bout metrics.
20. Detect movement bouts.
21. Merge and filter short bouts.
22. Segment CS and US trials.
23. Assign trials to experimental blocks.
24. Perform per-trial P10/P90 vigor scaling.
25. Finalize columns and metadata.

Outputs:

```text
Processed data/<recording-id>/samples_legacy-v1.parquet
Quality checks/<recording-id>/legacy-v1_preprocessing_summary.json
Metadata/<recording-id>_legacy-v1_complete.json
```

This is still **per-recording**, not cohort-level.

## 6. Legacy standard-main analysis per recording

Function:

```text
build_legacy_standard_main()
```

For every successfully preprocessed recording:

1. Read `samples_legacy-v1.parquet`.
2. Retain configured 10-trial blocks.
3. Produce CS-aligned output.
4. Produce US-aligned output.
5. Verify artifact lineage.

Outputs:

```text
Processed data/<recording-id>/samples_legacy-standard-main-v1_CS.parquet
Processed data/<recording-id>/samples_legacy-standard-main-v1_US.parquet
Quality checks/<recording-id>/legacy-standard-main-v1_summary.json
Metadata/<recording-id>_legacy-standard-main-v1_complete.json
```

Still per-recording.

## 7. Legacy normalized-vigor analysis per recording

Function:

```text
build_legacy_normalized_vigor()
```

For every recording:

1. Read standard-main CS and US outputs.
2. Aggregate normalized response and baseline vigor outcomes.
3. Write CS and US normalized outcomes.
4. Verify lineage.

Outputs:

```text
Processed data/<recording-id>/legacy-normalized-vigor-v1_CS.parquet
Processed data/<recording-id>/legacy-normalized-vigor-v1_US.parquet
Quality checks/<recording-id>/legacy-normalized-vigor-v1_summary.json
Metadata/<recording-id>_legacy-normalized-vigor-v1_complete.json
```

Still per-recording.

## 8. First cohort-level analysis: legacy scaled vigor

This is the **first cohort-level analysis in the entire pipeline**.

Function:

```text
build_legacy_scaled_vigor_cohort()
```

It runs only after all legacy per-recording standard-main and normalized-vigor stages have completed.

It:

1. Verifies the standard-main artifacts for all recordings.
2. Confirms that recordings belong to the same experiment.
3. Pools recordings by condition.
4. Checks that fish IDs are not duplicated across recordings.
5. Groups by condition.
6. Processes both CS and US alignments.
7. Processes configured time-bin widths.
8. Aggregates scaled-vigor line data.
9. Creates count heatmaps.
10. Creates scaled-vigor heatmaps.

Outputs are under:

```text
Processed data/Analyses/delay-cohort-v1-legacy/
Quality checks/Analyses/delay-cohort-v1-legacy/
Metadata/delay-cohort-v1-legacy_legacy-scaled-vigor-v1_complete.json
```

This is cohort-level descriptive pooling. It is not yet inferential statistics.

## 9. Second legacy cohort-level analysis: legacy statistics

Because `legacy_run_statistics: true`, the pipeline then calls:

```text
build_legacy_statistics()
```

It:

1. Verifies normalized-vigor CS artifacts for all recordings.
2. Concatenates normalized outcomes across the cohort.
3. Builds model input.
4. Calculates block medians.
5. Runs nonparametric tests.
6. Runs the global mixed model.
7. Runs block-level models.
8. Runs trial-level models.

Outputs:

```text
legacy-statistics-v1_CS_model_input.parquet
legacy-statistics-v1_CS_block_medians.parquet
legacy-statistics-v1_CS_nonparametric.parquet
legacy-statistics-v1_CS_global_model.parquet
legacy-statistics-v1_CS_block_models.parquet
legacy-statistics-v1_CS_trial_models.parquet
```

Additional summary and marker files are written under the analysis-specific `Quality checks` and `Metadata` directories.

The legacy runner then writes:

```text
Metadata/delay-cohort-v1-legacy_legacy-runner-v1_manifest.json
```

---

# Candidate corrected route

The candidate route runs after the legacy route.

## 10. Resolve the candidate recipe

The config selects:

```text
candidate-corrected-runner-v1
```

This resolves to the corrected five-metric route:

| Stage | Recipe |
|---|---|
| Preprocessing | `corrected-preprocess-v1` |
| Activity metrics | `tail-candidate-corrected-v1` |
| Movement state | `movement-candidate-corrected-v1` |
| Temporal profiles | `candidate-temporal-outcomes-corrected-v2` |
| Trial outcomes | `candidate-trial-outcomes-corrected-v1` |
| Cohort comparison | `candidate-metric-comparison-corrected-v1` |

## 11. Corrected frame preprocessing per recording

Function:

```text
build_corrected_preprocessing()
```

For each recording it:

1. Preserves measured camera time.
2. Does not discard initial camera rows.
3. Does not discard the final tracking row.
4. Matches tracking frames to camera frames.
5. Applies body translation using the tail base.
6. Computes frame-validity masks.
7. Computes timestamp-validity masks.
8. Computes point-validity masks.
9. Computes derivative-validity masks.
10. Tracks gaps explicitly.
11. Processes tracking data in batches.

Output:

```text
Processed data/<recording-id>/frame_preprocessed_corrected-v1.parquet
Quality checks/<recording-id>/corrected-v1_preprocessing_summary.json
Metadata/<recording-id>_corrected-preprocess-v1_complete.json
```

Per-recording only.

## 12. Corrected activity metrics per recording

Function:

```text
build_candidate_activity_metrics_from_corrected()
```

It calculates five candidate metrics:

1. Segment absolute angular-speed sum.
2. All-segment angular RMS.
3. Whole-tail XY RMS speed.
4. Whole-tail XY mean speed.
5. Curvature-change RMS.

It also:

- Applies derivative and gap validity masks.
- Validates XY and angle geometry agreement.
- Writes metric-level QC summaries.
- Authenticates corrected-preprocessing lineage.

Outputs:

```text
Processed data/<recording-id>/frame_activity_candidates-corrected-v1.parquet
Quality checks/<recording-id>/candidate-corrected-v1_activity_summary.json
Metadata/<recording-id>_candidate-corrected-v1_complete.json
```

Per-recording only.

## 13. Movement-state detection per recording

Function:

```text
build_candidate_movement_state()
```

For each of the five metrics:

1. Smooth contiguous data with a median smoother.
2. Calibrate quiet-window thresholds.
3. Detect movement using hysteresis.
4. Generate moving and valid masks.
5. Assign movement-bout IDs.
6. Produce positive-control summaries.

Output:

```text
Processed data/<recording-id>/movement_state_candidates-corrected-v1.parquet
Quality checks/<recording-id>/movement-candidate-corrected-v1_summary.json
Metadata/<recording-id>_movement-candidate-corrected-v1_complete.json
```

Per-recording only.

## 14. Temporal event profiles per recording

Function:

```text
build_candidate_temporal_profiles()
```

For each CS and US event:

1. Create a −45 to +45 second event window.
2. Bin the data at 0.5-second resolution.
3. Calculate total activity.
4. Calculate movement probability.
5. Calculate fraction of time moving.
6. Calculate conditional intensity.
7. Calculate bout count and bout rate.
8. Calculate bout duration.
9. Record validity and coverage fields.

Output:

```text
Processed data/<recording-id>/candidate_temporal_outcomes-corrected-v2.parquet
Quality checks/<recording-id>/candidate-corrected-v2_temporal_outcomes_summary.json
Metadata/<recording-id>_candidate-temporal-outcomes-corrected-v2_complete.json
```

Despite containing event and trial information, this remains per-recording.

## 15. Trial outcomes per recording

Function:

```text
build_candidate_trial_outcomes()
```

It:

1. Authenticates intake, protocol, metric, and movement lineage.
2. Parses recording identity.
3. Aggregates trial-level outcomes.
4. Produces coverage information.
5. Writes outcomes for all five candidate metrics.

Outputs:

```text
Processed data/<recording-id>/candidate-trial-outcomes-corrected-v1.parquet
Processed data/<recording-id>/candidate-trial-outcomes-corrected-v1_coverage.parquet
Quality checks/<recording-id>/candidate-trial-outcomes-corrected-v1_summary.json
Metadata/<recording-id>_candidate-trial-outcomes-corrected-v1_complete.json
```

This is still per-recording. Failed candidate recordings are excluded from the candidate cohort comparison.

## 16. Candidate cohort-level analysis

This is the first cohort-level step in the **candidate route**.

Function:

```text
build_candidate_metric_comparison()
```

It runs only after all candidate per-recording stages finish.

It:

1. Verifies temporal profiles for each successful recording.
2. Concatenates all temporal profiles.
3. Summarizes metric windows at the recording level.
4. Adds condition IDs.
5. Groups summaries by cohort:
   - all recordings
   - control
   - delay
6. Produces descriptive summaries for all five metrics.

Outputs:

```text
Processed data/Analyses/delay-cohort-v1-candidate/candidate-metric-comparison-corrected-v1_recording_summary.parquet
Processed data/Analyses/delay-cohort-v1-candidate/candidate-metric-comparison-corrected-v1_cohort_summary.parquet
Quality checks/Analyses/delay-cohort-v1-candidate/candidate-metric-comparison-corrected-v1_summary.json
Metadata/delay-cohort-v1-candidate_candidate-metric-comparison-corrected-v1_complete.json
```

Important scientific limitation:

- No candidate metric is selected.
- No metric is paper-approved.
- No inferential candidate comparison is performed.
- No approved primary cohort is applied.

## 17. Candidate runner manifest

The candidate route writes:

```text
Metadata/delay-cohort-v1-candidate_candidate-corrected-runner-v1_manifest.json
```

It records:

- Per-recording stage status.
- Cohort comparison status.
- Artifact lineage hashes.
- `paper_approved: false`.
- Blocked future steps, including reviewed annotations, metric selection, validation partition, approved cohort, and confirmatory statistics.

## 18. Optional cohort figure

Because `run_figures: true`, the pipeline generates a candidate cohort figure for:

```text
movement-probability
```

Function:

```text
build_metric_comparison_figure()
```

It:

1. Reads the candidate recording summary.
2. Selects CS-aligned movement-probability data.
3. Creates fish-equal standardized-difference points and bars.
4. Exports the static PNG.

Output location:

```text
F:\Results_refactored\Figures\PNG\Analyses\delay-cohort-v1-candidate\
```

This is a **cohort-level visualization**, but not a new inferential analysis.

## 19. Final pipeline summary

Finally:

```text
F:\Results_refactored\Metadata\delay-cohort-v1_pipeline_run.json
```

is written.

It contains:

- Resolved configuration.
- Final recording IDs.
- Intake completed/skipped/failed lists.
- Legacy preprocessing statuses.
- Legacy runner status.
- Candidate runner manifest.
- Generated figure paths.
- Generation timestamp.

## Cohort-level answer in one view

The actual cohort-level sequence is:

1. **Legacy scaled-vigor cohort pooling**
   - First cohort-level operation overall.
   - `build_legacy_scaled_vigor_cohort()`

2. **Legacy cohort statistics**
   - `build_legacy_statistics()`
   - Enabled by `legacy_run_statistics: true`

3. **Candidate five-metric cohort comparison**
   - First cohort-level operation for the candidate route.
   - `build_candidate_metric_comparison()`

4. **Candidate cohort figure**
   - `build_metric_comparison_figure()`
   - Visualization only.

The standalone `cohort.py` reviewed-manifest workflow is **not called** by `run-pipeline`. In particular, `freeze_cohort_manifest()` and `apply_cohort()` are separate operations. Therefore, this command does not yet apply a reviewed, approved primary cohort.

The implementation and orchestration tests confirm that all per-recording stages run before their corresponding cohort operation. Targeted tests could not be executed in the current environment because `pytest` is not installed, but the test source and implementation were inspected directly.