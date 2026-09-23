# Current pipeline guide

The supported routine entry point is `classical-conditioning run-pipeline
--config <strict-json-file>`. See the [README](../../README.md) for every JSON
field and a complete user workflow. This guide maps the active source modules
and their trust boundaries for manual review.

## Call graph and artifacts

```text
cli.py -> run_config.py -> pipeline.py
  -> inventory.py                  all raw groups, source SHA-256
  -> intake.py                     verified/rebuilt triplet, QC, durable status
  -> analysis/candidate_runner.py  corrected three-metric recipe family
       -> corrected_frame_preprocessing.py
       -> candidate_metrics_from_corrected_frames.py
       -> analysis/movement_state.py
       -> analysis/temporal_profiles.py
       -> analysis/trial_outcomes.py
       -> analysis/discarding.py    technical + exploratory assessment of all records
       -> analysis/metric_comparison.py
  -> bounded two-process renderer
       -> figures/temporal_profiles.py (4 families × CS/US × ready fish)
       -> analysis/review/trace_review.py (ready fish)
       -> figures/metric_comparison.py (5 outcomes × CS/US)
       -> figures/cohort_response.py (5 families when cohort+metric selected)
       -> figures/population_heatmap.py (matched frozen cohort + fish coverage)
       -> figures/learning_diagnostics.py and learning_onset.py (when ready)
  -> Metadata/<analysis_id>_pipeline_run.json
```

The direct-from-intake candidate calculation in
`preprocessing/benchmarks/candidate_metrics_from_intake.py` is a separately
selected development benchmark, not an alternative routine JSON route.

## Trust boundaries

1. `inventory.py` recursively groups camera, tracking, and protocol sources.
   It records complete, incomplete, ambiguous, and invalid-name groups with
   content hashes. This scan is unconditional.
2. `intake.py` compares each selected complete triplet with the source
   manifest and verifies all three lossless Parquet hashes. Unchanged verified
   intake is reused; new or changed triplets are published transactionally.
   `Metadata/intake_status.json` records ready, incomplete, and failed reasons
   after each recording. An unchanged failure requires `retry-intake` or a
   changed source. `analysis/discarding.py` then assesses all inventory
   records, including processing failures, before cohort comparison. Its
   technical and exploratory outputs do not make a reviewed inclusion decision.
3. `candidate_runner.py` authenticates each upstream stage and reruns a stale
   stage at its stable path. Corrected preprocessing uses measured timestamps,
   frame gaps and valid-point masks; the metric kernel calculates three
   candidates in native units. A single detector supplies movement/bout state
   to every metric. Trial outcomes and comparison retain coverage and fish
   identity. Runner invocation manifests are replaceable; frozen cohort
   manifests in `cohort.py` are immutable.
4. `pipeline.py` submits each ready render to at most two worker processes.
   Per-fish detector review and profiles are submitted at their input-ready
   callbacks; comparison and cohort families are submitted after their
   respective data are available. Failure of one render is recorded and does
   not make another figure optional. LME diagnostics run after a model fit;
   required gates control the onset plot. Learner plots and paper panels are
   blocked with explicit reasons until their scientific contracts exist.
5. `figures/export.py` publishes PNG or SVG/PDF and a provenance sidecar
   atomically. Publication SVG contains semantic artist IDs and embedded
   sidecar authentication; structural checks reject missing/duplicate IDs.

`Metadata/<analysis_id>_pipeline_run.json` is always written, including after
inventory, intake, analysis, or figure failure. Each required figure has a
`completed`, `failed`, or `blocked` disposition with a reason where relevant.
Its `figure_authentication` binds selected cohort, metric definition, settings,
and raw inventory. The [paper registry](../../configs/paper-figures/behavior-paper.json)
declares intended panel roles without pretending descriptive plots are final
paper panels.

## Scientific boundaries

The reviewed cohort is a separate immutable decision. Candidate comparisons
are descriptive, and model outputs are diagnostic until their parameters and
inference are approved. Figure 2 population heatmaps require frozen-cohort
fish coverage; per-fish profiles are not substitutes. Figure 4 signed,
baseline-centered learner timing must be defined independently of classifier
features. See the [draft comparison](figures/PAPER_DRAFT_COMPARISON.md) for
panel-level gaps and the [cohort plan](../../Plans/SINGLE_COHORT_AND_EXCLUSION.md)
for remaining assessment gates.
