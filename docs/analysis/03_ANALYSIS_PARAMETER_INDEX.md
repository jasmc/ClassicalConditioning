# Analysis parameter index

This is the fastest place to see what controls an analysis. It separates **run inputs**, **fixed recipe values**, and **paper-review choices**. The resolved run JSON and artifact summaries record the values actually used; code is authoritative if a default changes. The [LME guide](07_LEARNING_ONSET_LME.md#complete-public-parameter-reference) contains the full option-by-option learning-onset table, including constraints and effects.

## 1. Routine run inputs

Set these in [`configs/example-run.json`](../../configs/example-run.json) and run `classical-conditioning run-pipeline --config <file>`. Unknown and obsolete keys are rejected by [`run_config.py`](../../src/classical_conditioning/run_config.py). `null` means no optional selection.

| JSON field | Default | Controls |
| --- | --- | --- |
| `raw_dir` | required | Raw triplet inventory root; raw bytes are read and hashed. |
| `save_dir` | required | Derived-data project root. |
| `experiment` | required | `allDelay`, `all3sTrace`, or `all10sTrace`; loads the fixed protocol below. |
| `analysis_id` | required | Stable output identity for this run. |
| `keep_conditions` | all | Condition IDs retained from inventory. |
| `recording_ids` | all matching | Explicit fish subset. |
| `overwrite` | `false` | Rebuild otherwise reusable derived stages. |
| `continue_on_error` | `true` | Continue other fish after an individual failure; required-stage failure still fails the invocation. |
| `batch_size` | `250000` | Intake and candidate-metric processing rows per chunk. |
| `figure_mode` | `static` | PNG, or `publication` SVG/PDF. |
| `show_progress` | `true` | Terminal progress; `--quiet` suppresses it. |
| `cohort_id` and `metric` | both `null` | Supply together to enable frozen-cohort descriptive figures and a technical learning fit. |
| `learner_representation_id` | `null` | Reserved frozen learner identity; does not by itself approve Figure 3/4. |
| `assessment_metric` | selected `metric`, else `legacy_distal_angular_speed` | Technical/exploratory discarding evidence metric. |
| `technical_policy` | `null` | Reviewed technical policy JSON path; absent policy means draft assessment only. |
| `disabled_discard_checks` | `[]` | Named historical checks omitted from exploratory sensitivity assessment only. |

Use `classical-conditioning resolve-config --experiment <id> --project-dir <save-dir>` to inspect resolved recipe and trial map. Run fields and accepted types are also explained in the [root README](../../README.md#configure-one-run).

## 2. Experiment and trial definition

[`experiments.py`](../../src/classical_conditioning/config/experiments.py) fixes these values; they are **not JSON overrides**. All assays require at least 94 CS cycles and 78 US cycles, use a 10 s CS, CS blocks 5–14 Pre-train, 15–64 Train 1–5, and 65–94 Test 1–3. CS catch global trials are 25, 39, 53, 59, and 65. The selected paper block comparison uses the final Pre-train 10–14, Early Test 65–69, and Late Test 90–94; these five-trial subsets are distinct from the ten-trial model blocks.

| Assay | Condition IDs | Conditioned response window, s | Configured paired US latency, s |
| --- | --- | ---: | ---: |
| `allDelay` | `control`, `delay` | [0, 9) | 9 |
| `all3sTrace` | `control`, `trace` | [0, 13) | 9 |
| `all10sTrace` | `control`, `trace` | [0, 20) | 20 |

The 3sTrace configured 9 s US latency conflicts with the paper scaffold's 13 s expectation. Check raw protocol events before using expected-US guides in paper panels. US-aligned block numbering differs from CS numbering; inspect the generated `Metadata/trial_map_<experiment>.json`.

## 3. Fixed per-recording recipe settings

These are current code defaults, **not independent `run-pipeline` parameters**. Changing them requires a new recipe/artifact identity and scientific review. The linked configuration class contains the full field list.

| Stage and config | Values that determine the analysis |
| --- | --- |
| [Corrected frames](../../src/classical_conditioning/preprocessing/corrected_frame_preprocessing.py) `CorrectedPreprocessConfig` | 16 points; measured camera elapsed time; derivatives invalid across frame gaps or intervals >10 ms; no interpolation, temporal filter, or spatial filter; subtract tail base translation; no rotation correction; at least 0.8 valid point fraction; timestamp jitter tolerance 0.5 ms; duplicate frames and nonmonotonic timestamps rejected; no initial/trailing row dropped. |
| [Three candidate metrics](../../src/classical_conditioning/preprocessing/candidate_metric_kernel.py) `CandidateMetricConfig` | 16 points; at least 0.8 valid tail fraction; terminal angle placeholder; measured time; derivative invalid unless frame step is one; tail-base translation correction; no rotation correction; geometry error bound 0.001 rad over at least 0.99 validation fraction; XY speed normalized by recording median tail length. Metric IDs and units are in [05](05_METRICS_AND_BOUTS.md#current-recipe-reference). |
| [Shared movement/bout detector](../../src/classical_conditioning/analysis/movement_state.py) `MovementCalibrationConfig` | Distal cumulative-angle speed; 10 ms median smoothing; centered envelope windows 28.571 and 571.429 ms; envelope threshold 4 deg/ms; amplitude 1 deg/ms; merge gaps <14.286 ms; minimum bout 57.143 ms; minimum tail coverage 0.80. Positive-control window 500 ms; control baseline −5000 to −1000 ms. [05](05_METRICS_AND_BOUTS.md#current-shared-detector-reference) explains the rule. |
| [CS/US temporal profiles](../../src/classical_conditioning/analysis/temporal_profiles.py) `TemporalProfileConfig` | −45 to +45 s, 0.5 s left-closed bins, mean total activity; two-layer scale using bins before −15 s and all pre-onset bins, P10–P90, clipped to [0, 1]. Signed paper heatmaps use a **separate** −20 to 0 s bout-log baseline recipe; see [paper provenance](figures/01_PAPER_PANEL_PROVENANCE.md). |
| [Per-trial outcomes](../../src/classical_conditioning/analysis/trial_outcomes.py) `TrialOutcomeConfig` | Baseline [−15, 0) s; response window comes from `ExperimentSpec`: [0, 9), [0, 13), or [0, 20) s. These rows feed cohort trial ratios and learning-onset LME. |
| [Metric comparison](../../src/classical_conditioning/analysis/metric_comparison.py) `MetricComparisonConfig` | Baseline [−15, 0) s; response end follows the assay; left-closed windows; equal trial weight within recording and equal recording weight across the cohort. |
| [Cohort aggregation](06_COHORT_AGGREGATION.md) | Reviewed fish membership from frozen manifest; form per-fish values first and summarize fish equally. Preserve sample and contributing-fish coverage. No routine figure star is an approved test. |

The discarding stage uses `assessment_metric`, `technical_policy`, and `disabled_discard_checks` above; its rule order, thresholds, and distinction from a frozen cohort are in [04](04_DISCARDING_AND_SELECTION.md). The corrected route is `candidate-corrected-runner`; direct-from-intake `candidate-development-runner` is a separately selected benchmark.

### Exact fixed recipe fields

The compact list below includes every dataclass field in the current per-recording recipe configurations. These are serialized with recipe artifacts; their names make it easy to locate a setting in an output summary or source review.

| Config | Field = default |
| --- | --- |
| `CorrectedPreprocessConfig` | `point_count=16`; `time_source=camera_elapsed_time_ms`; `frame_gap_policy=invalidate_cross_gap_derivative`; `maximum_valid_derivative_interval_ms=10.0`; `long_gap_policy=invalidate`; `interpolation_enabled=false`; `interpolation_method=none`; `timestamp_jitter_tolerance_ms=0.5`; `duplicate_frame_policy=reject`; `nonmonotonic_timestamp_policy=reject`; `temporal_filter_enabled=false`; `spatial_filter_enabled=false`; `filter_edge_policy=not_applicable`; `body_translation_correction=subtract_tail_base`; `body_rotation_correction=none`; `minimum_valid_point_fraction=0.8`; `drop_initial_camera_rows=0`; `drop_trailing_tracking_summary_row=false`; `coordinate_source=measured`. |
| `CandidateMetricConfig` | `point_count=16`; `minimum_valid_tail_fraction=0.8`; `terminal_angle_is_placeholder=true`; `body_translation_correction=subtract_tail_base`; `body_rotation_correction=none`; `time_source=camera_elapsed_time_ms`; `gap_policy=invalidate_derivative_when_frame_step_is_not_one`; `maximum_mean_local_bend_geometry_error_rad=0.001`; `minimum_geometry_validation_fraction=0.99`; `spatial_normalization=recording_median_tail_length`. |
| `MovementCalibrationConfig` | `smoothing_window_ms=10.0`; `envelope_max_window_ms=20/700×1000` (28.571…); `envelope_min_window_ms=400/700×1000` (571.429…); `envelope_threshold_deg_per_ms=4.0`; `bout_amplitude_threshold_deg_per_ms=1.0`; `minimum_valid_tail_fraction=0.80`; `minimum_bout_duration_ms=40/700×1000` (57.143…); `maximum_interbout_gap_ms=10/700×1000` (14.286…); `positive_control_window_ms=500.0`; `control_baseline_start_ms=-5000.0`; `control_baseline_end_ms=-1000.0`. |
| `TemporalProfileConfig` | `window_start_s=-45.0`; `window_end_s=45.0`; `bin_width_s=0.5`; `interval_closure=left`; `aggregation=mean_total_activity`; `scaling_baseline_end_s=-15.0`; `scaling_onset_s=0.0`; `scaling_lower_quantile=0.1`; `scaling_upper_quantile=0.9`; `scaling_clip=(0.0,1.0)`. |
| `TrialOutcomeConfig` | `baseline_window_s=(-15.0,0.0)`; `response_window_s=(0.0,9.0)` class default for Delay, replaced by `trial_outcome_config_for_experiment()` with the experiment's conditioned-response window; `interval_closure=left`. |
| `MetricComparisonConfig` | `baseline_window_s=(-15.0,0.0)`; `response_window_s=(0.0,9.0)` class default, replaced by the experiment response window during the normal run; `interval_closure=left`; `recording_aggregation=equal_trial_weight`; `cohort_aggregation=equal_recording_weight`. |

**Response-window provenance:** `ExperimentSpec.conditioned_response_window` defines 9/13/20 s for Delay/3sTrace/10sTrace. Both [`comparison_config_for_experiment`](../../src/classical_conditioning/analysis/metric_comparison.py) and [`trial_outcome_config_for_experiment`](../../src/classical_conditioning/analysis/trial_outcomes.py) now apply those values. Older Trace trial-outcome artifacts may contain the former 9 s default; current per-fish and cohort loaders reject their mismatched saved configs. Rerun the corrected candidate stages and rebuild cohort outcomes, learning analyses, and ratio figures from the updated artifacts. Check `config.response_window_s` in each recording's trial-outcome summary.

The per-trial recipe currently applies that one experiment response window to both CS- and US-aligned outcome rows; no separate US response-window setting is declared.

Optional older exploratory CLI routes (`candidate-model-input`, `candidate-mixed-effects`, `candidate-fish-permutation`, `candidate-fish-bootstrap`) remain distinct from the condition-aware learning-onset analysis. Their defaults live in [`ModelInputConfig`](../../src/classical_conditioning/analysis/inference/model_input.py), [`MixedEffectsConfig`](../../src/classical_conditioning/analysis/inference/mixed_effects.py), [`FishPermutationConfig`](../../src/classical_conditioning/analysis/inference/fish_permutation.py), and [`FishBootstrapConfig`](../../src/classical_conditioning/analysis/inference/fish_bootstrap.py). They do not supply paper-approved inference. The permutation/bootstrap scaffold uses early blocks `Pre-train`, `Train 1`, late blocks `Train 5`, `Test 1–3`, seed 10, and 4,999 permutations or 1,999 bootstrap draws respectively; these are not learning-onset defaults.

## 4. Learning-onset LME: every public control at a glance

Run `classical-conditioning learning-onset` after freezing a cohort and building cohort trial outcomes. `--metric` selects **one** metric from the cohort table before eligibility or any model fit. `--metric-recipe` belongs to the upstream build command and selects an artifact family, not an LME metric. The [full LME reference](07_LEARNING_ONSET_LME.md) supplies exact constraints, model formulas, diagnostic gates, and output filenames.

| Group | Options and defaults |
| --- | --- |
| Identity | `--project-dir`, `--cohort-id`, `--analysis-id`, `--metric`, `--test-condition` required; `--control-condition control`. |
| Outcome and schedule | `--outcome total-activity` (`conditional-intensity` alternative); `--alignment CS` (`US` alternative); `--pretraining-block Pre-train`; repeated `--late-block` defaults to `Test 2`, `Test 3`. |
| Eligibility and scale | `--min-baseline-samples 1`; `--min-response-samples 1`; `--activity-offset 1e-6` for logged total activity. |
| Onset criterion | `--delta-min` required; `--persistence-trials 3`; `--confidence-level 0.95`. |
| Primary models | `--spline-df 5`; `--random-effects-formula '1 + trial_scaled'`; `--optimizer lbfgs`; random-intercept fallback enabled unless `--disable-random-intercept-fallback`. |
| Sensitivities | Categorical trial fit enabled unless `--skip-categorical-sensitivity`; `--sensitivity-optimizer powell` (`none` disables); random-intercept refit enabled unless `--skip-random-intercept-sensitivity`. |
| Fish uncertainty | `--bootstrap 499`; `--min-successful-bootstrap 100`; `--min-bootstrap-success-fraction 0.8`; `--permutations 9999`; `--seed 20260917`. |
| Replacement | `--overwrite` off; use a distinct analysis ID for a different specification. |

The route fits block and trial models, fish bootstrap bands, fish-level robustness, and leave-one-fish-out diagnostics. An onset requires a simultaneous lower band strictly above `delta_min` for the configured consecutive trials and passing required diagnostics. A successful fit alone does not approve paper inference.

## 5. Figure controls and status

Routine figure format comes from `figure_mode`. The paper review command `render-paper-panels` additionally accepts a metric, selected Delay/control fish and trials, figure set, mode, and optional inference review; see [02 paper specification](figures/02_PAPER_FIGURE_SPECIFICATION.md#rendering-interface). Figure 4 requires three cohort IDs and a frozen learner manifest; see [06 Figure 4](figures/06_FIGURE4_LEARNER_PROFILES.md). Every main and supplementary panel's source and calculation are in [01 paper provenance](figures/01_PAPER_PANEL_PROVENANCE.md). The registry's `blocked` status is a scientific readiness state, not a missing-file recovery instruction.
