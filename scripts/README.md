# Supported commands

Commands are grouped by figure/panel scope and role. Shared reusable implementation lives in `src/classical_conditioning/`; historical recipes retain their scientific definitions and are labelled reproduction. Folder membership does not approve an analysis.

Seven root compatibility paths remain for standing instructions, local checks and the immutable element specification. Other old source identities resolve through archive/relocation records; use the scoped paths below for execution.

- [Current implementation record](../records/repository-archive/script-organization-20261024/README.md).
- [Script lifecycle policy](../docs/project-maintenance/script-lifecycle.md).
- [Plan 13](../Plans/13_SCRIPT_COMMAND_ORGANIZATION.md).

## Normal routes

Use `python scripts/figure_working_copy.py display-target` for a specific selected figure and `python scripts/main_paper_figures.py build` / `display-target` for the registered collection. New freezes use `python scripts/freeze_figure.py --check-only` before explicit publication. Permanent artifacts stay external.

Browser presentation tools verify SVGs by default and write evidence. `--apply` explicitly edits candidate artwork; `--center-labels` explicitly edits centering. Neither may be combined with `--verify`. Browser collectors are not scientific selection authorities.

The metric workflow uses an external shared workspace and a hash-bound producer receipt. Supply `--processed-dir` / `--output-dir` for a new comparison; its consumer verifies that receipt or an explicit `--input-sha256`. Existing outputs are preserved.


## Compatibility

| Command | Scope |
|---|---|
| [check_repository_structure.py](check_repository_structure.py) | Stable compatibility facade |
| [figure_working_copy.py](figure_working_copy.py) | Stable compatibility facade |
| [freeze_figure.py](freeze_figure.py) | Stable compatibility facade |
| [main_paper_figures.py](main_paper_figures.py) | Stable compatibility facade |
| [populate_figure2_available.py](populate_figure2_available.py) | Stable compatibility facade |
| [repository_archive.py](repository_archive.py) | Stable compatibility facade |
| [review_library.py](review_library.py) | Stable compatibility facade |

## Maintenance

| Command | Scope |
|---|---|
| [maintenance/archive.py](maintenance/archive.py) | Audit repository content and transfer payloads only after JOAQUIM verification. |
| [maintenance/audit_preprocessing.py](maintenance/audit_preprocessing.py) | Read-only preprocessing evidence; writes only a new audit directory. |
| [maintenance/check_structure.py](maintenance/check_structure.py) | Read-only ownership, active-link, relocation and evidence checks. |

## Paper

| Command | Scope |
|---|---|
| [paper/figure_01/panel_a/build.py](paper/figure_01/panel_a/build.py) | Normalize the supplied vector preparation scheme to the plot font. |
| [paper/figure_01/panel_b/build_protocol.py](paper/figure_01/panel_b/build_protocol.py) | Draw an exploratory, fully vector Figure 1B from active experiment timing. |
| [paper/figure_01/panel_c/build_session.py](paper/figure_01/panel_c/build_session.py) | Build an editable SVG schematic of the Figure 1 session protocol. |
| [paper/figure_01/panel_e/edit_display.py](paper/figure_01/panel_e/edit_display.py) | Guarded E-only display edits using the current trace and selected frozen F rows. |
| [paper/figure_01/panels_d_e/reproduction/render_saved_traces.py](paper/figure_01/panels_d_e/reproduction/render_saved_traces.py) | Render Figure 1 example traces from the versioned allDelay-full-v1 SSD export. |
| [paper/figure_01/panels_d_e/reproduction/render_scaled_log_traces.py](paper/figure_01/panels_d_e/reproduction/render_scaled_log_traces.py) | Compare Figure 1 C/D traces with trial-scaled log vigor baselines. |
| [paper/figure_01/panels_e_f_g_h/reproduction/audit_camera_gaps.py](paper/figure_01/panels_e_f_g_h/reproduction/audit_camera_gaps.py) | Audit >10 ms intervals back to immutable camera/tracking text logs. |
| [paper/figure_01/panels_e_f_g_h/reproduction/audit_legacy_cadence.py](paper/figure_01/panels_e_f_g_h/reproduction/audit_legacy_cadence.py) | Compare arrival-clock and legacy presumed-cadence processing for Figure 1. |
| [paper/figure_01/panels_e_f_g_h/reproduction/audit_vigor_alignment.py](paper/figure_01/panels_e_f_g_h/reproduction/audit_vigor_alignment.py) | Reproduce Figure 1E/F frame, detector and bin alignment without changing panels. |
| [paper/figure_01/panels_f_g_h/build_version_comparison.py](paper/figure_01/panels_f_g_h/build_version_comparison.py) | Assemble a self-contained, unselected F–H history comparison without re-analysis. |
| [paper/figure_01/panels_f_g_h/recolor_trace10_comparison.py](paper/figure_01/panels_f_g_h/recolor_trace10_comparison.py) | Change the standalone 10sTrace shortlist comparison's shared heatmap scale from authenticated numerical caches; exact raster reconstruction, staged review and guarded publication. |
| [paper/figure_01/panels_f_g_h/reproduction/build_legacy_heatmaps.py](paper/figure_01/panels_f_g_h/reproduction/build_legacy_heatmaps.py) | Build matching, fully vector Figure 1 single-fish legacy-vigor heatmaps. |
| [paper/figure_01/panels_f_g_h/reproduction/render_saved_heatmaps.py](paper/figure_01/panels_f_g_h/reproduction/render_saved_heatmaps.py) | Render paired single-fish Figure 1 heatmaps from versioned SSD artifacts. |
| [paper/figure_01/panels_f_g_h/reproduction/render_trace_example.py](paper/figure_01/panels_f_g_h/reproduction/render_trace_example.py) | Render the historical 3sTrace example fish as a signed Figure 1G review heatmap. |
| [paper/figure_01/reproduction/verify_abcd_freeze.py](paper/figure_01/reproduction/verify_abcd_freeze.py) | Verify the historical Figure 1 A-D freeze without changing its artifacts. |
| [paper/figure_02/assembly/build.py](paper/figure_02/assembly/build.py) | Build the Figure 2 review layout, preserving SSD versions and source provenance. |
| [paper/figure_02/assembly/repair_canvas.py](paper/figure_02/assembly/repair_canvas.py) | Prepare a guarded, unselected canvas-overdraw repair; preserve all geometry/data and require fresh browser evidence before selection. |
| [paper/figure_02/panel_a/reproduction/build_pooled_heatmaps.py](paper/figure_02/panel_a/reproduction/build_pooled_heatmaps.py) | Build scoped Figure 2 A with the selected Figure 1 F–H pipeline and styles. |
| [paper/figure_02/panel_b/reproduction/render_trace_signed.py](paper/figure_02/panel_b/reproduction/render_trace_signed.py) | Render the 3sTrace/control Figure 2B signed heatmap and fish coverage review. |
| [paper/figure_02/panel_g/build_exploratory_presentation.py](paper/figure_02/panel_g/build_exploratory_presentation.py) | Separate, authenticated Figure 2 G presentation of saved exploratory results. |
| [paper/figure_02/panel_g/build_study_guide.py](paper/figure_02/panel_g/build_study_guide.py) | Build an evidence-bound teaching guide; never fit models or publish figure state. |
| [paper/figure_02/panel_g/reproduction/audit_delay_lmm.py](paper/figure_02/panel_g/reproduction/audit_delay_lmm.py) | Reproduce all saved Delay LMMs and independently check contrast algebra. |
| [paper/figure_02/panel_g/reproduction/render_historical_logmedian.py](paper/figure_02/panel_g/reproduction/render_historical_logmedian.py) | Matched historical LogMedian outcome; unchanged corrected frames and cohort. |
| [paper/figure_02/panel_g/reproduction/render_legacy_metric_lme.py](paper/figure_02/panel_g/reproduction/render_legacy_metric_lme.py) | Versioned Delay G mixed-effects review; fixed legacy metric, SSD outputs only. |
| [paper/figure_02/panel_g/reproduction/render_phase_lmm.py](paper/figure_02/panel_g/reproduction/render_phase_lmm.py) | Observed bout-only medians and separate global/phase-aware LMM contrasts. |
| [paper/figure_02/panel_g/run_independent_review.py](paper/figure_02/panel_g/run_independent_review.py) | Run the isolated Figure 2 G analysis without changing any selected figure. |
| [paper/figure_02/panel_g/run_technical_sensitivity.py](paper/figure_02/panel_g/run_technical_sensitivity.py) | Run the independent, source-only technical protocol sensitivity. |
| [paper/figure_02/panel_g/verify_study_guide.cjs](paper/figure_02/panel_g/verify_study_guide.cjs) | Retained technical workflow helper |
| [paper/figure_02/panels_g_h_i/audit_saved_statistics.py](paper/figure_02/panels_g_h_i/audit_saved_statistics.py) | Reconcile hash-bound saved outcomes, bootstrap draws, models and selected SVG statistical labels; produce an external audit without refitting or selecting science. |
| [paper/figure_02/panels_a_d_g/reproduction/render_saved_delay.py](paper/figure_02/panels_a_d_g/reproduction/render_saved_delay.py) | Render descriptive Figure 2 A/D/G Delay-control panels from the SSD cohort. |
| [paper/figure_02/panels_d_e/reproduction/audit_bout_statistics.py](paper/figure_02/panels_d_e/reproduction/audit_bout_statistics.py) | Recheck B rank tests and separately audit its zero-baseline reference. |
| [paper/figure_02/panels_d_e/reproduction/review_block_log_median.py](paper/figure_02/panels_d_e/reproduction/review_block_log_median.py) | Compare selected D/E ratios with the legacy log-median outcome on corrected frames. |
| [paper/figure_02/panels_d_e/reproduction/review_bout_only.py](paper/figure_02/panels_d_e/reproduction/review_bout_only.py) | Rebuild both active D/E outcomes under the standing bout-only vigor policy. |
| [paper/figure_02/panels_d_e/reproduction/verify_bout_review.py](paper/figure_02/panels_d_e/reproduction/verify_bout_review.py) | Validate active scientific review without changing frozen/historical files. |
| [paper/figure_02/panels_d_g/reproduction/render_legacy_statistics.py](paper/figure_02/panels_d_g/reproduction/render_legacy_statistics.py) | Figure 2D legacy-style stars and Figure 2G authenticated LME review. |
| [paper/figure_02/panels_e_h/reproduction/render_trace_legacy.py](paper/figure_02/panels_e_h/reproduction/render_trace_legacy.py) | Render exploratory Figure 2E/H using the archived normalized-vigor layout. |
| [paper/figure_02/reproduction/recover_saved_panels.py](paper/figure_02/reproduction/recover_saved_panels.py) | Recover Figure 2 review panels from authenticated, existing processed data. |
| [paper/figure_03/build_blup_study.py](paper/figure_03/build_blup_study.py) | Build an SSD-owned Markdown/HTML study from completed authenticated BLUP results; no model execution or scientific selection. |
| [paper/figure_03/build_main_draft.py](paper/figure_03/build_main_draft.py) | Build and guard the Figure 3 main-paper draft; no classifier fit or freeze. |
| [paper/figure_03/build_study_guide.py](paper/figure_03/build_study_guide.py) | Build a source-bound, offline educational guide; never select analysis/figure state. |
| [paper/figure_03/reproduction/render_trace_legacy_classifier.py](paper/figure_03/reproduction/render_trace_legacy_classifier.py) | Render an exploratory 3sTrace learner review from the archived WIP rule. |
| [paper/figure_03/run_blup_comparison.py](paper/figure_03/run_blup_comparison.py) | Authenticated, isolated Delay/control BLUP calibration and independent stress checks; no method selection or figure-state update. |
| [paper/figure_03/supplement/build_atlas.py](paper/figure_03/supplement/build_atlas.py) | Build/verify/publish a guarded exploratory Figure 3 atlas on external storage. |
| [paper/figure_03/supplement/verify_atlas.cjs](paper/figure_03/supplement/verify_atlas.cjs) | Retained technical workflow helper |
| [paper/figure_03/verify_main_draft.cjs](paper/figure_03/verify_main_draft.cjs) | Retained technical workflow helper |
| [paper/figure_03/verify_reference_roles.py](paper/figure_03/verify_reference_roles.py) | Verify exact Figure 3 heatmap/color roles and G observed-outcome transfer. |
| [paper/shared/apply_annotation_alignment.cjs](paper/shared/apply_annotation_alignment.cjs) | Explicit candidate annotation alignment; counterpart verify_annotations.cjs verifies saved artwork |
| [paper/shared/apply_axis_geometry.cjs](paper/shared/apply_axis_geometry.cjs) | Explicit candidate-axis edits; counterpart verify_axis_geometry.cjs verifies saved artwork |
| [paper/shared/assemble_svg.py](paper/shared/assemble_svg.py) | Compose replaceable SVG panels from a declarative paper-figure layout. |
| [paper/shared/build_legend_review.py](paper/shared/build_legend_review.py) | Authenticated Figure 1–2 text/layout candidates and draft legends in one self-contained external review; no recomputation or freeze. |
| [paper/shared/calibrate_layout_frames.py](paper/shared/calibrate_layout_frames.py) | Separate preset comparison; never rewrites selected figure artifacts or state. |
| [paper/shared/collect_layout_evidence.cjs](paper/shared/collect_layout_evidence.cjs) | Retained technical workflow helper |
| [paper/shared/collection.py](paper/shared/collection.py) | Apply explicit main-paper layout ownership to authenticated selected drafts. |
| [paper/shared/correct_relationships.py](paper/shared/correct_relationships.py) | Correct typed panel relationships and named axis-title formatting in drafts. |
| [paper/shared/freeze.py](paper/shared/freeze.py) | Standalone freeze gate, usable without importing the scientific libraries. |
| [paper/shared/layout.py](paper/shared/layout.py) | Apply explicit main-paper layout ownership to authenticated selected drafts. |
| [paper/shared/measure_layout.cjs](paper/shared/measure_layout.cjs) | Retained technical workflow helper |
| [paper/shared/measure_packing.cjs](paper/shared/measure_packing.cjs) | Retained technical workflow helper |
| [paper/shared/publish_axis_geometry.py](paper/shared/publish_axis_geometry.py) | Guarded publication of measured axis presentation corrections; no science edits. |
| [paper/shared/refine_composition.cjs](paper/shared/refine_composition.cjs) | Retained technical workflow helper |
| [paper/shared/reconcile_saved_draft.py](paper/shared/reconcile_saved_draft.py) | Reconcile authenticated saved artwork, caption and scoped state through existing guards; preserve SVG/HTML and frozen selections. |
| [paper/shared/register_axes.py](paper/shared/register_axes.py) | Extend existing draft registries with explicit axis identities; never edit SVGs. |
| [paper/shared/stage_annotations.py](paper/shared/stage_annotations.py) | Stage author-requested legend/heading semantics on measured axis candidates. |
| [paper/shared/stage_nature.py](paper/shared/stage_nature.py) | Stage and guard the selected main-paper SVG presentation migration. |
| [paper/shared/style_nature_presentation.cjs](paper/shared/style_nature_presentation.cjs) | Explicit profile-bound candidate styling; counterpart verify_nature_presentation.cjs verifies saved artwork |
| [paper/shared/verify_annotations.cjs](paper/shared/verify_annotations.cjs) | Retained technical workflow helper |
| [paper/shared/verify_axis_geometry.cjs](paper/shared/verify_axis_geometry.cjs) | Retained technical workflow helper |
| [paper/shared/verify_collection.cjs](paper/shared/verify_collection.cjs) | Retained technical workflow helper |
| [paper/shared/verify_layout_frames.py](paper/shared/verify_layout_frames.py) | Read-only frame checks, browser review generation and bound evidence collection. |
| [paper/shared/verify_nature_presentation.cjs](paper/shared/verify_nature_presentation.cjs) | Retained technical workflow helper |
| [paper/shared/working_copy.py](paper/shared/working_copy.py) | Prepare/reuse hash-verified editable frozen/draft SVGs, measure effective SVG properties and verify the scoped current state. |

## Reviews

| Command | Scope |
|---|---|
| [reviews/library.py](reviews/library.py) | Check review links and publish a portable SSD reading library without rewriting existing reviews or scientific state. |

## Workflows

| Command | Scope |
|---|---|
| [workflows/macos/run_legacy_with_paths.py](workflows/macos/run_legacy_with_paths.py) | Run an existing script on macOS with opt-in Windows path relocation. |
| [workflows/metrics/compare_response_magnitude.py](workflows/metrics/compare_response_magnitude.py) | Descriptive fish-level suppression comparison with paired-metric bootstrap. |
| [workflows/metrics/compare_train_test_windows.py](workflows/metrics/compare_train_test_windows.py) | Compare Pre-Train with pooled late-Train/early-Test CS trials for every fish. |
| [workflows/trace/audit_increasing_trace_protocol.py](workflows/trace/audit_increasing_trace_protocol.py) | Classify every incTrace protocol using only Cycle and Reinforcer event times. |
| [workflows/trace/prepare_runs.py](workflows/trace/prepare_runs.py) | Prepare an all-fish fixed-trace pipeline config from the raw folder. |
| [workflows/trace/reproduction/cohort_and_legacy_labels.py](workflows/trace/reproduction/cohort_and_legacy_labels.py) | Freeze the complete 3sTrace cohort and package provisional WIP labels. |
| [workflows/trace/reproduction/run_legacy_full59.py](workflows/trace/reproduction/run_legacy_full59.py) | Finish exploratory 3sTrace Figures 2–4 for all 59 complete fish. |
| [workflows/windows/common.ps1](workflows/windows/common.ps1) | Shared manifest-completion helper; sourced by stages, not a standalone workflow |
| [workflows/windows/delay.ps1](workflows/windows/delay.ps1) | Maintained Delay launcher: -Stage pipeline-analysis, resume or complete-analysis; forwards original stage parameters |
| [workflows/windows/stages/delay_complete_analysis.ps1](workflows/windows/stages/delay_complete_analysis.ps1) | Complete the technical allDelay pipeline and LME stages |
| [workflows/windows/stages/delay_pipeline_analysis.ps1](workflows/windows/stages/delay_pipeline_analysis.ps1) | Run the resumable allDelay technical workflow |
| [workflows/windows/stages/delay_resume.ps1](workflows/windows/stages/delay_resume.ps1) | Resume authenticated allDelay technical stages |
| [workflows/windows/stages/trace_pipeline.ps1](workflows/windows/stages/trace_pipeline.ps1) | Prepare and run the fixed-trace technical pipeline |
| [workflows/windows/stages/trace_process_ingested.ps1](workflows/windows/stages/trace_process_ingested.ps1) | Process complete ingested fixed-trace/control triplets |
| [workflows/windows/trace.ps1](workflows/windows/trace.ps1) | Maintained Trace launcher: -Stage pipeline or process-ingested; preserves separate protocol/cohort checks |

## Recovered and retired sources

Twenty-six completed recipes were retired after native transfer/hash verification. After extracting their reusable helpers, three further completed drivers and one redundant helper facade were retired. This is 29 completed drivers plus one helper facade; exact pre-refactor originals for all 111 former files remain recoverable. No numerical dataset or scientific definition was deleted.

- [Initial retirement inventory](../records/repository-archive/script-organization-20261024/transfer-inventory.json).
- [Post-extraction retirement inventory](../records/repository-archive/script-extracted-drivers-20261024/transfer-inventory.json).
- [Source identity and relocation record](../records/repository-archive/script-organization-20261024/README.md).

The dated audit is assessment evidence; this catalogue describes the current physical command tree.

- [Ordered figure browser](reviews/ordered_library.py): guarded figure-page,
  catalogue, reading-snapshot and recovery publication through review_library.py.
- [Version discovery](reviews/versions.py): source-bound saved-artwork inventory,
  embedded payload adapters, declared panel views and single-version controls.

## Figure 4 local exploratory model pilots

- [build_figure4_delay_model_draft.py](build_figure4_delay_model_draft.py): exploratory pilot or exact-render check; no scientific freeze.
- [check_figure4_delay_model_draft.cjs](check_figure4_delay_model_draft.cjs): exploratory pilot or exact-render check; no scientific freeze.
- [build_figure4_3strace_model_draft.py](build_figure4_3strace_model_draft.py): exploratory pilot or exact-render check; no scientific freeze.
- [check_figure4_3strace_model_draft.cjs](check_figure4_3strace_model_draft.cjs): exploratory pilot or exact-render check; no scientific freeze.

## Retained PC-side commands

These existing PC-side workflows are preserved for compatibility and further lifecycle review. Their presence does not establish scientific approval.

- [align_figure2_columns.py](align_figure2_columns.py)
- [analyze_figure1_trace3_learner_candidates.py](analyze_figure1_trace3_learner_candidates.py)
- [analyze_figure2_trace10.py](analyze_figure2_trace10.py)
- [analyze_figure2_trace3_matched.py](analyze_figure2_trace3_matched.py)
- [analyze_figures34_3strace.py](analyze_figures34_3strace.py)
- [analyze_figures34_trace10.py](analyze_figures34_trace10.py)
- [analyze_trace10_learners.py](analyze_trace10_learners.py)
- [assemble_figure2_trace10.py](assemble_figure2_trace10.py)
- [assemble_figure2_trace3_matched.py](assemble_figure2_trace3_matched.py)
- [audit_figure2_heatmap_ab.py](audit_figure2_heatmap_ab.py)
- [audit_figure2_heatmap_ab_followup.py](audit_figure2_heatmap_ab_followup.py)
- [audit_trace10_learner_shortlist.py](audit_trace10_learner_shortlist.py)
- [build_figure2_trace3_panels.py](build_figure2_trace3_panels.py)
- [build_figure3_3strace_ae.py](build_figure3_3strace_ae.py)
- [build_figure4_3strace_preliminary.py](build_figure4_3strace_preliminary.py)
- [build_figure4_3strace_score_draft.py](build_figure4_3strace_score_draft.py)
- [check_figure3_3strace_ae.cjs](check_figure3_3strace_ae.cjs)
- [check_figure4_3strace_preliminary.cjs](check_figure4_3strace_preliminary.cjs)
- [check_figure4_3strace_score_draft.cjs](check_figure4_3strace_score_draft.cjs)
- [check_figures34_3strace.cjs](check_figures34_3strace.cjs)
- [check_figures34_trace10.cjs](check_figures34_trace10.cjs)
- [complete_figures34_trace10.py](complete_figures34_trace10.py)
- [explore_figure2_columns.py](explore_figure2_columns.py)
- [figure2_median_pooling.py](figure2_median_pooling.py)
- [finalize_figures34_3strace.py](finalize_figures34_3strace.py)
- [finalize_trace3_additions.py](finalize_trace3_additions.py)
- [finish_figure2_trace10.py](finish_figure2_trace10.py)
- [finish_trace10_example.py](finish_trace10_example.py)
- [finish_trace10_learner_selection.py](finish_trace10_learner_selection.py)
- [maintenance/storage_migration.py](maintenance/storage_migration.py)
- [measure_figure1_trace3_replacement.cjs](measure_figure1_trace3_replacement.cjs)
- [measure_figure2_exploration.cjs](measure_figure2_exploration.cjs)
- [measure_saved_figure.cjs](measure_saved_figure.cjs)
- [measure_trace10_example.cjs](measure_trace10_example.cjs)
- [polish_trace10_report.py](polish_trace10_report.py)
- [process_delay_additions.py](process_delay_additions.py)
- [process_trace10_additions.py](process_trace10_additions.py)
- [process_trace3_additions.py](process_trace3_additions.py)
- [publish_figure2_trace10.py](publish_figure2_trace10.py)
- [publish_figure3_3strace_ae.py](publish_figure3_3strace_ae.py)
- [publish_figure4_3strace_preliminary.py](publish_figure4_3strace_preliminary.py)
- [publish_figure4_3strace_score_draft.py](publish_figure4_3strace_score_draft.py)
- [refine_figure2_axes.py](refine_figure2_axes.py)
- [render_figures34_3strace.py](render_figures34_3strace.py)
- [render_figures34_trace10.py](render_figures34_trace10.py)
- [render_trace10_de_correction.py](render_trace10_de_correction.py)
- [render_trace10_example.py](render_trace10_example.py)
- [render_trace10_learner_comparison.py](render_trace10_learner_comparison.py)
- [repair_figures34_3strace_layout.py](repair_figures34_3strace_layout.py)
- [replace_figure1_trace3_example.py](replace_figure1_trace3_example.py)
- [resume_trace10_additions.py](resume_trace10_additions.py)
- [review_figure1_trace3_runner_up.py](review_figure1_trace3_runner_up.py)
- [review_figure2_exploration.py](review_figure2_exploration.py)
- [run_trace3_additions.py](run_trace3_additions.py)
- [show_figure1_panel_h_five_options.py](show_figure1_panel_h_five_options.py)
- [trace10_panel_statistics.py](trace10_panel_statistics.py)
- [verify_figures34_3strace.py](verify_figures34_3strace.py)
