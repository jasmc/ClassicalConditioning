# Supported scripts

This folder keeps supported commands, audits, reproduction tools and their
dependency closure. Earlier cleanup counts describe their dated checkpoints;
they are not the current inventory. Every retained tool is linked below.
The [lifecycle policy](../docs/project-maintenance/script-lifecycle.md) separates
supported work from historical candidates. The
[dated inventory and consumer candidates](../records/repository-history/2026-10-17-organization-reconciliation/script-inventory.json)
records the initial assessment. The [23 October follow-up](../records/repository-history/2026-10-23-script-lifecycle/README.md) completes its pending classifications. Retention does not approve scientific use.

## Authority and normal entry points

Use `classical-conditioning render-paper-panels` for the package’s supported paper-review route. Use `python scripts/freeze_figure.py` or `classical-conditioning freeze-figure` for new freezes, with `--check-only` first. An old preparation or registration recipe does not authorize a new freeze. Current scoped scientific selections are listed in the [freeze index](../records/frozen-analyses/selections.md); exploratory learner/trace outputs are not publication approval.

Permanent exports belong on JOAQUIM or another explicit external destination. Follow the [archive/storage instructions](../records/repository-archive/README.md) and AGENTS.md. A new review should be one consolidated artifact. Temporary synthetic test files are allowed. Historical configurations and provenance may name an archived script; `resolve_artifact` resolves those references through the inventories without rewriting frozen bytes.

## Retained tool catalogue

Descriptions identify scope, not blanket scientific approval. Windows launchers support the named technical workflows; they are not replacements for reviewed cohort decisions. Renderers and historical audits may require the documented SSD inputs and path adapter.

### Storage and transfer tools

| Script | Scope |
|---|---|
| [mac_ssd_run.py](mac_ssd_run.py) | Run an existing script on macOS with opt-in Windows path relocation. |
| [repository_archive.py](repository_archive.py) | Audit repository content and transfer payloads only after JOAQUIM verification. |

### Assembly and schematic builders

| Script | Scope |
|---|---|
| [figure_working_copy.py](figure_working_copy.py) | Prepare/reuse hash-verified editable frozen/draft SVGs, measure effective SVG properties and verify the scoped current state. |
| [build_figure_legend_review.py](build_figure_legend_review.py) | Authenticated Figure 1–2 text/layout candidates and draft legends in one self-contained external review; no recomputation or freeze. |
| [assemble_svg_figure.py](assemble_svg_figure.py) | Compose replaceable SVG panels from a declarative paper-figure layout. |
| [recolor_trace10_comparison.py](recolor_trace10_comparison.py) | Change the standalone 10sTrace shortlist comparison's shared heatmap scale from authenticated numerical caches; exact raster reconstruction, staged review and guarded publication. |
| [build_figure1_panel_a.py](build_figure1_panel_a.py) | Normalize the supplied vector preparation scheme to the plot font. |
| [build_figure1_panel_b.py](build_figure1_panel_b.py) | Draw an exploratory, fully vector Figure 1B from active experiment timing. |
| [build_figure1_panel_c.py](build_figure1_panel_c.py) | Build an editable SVG schematic of the Figure 1 session protocol. |
| [build_figure2_assembly.py](build_figure2_assembly.py) | Build the Figure 2 review layout, preserving SSD versions and source provenance. |
| [update_latest_figure_assemblies.py](update_latest_figure_assemblies.py) | Compose the latest selected panels into one self-contained review HTML. |

### Scientific audits and verification

| Script | Scope |
|---|---|
| [audit-inc-trace-stim.py](audit-inc-trace-stim.py) | Classify every incTrace protocol using only Cycle and Reinforcer event times. |
| [audit_figure1_camera_gaps.py](audit_figure1_camera_gaps.py) | Audit >10 ms intervals back to immutable camera/tracking text logs. |
| [audit_figure1_legacy_cadence.py](audit_figure1_legacy_cadence.py) | Compare arrival-clock and legacy presumed-cadence processing for Figure 1. |
| [audit_figure1_vigor_alignment.py](audit_figure1_vigor_alignment.py) | Reproduce Figure 1E/F frame, detector and bin alignment without changing panels. |
| [audit_figure2_bout_only_statistics.py](audit_figure2_bout_only_statistics.py) | Recheck B rank tests and separately audit its zero-baseline reference. |
| [audit_figure2_delay_lmm.py](audit_figure2_delay_lmm.py) | Reproduce all saved Delay LMMs and independently check contrast algebra. |
| [audit_preprocessing_contract.py](audit_preprocessing_contract.py) | Read-only preprocessing evidence; writes only a new audit directory. |
| [verify_figure2_bout_only_review.py](verify_figure2_bout_only_review.py) | Validate active scientific review without changing frozen/historical files. |

### Analysis and review renderers with retained dependencies

| Script | Scope |
|---|---|
| [build_figure1_legacy_vigor_heatmaps.py](build_figure1_legacy_vigor_heatmaps.py) | Build matching, fully vector Figure 1 single-fish legacy-vigor heatmaps. |
| [plot_metric_response_magnitude.py](plot_metric_response_magnitude.py) | Descriptive fish-level suppression comparison with paired-metric bootstrap. |
| [plot_train_test_transition_metrics.py](plot_train_test_transition_metrics.py) | Compare Pre-Train with pooled late-Train/early-Test CS trials for every fish. |
| [populate_figure2_available.py](populate_figure2_available.py) | Recover Figure 2 review panels from authenticated, existing processed data. |
| [render_figure1_3strace_example.py](render_figure1_3strace_example.py) | Render the historical 3sTrace example fish as a signed Figure 1G review heatmap. |
| [render_figure1_cd_scaled_log_review.py](render_figure1_cd_scaled_log_review.py) | Compare Figure 1 C/D traces with trial-scaled log vigor baselines. |
| [render_figure2_3strace_legacy_layout.py](render_figure2_3strace_legacy_layout.py) | Render exploratory Figure 2E/H using the archived normalized-vigor layout. |
| [render_figure2_3strace_signed_review.py](render_figure2_3strace_signed_review.py) | Render the 3sTrace/control Figure 2B signed heatmap and fish coverage review. |
| [render_figure2_delay_legacy_metric_lme.py](render_figure2_delay_legacy_metric_lme.py) | Versioned Delay G mixed-effects review; fixed legacy metric, SSD outputs only. |
| [render_figure2_delay_logmedian.py](render_figure2_delay_logmedian.py) | Matched historical LogMedian outcome; unchanged corrected frames and cohort. |
| [render_figure2_delay_phase_lmm.py](render_figure2_delay_phase_lmm.py) | Observed bout-only medians and separate global/phase-aware LMM contrasts. |
| [render_figure2_legacy_stats_lme_review.py](render_figure2_legacy_stats_lme_review.py) | Figure 2D legacy-style stars and Figure 2G authenticated LME review. |
| [render_figure3_3strace_review.py](render_figure3_3strace_review.py) | Render an exploratory 3sTrace learner review from the archived WIP rule. |
| [render_legacy_ssd_example_heatmaps.py](render_legacy_ssd_example_heatmaps.py) | Render paired single-fish Figure 1 heatmaps from versioned SSD artifacts. |
| [render_legacy_ssd_example_traces.py](render_legacy_ssd_example_traces.py) | Render Figure 1 example traces from the versioned allDelay-full-v1 SSD export. |
| [render_legacy_ssd_figure2_delay.py](render_legacy_ssd_figure2_delay.py) | Render descriptive Figure 2 A/D/G Delay-control panels from the SSD cohort. |
| [review_figure2_block_log_median.py](review_figure2_block_log_median.py) | Compare selected D/E ratios with the legacy log-median outcome on corrected frames. |
| [review_figure2_bout_only.py](review_figure2_bout_only.py) | Rebuild both active D/E outcomes under the standing bout-only vigor policy. |

### Freeze gate, exact extraction and scoped reproduction

| Script | Scope |
|---|---|
| [build_panel_e_frozen_v12_heatmap_rows_review.py](build_panel_e_frozen_v12_heatmap_rows_review.py) | Single HTML Panel E review using the historical V12 half-second bins. |
| [freeze_figure.py](freeze_figure.py) | Standalone freeze gate, usable without importing the scientific libraries. |
| [freeze_figure1_panels.py](freeze_figure1_panels.py) | Verify the historical Figure 1 A-D freeze without changing its artifacts. |
| [prepare_figure1_fgh_v12_freeze.py](prepare_figure1_fgh_v12_freeze.py) | Prepare and package the author-selected F/G/H V12 in the single HTML review. |
| [prepare_figure2_B_freeze.py](prepare_figure2_B_freeze.py) | Prepare the selected bout-only B panels with corrected fish-level statistics. |
| [prepare_panel_e_heatmap_rows_freeze.py](prepare_panel_e_heatmap_rows_freeze.py) | Prepare the selected Panel E for the explicit freeze gate; archive into one HTML. |

### Supported analysis workflows

| Script | Scope |
|---|---|
| [complete-allDelay-to-lme-windows.ps1](complete-allDelay-to-lme-windows.ps1) | Complete the technical allDelay pipeline and LME stages |
| [finalize_3strace_exploratory.py](finalize_3strace_exploratory.py) | Freeze the complete 3sTrace cohort and package provisional WIP labels. |
| [prepare-trace-runs.py](prepare-trace-runs.py) | Prepare an all-fish fixed-trace pipeline config from the raw folder. |
| [process-ingested-trace-fish.ps1](process-ingested-trace-fish.ps1) | Process complete ingested fixed-trace/control triplets |
| [resume-allDelay-full-windows.ps1](resume-allDelay-full-windows.ps1) | Resume authenticated allDelay technical stages |
| [run-all3sTrace-full-windows.ps1](run-all3sTrace-full-windows.ps1) | Prepare and run the fixed-trace technical pipeline |
| [run-allDelay-full-windows.ps1](run-allDelay-full-windows.ps1) | Run the resumable allDelay technical workflow |
| [run_3strace_legacy_full59.py](run_3strace_legacy_full59.py) | Finish exploratory 3sTrace Figures 2–4 for all 59 complete fish. |

## Maintenance rules

Keep a script here when a supported command/test uses it, an active plan needs it, or it provides an explicit reusable audit/extraction contract. Keep its required imports as well. Put exact frozen code under `records/frozen-analyses/`; archive dated candidate recipes and completed transfer/continuation helpers on JOAQUIM with hashes. Do not rename supported scripts merely to group files, and do not remove unique originals before archive verification. Add new supported tools to this catalogue with their purpose and consumers.

The unused G preparation script was an exact copy of the [retained frozen G source](../records/frozen-analyses/figure-02/panel-g/2026-10-09-historical-log-median/code/renderer_source.py). Its old path resolves through the relocation map. The block-log-median script also matches a frozen snapshot but stays here because supported D/E preparation and statistics tools import it.

Assembly use: see [preflight and panel replacement](../docs/paper/figures/shared/assembly-preflight.md). The assembler supports no-write `--plan`, an explicit external `--output`, verified `--baseline-manifest` comparison and candidate-only `--overwrite`. Current figure-state artifacts continue through the existing guarded working-copy/review workflow.

## Structure validation

[check_repository_structure.py](check_repository_structure.py) performs read-only folder ownership, active-link, relocation, immutable evidence and scoped-state checks. Add `--external` for selected SSD artifacts, transaction payloads and recovered expected versions. Current folder rules are in [project maintenance](../docs/project-maintenance/structure.md); archive control records are under `records/repository-archive/`.

## Additional retained tools

Current display and Nature entry points are `main_paper_figures.py`,
`figure_working_copy.py` and `nature_main_figures.py`; follow the figure workflow
for scope and guarded publication. Historical-candidate rows below retain code
for provenance; they are not entry points for new styling.

| Tool | Purpose | Lifecycle |
|---|---|---|
| [align_figure1_fgh_fonts.py](align_figure1_fgh_fonts.py) | Match working F/G/H axis labels to the existing 8 pt bold Figure 1 labels. | historical-presentation-recipe |
| [align_paper_annotations.cjs](align_paper_annotations.cjs) | Retained technical workflow helper | supported-or-dependency |
| [archive_figure1_singlefish_versions.py](archive_figure1_singlefish_versions.py) | Archive author-disposed single-fish reviews using the verified transfer tools. | completed-archive-recipe |
| [build_figure1_fgh_version_comparison.py](build_figure1_fgh_version_comparison.py) | Assemble a self-contained, unselected F–H history comparison without re-analysis. | supported-or-dependency |
| [build_figure1_panel_f_bin_variants.py](build_figure1_panel_f_bin_variants.py) | Panel F comparison from authenticated frames; leaves active assembly unchanged. | historical-comparison-recipe |
| [build_figure2_g_analysis_guide.py](build_figure2_g_analysis_guide.py) | Build an evidence-bound teaching guide; never fit models or publish figure state. | supported-or-dependency |
| [build_figure2_g_exploratory.py](build_figure2_g_exploratory.py) | Separate, authenticated Figure 2 G presentation of saved exploratory results. | supported-or-dependency |
| [build_figure2_heatmap_row.py](build_figure2_heatmap_row.py) | Prepare the author-approved F/G/H 1 s physical log median freeze candidate. | supported-or-dependency |
| [build_figure2_pooled_heatmaps.py](build_figure2_pooled_heatmaps.py) | Build scoped Figure 2 A with the selected Figure 1 F–H pipeline and styles. | supported-or-dependency |
| [build_figure3_explanation.py](build_figure3_explanation.py) | Build a source-bound, offline educational guide; never select analysis/figure state. | supported-guide-builder |
| [build_figure3_individual_atlas.py](build_figure3_individual_atlas.py) | Build/verify/publish a guarded exploratory Figure 3 atlas on external storage. | supported-or-dependency |
| [build_figure3_main_paper.py](build_figure3_main_paper.py) | Build and guard the Figure 3 main-paper draft; no classifier fit or freeze. | supported-or-dependency |
| [calibrate_layout_frames.py](calibrate_layout_frames.py) | Separate preset comparison; never rewrites selected figure artifacts or state. | supported-or-dependency |
| [check_figure3_atlas.cjs](check_figure3_atlas.cjs) | Retained technical workflow helper | supported-or-dependency |
| [check_figure3_main_paper.cjs](check_figure3_main_paper.cjs) | Retained technical workflow helper | supported-or-dependency |
| [check_layout_frames.py](check_layout_frames.py) | Read-only frame checks, browser review generation and bound evidence collection. | supported-or-dependency |
| [check_main_paper_layout.cjs](check_main_paper_layout.cjs) | Retained technical workflow helper | supported-or-dependency |
| [check_nature_presentation.cjs](check_nature_presentation.cjs) | Retained technical workflow helper | supported-or-dependency |
| [correct_panel_relationships.py](correct_panel_relationships.py) | Correct typed panel relationships and named axis-title formatting in drafts. | supported-or-dependency |
| [figure1_layout_spacing.py](figure1_layout_spacing.py) | Working presentation overrides for Figure 1 C cards and E colourbar spacing. | supported-or-dependency |
| [figure2_g_guide_notebook.js](figure2_g_guide_notebook.js) | Retained technical workflow helper | supported-or-dependency |
| [layout_figure2_compact_rows.py](layout_figure2_compact_rows.py) | Presentation-only shared D/E scale and compact G/H/I trial layout. | historical-presentation-recipe |
| [layout_main_paper_figures.py](layout_main_paper_figures.py) | Apply explicit main-paper layout ownership to authenticated selected drafts. | supported-or-dependency |
| [main_paper_figures.py](main_paper_figures.py) | Build or resolve the authenticated, self-contained main-paper figure collection. | supported-or-dependency |
| [measure_main_paper_layout.cjs](measure_main_paper_layout.cjs) | Retained technical workflow helper | supported-or-dependency |
| [measure_nature_packing.cjs](measure_nature_packing.cjs) | Retained technical workflow helper | supported-or-dependency |
| [move_figure2_block_notes.py](move_figure2_block_notes.py) | Move scoped D/E/F sample counts and method notes to the bound legend. | historical-presentation-recipe |
| [nature_main_figures.py](nature_main_figures.py) | Stage and guard the selected main-paper SVG presentation migration. | supported-or-dependency |
| [prepare_figure1_historical_fgh_freeze.py](prepare_figure1_historical_fgh_freeze.py) | Prepare the author-approved F/G/H 1 s physical log median freeze candidate. | supported-or-dependency |
| [prepare_figure1_panel_e_current_freeze.py](prepare_figure1_panel_e_current_freeze.py) | Prepare the authenticated current E alone for the explicit freeze gate. | historical-freeze-preparation |
| [publish_axis_geometry.py](publish_axis_geometry.py) | Guarded publication of measured axis presentation corrections; no science edits. | supported-or-dependency |
| [publish_figure1_fgh_one_second.py](publish_figure1_fgh_one_second.py) | Guarded installation of the scoped 1 s F/G/H freeze and linked E bands. | supported-or-dependency |
| [publish_figure1_historical_fgh_freeze.py](publish_figure1_historical_fgh_freeze.py) | Select and integrate the author-frozen historical-fish F/G/H row. | historical-freeze-publication |
| [publish_figure1_panel_e_current_freeze.py](publish_figure1_panel_e_current_freeze.py) | Publish the checked E-only freeze into the guarded current Figure 1 display. | historical-freeze-publication |
| [publish_figure2_heatmap_row.py](publish_figure2_heatmap_row.py) | Integrate the selected compact A freeze and retained B into Figure 2. | historical-freeze-publication |
| [readjust_figure2_g.py](readjust_figure2_g.py) | G draft: native D/Figure-1 plot typography and strokes, compact stats only. | historical-presentation-recipe |
| [refine_nature_layout.cjs](refine_nature_layout.cjs) | Retained technical workflow helper | supported-or-dependency |
| [refine_paper_annotations.py](refine_paper_annotations.py) | Stage author-requested legend/heading semantics on measured axis candidates. | supported-or-dependency |
| [register_figure_axes.py](register_figure_axes.py) | Extend existing draft registries with explicit axis identities; never edit SVGs. | supported-or-dependency |
| [render_figure1_best_learners_version08.py](render_figure1_best_learners_version08.py) | Render two candidate learners with numbered #08's one-second mean analysis. | supported-or-dependency |
| [render_figure1_historical_fish_one_second.py](render_figure1_historical_fish_one_second.py) | Apply the approved 1 s recipe to the authenticated legacy F/G/H fish. | historical-scientific-reproduction |
| [reorder_control_first.py](reorder_control_first.py) | Guarded presentation-only control-first ordering for Figure 1 F–H and Figure 2 D–F. | historical-presentation-recipe |
| [review_figure2_g_independent.py](review_figure2_g_independent.py) | Run the isolated Figure 2 G analysis without changing any selected figure. | supported-or-dependency |
| [review_figure2_g_technical.py](review_figure2_g_technical.py) | Run the independent, source-only technical protocol sensitivity. | supported-or-dependency |
| [style_axis_geometry.cjs](style_axis_geometry.cjs) | Retained technical workflow helper | supported-or-dependency |
| [style_figure2_placeholders.py](style_figure2_placeholders.py) | Match Figure 2 F/I pending placeholders to the current C presentation. | historical-presentation-recipe |
| [update_figure1_colourbar_row_alignment.py](update_figure1_colourbar_row_alignment.py) | Guarded rigid translations for Figure 1 E/H colourbars and FGH spacing. | historical-presentation-recipe |
| [update_figure1_f_arrow_position.py](update_figure1_f_arrow_position.py) | Guarded local arrow presentation edit, preserving the active Figure 1 pair. | historical-presentation-recipe |
| [update_figure1_layout_spacing.py](update_figure1_layout_spacing.py) | Guarded update of C box sizes and E colourbar title clearance only. | historical-presentation-recipe |
| [update_figure1_panel_e_display.py](update_figure1_panel_e_display.py) | Guarded E-only display edits using the current trace and selected frozen F rows. | supported-or-dependency |
| [update_figure2_panel_a_freeze.py](update_figure2_panel_a_freeze.py) | Prepare and integrate the explicitly requested Figure 2 A scoped freeze. | supported-or-dependency |
| [verify_figure2_g_analysis_guide.cjs](verify_figure2_g_analysis_guide.cjs) | Retained technical workflow helper | supported-guide-verifier |
| [verify_figure3_reference_style.py](verify_figure3_reference_style.py) | Verify exact Figure 3 heatmap/color roles and G observed-outcome transfer. | supported-scoped-verifier |
| [verify_main_paper_collection.cjs](verify_main_paper_collection.cjs) | Retained technical workflow helper | supported-or-dependency |
| [repair_figure3_guide_downloads.py](repair_figure3_guide_downloads.py) | Guarded, dated repair of embedded Figure 3 guide downloads; preserves embedded scientific payloads and notes. | completed-guide-repair |
| [complete_figure2_g_analysis_guide.py](complete_figure2_g_analysis_guide.py) | Append a source-bound chat supplement to the existing Figure 2 G guide; preserve the original scientific snapshot and notebook keys. | completed-guide-supplement |
| [preserve_figure3_guide_conversation.py](preserve_figure3_guide_conversation.py) | Preserve an authenticated conversation checkpoint in the Figure 3 guide without rebuilding scientific panels. | completed-guide-supplement |
| [review_library.py](review_library.py) | Check review links and publish a portable SSD reading library without rewriting existing reviews or scientific state. | supported-navigation-tool |
