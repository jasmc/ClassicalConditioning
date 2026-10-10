# Supported scripts

This folder keeps supported commands, scientific audits, freeze/extraction tools and their dependency closure. The 2026-10-10 follow-up archived 41 historical scripts, leaving 52 supported scripts. [Assessment and exact archive map](../docs/maintenance/archive/scripts-cleanup-20261010/README.md) records every disposition. Paths of retained tools stay stable for package imports, tests and provenance.

## Authority and normal entry points

Use `classical-conditioning render-paper-panels` for the package’s supported paper-review route. Use `python scripts/freeze_figure.py` or `classical-conditioning freeze-figure` for new freezes, with `--check-only` first. An old preparation or registration recipe does not authorize a new freeze. Current scoped scientific selections are listed in the [freeze index](../docs/analysis/figures/freezes/README.md); exploratory learner/trace outputs are not publication approval.

Permanent exports belong on JOAQUIM or another explicit external destination. Follow the [archive/storage instructions](../docs/maintenance/archive/README.md) and AGENTS.md. A new review should be one consolidated artifact. Temporary synthetic test files are allowed. Historical configurations and provenance may name an archived script; `resolve_artifact` resolves those references through the inventories without rewriting frozen bytes.

## Retained tool catalogue

Descriptions identify scope, not blanket scientific approval. Windows launchers support the named technical workflows; they are not replacements for reviewed cohort decisions. Renderers and historical audits may require the documented SSD inputs and path adapter.

### Storage and transfer tools

| Script | Scope |
|---|---|
| [archive_delay_review_versions.py](archive_delay_review_versions.py) | Build one self-contained review with a verified, content-deduplicated archive. |
| [audit_mac_ssd_handoff.py](audit_mac_ssd_handoff.py) | Read-only SSD data-coverage audit; write only a small handoff report. |
| [mac_ssd_run.py](mac_ssd_run.py) | Run an existing script on macOS with opt-in Windows path relocation. |
| [prepare_mac_ssd_handoff.py](prepare_mac_ssd_handoff.py) | Prepare the requested SSD handoff without altering existing source artifacts. |
| [repository_archive.py](repository_archive.py) | Audit repository content and transfer payloads only after JOAQUIM verification. |

### Assembly and schematic builders

| Script | Scope |
|---|---|
| [assemble_svg_figure.py](assemble_svg_figure.py) | Compose replaceable SVG panels from a declarative paper-figure layout. |
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
| [build_panel_e_frozen_v12_heatmap_rows_review.py](build_panel_e_frozen_v12_heatmap_rows_review.py) | Single HTML Panel E review using the current frozen V12 half-second bins. |
| [freeze_figure.py](freeze_figure.py) | Standalone freeze gate, usable without importing the scientific libraries. |
| [freeze_figure1_panels.py](freeze_figure1_panels.py) | Verify the historical Figure 1 A-D freeze without changing its artifacts. |
| [prepare_figure1_fgh_v12_freeze.py](prepare_figure1_fgh_v12_freeze.py) | Prepare and package the author-selected F/G/H V12 in the single HTML review. |
| [prepare_figure2_B_freeze.py](prepare_figure2_B_freeze.py) | Prepare the selected bout-only B panels with corrected fish-level statistics. |
| [prepare_figure2_G_logmedian_freeze.py](prepare_figure2_G_logmedian_freeze.py) | Prepare G from the selected, already fitted historical LogMedian revision. |
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
