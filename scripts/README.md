# Script inventory

Use `classical-conditioning render-paper-panels` for the current paper review.
It records its commands, settings, outputs and sidecars in `paper-panel-run.json`.
The scientific panel registry remains blocked until the approvals in
[`PAPER_FIGURE_FREEZE.md`](../docs/analysis/figures/PAPER_FIGURE_FREEZE.md)
are complete.

| Role | Scripts | Maintenance rule |
| --- | --- | --- |
| Current paper review adapters | `render_legacy_ssd_example_traces.py`, `render_legacy_ssd_example_heatmaps.py`, `render_legacy_ssd_figure2_delay.py`, `render_figure2_legacy_stats_lme_review.py` | Keep these paths stable because run manifests and provenance sidecars name them. Figure 2 A now pools the same signed bins as Figure 1 E; inference remains opt-in and exploratory. |
| Earlier figure comparisons | `render_figure1_all_frame_heatmaps.py`, `render_figure1_managua_review.py`, `render_figure1_raw_vigor_y_focus.py`, `render_figure2_managua_review.py`, `render_figure2_stats_review.py`, `render_log_scaled_vigor_heatmaps.py`, `render_ssd_per_trial_example_heatmaps.py` | Run directly only to reproduce a named comparison. Their 0–1 measures do not define the current paper heatmap. Retain old outputs and sidecars. |
| Analysis workflow launchers | `complete-allDelay-to-lme-windows.ps1`, `resume-allDelay-full-windows.ps1`, `run-allDelay-full-windows.ps1`, `run-all3sTrace-full-windows.ps1`, `process-ingested-trace-fish.ps1` | These operate the analysis pipeline rather than render paper panels. The trace per-fish worker can run while cohort preflight is blocked by incomplete raw triplets. |

The active Figure 1 and Figure 2 heatmaps share
[`signed_bout_heatmap.py`](../src/classical_conditioning/figures/signed_bout_heatmap.py):
positive moving-bout frames, log vigor, fish/trial pre-CS median centering over
−20≤t<0 s, bout medians, and 0.5-s bin means. Figure 2 then averages fish
equally. Both vigor heatmaps use `managua_r` at −0.25…+0.25. Supplementary
coverage also uses `managua_r`, with a separately labeled 0…1 fraction scale.

Do not rename a script or overwrite a historical output merely to tidy the
directory: old sidecars contain its source path and hash. The complete variant
inventory is [`review-variants.json`](../configs/paper-figures/review-variants.json).
