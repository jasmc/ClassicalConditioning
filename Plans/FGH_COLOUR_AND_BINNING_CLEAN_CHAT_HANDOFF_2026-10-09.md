# Clean-chat handoff: single-fish heatmap versions and colour/binning experiments

Prepared 9 October 2026. Repository: `C:/Users/joaquim/Documents/ClassicalConditioning` (PowerShell). Read the current `AGENTS.md` before working. This is a handoff for a fresh chat, not a request to refreeze existing panels.

## User's new request and authorized scope

The user approved the latest fourth version and wants a new clean chat to:

1. Make colours easier to distinguish, comparing reviewable candidates.
2. Study a fifth version based on current Version 4 but with **1 s bins**.
3. Study a sixth version that replaces continuous **managua_r** with **four discrete colours based on quartiles and the median**.
4. Show the comparisons and update a review HTML summary with exact processing bullets, source provenance and numeric verification.

Preserve the existing four versions and their numeric definitions. Create separate candidates for the new work. Do not change panel E, A-D, cohorts, detector settings or the whole-figure assembly. Existing authorizations allow generating candidates and their local artifacts without redundant permission. Ask questions only where scientific or presentation choices remain unresolved. The user was frustrated when dependent recalculation proceeded before their clarification replies: keep necessary questions pending and wait for answers, while doing independent work.

## Current review and current-selection records

- Portable HTML: `C:/Users/joaquim/Documents/ClassicalConditioning/reviews/fgh_latest_summary_20261009/index.html`.
- HTML builder: same folder, `build_summary.py`.
- HTML verifier: same folder, `check_summary.cjs`; its current checks expect four figures/PDF downloads, so adapt it if adding rows.
- Current scoped selection: `C:/Users/joaquim/Documents/ClassicalConditioning/configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json`.
- Analysis history and assessment: `C:/Users/joaquim/Documents/ClassicalConditioning/Plans/FGH_REQUESTED_METHOD_CORRECTION_2026-10-09.md`.
- Earlier critical assessment: `C:/Users/joaquim/Documents/ClassicalConditioning/Plans/FGH_ANALYSIS_STEP_ASSESSMENT_2026-10-08.md`.

Important naming: the HTML contains **four current rows**, but their historical names are C Version 1, D Version 1, C Version 2, and Version 4. There is no separately established scientific recipe called Version 3. Keep those names explicit rather than silently renumbering their formulas.

## Shared fish, detector and eligibility

- F: **Delay, 20221115_07**.
- G: **3 s Trace, 20230310_08**. The user explicitly replaced the earlier fish 20230307_12. All current G data are from 20230310_08; hashes and reconstructed samples were verified.
- H: **Control, 20221115_09**.
- Display trials **5-94**, aligned to CS onset over **[-20,20) s**.
- Blocks: Pre trials 5-14, Train 15-64, Test 65-94.
- Direct raw vigor is calculated **before bout detection**: sum 16 corrected tail-angle components, take the wrapped consecutive-frame change, absolute value, and divide by inferred camera interval. Units: rad/ms. It is not a bout scalar.
- The common detector uses median smoothing and a rolling maximum-minus-minimum envelope, plus amplitude/duration/gap rules. Settings: smoothing 10 ms (7 samples); max 28.57 ms (21); min 571.43 ms (401); envelope threshold 4 degrees/ms; raw peak threshold 1 degree/ms; minimum duration 57.14 ms; maximum interbout gap 14.29 ms. Cadence/window sample counts are in the existing manifests.
- Eligible moving samples require valid consecutive frames, finite detector envelope, at least 80% angular tail coverage, detected bout membership and finite positive raw vigor.
- Log transform is **natural logarithm** of eligible raw vigor. Current versions retain non-bout and invalid/ineligible frames as **NaN**. No zero replacement, finite floor or negative-infinity replacement is currently applied.
- In binned versions, any one eligible finite sample is enough. Ignore NaNs regardless of their proportion; all-NaN bins are missing. There is no minimum bin coverage gate.

## The four current recipes

| Current row | Value before centring | Trial baseline | Transformation | Colour limits |
|---|---|---|---|---|
| Version 1 C samples | Full-bout median log vigor repeated on each eligible frame; no binning | P10, m=P50, P90 from eligible timepoints in [-15,0), carrying their corresponding complete-bout median | clip((x-m)/((P90-P10)/2), -1,+1) | [-1,+1], continuous managua_r |
| Version 1 D samples | Same full-bout median samples; no binning | Same timepoint baseline as C | clip((x-m)/max(m-P10,P90-m), -1,+1) | [-1,+1], continuous managua_r |
| Version 2 C direct means | Mean of direct eligible framewise log vigor per 0.5 s bin; no bout-median substitution | P10/P50/P90 from finite unscaled baseline-bin means in [-15,0); one vote per finite bin | clip((bin_mean-m)/((P90-P10)/2), -1,+1) | [-1,+1], continuous managua_r |
| Current Version 4 direct means | Mean of direct eligible framewise log vigor per **0.25 s** bin; no bout-median substitution | **Arithmetic mean** of finite unscaled baseline-bin means in **[-20,0)**; one vote per finite bin | bin_mean - mean(finite baseline-bin means); **no scaling, P10/P90 or numerical clipping** | **[-0.25,+0.25] log units**, continuous managua_r |

For C/D, each median uses the **complete detected bout beyond the plotting crop**. Replace that bout's eligible values with its median, then construct trial displays and baselines. Baseline timepoints are not reduced to one vote per bout: longer eligible durations contribute more timepoints. Full-bout support was explicitly verified against expanded read context. Some baseline bouts cross CS onset, so their complete median includes post-onset samples; this is documented, not silently changed.

C/D are median-centred percentile scales, **not ordinary min-max**. Mapping baseline P10/P90 exactly to -1/+1 would put zero at their midpoint, which need not equal the baseline median. The user wanted the median at zero, so the stated C/D formulas were retained.

Version 2 originally looked predominantly blue when its direct means were centred against a bout-median reference. The user explicitly approved computing its baseline statistics from its own finite bin means instead. Do not restore the old mismatched reference.

Version 4 has **160 bins per trial**, up to **80 baseline bins**. All **270 trial baselines are defined**; centred baseline means are zero within floating-point precision (exported maximum absolute residual about 9.7e-16). Mean-zero does not imply median-zero or equal sign counts. Only 8 current trial baselines have equal above/below-zero counts; this is compatible with correct mean centring.

For all versions, missing references make the trial undefined. C/D additionally require positive finite denominators. The three scaled rows have Control trial 16 undefined; current Version 4 has all trial references defined, including that trial. Fixed colour saturation is separate from numeric clipping: Version 4 values outside +/-0.25 remain unchanged in exports and use endpoint colours.

## Exact current numeric files and renderers

### C/D complete-bout sample sources

`C:/Users/joaquim/Documents/ClassicalConditioning/reviews/fgh_full_bouts_baseline_samples_20261009/`

- `PanelF_complete_bout_sample_data.parquet`, and G/H equivalents: columns include trial, FrameID, time_s, eligible, bout_id, raw_vigor, **log_vigor**, bout_median_log, C_sample, D_sample.
- Read **log_vigor** for direct new versions; do not accidentally read bout_median_log.
- `PanelF_C_sample_runs.csv`, `PanelF_D_sample_runs.csv`, and G/H equivalents.
- Full-bout support parquet, baseline_statistics.csv, data_manifest.json and numeric_verification.json.
- Its `PanelX_direct_bins.csv` files retain the historical mismatched-reference direct version. **Do not use their scaled values as current Version 2**.

Direct-frame parquet SHA-256 values:

- F: `0ea6642119308fec55a6cb00c370b003bc6b3cc6c9f1a043dca1d3cc90bee030`
- G: `a6e627f1acb7f58092494320cb43855c4a542f74d239f084cfd905c768e6c76d`
- H: `13c4c40754cb0b95f5a97eeea13c179e617d2da24b666730a1fd9a9f1c9c4c92`

### Current Version 2 data

`C:/Users/joaquim/Documents/ClassicalConditioning/reviews/fgh_direct_bins_own_reference_20261009/`

- `PanelF_direct_bins.csv` and G/H equivalents; mean_log_vigor, eligible_frame_count, C, trial/bin edges and own reference quantiles.
- `baseline_statistics.csv`, `data_manifest.json`, `numeric_verification.json`.

### Current Version 4 data and figures

`C:/Users/joaquim/Documents/ClassicalConditioning/reviews/fgh_version4_quartersecond_means_20261009/`

- **`build_version4.py`**: current complete recipe and registration code; use this as the starting point for Version 5.
- `PanelF_mean_bins.csv` and G/H: 14,400 rows each; trial, bin_index, start_s/end_s, total_frame_count, eligible_frame_count, nan_sample_count, mean_log_vigor, baseline_mean_log_vigor, trial_defined, delta_log_vigor.
- `baseline_statistics.csv`, `per_trial_baseline_balance.csv`, data_manifest.json, numeric_verification.json, exported_data_verification.json, README.md.
- `FGH_Version4_DirectBinMeans_legacy_layout.svg/png/pdf`.
- `Version4_DirectBinMeans_validation.json` binds outputs and source tables to hashes.

### Current C/D/Version 2 figures and common layout code

`C:/Users/joaquim/Documents/ClassicalConditioning/reviews/fgh_legacy_layout_20261009/`

- **`render_layout.py`**: shared plotting/layout and SVG verification. `render(kind, source, value_col, binned)` reads source CSVs and data_manifest hashes. Current Version4_DirectBinMeans expects PanelX_mean_bins.csv; colour limits depend on Version4_ prefix. Adapt carefully for new variant IDs/colour norms.
- Current figures: `FGH_C_BoutSamples_legacy_layout`, `FGH_D_BoutSamples_legacy_layout`, `FGH_C_DirectBins_legacy_layout`, each SVG/PNG/PDF with matching validation JSON.
- `example_trials.json` records the five G marker positions and original selection rationale.
- Geometry helper: `C:/Users/joaquim/Documents/ClassicalConditioning/reviews/fgh_v1_layout_freeze_20261009/shared_colourbar.py`. It verifies cell geometry/colour against the input tables; its global SIZE/LEFTS/WIDTH/NORM are set by the current renderer.

Use local hash-bound frame tables for new aggregation; no new external data reads should be required.

## Preserve the approved visual conventions

- Tall aligned F/G/H heatmaps; one shared colourbar at the far right per three-panel row.
- Current figure size 9 x 5.1 inches; panel lefts [.09,.355,.62], width .22; bottom .13, height .75.
- Thin boxed spines; outward ticks; x ticks -20,0,20; y ticks 10,20,...,90. No "Global CS trial" y-axis title.
- **Pre/Train/Test names left of F**, bold and vertical; white block separators at trial coordinates 14.5 and 64.5.
- **Strong green CS boundaries at 0 and 10 s** in all panels: #0d8136, width 2.4 pt, alpha .8. Keep them clearly visible. Legacy reference had 2 pt and alpha .75; the user explicitly asked for stronger boundaries.
- Training US dotted guide at 9 s for Delay and 13 s for 3 s Trace; no Control US guide.
- G right-side black **left-pointing arrowheads at trials 9,17,63,66,93**, provisional examples for a later panel E adaptation. Do not change panel E now.
- Panel names/fish IDs remain visible; fish name colours pink Delay, orange Trace, blue Control.
- Version 4 colourbar label: **Log vigor relative to baseline**; C/D labels: Vigor relative to baseline.
- Missing values are black and must stay distinguishable from a finite neutral value or a discrete low class.

Legacy style file, already inspected:
`F:/Results (paper)/2025_delay/Processed data/20221115_07_delay_blue-1_mitfaminusminus,elavl3gff,10uasgcamp6fef05_6dpf_scaled vigor heatmap aligned to CS_cmap_managua_r_vlim_auto.svg`.

## Version 5: authorized independent implementation

Base it on **current** Version 4, changing the bin width to **1 s**:

- Direct eligible framewise log vigor, non-bout/invalid NaNs ignored.
- Arithmetic mean within each 1 s bin.
- 40 bins per trial over [-20,20), up to 20 baseline bins in [-20,0).
- Baseline is the mean of finite **new 1 s baseline-bin means**, one vote per bin.
- Subtract baseline, no P10/P90 or scaling, no numeric clipping.
- Initially keep continuous managua_r limits +/-0.25 to isolate the bin-width effect. Preserve layout and missing-cell handling.

**Recompute from frames. Do not average the current quarter-second means into one-second means without sample-count weighting.** Eligible counts differ and some quarter bins are missing, so averaging bin means would define a different statistic. The trial baseline, however, intentionally gives each finite new bin one vote.

## Version 6 and colour clarity: unresolved choices to ask in the new chat

The user wants four managua_r colours based on quartiles and median. A usual empirical four-class mapping has thresholds **P25, P50, P75**, but the reference population is not yet specified. Ask targeted questions early, while independently building Version 5:

1. Should Version 6 use current Version 4's **0.25 s** values or new Version 5's **1 s** values?
2. Are the quartile thresholds empirical **data quartiles**, and should they be computed from pooled finite baseline values, all displayed values, or per-trial distributions? A shared pooled reference preserves the same colour meaning across fish; per-trial quartiles change it. If the user instead means quarter divisions of the fixed +/-0.25 colour range, clarify that this is not empirical quartiles.
3. Should the central split follow empirical **P50** or **zero relative to baseline**? They may differ under mean centring. Do not claim mean-centred zero is necessarily the median.

Do not silently choose which population's quartiles define the classes or silently apply per-trial colour normalization. Decide how endpoint/tie intervals are assigned and record class thresholds, colour hex values, reference population and bin width. Quartile ties can collapse categories; preserve and disclose such cases rather than inventing four equally populated classes. NaNs must remain a separate missing colour, not enter the lowest quartile.

Once clarified, use four clearly separated colours sampled from managua_r (for example four evenly spaced interior/endpoint positions, subject to visual review). Use a discrete legend/colourbar with the actual class boundaries and clear labels. Keep colour quantization as a display operation; retain unscaled delta_log_vigor in CSVs. If stronger continuous contrast/alternative sampled colours are proposed, separate colour changes from numeric transformations and label candidates clearly.

## Verification, dependencies and preservation

- Python: `C:/Users/joaquim/Documents/ClassicalConditioning/.venv-trace/Scripts/python.exe`, with NumPy/pandas/Matplotlib/PyArrow/PIL.
- pdftoppm is available for PDF readback. Apply the PDF skill and its required operation marker before authoring PDFs. Inspect the rendered figures, not only exit status.
- Node/Playwright dependency path used by HTML checker: `C:/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright`; headless Edge executable `C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe`. Headless browser launch required approved escalation in this workspace.
- Verify source hashes, finite/NaN masks, bin widths and edges, grouped raw-frame means, equal-bin baseline reference and mean-zero residuals. For discrete colours also verify thresholds, class assignment, tie policy and representative SVG fills.
- Export SVG/PNG/PDF and numeric CSV/provenance for candidates, and make review comparisons portable in HTML. Preserve the existing four rows or link their historical summary explicitly rather than overwrite their recipes.
- Do not confuse colour limits with numeric clipping. Mean-centred versions do not promise equal above/below counts; show the audit accurately.
- Follow AGENTS.md's figure element contract if the user later requests a freeze; this current request is for experiments/candidates.
- Earlier Version 4 folders with bin medians or -infinity trials are **historical**, not the current starting point. In particular, `fgh_version4_means_baseline20_20261009` has all trials undefined due to -infinity; do not start from that recipe.

## Suggested first message in the new chat

"I’ll preserve the four current versions and build the 1 s candidate from the current quarter-second mean version. I’ll compare colour clarity separately and clarify the quartile reference for the four-colour candidate before calculating its thresholds."
