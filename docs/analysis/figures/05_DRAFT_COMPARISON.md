# Paper draft-to-current figure comparison

This is a documented comparison, **not** an approval gate. The written [paper scaffold](../../../../ClassicalConditioningPaper/Helpers/Paper_Scaffold.md) and figure list in the separate paper repository define the intended panel roles. The supplied `Fig1_test2.png`, `Fig2_test1.png`, and `Fig4_test1.png` are draft evidence, not templates to copy. Figure 3 has no supplied draft. The paper repository and PNGs are not modified here. Proposed panel IDs and current gates are machine-readable in [`configs/paper-figures/behavior-paper.json`](../../../configs/paper-figures/behavior-paper.json).

## Evidence and comparison rules

- Draft pixels and panel lettering were inspected directly. Archived calculation evidence is in `legacy/scripts/2_ExampleFishPlotting.py` (individual traces and heatmaps), `4_ScaledVigorPlotting.py` (pooled scaled heatmaps, blocks, catches), and `5_NormalizedVigorPlotting.py` (trial ratio, selected blocks, statistics). The first script draws raw tail angle and vigor in degrees/deg/ms; the second masks non-bout samples before its scaled-vigor heatmap, takes fish-trial-bin medians then across-fish means, applies a baseline quantile scale, and offers signed baseline-subtracted clipped lines. The third calculates baseline/response means per trial, groups selected fish, and contains exploratory tests and mixed models. These are evidence of calculation, not proof that an exact draft PNG used every current switch.
- Current source evidence is `src/classical_conditioning/preprocessing/candidate_metric_kernel.py` (three corrected metric definitions), `analysis/temporal_profiles.py` (binning, masks and 0–1 scaling), `figures/temporal_profiles.py` (five CS and four US routine per-fish profile families), `figures/example_traces.py` (focused Figure 1 C/D paired traces), `figures/population_heatmap.py` (frozen-cohort all-valid-frame heatmap with fish coverage), `figures/cohort_response.py` (frozen-cohort ratios and 0–1 catch/block profiles), `analysis/trial_outcomes.py` ([−15,0) baseline and experiment-specific 9/13/20 s response windows), and `figures/export.py` (semantic export). The signed Figure 4 learner-panel analysis and renderer now exist in `analysis/figure4.py` and `figures/figure4.py`; they still require frozen Gate L inputs and timing reconciliation.
- No archived figure embeds a trustworthy machine-readable fish list, metric implementation hash, or inference table. Differences in cohort membership, uncertainty, and significance cannot be quantified from image pixels alone. A reviewed manifest and approved results must be authenticated before a paper panel can be declared completed.

## Figure 1 — setup and individual evidence

The intended visual sequence is **tail angle versus time (C) → vigor versus
time from the same fish and selected trials (D) → signed log-vigor heatmaps
(E–G)**. The March 24, 2026 single-fish implementation in
`2_ExampleFishPlotting.py` explicitly selected `managua_r`, with CS limits
−0.25 to +0.25, for bout-only log vigor centred on each trial's pre-stimulus
median. The February 14 version had used `spring`; March changed the palette
and limits. The refactored `signed-log-vigor` candidate figure now uses the
March palette and fixed CS limits with corrected bout data in 0.5-s bins.
The user-selected Figure 1E review is signed log vigor relative to each
trial's −20 to 0 s pre-CS median, displayed with `managua_r` at −0.25 to
+0.25. The 0–1 P10–P90 variants remain as historical alternatives. The current
Figure 2A review pools these same signed fish/trial bins equally across cohort
fish and uses the same `managua_r` limits.

| Intended panel | Draft evidence | Current counterpart and difference |
| --- | --- | --- |
| A setup | Draft A shows the restrained fish, basal illumination, CS LEDs, and US. | No registered apparatus artwork. The drawing lacks a reproducible source-artifact mapping; panel remains blocked. |
| B protocol | Draft B shows priming, 10 CS-only Pre-Train trials, 46 paired training trials, 30 CS-only Test trials and the four contingencies. | Experiment definitions and trial map encode timing. The written target also needs short 50-ms versus long 100-ms US pulses and randomized control schedule; these are not all explicit in the draft artwork. Panel remains blocked pending complete artwork. |
| C raw tracking | Draft C shows tail angle over time across selected example trials. | The user-selected paired trace renderer now plots distal cumulative angle centered on each trial's pre-CS median. C and D use the same fish, trials and measured frames. Example selection and final paper composition remain open. |
| D movement calculation | Draft D shows the corresponding vigor over time. | The paired renderer plots exactly one selected corrected vigor metric next to C, with the same event markers. The figure still needs final example selection and an explanatory calculation inset if retained. |
| E Delay fish | Draft E shows Delay versus control heatmaps, not just a Delay single-fish example. | The refactored `signed-log-vigor` figure is a **per-fish CS heatmap** with the March palette and limits, with three corrected candidate metrics per output. A one-signal paper panel and example selection are still needed. Record the extra draft control explicitly. |
| F 3sTrace fish | **Missing in draft.** | The written target requires a distinct 3sTrace example. A per-fish candidate profile can inform selection but cannot fill this panel until fish and metric are prespecified. |
| G control fish | **Missing as a separate draft panel** (only paired with Delay in E). | A prespecified unpaired-control fish and its coverage-masked profile are needed. |

The draft combines conditions in E while the written A–G layout separates three examples. The new signed-vigor figure implements the March colour rendering on corrected profile data but does not compose C–G into the planned sequence. Neither draft nor current candidate plots authenticate example-fish selection or a reviewed cohort.

An exploratory Figure 1D renderer now shows the five selected Delay/control
trials over the full −20 to +20 s window for each of the three candidate vigor
metrics. Its focused-Y version overlays the exact 0.5-s **signed log-vigor**
heatmap values as orange outlined step runs on a separate −0.25 to +0.25 axis.
Each run descends to zero at a neighboring NaN bin;
missing bins remain blank. The black frame-level vigor remains unscaled.
The conditional heatmap is sparse because
the legacy detector's centered 403-frame envelope loses validity around
isolated invalid derivatives, particularly in the selected Delay fish; those
gaps must not be filled by interpolation or called zero vigor. A continuous
all-frame variant remains archived for comparison. These remain review
variants, not a final metric choice.

**Protocol discrepancy requiring review:** the paper scaffold describes the
3sTrace long US at 13 s after CS onset (three seconds after the 10-s CS), but
the active `all3sTrace` `ConditionSpec.us_latency_s` is currently 9 s, matching
the archived `ALL_3S_TRACE` entry; a separate archived fixed-trace entry uses
13 s. We do not silently alter the experiment definition or put an expected-US
line at either time in a paper panel until the raw protocol identity is
reconciled. This is why Figure 1B and Figure 4 expected-US alignment remain
blocked, even though descriptive CS-aligned profiles can be rendered.

## Figure 2 — three matched population comparisons

| Intended panels | Draft evidence | Current counterpart and difference |
| --- | --- | --- |
| A–C heatmaps | Control/Delay, Control/3sTrace, Control/10sTrace; CS-aligned Pre/Train/Test stacks with a 0–1 scaled-vigor colour bar. | The Delay/control **paper-review adapter** now calculates the same signed, −20 to 0 s baseline-centered bout-log-vigor bins as Figure 1E for every fish, then averages fish equally. It displays `managua_r` at −0.25…+0.25. Coverage is exported separately with its own 0…1 fraction scale. Earlier two-stage P10/P90 output is retained as a historical comparison. The general-purpose `population_heatmap.py` route remains an all-valid-frame total-activity diagnostic. Trace cohorts and the final metric still require review. |
| D–F selected blocks | Three fish-level paired-line panels at PTr, ETe, LTe, with draft stars in Delay and 3sTrace. | Current `selected-block-ratio` derives finite response/baseline ratios for one selected metric, takes per-fish block medians, then condition medians/IQR, and records eligible trial/fish counts. Paper membership is **10–14, 65–69, 90–94**; the archived nine-block setting chose **Early Pre-Train** (the earlier five, 5–9), whereas its twelve-block branch chose Late Pre-Train. Thus the draft PTr values cannot be assumed equivalent. Current output is descriptive and does not copy the stars. |
| G–I trial trajectories | Paired-condition trial curves with shaded uncertainty and many annotated marks in Delay and 3sTrace; near-flat 10sTrace. | Current `trial-ratio` aggregates fish equally at each trial and shows coverage/uncertainty from available fish; it does not reproduce unexplained draft marks. Trial windows, cohort, metric, normalization, and model contrasts must match an approved analysis before inference is placed on the panel. Present 10sTrace as inconclusive, not a demonstrated positive or absolute limit. |

The archived normalized-vigor script can filter fish, uses mean baseline/response windows and exploratory within/between-condition tests; the current route uses corrected measured-time validity and a frozen cohort. Therefore visual agreement alone is not numerical equivalence. The displayed draft stars and line-plot marks must be regenerated from approved results, not traced from the PNG.

For Delay/control review, `scripts/render_legacy_ssd_example_heatmaps.py`
renders the selected signed Figure 1E variant. The separate
`scripts/render_log_scaled_vigor_heatmaps.py` retains an exploratory 0–1
Figure 1E variant and an older pooled Figure 2A from log-transformed corrected
bin means with a second pooled trial scale. The active Figure 2 A renderer
uses the shared signed bins. Earlier palette and `magma` images remain.
`scripts/render_figure2_stats_review.py`
adds exploratory fish-level condition comparisons to D and G: each fish's
post-pretrain change is compared by whole-fish condition-label permutation,
with maximum-contrast family-wise correction over two D blocks or 80 G trials.
The resulting CSV p values are separate from the planned LME tests; the
existing LME onset analysis has not passed its diagnostic gate.

The later `render_figure2_legacy_stats_lme_review.py` adapter adds D's
legacy-style Holm-corrected Mann–Whitney and paired Wilcoxon stars on the
current three-block fish ratios. Its G review combines the authenticated
current total-activity LME's global condition × block test, Holm-supported
block contrasts, and simultaneous trial band with exploratory legacy-style
within-block mean/rate LMEs and BH-adjusted pointwise trial marks. The latter
are secondary markers; the failed influence gate and absent simultaneous
trial onset remain visible. This combined G currently exists only for tail
length weighted angular L1, the metric of the saved LME run.

## Figure 3 — learner representation

There is **no draft image**. Written A–G roles are workflow, continuous scores, threshold/stability, learner fractions, individual change, representative trajectories, and all-fish validation. Current corrected per-fish outcomes are inputs, not a frozen learner representation. All seven paper panels remain blocked until Gate L and an independent validation mode are frozen.

## Figure 4 — earlier draft and replacement learner profiles

The table below records the comparison made before the signed learner-stratified Figure 4A–C route was implemented. The current registry assigns A, B and C to Delay, 3sTrace and 10sTrace block/catch figures respectively. Individual catches, movement and coverage are supplementary; independent timing is governed by the [supplementary plan](../../../Plans/10_SUPPLEMENTARY_FIGURES.md). This historical draft does not define the current figure calculation.

| Intended panel | Draft evidence | Current counterpart and difference |
| --- | --- | --- |
| A Delay learner dynamics | Draft A is **condition-wide** Pre-Train, Train 1–5, Test 1–3 traces versus control. | The newer signed learner-stratified Figure 4A analyzer/renderer exists and differs from this 0–1 condition-wide draft. A frozen learner manifest and authenticated analysis are still required. |
| B 3sTrace learner dynamics | Draft B has the same condition-wide rows and a conspicuous post-US rise. | New signed Figure 4B code exists, but Gate L and paired-US timing reconciliation remain. Separate anticipatory CS/trace behavior from direct post-US response. |
| C expected-US alignment | Draft C is instead a 10sTrace condition-wide profile. | New signed Figure 4C represents 10sTrace learner strata. Independent expected-US response timing remains a separate supplementary analysis; do not relabel the old draft as that analysis. |
| D timing metrics | Draft D is pooled Delay catch trials. | Current catch profile uses 0–1 activity, not signed onset/peak/center-of-mass/offset estimates with uncertainty. |
| E direct timing comparison | Draft E is pooled 3sTrace catch trials. | No authenticated Delay-versus-3sTrace timing contrast exists. |
| F all-fish validation | Draft F is pooled 10sTrace catch trials. | All-eligible-fish signed validation is missing; the draft is a different condition and calculation. |
| G robustness | **Missing in draft.** | Alternative windows/metrics, controls, and held-out timing checks remain planned. |

The archived script optionally subtracts a pre-CS baseline median and clips the result, so signed negative draft traces are mathematically possible. The older condition-wide catch/block plots use the **0–1 scaled total-activity** field and are not numerical replacements for the signed learner-stratified route. The replacement calculates −20 to 0 s baseline-centered bout-log-vigor bins, preserves no-bout gaps, and saves panel-data provenance. CS onset, CS offset, and each verified expected-US time have distinct guides.

Catch provenance has two numbering systems: training catches **11, 25, 39, 45** map to global CS trials **25, 39, 53, 59**. The archived `resolve_catch_trials()` and current experiment definition also pool global **65**, the first Test trial, in “all catch” views. The eventual panel-data sidecar and legend must state this **five-trial** rule.

## Export and remaining gates

Routine PNGs use the same registered data/artist semantics as publication SVG/PDF. `figures/export.py` attaches semantic IDs, artist mappings and source/input hashes in sidecars, embeds SVG provenance, and checks unique/required IDs. A paper export still needs panel-specific physical units and approved source data. The registry's `blocked` state is intentional where those requirements are absent; it is not a silent optional figure.
