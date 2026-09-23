# Paper draft-to-current figure comparison

This is a documented comparison, **not** an approval gate. The written [paper scaffold](../../../../ClassicalConditioningPaper/Helpers/Paper_Scaffold.md) and figure list in the separate paper repository define the intended panel roles. The supplied `Fig1_test2.png`, `Fig2_test1.png`, and `Fig4_test1.png` are draft evidence, not templates to copy. Figure 3 has no supplied draft. The paper repository and PNGs are not modified here. Proposed panel IDs and current gates are machine-readable in [`configs/paper-figures/behavior-paper.json`](../../../configs/paper-figures/behavior-paper.json).

## Evidence and comparison rules

- Draft pixels and panel lettering were inspected directly. Archived calculation evidence is in `Archive/historical-scripts/2_ExampleFishPlotting.py` (individual traces and heatmaps), `4_ScaledVigorPlotting.py` (pooled scaled heatmaps, blocks, catches), and `5_NormalizedVigorPlotting.py` (trial ratio, selected blocks, statistics). The first script draws raw tail angle and vigor in degrees/deg/ms; the second masks non-bout samples before its scaled-vigor heatmap, takes fish-trial-bin medians then across-fish means, applies a baseline quantile scale, and offers signed baseline-subtracted clipped lines. The third calculates baseline/response means per trial, groups selected fish, and contains exploratory tests and mixed models. These are evidence of calculation, not proof that an exact draft PNG used every current switch.
- Current source evidence is `src/classical_conditioning/preprocessing/candidate_metric_kernel.py` (three corrected metric definitions), `analysis/temporal_profiles.py` (binning, masks and 0–1 scaling), `figures/temporal_profiles.py` (four per-fish profile families), `figures/population_heatmap.py` (frozen-cohort all-valid-frame heatmap with fish coverage), `figures/cohort_response.py` (frozen-cohort ratios and 0–1 catch/block profiles), `analysis/trial_outcomes.py` (exact response/baseline windows), and `figures/export.py` (semantic export). Signed learner-panel renderers are **not implemented**.
- No archived figure embeds a trustworthy machine-readable fish list, metric implementation hash, or inference table. Differences in cohort membership, uncertainty, and significance cannot be quantified from image pixels alone. A reviewed manifest and approved results must be authenticated before a paper panel can be declared completed.

## Figure 1 — setup and individual evidence

| Intended panel | Draft evidence | Current counterpart and difference |
| --- | --- | --- |
| A setup | Draft A shows the restrained fish, basal illumination, CS LEDs, and US. | No registered apparatus artwork. The drawing lacks a reproducible source-artifact mapping; panel remains blocked. |
| B protocol | Draft B shows priming, 10 CS-only Pre-Train trials, 46 paired training trials, 30 CS-only Test trials and the four contingencies. | Experiment definitions and trial map encode timing. The written target also needs short 50-ms versus long 100-ms US pulses and randomized control schedule; these are not all explicit in the draft artwork. Panel remains blocked pending complete artwork. |
| C raw tracking | Draft C shows tail angle in degrees across example trials. | Intake tracking and corrected frames support raw kinematics, but no prespecified representative fish/trials or approved tracking point. Three corrected metrics are in `rad/ms` or `tail lengths/ms`; a radians-derived value must not be labelled degrees. |
| D movement calculation | Draft D shows a vigor trace in `deg/ms`. | The corrected kernel yields three candidate metrics, none selected for the paper. Converting angular units and specifying the XY-derived metric are essential before a calculation schematic can be labelled. |
| E Delay fish | Draft E shows Delay versus control heatmaps, not just a Delay single-fish example. | Current candidate profile heatmaps are **per fish**, CS/US-aligned and coverage-masked; they are descriptive candidates only. The Delay fish and final metric are not prespecified. Record the extra draft control explicitly, rather than silently relabelling it. |
| F 3sTrace fish | **Missing in draft.** | The written target requires a distinct 3sTrace example. A per-fish candidate profile can inform selection but cannot fill this panel until fish and metric are prespecified. |
| G control fish | **Missing as a separate draft panel** (only paired with Delay in E). | A prespecified unpaired-control fish and its coverage-masked profile are needed. |

Signal and presentation differences are therefore substantial: raw tail angle, vigor calculation, and heatmap scales are not interchangeable; the draft combines conditions in E while the written A–G layout separates three examples. Neither draft nor current candidate plots authenticate example-fish selection or a reviewed cohort.

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
| A–C heatmaps | Control/Delay, Control/3sTrace, Control/10sTrace; CS-aligned Pre/Train/Test stacks with a 0–1 scaled-vigor colour bar. | A **descriptive frozen-cohort population heatmap** now pools one scaled-total-activity value per fish and bin, then takes an equal-fish mean. Its companion strip shows the fraction of frozen fish contributing; the panel-data Parquet records counts. Its signal is **activity across all valid frames, including valid zeros**, not the archived pooled script's vigor among detected movers. The archived script discards non-bout values and aggregates fish-trial-bin medians into pooled scaled-vigor values. Per-fish candidate heatmaps remain distinct. Paper status stays blocked until matched cohorts, metric and this signal definition are approved. |
| D–F selected blocks | Three fish-level paired-line panels at PTr, ETe, LTe, with draft stars in Delay and 3sTrace. | Current `selected-block-ratio` derives finite response/baseline ratios for one selected metric, takes per-fish block medians, then condition medians/IQR, and records eligible trial/fish counts. Paper membership is **10–14, 65–69, 90–94**; the archived nine-block setting chose **Early Pre-Train** (the earlier five, 5–9), whereas its twelve-block branch chose Late Pre-Train. Thus the draft PTr values cannot be assumed equivalent. Current output is descriptive and does not copy the stars. |
| G–I trial trajectories | Paired-condition trial curves with shaded uncertainty and many annotated marks in Delay and 3sTrace; near-flat 10sTrace. | Current `trial-ratio` aggregates fish equally at each trial and shows coverage/uncertainty from available fish; it does not reproduce unexplained draft marks. Trial windows, cohort, metric, normalization, and model contrasts must match an approved analysis before inference is placed on the panel. Present 10sTrace as inconclusive, not a demonstrated positive or absolute limit. |

The archived normalized-vigor script can filter fish, uses mean baseline/response windows and exploratory within/between-condition tests; the current route uses corrected measured-time validity and a frozen cohort. Therefore visual agreement alone is not numerical equivalence. The displayed draft stars and line-plot marks must be regenerated from approved results, not traced from the PNG.

## Figure 3 — learner classification

There is **no draft image**. Written A–G roles are workflow, continuous scores, threshold/stability, learner fractions, individual change, representative trajectories, and all-fish validation. Current corrected per-fish outcomes are inputs, not a frozen learner representation. All seven paper panels remain blocked until Gate L and an independent validation mode are frozen.

## Figure 4 — signed learner dynamics and timing

| Intended panel | Draft evidence | Current counterpart and difference |
| --- | --- | --- |
| A Delay learner dynamics | Draft A is **condition-wide** Pre-Train, Train 1–5, Test 1–3 traces versus control. | Current `block-profile` is a frozen-cohort **0–1 scaled total activity** summary, not signed learner-pooled vigor change. Learner assignment and signed panel data remain undefined. |
| B 3sTrace learner dynamics | Draft B has the same condition-wide rows and a conspicuous post-US rise. | Current 0–1 block profile is only descriptive. CS/trace anticipatory suppression must be separated from the direct paired-training post-US response. |
| C expected-US alignment | Draft C is instead a 10sTrace condition-wide profile. | No independent expected-US aligned learner timing analysis exists. Retain 10sTrace descriptive profiles, but do not relabel draft C or add it silently to the main Delay/3sTrace learner comparison. |
| D timing metrics | Draft D is pooled Delay catch trials. | Current catch profile uses 0–1 activity, not signed onset/peak/center-of-mass/offset estimates with uncertainty. |
| E direct timing comparison | Draft E is pooled 3sTrace catch trials. | No authenticated Delay-versus-3sTrace timing contrast exists. |
| F all-fish validation | Draft F is pooled 10sTrace catch trials. | All-eligible-fish signed validation is missing; the draft is a different condition and calculation. |
| G robustness | **Missing in draft.** | Alternative windows/metrics, controls, and held-out timing checks remain planned. |

The archived script optionally subtracts a pre-CS baseline median and clips the result, so signed negative draft traces are mathematically possible. Current catch/block plots use the **0–1 scaled total-activity** field, mask bins below coverage threshold, pool configured trials within fish, then summarize fish equally with fish/trial counts. They are not numerical replacements. Figure 4 must specify signed panel-data calculation and provenance before rendering. Use distinct guides for CS onset, CS offset, and each condition's expected US time.

Catch provenance has two numbering systems: training catches **11, 25, 39, 45** map to global CS trials **25, 39, 53, 59**. The archived `resolve_catch_trials()` and current experiment definition also pool global **65**, the first Test trial, in “all catch” views. The eventual panel-data sidecar and legend must state this **five-trial** rule.

## Export and remaining gates

Routine PNGs use the same registered data/artist semantics as publication SVG/PDF. `figures/export.py` attaches semantic IDs, artist mappings and source/input hashes in sidecars, embeds SVG provenance, and checks unique/required IDs. A paper export still needs panel-specific physical units and approved source data. The registry's `blocked` state is intentional where those requirements are absent; it is not a silent optional figure.
