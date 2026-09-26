# Paper figure creation and provenance, panel by panel

This file answers **what was used, how a panel is calculated, which code renders it, and what remains before publication**. The panel map is the proposed [paper specification](02_PAPER_FIGURE_SPECIFICATION.md) and [machine registry](../../../configs/paper-figures/behavior-paper.json). All registry panels are currently `blocked`; a review renderer existing is not equivalent to a completed manuscript panel. The separate [review variant registry](../../../configs/paper-figures/review-variants.json) identifies historical images and parameters.

## Shared construction path

1. Read and hash raw camera, tail-tracking, and stimulus-control triplets; produce lossless intake and corrected measured-time frames. Derive three candidate metrics and one shared movement/bout state. [02 pipeline](../02_CURRENT_PIPELINE.md) and [03 parameters](../03_ANALYSIS_PARAMETER_INDEX.md) give the stage settings.
2. For a population panel, authenticate a frozen cohort and each contributing fish's corrected temporal or trial-outcome artifact. Form per-fish values before equal-fish aggregation; retain valid-sample and contributing-fish counts. Do not infer cohort membership from a plotted image.
3. Keep the measures separate: **signed bout-log-vigor** is moving-frame log vigor centered on each trial's [−20, 0) s median, then bout median and 0.5 s bin mean; the paper Figure 1 F/H and Figure 2 A review share these fish/trial bins. **Response/baseline ratio** comes from per-trial outcome tables with [−15, 0) s baseline and experiment-specific response ends at 9/13/20 s. Older Trace artifacts with the former 9 s default are rejected by current loaders and must be rebuilt. Routine **0–1 total-activity** profiles are another descriptive family. None of these scales can substitute for another.
4. Render from saved, authenticated panel data; inspect the adjacent `.figure.json` for source code hash, input hashes, cohort/metric identity, artist mappings, units, and reproduction command. Publication SVG/PDF uses [export.py](../../../src/classical_conditioning/figures/export.py). A `paper-panel-run.json` records what `render-paper-panels` actually attempted and its blocked panels. The authoritative analysis output and figure files belong together.

The supported integrated command and example are in [02 specification § Rendering interface](02_PAPER_FIGURE_SPECIFICATION.md#rendering-interface). Its default review set renders Figure 1 D/E and paired Delay/control heatmaps, plus Delay Figure 2 A/D/G and separate heatmap coverage. The optional inference overlay is exploratory. Figure 4 uses `figure4-analyze` then `figure4-render` with three reviewed cohorts and a frozen learner manifest.

## Figure 1 — protocol and individual behavior

| Panel | How it is created, and source | Current result / missing input |
| --- | --- | --- |
| **1A** preparation | Approved apparatus artwork must depict the head-fixed larva, basal illumination, green CS and violet/optovin US; cross-check against [`experiments.py`](../../../src/classical_conditioning/config/experiments.py). | Artwork source and approval absent; no code renderer can reconstruct it from tracking data. |
| **1B** condition timing | Compare the 10 s CS and paired US onsets at 9, 13 and 20 s for Delay, 3sTrace and 10sTrace; show unpaired control without a fixed US latency. The vector source is described in [Figure 1B redesign](FIGURE1_PANEL_B_REDESIGN.md). | Exploratory vector review exists; final artwork approval remains open. |
| **1C** session protocol | Show Pre-Train, Train and Test phases, trial counts and the 50/100 ms US legend from `ExperimentSpec` and the trial map. The assembly source is in [Figure 1 assembly](FIGURE1_ASSEMBLY.md). | Session timeline artwork exists as a review source; final protocol approval remains open. |
| **1D** tail angle | For selected fish and global CS trials, [`example_traces.py`](../../../src/classical_conditioning/figures/example_traces.py) reads corrected frames, centers distal cumulative angle on the trial pre-CS median, and plots measured frames with event guides. Integrated adapter: [`render_legacy_ssd_example_traces.py`](../../../scripts/render_legacy_ssd_example_traces.py). | Paired D/E review exists; representative fish/trials and final composition remain open. |
| **1E** vigor trace | Same fish/trials and time axis as 1D; plot one selected corrected frame metric. The optional orange step overlay is the exact 1F/H signed heatmap bins on a separately labeled axis; invalid gaps remain open. Renderer above; focused review: [`render_figure1_raw_vigor_y_focus.py`](../../../scripts/render_figure1_raw_vigor_y_focus.py). | Review exists; metric and examples need selection. Frame vigor is not silently rescaled to the heatmap color range. |
| **1F** Delay fish | [`signed_bout_heatmap.py`](../../../src/classical_conditioning/figures/signed_bout_heatmap.py) computes moving positive bout-frame log vigor relative to each trial's [−20, 0) s median, then 0.5 s signed bins. [`render_legacy_ssd_example_heatmaps.py`](../../../scripts/render_legacy_ssd_example_heatmaps.py) draws the provisional Delay/control pair with `managua_r`, fixed −0.25…+0.25, NaNs distinct. | Paired review exists for provisional fish `20221115_07`/`20221115_09`; single final panel, metric and example approval are open. |
| **1G** 3sTrace fish | Same signed single-fish method as 1F, applied to a prespecified 3sTrace fish and authenticated corrected artifacts. | No final fish selection or integrated paper-panel renderer. |
| **1H** unpaired control fish | Same signed method as 1F for a prespecified control fish; retain coverage mask and trial order. | Control half of the paired F/H review is a candidate, but no separately composed final H. |

## Figure 2 — matched population comparisons

Each column compares one conditioned assay with its matched control. The intended 10sTrace interpretation is inconclusive. See [06 cohort aggregation](../06_COHORT_AGGREGATION.md) for ratio semantics and coverage.

| Panel | How it is created, and source | Current result / missing input |
| --- | --- | --- |
| **2A** Delay/control heatmaps | Authenticate reviewed Delay cohort. Compute the **same signed fish/trial bins as 1F/H**, then average available fish equally at each condition/trial/bin; `managua_r` −0.25…+0.25. [`render_legacy_ssd_figure2_delay.py`](../../../scripts/render_legacy_ssd_figure2_delay.py) uses [`signed_bout_heatmap.py`](../../../src/classical_conditioning/figures/signed_bout_heatmap.py). Export contributing-fish fraction as separate supporting data. | Delay review exists. Final metric and paper cohort are unapproved. The routine all-valid-frame 0–1 heatmap in [`population_heatmap.py`](../../../src/classical_conditioning/figures/population_heatmap.py) is a distinct diagnostic. |
| **2B** 3sTrace/control heatmaps | Apply the approved population signal and equal-fish construction to an authenticated matched 3sTrace cohort; retain coverage. | Routine 0–1 descriptive heatmap exists; signed paper-equivalent panel and selections remain open. |
| **2C** 10sTrace/control heatmaps | Same construction with authenticated 10sTrace/control cohort; show valid coverage and inconclusive result clearly. | Routine descriptive precursor exists; matched paper cohort, signal and interpretation await review. |
| **2D** Delay selected-block ratios | Take finite response/[−15,0) baseline ratios from selected metric trial outcomes, then per-fish medians at 10–14, 65–69, 90–94 and condition median/IQR. [`cohort_response.py`](../../../src/classical_conditioning/figures/cohort_response.py) and Delay review adapter above render fish lines and summary. | Descriptive review exists. [`render_figure2_legacy_stats_lme_review.py`](../../../scripts/render_figure2_legacy_stats_lme_review.py) can add **exploratory** legacy-style test marks; approved inference is absent. |
| **2E** 3sTrace selected-block ratios | Same selected global trial groups and equal-fish summary as 2D; the experiment response window is [0, 13) s. | General cohort renderer exists; final matched cohort, metric, and inference missing. Rebuild any old 9 s Trace artifacts. |
| **2F** 10sTrace selected-block ratios | Same as 2D; the experiment response window is [0, 20) s and the panel is inconclusive. | General descriptive renderer exists; final inputs and inference missing. Rebuild any old 9 s Trace artifacts. |
| **2G** Delay trial ratios | At every scheduled CS trial, summarize per-fish response/baseline ratios by condition with coverage and uncertainty. [`cohort_response.py`](../../../src/classical_conditioning/figures/cohort_response.py) and Delay adapter render the descriptive review. Optional saved [`learning-onset`](../07_LEARNING_ONSET_LME.md) tables can annotate global/block/trial contrasts via the inference adapter. | Descriptive and exploratory LME reviews exist only for the saved tail-length-weighted angular L1 fit; its influence gate failed and simultaneous onset was not localized. Do not copy review marks as paper results. |
| **2H** 3sTrace trial ratios | Same fish-weighted trajectory method as 2G, using [0, 13) s response; approved condition-aware model output would supply inferential marks. | Descriptive route exists; paper cohort, metric, model and diagnostics missing. |
| **2I** 10sTrace trial ratios | Same as 2G; present inconclusive evidence and coverage. | Descriptive route exists; approved model and interpretation missing. |

## Figure 3 — learner representation

These are proposed **outputs of a frozen Gate L representation and validation protocol**, not reconstructions from old draft pixels. A classifier and class-fraction panels are conditional on Gate L approving categories. No Figure 3 manuscript renderer or approved learner artifact exists. [`legacy_learners.py`](../../../src/classical_conditioning/analysis/legacy_learners.py) and [legacy references](../legacy/README.md) preserve historical behavior only.

| Panel | Planned construction and required artifact |
| --- | --- |
| **3A** | Draw the approved learner-representation workflow from the metric, eligibility, validation split, and frozen manifest; include a threshold only if approved. |
| **3B** | Plot continuous frozen fish learning scores by assay/condition, including eligible-fish counts. |
| **3C** | Show uncertainty and stability assessments; add a frozen threshold and false-positive controls only if categories are approved. |
| **3D** | If categories are approved, compute learner fractions from eligible fish assignments by assay/condition with fish-level uncertainty; otherwise revise this panel around the approved continuous representation. |
| **3E** | Plot all-fish paired pre-to-late change from reviewed cohort outcomes, with explicit metric and inclusion rule. |
| **3F** | Render prespecified positive, negative, intermediate, and borderline trajectories from authenticated fish outcomes and example-selection record. |
| **3G** | Summarize the all-eligible-fish validation analysis and its independent validation mode. |

## Figure 4 — learner-stratified signed CR profiles

[`figure4-analyze`](06_FIGURE4_LEARNER_PROFILES.md) authenticates **three** reviewed cohorts and a frozen classifier manifest, verifies paired-training US timing, and saves trial, fish, group, and sample-flow tables. For each CS trial it makes 0.5 s signed bout-log-vigor bins over −20…+20 s with [−20, 0) s baseline, masks bins below 0.9 valid expected-frame coverage, retains no-bout bins as missing, forms trial medians within fish, then equal-fish group medians and IQRs. [`figure4.py`](../../../src/classical_conditioning/figures/figure4.py) renders the nine declared ten-trial blocks plus one pooled-catch row. The four displayed groups are conditioned learner/nonlearner and control learner-flagged/nonlearner. Same-data strata are descriptive, with no independent p-values.

| Panel | Assay and output | Current gate |
| --- | --- | --- |
| **4A** | Delay; `figure-4_<metric>` under `allDelay/` from `figure4-render`. | Frozen Gate L manifest and complete authenticated analysis. |
| **4B** | 3sTrace; same output under `all3sTrace/`. | Gate L plus reconciliation of configured versus raw paired-US timing. |
| **4C** | 10sTrace; same output under `all10sTrace/`. | Gate L and complete authenticated analysis. |

The renderer also exports individual catches, movement probability, and signed-signal coverage for each assay as supplementary *data/figure families*. Their final supplementary numbering is not frozen. The pooled catch row uses global CS 25, 39, 53, 59, and 65; the first four correspond to training catch numbers 11, 25, 39, and 45. Expected-US guides on catch rows reflect verified paired-training expectation, not an observed catch-trial US.

## Supplementary support by main figure

The [paper specification](02_PAPER_FIGURE_SPECIFICATION.md#supplementary-figures-and-data)
and [Plan 10](../../../Plans/10_SUPPLEMENTARY_FIGURES.md) group supporting
outputs by the main figure they explain. Final supplementary numbers and panel
letters are assigned after analysis and layout review.

| Parent figure | Creation route and source | Current state |
| --- | --- | --- |
| **Figure 1: US and illumination controls** | Use authenticated US-aligned corrected profiles, optovin/violet pulse-length records, and violet-only/sham conditions. Compare prespecified 50/100-ms US conditions and identify direct illumination artifacts. | Technical/control approvals and final composition absent. |
| **Figure 1: protocol, baseline, and individual QC** | Draw the detailed schedule from `ExperimentSpec`/trial map; show pre-training baseline, detector/metric definitions from [05](../05_METRICS_AND_BOUTS.md), per-fish three-metric review, and trial/recording coverage. | Components exist; reviewed artwork, selected examples, and cohort audit absent. |
| **Figure 2: US response** | Use CS/US temporal outcome tables and paired/unpaired condition labels; separate direct post-US behavior from anticipatory CR. | Descriptive profiles exist; matched analysis and interpretation open. |
| **Figure 2: population heatmap coverage** | From each Figure 2 heatmap panel data, render contributing-fish fraction on its own 0…1 `managua_r` scale and show counts. Delay review writes `supplementary/figure-2A-coverage_*` via [`render_legacy_ssd_figure2_delay.py`](../../../scripts/render_legacy_ssd_figure2_delay.py). | Delay review exists; Trace cohorts and final cohort/metric remain. |
| **Figure 2: trajectories and inference checks** | Render authenticated per-fish trial outcomes, example-selection context, block/trial diagnostics, and sensitivity with cohort membership and eligibility visible. | Trial outcomes and Delay review traces exist; approved inference and any extinction estimator remain open. |
| **Figure 3: learner validation** | Render fish scores, eligibility, uncertainty, control calibration, method sensitivity, and independent validation from the frozen Gate L artifacts. | Learner representation and final validation artifacts are open. |
| **Figure 4: catches and response timing** | Use configured catch IDs (global 25/39/53/59 and, for the pooled set, 65). Figure 4 already renders individual-catch signed/movement/coverage families. Independent onset/peak/offset timing and catch-selection sensitivity follow the [supplementary plan](../../../Plans/10_SUPPLEMENTARY_FIGURES.md). | Descriptive catches exist; independent timing estimator and final composition open. |
| **Figure 4: visual controls** | Compare approved red-CS/visual-control cohorts and metric-consistent outcomes with cohort and protocol provenance. | Required cohort and analysis approval absent. |

## Checking a particular exported figure

Open its `.figure.json` and record `figure_id`, `analysis_recipe`, `source_file`/`source_hash`, `input_artifacts` hashes, `cohort_hash`, `analysis_identity`, `artist_mappings`, and `reproduction_snippet`. Check `paper-panel-run.json` for the actual command and status, and the analysis completion marker for table hashes. For a planned panel with no exported sidecar, the table above is a construction specification, not evidence that the figure was produced.
