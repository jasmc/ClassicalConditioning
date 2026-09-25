# Supplementary Figures and Data

**Status:** Active plan; review outputs exist, but final supplementary composition and scientific gates are open.
**Scope:** Complete the data, controls, sensitivity analyses, and figures that support each main Figure 1–4. Final supplementary numbering and panel letters are assigned during paper composition, not used as analysis-step names.
**Figure infrastructure:** [Figures and Reproducible Reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md) owns rendering, provenance, and QC. The [paper figure specification](../docs/analysis/figures/02_PAPER_FIGURE_SPECIFICATION.md) records the current proposed slot map.

## Order and shared rules

Build each supplement from the same authenticated inputs, selected metric, reviewed cohort, outcome definitions, and panel data as its parent figure. Record fish/trial counts, missingness, source hashes, and the descriptive or inferential status of every result. Exploratory comparisons may show all three candidate vigor metrics; final paper panels use the approved single metric consistently. A supplement cannot silently change a fish inclusion rule or replace a missing response with zero.

Prepare the supplementary data for a figure alongside that figure. Figure 4 supplements that depend on learner groups wait for the frozen learner representation from Figure 3. Final publication numbering follows approved content and layout.

## Figure 1 support: preparation and individual behavior

- Show the full CS/US schedule, baseline and trial availability, detector and metric definitions, and raw-data/QC coverage.
- Compare optovin/violet pulse durations, violet-only and sham controls, and any relevant direct US or illumination artifacts using authenticated source records.
- Keep selected-fish tail-angle and vigor traces on the same measured-time axis. Retain the exploratory three-metric comparison and per-fish heatmaps as review data; identify the selected metric, fish, and trials in any final supplement.

## Figure 2 support: population learning

- Export contributing-fish counts, valid-bin fractions, and QC for every pooled heatmap on their own labeled scales.
- Show individual-fish trial trajectories and the context for any selected examples, without changing the reviewed cohort.
- Provide US-aligned paired and unpaired summaries that distinguish direct post-US behavior from anticipatory conditioned responses.
- Preserve fish-level block and trial tables, model diagnostics, uncertainty, multiplicity rules, and sensitivity to the selected metric, cohort, coverage, and outcome. Supplementary claims about learning onset or loss of an established response require prespecified definitions and passing inference gates in [analysis and statistics](./02_ANALYSIS_AND_STATISTICS.md) and [learning-onset completion](./03_LEARNING_ONSET_IMPLEMENTATION.md).

## Figure 3 support: individual learning

- Export the full distribution of continuous fish effects or scores, their uncertainty and eligibility reasons, and any approved classification threshold and calibration evidence.
- Show method and representation sensitivity, control false-positive checks, and held-out or cross-fitted validation when an inferential learner claim is made.
- Retain all eligible fish in validation summaries; never treat the same trials used to define a learner label as independent confirmation of that label.

## Figure 4 support: conditioned-response profiles and timing

- Export individual catch-trial profiles, movement probability, signed-signal coverage, and red-CS or visual-control comparisons where approved. The pooled catch set and any sensitivity set must state their global trial IDs explicitly.
- Keep same-data learner-stratified block and catch curves descriptive. A response-timing claim needs evaluation trials disjoint from those that determined each fish's label, or a reviewed cross-fitted or independent-cohort design.

### Independent response timing

Use authenticated Figure 4 trial-bin and fish-bin tables, the frozen learner manifest, recorded paired-training US times, and explicitly identified evaluation trials. Analyze anticipatory responses only before the expected US. No-bout signed-vigor bins remain missing; movement probability supplies the complementary occurrence outcome.

Before fitting, freeze numerical definitions for response onset, peak, negative-response center, and offset, plus a rule for traces with no measurable response. Report each estimate relative to CS onset and expected US. Do not estimate conditioned-response timing from post-US paired-training activity. Verify classifier-training and evaluation trial sets are disjoint for each fish. Resample fish for group uncertainty, with trials resampled within fish where justified. Compare Delay and 3sTrace only after the recorded 3sTrace US time agrees with the approved protocol. Include controls, fish without an approved learner label, all-fish results, and alternate-metric sensitivity.

Save fish-level estimates, group summaries, uncertainty draws, trial-set provenance, diagnostics, and the resulting supplementary panel data with source hashes. Keep Figure 4 free of timing p-values or onset/offset claims until definitions, independence, and diagnostics are reviewed. An unresolved US timestamp or no measurable response gets an explicit missing reason.

## Completion gate

For each main figure, list its approved supplements, data tables, calculation, renderer, inputs, cohort/metric identities, counts, inference status, visual review, and final manuscript cross-reference. Close a proposed supplement only when its parent figure and relevant scientific gates are approved; record omitted proposals with reasons rather than silently dropping them.
