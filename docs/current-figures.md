# Current figures: accepted frozen versions

The accepted frozen versions define **current**. An older frozen panel remains current when no later accepted replacement is recorded. Superseded recipes are legacy. A panel without an accepted freeze is **proposed**, even if an exploratory export exists.

This guide summarizes the immutable selections; it does not create, restyle or refreeze anything. Exact definitions, code, settings, exceptions and hashes remain in the linked records. The first release is Figures 1–4. Whole-figure assembly and release approval remain separate from panel freezes.

## Current frozen panels

| Panels | Current accepted version | Exact authority |
|---|---|---|
| **1A–D** | Retained frozen preparation, contingency inset, session protocol and measured tail-angle panels | [A–D selection](../configs/paper-figures/selections/figure1-freeze.json) |
| **1E** | Raw frame-vigor traces with exact frozen V12 heatmap rows | [E selection](../configs/paper-figures/selections/figure1-panel-e-heatmap-rows-freeze-20261009.json), [E records](../records/frozen-analyses/figure-01/panel-e/README.md) |
| **1F/G/H** | Author-selected V12 Delay, 3sTrace and control heatmaps | [V12 selection](../configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json), [F/G/H records](../records/frozen-analyses/figure-01/panels-f-g-h/README.md) |
| **2D/E** | Frozen B bout-log response-minus-baseline block comparisons and corrected statistics | [D/E B selection](../configs/paper-figures/selections/figure2-DE-B-freeze-20261009.json), [shared records](../records/frozen-analyses/figure-02/panels-d-e/README.md) |
| **2G** | Frozen Native Historical LogMedian difference and phase-aware LMM | [G selection](../configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json), [G records](../records/frozen-analyses/figure-02/panel-g/README.md) |

### Figure 1

A shows preparation/illumination; B shows the four within-trial contingencies; C shows the session timeline; D shows the selected measured tail-angle traces. Their accepted source bytes come from the retained A–D freeze. Older approval dates alone do not make those panels obsolete.

E preserves the selected raw traces and exact V12 heatmap rows. F/G/H use finite, positive, valid adjacent detected-bout frames, transformed by natural log. Each left-inclusive/right-exclusive 0.5 s bin is the median of eligible original log frames. The baseline is the median of finite bin medians over **[-15,0) s**, with each bin one vote. If `d` is the bin value minus that baseline, the frozen normalization is:

```text
z = 0.7 * d / max(-P10(d_baseline), P90(d_baseline))
```

At least ten finite baseline bins are required. Insufficient/collapsed baselines remain missing; there is no numerical clipping. Display limits are −1…+1 with NaNs black. These descriptive heatmaps use the frozen authenticated cadence-reconstructed timing and original movement-mask recipe. Cleanup does not replace that timing with a new reconstruction or add smoothing. E's raw vigor remains in rad/ms; its heatmap rows retain the separate frozen dimensionless scale.

### Figure 2 D/E

Frozen B uses the median natural log of eligible positive valid bout-frame vigor in the response window minus the corresponding baseline median. The baseline is **[-15,0) s**; response windows are **[0,9) s for D** and **[0,13) s for E**. The per-fish/block value is the median over eligible trials. The recorded recipe uses PT10–14, ET65–69 and LT90–94, with minimum eligible trials D=3/E=1.

The accepted family is **36 two-sided tests across D/E: exact sign and Brunner–Munzel t, Holm36**. Exact eligibility, source counts, statistics and presentation are in the unchanged manifests/settings. Older mean-response/baseline ratios, Wilcoxon/Mann–Whitney review recipes, Holm24 and the ratio reference at 1 are legacy within the replaced scope. They are not instructions for current B.

### Figure 2 G

The frozen outcome is `median(ln positive response bout frames) − median(ln positive baseline bout frames)`, summarized across fish. Baseline is [-15,0) s, response [0,9) s, and scheduled CS trials are 5–94. The exact selected record has 29 Delay and 28 control fish; those counts apply to this scoped selection and do not approve other populations.

The selected scientific record preserves the whole-fish trajectory bootstrap (5,000 resamples, seed 10, pointwise percentile 95% intervals) and the recorded block/local/trial model families. Its trial LMM is phase-aware and baseline-adjusted, with fish random intercept/slope and the frozen contrast/reference/correction definitions. Consult the [exact scientific selection](../records/frozen-analyses/figure-02/panel-g/2026-10-09-historical-log-median/manifests/scientific-selection.json) for model terms and multiplicity.

**Frozen limitations remain:** exploratory analysis, heavy residual tails, unavailable Test3 local M/R, and no learning-onset claim. Freezing selects an exact result with its limitations; it does not remove them.

## Proposed panels: no accepted freeze recorded here

These retain planned composition only. They have no current frozen method under the selection records above.

| Panels | Proposed purpose | Owning work |
|---|---|---|
| **2A/B/C** | Delay/control, 3sTrace/control and 10sTrace/control population heatmaps | [Figures/reporting](../Plans/09_FIGURES_AND_REPRODUCIBLE_REPORTING.md), [cohorts](../Plans/02_COHORT_IMPLEMENTATION.md) |
| **2F** | 10sTrace/control block comparison | [Analysis/statistics](../Plans/03_ANALYSIS_AND_STATISTICS.md) |
| **2H/I** | 3sTrace and 10sTrace trial trajectories | [Analysis/statistics](../Plans/03_ANALYSIS_AND_STATISTICS.md) |
| **3A–G** | Learner representation, distributions, uncertainty, fractions if validated, paired change, examples and validation | [Learner method](../Plans/05_LEARNER_ANALYSIS_AND_VALIDATION.md), [learner analysis](../Plans/05_LEARNER_ANALYSIS_AND_VALIDATION.md) |
| **4A/B/C** | Delay, 3sTrace and 10sTrace learner-stratified response profiles | [Integrated analysis](../Plans/08_INTEGRATED_ANALYSIS_AND_CR_PROFILES.md) |

Population adaptation of V12 is a proposal until specifically accepted; a single-fish freeze does not select pooling. The old ratio routes for 2F/H/I are legacy proposals, not current accepted definitions. Figures 3–4 depend on their own cohort/learner/validation choices. Do not extrapolate a scoped freeze into those panels.

Supplementary content stays grouped by parent figure: protocol/QC/controls for 1, population coverage and inference diagnostics for 2, learner validation for 3, and catch/movement/coverage/timing controls for 4. Its composition remains in [Plan 11](../Plans/11_SUPPLEMENTARY_FIGURES.md), separate from first-release status.

## Sources, review and reproduction

- [Panel review](panel-review.md) records current concerns and explicit unreviewed placeholders.
- [Paper review](paper-review.md) records cross-panel questions; [scientific decisions](scientific-decisions.md) preserves scoped choices.
- [Freeze index](analysis/figures/freezes/README.md) links current and accepted historical selections.
- [Technical references](analysis/reference/README.md) explain supported commands and assembly scaffolds; they do not override frozen methods.
- [Figure history](analysis/figures/history/README.md) preserves superseded recipes and comparisons.

Exact manifests/settings/code are local under `records/frozen-analyses/`. Figure exports and numerical payloads are indexed on JOAQUIM. Resolve historical paths with `resolve_artifact(reference, expected_sha256=...)`; an unavailable drive or hash mismatch must fail explicitly. A historical whole-assembly source does not override a newer scoped panel selection. Current assembly tooling is `scripts/update_latest_figure_assemblies.py`; old scaffolds describe their own versions.

The versioned [element/style contract](analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md) remains byte-preserved and governs an explicitly requested new freeze. This documentation update introduces no new freeze.
