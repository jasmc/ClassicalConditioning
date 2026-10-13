# Figure 2: Population responses and inference

D/E B and G LogMedian have accepted scoped freezes. A/B/C/F/H/I remain proposed.

- [Review concerns](review.md)

- [Assembly instructions](assembly.md)
- [Editable legend](legend.md)

Scientific authority: [scoped selection index](../../../../records/frozen-analyses/selections.md). Shared definitions: [methods](../../methods/README.md).

## Figure 2 D/E

Frozen B uses the median natural log of eligible positive valid bout-frame vigor in the response window minus the corresponding baseline median. The baseline is **[-15,0) s**; response windows are **[0,9) s for D** and **[0,13) s for E**. The per-fish/block value is the median over eligible trials. The recorded recipe uses PT10–14, ET65–69 and LT90–94, with minimum eligible trials D=3/E=1.

The accepted family is **36 two-sided tests across D/E: exact sign and Brunner–Munzel t, Holm36**. Exact eligibility, source counts, statistics and presentation are in the unchanged manifests/settings. Older mean-response/baseline ratios, Wilcoxon/Mann–Whitney review recipes, Holm24 and the ratio reference at 1 are legacy within the replaced scope. They are not instructions for current B.


## Figure 2 G

The frozen outcome is `median(ln positive response bout frames) − median(ln positive baseline bout frames)`, summarized across fish. Baseline is [-15,0) s, response [0,9) s, and scheduled CS trials are 5–94. The exact selected record has 29 Delay and 28 control fish; those counts apply to this scoped selection and do not approve other populations.

The selected scientific record preserves the whole-fish trajectory bootstrap (5,000 resamples, seed 10, pointwise percentile 95% intervals) and the recorded block/local/trial model families. Its trial LMM is phase-aware and baseline-adjusted, with fish random intercept/slope and the frozen contrast/reference/correction definitions. Consult the [exact scientific selection](../../../../records/frozen-analyses/figure-02/panel-g/2026-10-09-historical-log-median/manifests/scientific-selection.json) for model terms and multiplicity.

**Frozen limitations remain:** exploratory analysis, heavy residual tails, unavailable Test3 local M/R, and no learning-onset claim. Freezing selects an exact result with its limitations; it does not remove them.

## Panel composition

These roles describe the paper composition; proposed panels have no accepted freeze. Exact selected definitions remain in the linked scope index.

| Panel | Purpose |
|---|---|
| A | Control versus Delay equal-fish pooled signed bout-log-vigor heatmaps, managua_r at −0.25 to +0.25 |
| B | frozen-cohort Control versus 3sTrace CS-aligned population heatmaps |
| C | frozen-cohort Control versus 10sTrace CS-aligned population heatmaps; inconclusive |
| D | Delay selected-block median log positive valid bout-frame response minus baseline (frozen B) |
| E | 3sTrace selected-block median log positive valid bout-frame response minus baseline (frozen B) |
| F | 10sTrace block CR ratios at trials 10–14, 65–69, 90–94; inconclusive |
| G | Delay NativeHistoricalLogMedian response-minus-baseline trajectory with phase-aware LMM (scoped freeze) |
| H | 3sTrace trial-by-trial CR ratio |
| I | 10sTrace trial-by-trial CR ratio; inconclusive |
