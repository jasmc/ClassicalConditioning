# Figure 1: Protocol and individual behavioral examples

A–D, E and F/G/H V12 have accepted scoped freezes. The whole figure is not frozen.

- [Review concerns](review.md)

- [Current presentation state](current-state.json): frozen display and active draft, separately authenticated.
- [Assembly instructions](assembly.md)
- [Editable legend](legend.md)

Scientific authority: [scoped selection index](../../../../records/frozen-analyses/selections.md). Shared definitions: [methods](../../methods/README.md).

## Figure 1

A shows preparation/illumination; B shows the four within-trial contingencies; C shows the session timeline; D shows the selected measured tail-angle traces. Their accepted source bytes come from the retained A–D freeze. Older approval dates alone do not make those panels obsolete.

E preserves the selected raw traces and exact V12 heatmap rows. F/G/H use finite, positive, valid adjacent detected-bout frames, transformed by natural log. Each left-inclusive/right-exclusive 0.5 s bin is the median of eligible original log frames. The baseline is the median of finite bin medians over **[-15,0) s**, with each bin one vote. If `d` is the bin value minus that baseline, the frozen normalization is:

```text
z = 0.7 * d / max(-P10(d_baseline), P90(d_baseline))
```

At least ten finite baseline bins are required. Insufficient/collapsed baselines remain missing; there is no numerical clipping. Display limits are −1…+1 with NaNs black. These descriptive heatmaps use the frozen authenticated cadence-reconstructed timing and original movement-mask recipe. Cleanup does not replace that timing with a new reconstruction or add smoothing. E's raw vigor remains in rad/ms; its heatmap rows retain the separate frozen dimensionless scale.

## Panel composition

These roles describe the paper composition; proposed panels have no accepted freeze. Exact selected definitions remain in the linked scope index.

| Panel | Purpose |
|---|---|
| A | head-fixed preparation, CS, basal illumination and optovin US |
| B | within-trial CS–US protocol contingencies: Delay, 3sTrace, 10sTrace and unpaired control |
| C | across-session protocol, phase trial counts and US pulse durations |
| D | representative measured tail angle for selected fish and trials |
| E | matching raw frame vigor trace with exact selected V12 heatmap rows on a separate axis |
| F | Delay single-fish V12 baseline-spread normalized bout-log heatmap |
| G | 3sTrace single-fish V12 baseline-spread normalized bout-log heatmap |
| H | unpaired-control single-fish V12 baseline-spread normalized bout-log heatmap |
