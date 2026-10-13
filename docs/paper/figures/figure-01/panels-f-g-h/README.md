# Figure 1 F/G/H

The accepted V12 scope is joint. Frozen code and definitions stay in the joint record; no independent per-panel freeze is created.

Use the owning [figure overview](../README.md), [review](../review.md) and [scoped selection index](../../../../../records/frozen-analyses/selections.md).

## Frozen scientific definition

E preserves the selected raw traces and exact V12 heatmap rows. F/G/H use finite, positive, valid adjacent detected-bout frames, transformed by natural log. Each left-inclusive/right-exclusive 0.5 s bin is the median of eligible original log frames. The baseline is the median of finite bin medians over **[-15,0) s**, with each bin one vote. If `d` is the bin value minus that baseline, the frozen normalization is:

```text
z = 0.7 * d / max(-P10(d_baseline), P90(d_baseline))
```

At least ten finite baseline bins are required. Insufficient/collapsed baselines remain missing; there is no numerical clipping. Display limits are −1…+1 with NaNs black. These descriptive heatmaps use the frozen authenticated cadence-reconstructed timing and original movement-mask recipe. Cleanup does not replace that timing with a new reconstruction or add smoothing. E's raw vigor remains in rad/ms; its heatmap rows retain the separate frozen dimensionless scale.
