# Matched pre-CS baseline heatmap review

These are exploratory exports for the `tail_length_weighted_angular_l1` metric. Both variants use positive moving-bout log vigor centred on each trial's baseline median. Figure 1 E/G shows Delay `20221115_07` and Control `20221115_09`; Figure 2A pools 29 Delay and 28 Control fish equally. All heatmap images share the `managua_r` palette and ±0.25 signed-log display limits.

| Baseline | Figure 1 E/G | Figure 2A | Figure 2A coverage |
| --- | --- | --- | --- |
| [−15, 0) s | [PNG](figure1-pre15/figure-1-EG_delay-control_tail_length_weighted_angular_l1.png) | [PNG](figure2-pre15/figure-2A_delay-control_tail_length_weighted_angular_l1.png) | [PNG](figure2-pre15/supplementary/figure-2A-coverage_delay-control_tail_length_weighted_angular_l1.png) |
| [−20, 0) s | [PNG](figure1-pre20/figure-1-EG_delay-control_tail_length_weighted_angular_l1.png) | [PNG](figure2-pre20/figure-2A_delay-control_tail_length_weighted_angular_l1.png) | [PNG](figure2-pre20/supplementary/figure-2A-coverage_delay-control_tail_length_weighted_angular_l1.png) |

Each PNG has a `.figure.json` sidecar and a matching Parquet panel-data file in its directory. Figure 2D/G exports appear in both Figure 2 directories because the adapter generates a complete descriptive set; they use the same stored trial outcomes in both runs and do not depend on this heatmap baseline option. The paper render plan still specifies [−20, 0) s for these signed heatmaps. The baseline decision remains open.

The [baseline audit](../../docs/analysis/figures/BASELINE_WINDOW_REVIEW.md) records the quantitative comparison and other repository baseline paths. Figure 1 C/D's per-trial scaled-log comparison is in [its own directory](../figure1-cd-baseline-review/).
