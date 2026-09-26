# Figure 1 C/D baseline comparison

These are review outputs for Delay recording `20221115_07`, global CS trials 9, 17, 63, 66 and 93, and `tail_length_weighted_angular_l1`. C is unchanged measured tail angle. D is per-trial log-vigor P10/P90 scaling, clipped to 0–1, then averaged in 0.5 s bins. The grey region shows the quantile reference interval. Each PNG has an adjacent `.figure.json` source/input hash record and a per-frame Parquet table.

| Signal | [−15, 0) s | [−20, 0) s |
| --- | --- | --- |
| Positive moving-bout frames, log bout-mean raw vigor | [PNG](figure-1-CD-scaled-log_20221115_07_tail_length_weighted_angular_l1_moving-bouts_pre15.png) | [PNG](figure-1-CD-scaled-log_20221115_07_tail_length_weighted_angular_l1_moving-bouts_pre20.png) |
| All positive valid frames, log raw frame vigor | [PNG](figure-1-CD-scaled-log_20221115_07_tail_length_weighted_angular_l1_all-valid_pre15.png) | [PNG](figure-1-CD-scaled-log_20221115_07_tail_length_weighted_angular_l1_all-valid_pre20.png) |

The moving-bout signal has gaps where no eligible frames occur. The all-valid signal is more continuous and can be dominated visually by repeated activity. These are different measures. Neither signal nor baseline has been frozen. See the [baseline impact audit](../../docs/analysis/figures/BASELINE_WINDOW_REVIEW.md) before propagating a choice to heatmaps or other analyses.
