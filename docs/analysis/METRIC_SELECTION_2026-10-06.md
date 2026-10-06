# Frozen paper activity metric — 2026-10-06

The author selected **tail bend angular speed** as the activity metric for the
final paper on 2026-10-06. Its stable storage/CLI ID is
`legacy_distal_angular_speed`; its frame column is
`legacy_distal_angular_speed_rad_per_ms`. This is the existing corrected,
measured-time implementation of the legacy-derived metric, not a switch back
to historical interpolated timing or legacy preprocessing.

## Definition and name

For frame t, let B(t) be the sum of the local tail angles. Activity is
`abs(atan2(sin(B(t)-B(t-1)), cos(B(t)-B(t-1)))) / delta_time_ms`.
The units are rad/ms. The existing validity, gap and placeholder policies apply.
The name includes **angular speed** because the selected metric is a derivative;
"bend angle" alone would describe B(t), a different quantity. Opposing local
angle changes can cancel in B(t). The calculation and stable IDs are preserved.

## Evidence supporting the author's choice

The saved comparison includes all 57 processed fish from the Delay project
(29 Delay, 28 controls). It compares Pre-Train CS trials 5–14 with a single pooled
window containing Train trials 60–64 and Test trials 65–69. Both windows contain
10 trials per fish. Within each window, the score is the ratio of the mean
CS total activity to the mean activity in each trial's preceding baseline.
The source trial outcomes use baseline [−15, 0) s and CS response [0, 9) s;
all 57 source summaries were checked against their saved Parquet hashes and
identical window settings. This is zero-inclusive total activity, not
movement-only intensity.

Suppression is `log2(Pre ratio) - log2(pooled late ratio)`. The descriptive
contrast is the equal-fish mean suppression in Delay minus that in controls.
Bootstrap sampling resamples fish separately within conditions, keeping each
fish's three metric values together (10,000 replicates; seed 20261006).

| Metric | Mean Delay minus control | Pointwise 95% bootstrap interval | Difference of group medians |
| --- | ---: | ---: | ---: |
| Weighted angular L1 | 0.1199 | 0.0597–0.1841 | 0.0443 |
| Whole-tail XY | 0.2427 | 0.1194–0.3707 | 0.1535 |
| **Tail bend angular speed** | **0.2163** | **0.1138–0.3210** | **0.1557** |

Tail bend angular speed reflects a larger suppression than angular L1
(paired difference 0.0964; interval 0.0389–0.1615). XY has the largest mean
contrast, but its advantage over the selected metric is unresolved
(0.0264; interval −0.0456–0.0992). The selected metric has the slightly larger
median contrast and retains continuity with the historical bend-angle measure.
The author chose it after reviewing this evidence. This is an exploratory,
data-informed selection, not independent confirmation of a conditioned response
or proof that it is uniquely superior. The intervals are not multiplicity adjusted.

## Code and analysis consequences

- Routine pipeline metric and assessment defaults are the selected metric.
- Paper render plans enforce it. CLI analyses and standalone renderers default
  to it; explicit alternative metrics remain available for sensitivity work.
- All three frame-level metrics remain in saved tables and comparison tools.
  Removing them would destroy review/sensitivity capabilities.
- Existing L1 models, learner labels and figures remain historical artifacts.
  They must be rerun with the selected metric before use as selected-metric paper
  results. The saved L1-only inference renderer continues to reject a mismatched
  metric; it cannot supply inference for the selected metric.
- Scientific selection of the metric does not itself approve preprocessing,
  detector parameters, cohort inclusion, statistical models or learner rules.

## Preserved evidence

The bundle is [outputs/pre-vs-late-train-early-test-review](../../outputs/pre-vs-late-train-early-test-review/).
It contains the selected trial inputs, per-fish window values and paired changes,
both PNG/PDF figures, magnitude table, paired-bootstrap comparison JSON, and a
SHA-256 manifest recording original input paths and evidence hashes.

Reproduce the descriptive comparisons with
`scripts/plot_train_test_transition_metrics.py` followed by
`scripts/plot_metric_response_magnitude.py`. The original data root is
`F:\Digested Data\allDelay-full-v1\Processed data`.
