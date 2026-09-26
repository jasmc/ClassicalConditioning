# Figure 1 C/D trace versions

**Panel lettering for this decision:** B is the protocol, C is tail angle, and D is vigor. An exploratory eight-panel assembly used D/E for these traces; its files are historical layout variants, not the panel numbering used here. Interpret historical filenames with their sidecars rather than renaming them.

## Shared data path

The paired trace adapter [`render_legacy_ssd_example_traces.py`](../../../scripts/render_legacy_ssd_example_traces.py) verifies completion-marker hashes for corrected frames and candidate activity metrics, and the source-manifest hash for the stimulus events. It selects the same measured frames for C and D around each chosen global CS trial. The shared [`prepare_example_trace_data`](../../../src/classical_conditioning/figures/example_traces.py) routine aligns timestamps to that trial's CS onset. Green guides mark 0 and 10 s; purple guides are actual reinforcer onsets read from the event record. The current review examples use Delay fish `20221115_07` or control fish `20221115_09`, trials 9, 17, 63, 66, 93, and tail point 15. Those are examples, not an approved final fish/trial choice.

## C: measured tail-angle trace

At each corrected frame the code sums `angle0` through `angle15` when all are finite. It subtracts that trial's pre-CS median angle. The paired review plots the resulting frame values in **radians**; no vigor normalization or movement mask is applied to C.

The C review uses the full **−20 to +20 s** window. The paired renderer automatically chooses the C Y range. Separate vector trace artwork made during the temporary D/E assembly converts the same angle to degrees and fixes a shared ±240° range; it is a layout alternative for C, not a new angle calculation.

## D: per-trial scaled log vigor

D starts from one corrected candidate metric. The intended D review now takes its natural log, calculates the trial's pre-CS log-vigor P10 and P90, maps those quantiles to 0 and 1, clips outside values to 0–1, then averages valid scaled frames into 0.5 s bins for display. The metric is a scientific choice:

| Metric ID | What the trace measures | Units |
| --- | --- | --- |
| `tail_length_weighted_angular_l1` | Tail-length-weighted angular activity across the tail | rad/ms |
| `whole_tail_xy_mean_speed_normalized` | Mean whole-tail XY speed normalized by tail length | tail lengths/ms |
| `legacy_distal_angular_speed` | Historical distal angular speed | rad/ms |

[`render_figure1_cd_scaled_log_review.py`](../../../scripts/render_figure1_cd_scaled_log_review.py) saves matched C/D plots, per-frame panel data and provenance for both **[−15, 0) s** and **[−20, 0) s** references. It can compare positive moving-bout frames (log of each bout's mean raw vigor assigned to its frames) with all positive valid frames (log of each frame's raw vigor). Those are different signals; the moving-bout route is the one used in the retained frame-first scaled heatmap comparison. Neither baseline nor signal mask has been frozen.

Earlier D versions are retained for provenance:

| Output family | What changes | Review status |
| --- | --- | --- |
| `outputs/figure1-examples/` | Paired C/D full window: raw frame vigor next to C, −20 to +20 s; automatic nonnegative Y range. | Historical raw-vigor review. |
| `outputs/figure1-D-raw-vigor-zoom/` full Y | Delay/control rows at −20 to +20 s; black raw frame trace, Y ceiling at observed maximum. | Historical raw-vigor view. |
| `outputs/figure1-D-raw-vigor-zoom/` focus Y | Same black raw trace, Y ceiling at the across-panel 99.5th percentile. Orange steps on a separate axis show the selected signed Figure 1E heatmap bins; missing bins remain blank. | Historical raw-vigor view. |
| `previous-*` subfolders beneath the D-only family | Conditional 0–1, all-frame 0–1, bin-first log, short-baseline log, or pre-20 P10/P90 orange bins. | Historical comparisons; they are different heatmap calculations, not different raw black traces. |
| `outputs/figure1-vigor-zoom/withdrawn-scaled-bins/` | Earlier scaled-bin surrogate with a different or unauthenticated recipe. | Historical; the new baseline comparison has explicit inputs and scaling. |

The historical focused D renderer is [`render_figure1_raw_vigor_y_focus.py`](../../../scripts/render_figure1_raw_vigor_y_focus.py). It reads the selected heatmap's saved panel data for the orange axis. That overlay is a different figure construction from the new single-axis scaled-log D.

## What to select before freezing

1. One fish and the global CS trials, shared by C and D.
2. C's X window and whether radians with automatic Y range or degrees with a fixed shared Y range are clearer.
3. One D vigor metric, the moving-bout versus all-valid-frame signal, and [−15, 0) versus [−20, 0) s for scaling.
4. Review the baseline choice in Figure 1/2 heatmaps and other analysis outcomes before setting a repository-wide default. The older profile option called “−15” selects samples **before −15 s**; it does not implement [−15, 0) s.
5. A single output run with source/input hashes, exact settings, sidecars, and reviewed PNG/SVG/PDF exports. Historical review images remain available by their original names.

The [review variant inventory](../../../configs/paper-figures/review-variants.json) and [version comparison](04_REVIEW_VARIANTS.md) preserve the older output-family names. Many of those PNGs are absent from this checkout's `outputs` directory; descriptions here are based on the renderers, registry, and documented sidecars. No C/D version is scientifically frozen by this index.
