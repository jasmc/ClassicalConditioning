# Paper figure review versions (24 September 2026)

This compares the output families created during this figure review. It records **what was plotted**, not an approval of the signal or statistics. The authoritative file-level audit trail is each PNG's adjacent `.figure.json` provenance sidecar and the machine [variant inventory](../../../configs/paper-figures/review-variants.json). The supplied legacy draft PNGs are visual references; their exact settings cannot be recovered from pixels.

The current integrated renderer uses the same −20 to 0 s signed bout-log-vigor bins for Figure 1 E and Figure 2 A, then averages fish equally for Figure 2 A. The [selected-pre20 review run](../../../outputs/paper-figures/selected-pre20-review/paper-panel-run.json) is a retained **0–1 scaled** version: its fish and pooled P10/P90 stages both use −20 to 0 s, but its measure differs from Figure 1 E. The earlier [selected-metric run](../../../outputs/paper-figures/selected-metric-review/paper-panel-run.json) retained the previous first-stage profile default (all available bins before −15 s, including times earlier than −20 s). Its Figure 1 E control half is an uncomposed candidate for paper panel G.
The [integrated inference review](../../../outputs/paper-figures/selected-pre20-inference-review/paper-panel-run.json) uses that historical pre-20 scaled heatmap recipe and adds the retained legacy-style D and combined LME G exploratory statistics; it has no validated learning-onset claim.

All three candidate metric versions use `tail_length_weighted_angular_l1`, `whole_tail_xy_mean_speed_normalized`, or `legacy_distal_angular_speed`, unless a row says otherwise. The example recordings were `20221115_07` (Delay) and `20221115_09` (control). Figure 1 traces select global CS trials 9, 17, 63, 66, and 93. All current heatmaps span −20 to +20 s relative to CS onset, with 0.5-s bins; phase rows run Pre-Train 5–14, Train 15–64, Test 65–94, top to bottom. The selected fish and metric are **review examples**, not approved manuscript selections.

## Figure 1 C and D: measured traces

| Output family | Signal and display parameters | Decision |
| --- | --- | --- |
| [`figure1-examples`](../../../outputs/figure1-examples/) | Paired tail angle (distal cumulative angle, trial pre-CS median centered) and **unscaled corrected frame vigor** for the same fish/trials. Tail point 15; x = −20…+20 s; CS 0–10 s. One output per fish × metric. | Current paired C/D review. |
| [`figure1-examples-zoom`](../../../outputs/figure1-examples-zoom/) | Same measured data and fish/trials, but x = −10…+12 s. | Superseded: zooming x obscured baseline comparison. |
| [`figure1-D-raw-vigor-zoom/previous-x-zoom`](../../../outputs/figure1-D-raw-vigor-zoom/previous-x-zoom/) | Earlier D-only raw-vigor full/focus views with x zoom. | Superseded. |
| [`figure1-D-raw-vigor-zoom`](../../../outputs/figure1-D-raw-vigor-zoom/) | D-only x = −20…+20 s. Black = unscaled corrected frame vigor. Full-y uses observed maximum; focus-y caps only the raw axis at the across-panel 99.5th percentile. Orange secondary-axis steps reproduce **the exact signed log heatmap bins** from the selected Figure 1E panel data, fixed at −0.25…+0.25. Consecutive finite 0.5-s bins touch; missing bins remain blank, with a vertical boundary at each finite run. | Current exploratory D comparison; keep black and orange axis meanings distinct. |
| `previous-conditional-bout-only`, `previous-linear-conditional`, `previous-all-frame-continuous`, `previous-bin-first-log`, `previous-short-baseline-log`, `previous-pre20-p10p90` beneath [`figure1-D-raw-vigor-zoom`](../../../outputs/figure1-D-raw-vigor-zoom/) | Same D-only layout and raw black trace, but orange steps came respectively from conditional 0–1 bins, an earlier linear-conditional rendering, all-valid-frame 0–1 bins, bin-first log 0–1 bins, short-baseline log 0–1 bins, and −20…0 s P10/P90 log 0–1 bins. The all-frame version looks continuous because it includes non-bout frames; that is a different estimand. The first two directories may share panel-data lineage but differ in renderer revision. | Historical comparisons, not current signal. Their sidecars preserve the source hashes; exact old renderer revisions are not all available. |
| [`figure1-vigor-zoom/withdrawn-scaled-bins`](../../../outputs/figure1-vigor-zoom/withdrawn-scaled-bins/) | Earlier **scaled-bin** D surrogate, rather than measured raw vigor over time. | Withdrawn after the user requested vigor traces. |

## Figure 1 E: single-fish heatmaps

| Output family | Transform → per-trial reference → masking/binning → palette/range | Decision |
| --- | --- | --- |
| [`signed-log-pre20`](../../../outputs/figure1-heatmaps/signed-log-pre20/) | Positive **moving-bout frames only**; log metric; subtract that fish/trial's median log vigor from −20≤t<0 s; median within each bout, then average available bout values in 0.5-s bins. NaN for no valid bout/baseline. `managua_r`, fixed −0.25…+0.25 (display clipping only). | **Selected review method** from the user's last decision. This is signed baseline-centering, not P10/P90 scaling. |
| [`previous-signed-log`](../../../outputs/figure1-heatmaps/previous-signed-log/) | Same signed bout-log recipe and palette/range, but baseline ended at **−15 s** in the archived implementation. | Superseded baseline. |
| [`previous-pre20-p10p90-log-managua-r`](../../../outputs/figure1-heatmaps/previous-pre20-p10p90-log-managua-r/) | Compute each bout's **mean raw metric**, take its log on moving frames; each fish/trial uses **−20…0 s frame P10/P90**, scales/clips to 0–1 **before** 0.5-s binning. `managua_r`, 0…1. | Rejected in favor of signed log. |
| [`previous-short-baseline-log-managua-r`](../../../outputs/figure1-heatmaps/previous-short-baseline-log-managua-r/) | Earlier log P10/P90 recipe with the shorter pre-CS reference (ending −15 s); 0…1 `managua_r`. | Superseded. |
| [`previous-bin-first-log-managua-r`](../../../outputs/figure1-heatmaps/previous-bin-first-log-managua-r/) | Log corrected **0.5-s conditional bin means first**, then fish/trial P10/P90 scaling against pre-CS bins (short baseline in this version), 0…1 `managua_r`. This order differs from frame-first scaling above. | Superseded; retained to diagnose the blue Delay rows. |
| [root Figure 1 heatmaps](../../../outputs/figure1-heatmaps/) | Earlier corrected conditional intensity, per-fish/trial P10/P90 (0…1), `magma`; no signed centering. | Historical. |
| [`managua-r-review`](../../../outputs/figure1-heatmaps/managua-r-review/) | Palette-only rerender of that saved 0–1 panel data: `managua_r` instead of `magma`, same scaling and mask. | Palette comparison only. |
| [`all-frame-review`](../../../outputs/figure1-heatmaps/all-frame-review/) and its `managua-r-review` child | All valid frames, including non-bout activity, per-trial P10/P90 to 0…1; original `magma`, palette variant `managua_r`. | Diagnostic different signal. |

`managua_r` is the Matplotlib colormap spelled exactly that way in the March legacy code. The draft's phrase “scaled vigor” was ambiguous; the selected Figure 1 method is the signed log definition above. Values outside ±0.25 retain their calculated panel-data values and saturate at the display limits. The heatmap is sparse where movement validity is absent; blue/purple low values are not silently filled from missing cells.

## Figure 2 A: pooled Delay/control heatmap and coverage

| Output family | Signal, normalization order and display | Decision |
| --- | --- | --- |
| [`signed-bout-aligned-review`](../../../outputs/paper-figures/signed-bout-aligned-review/) | Each cohort fish uses the exact Figure 1 E frame/bout calculation: positive moving-bout frames, log vigor, subtract fish/trial −20…0 s median, bout median, then 0.5-s bin mean. Average available fish bins equally; no second scaling. `managua_r`, fixed −0.25…+0.25. Missing fish bins do not become zero. | Current aligned review method; metric and manuscript cohort remain to be approved. |
| [root Figure 2 Delay](../../../outputs/figure2-delay/) | Earlier saved `magma`, 0…1 image. Root PNGs predate the current renderer revision; their sidecars, panel data and current source must not be assumed identical. | Historical image set; keep. |
| [`managua-r-review`](../../../outputs/figure2-delay/managua-r-review/) | Palette-only `managua_r` rerender of saved root panel data, 0…1. | Earlier palette comparison; does not fix scaling. |
| [`log-managua-r`](../../../outputs/figure2-delay/log-managua-r/) | Log of positive corrected **conditional 0.5-s bin mean**, require ≥0.9 valid expected fraction; fish/trial pre-CS P10/P90 (the earlier profile scaler's all-available pre-−15 s reference) without clipping; average fish equally per condition/trial/bin; **second pooled/trial** P10/P90 over −20…0 s; clip 0…1; `managua_r`. | Earlier pooled review. It matches legacy *order* but uses corrected bin means, not the Figure 1 frame/bout-median signal. |
| [`selected-pre20-review`](../../../outputs/paper-figures/selected-pre20-review/) | Same corrected conditional bin-mean and two-stage order, but the **first fish/trial P10/P90 uses only −20≤t<0 s**; the pooled/trial P10/P90 also uses −20≤t<0 s. Log, ≥0.9 coverage, 0…1 `managua_r`. | Superseded 0–1 review setting. |

| [`previous-fish-first-scaled`](../../../outputs/figure2-delay/previous-fish-first-scaled/) | Earlier 0…1 fish-first scaled implementation; saved output only. | Superseded; exact old code revision unavailable. |
| [`previous-all-valid-activity`](../../../outputs/figure2-delay/previous-all-valid-activity/) | Earlier all-valid-frame activity implementation, a different signal from conditional bout vigor. | Diagnostic output only. |
| [`supplementary`](../../../outputs/figure2-delay/supplementary/) | Historical contributing-fish coverage / cohort fish, 0…1 `cividis`. New aligned coverage uses `managua_r` on its own 0…1 fraction scale. | Planned Supplement S5, separate from main A. |

For tail-length-weighted angular L1, a direct join of the two **older integrated** Figure 2 A panel-data tables found **11,208 changed finite cells among 14,400 condition × trial × time cells** (maximum absolute display-value change 1.0). That comparison measures the older baseline correction, not the new signed-bin alignment.

The present `render_legacy_ssd_figure2_delay.py` source uses the same shared signed-bin calculation as Figure 1 E. Earlier root PNGs still show `magma` because they were created before the palette and signal revisions. Use each sidecar's source SHA-256 to identify its rendered revision.

## Figure 2 D and G: response ratios and inference

All saved D/G families use the reviewed `allDelay-full-v1` cohort, a fish-level response/pre-CS ratio, and global CS trials 5–94. D's planned blocks are Late Pre-Train 10–14, Early Test 65–69, and Late Test 90–94. G covers the full trial sequence. These ratios are a different outcome from the heatmap color value.

| Output family | Summary/statistical parameters | Decision |
| --- | --- | --- |
| [root D/G](../../../outputs/figure2-delay/) | Fish-level block medians and cohort median/IQR in D; equal-fish trial median/IQR in G. No inferential marks. | Retained descriptive baseline. |
| [`stats-review`](../../../outputs/figure2-delay/stats-review/) | Whole-fish condition-label permutation of each fish's post-minus-pretrain change, maximum-contrast family-wise correction over two D blocks or 80 G trials. Stars/marks use these exploratory corrected p-values. | Retained previous inference version. |
| [`legacy-stats-lme-review`](../../../outputs/figure2-delay/legacy-stats-lme-review/) D | Legacy-style Holm-corrected Mann–Whitney between conditions and paired Wilcoxon within condition on the current three-block fish ratios; all three metrics. | Retained newer review version. |
| [`legacy-stats-lme-review`](../../../outputs/figure2-delay/legacy-stats-lme-review/) G | Only `tail_length_weighted_angular_l1`: authenticated current LME global condition×block test, Holm block contrasts, simultaneous trial band, plus exploratory legacy local block mean/rate LMEs and BH/FDR trial marks. | Retained combined review. The saved influence diagnostic failed and simultaneous onset was not localized; marks must not be reported as a validated learning onset. |

The legacy draft `Fig2_test1.png` shows stars and a trial significance row, but contains no authenticated cohort/metric/model settings. Its marks cannot be copied. The supplied `Fig1_test2.png` and `Fig4_test1.png` are likewise visual references, not versioned outputs from this repository.

## How to compare a particular pair

Open both PNGs, then inspect each adjacent `.figure.json`: `analysis_recipe`, `source_hash`, `input_artifacts`, `artist_mappings` (signal, colormap and limits), and `reproduction_snippet`. Compare the source hashes and the saved `*_panel-data.parquet` rather than relying on folder names. The [paper-panel command](PAPER_FIGURE_FREEZE.md#rendering-interface) produces a new run manifest. It does not overwrite any of the historical comparison folders by default.
