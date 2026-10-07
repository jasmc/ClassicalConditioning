# F/G/H correction: trial baseline of the displayed signal

The user approved moving the reference to the displayed-bin stage after observing a blue baseline in later rows. This supersedes the prior frame-reference-centred F/G/H review for this requested presentation. Earlier data, panels and builders remain preserved.

Builder: `remake_fgh_display_centred.py`; numerical implementation: `display_bin_centring.py`; regression tests: `test_display_bin_centring.py`, all in this task's review folder. New artifacts are directly at `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/fgh-trial-baseline-bin-centred-20261007/`. Only F, G and H were produced. No full assembly or frozen assets were modified.

## New scientific definition

For each trial j:

1. Retain the verified reconstructed acquisition clock, selected legacy_distal_angular_speed metric in rad/ms, detector and eligible-frame mask from the independent F/G/H reconstruction. Its source manifest and panel-data hashes are verified before reuse.
2. On finite positive eligible moving frames, calculate ln(vigor), take each bout's median log across its eligible support in the displayed [−20,20) trial window, and repeat that number only on those eligible frames.
3. Average the finite repeated bout values in each left-closed/right-open 0.5 s bin. Weight by contributing eligible-frame count. Empty bins remain NaN. This produces uncentred log-bin values U[j,k].
4. Set R[j] to the median of FINITE U[j,k] bins entirely within the SAME approved [−15,0) window. Each supported baseline bin contributes one equally weighted observation, irrespective of the number of eligible frames inside it.
5. Display H[j,k] = U[j,k] − R[j]. A missing reference makes the whole centred row NaN; no trial among these examples has a missing reference.
6. Render H using managua_r at [−0.25,+0.25], with black for NaN. Colours saturate outside the range; stored values remain uncapped.

The renderer consumes the previously independently rebuilt frame-reference-centred bins S and old frame reference b. Since U=S+b, the corrected value can equivalently be evaluated as `S − median(S_baseline_bins)`. This is a replacement reference, not two independent normalizations. The new Parquets retain original S, recover U, and store the new R and correction offset so the change is fully auditable. Baseline counts and all contribution counts are retained. There is no P10–P90 division, min–max scaling, resampling or smoothing in this correction.

Median baseline centring guarantees the median supported baseline cell is zero. It does not force every cell, or their arithmetic mean, to zero. Positive and negative baseline variations remain. Likewise, response-region changes outside the baseline may still produce many cells of one sign.

This is a scientifically explicit change in reference weighting: from eligible FRAME logs before bout aggregation to supported DISPLAYED BINS after aggregation. It is appropriate for the user's requested row-wise displayed baseline centring. It does not establish that equal-bin weighting is an optimal inference recipe. Frame coverage is exported separately; low-support cells receive equal reference weight here because the reference is defined over displayed cells.

## Critical comparison to the requested legacy scripts

`legacy/scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py` presently reconstructs camera timing, resamples to 700 FPS, forms cumulative angles, applies width-3 spatial and 10-frame temporal means, then calculates an unwrapped absolute angle difference in deg/ms at the expected rate. The current figure timing-only recipe uses original acquired frames and a wrapped distal-angle difference in rad/ms. These pipelines must not be labelled bit-identical. This figure correction does not silently adopt the legacy filtering/resampling or change the pending preprocessing audit.

The preprocessing script's stored scaled column (around lines 281–305) chooses baseline observations with `time_col < -baseline_window_frames`; it divides by P90−P10 of raw vigor. Its grouping key is Trial number alone at that stage, rather than an explicitly CS-only heatmap key. The independently recomputed plotting route is therefore a different operation from simply displaying that stored scaled column.

The scaled heatmap routines in the preprocessing script (around lines 794–849) and `legacy/scripts/2_ExampleFishPlotting.py` (around lines 1315–1355) mask off-bout raw vigor with NaN, replace bout values by mean RAW vigor, choose reference observations with `Trial time (s) < -baseline_window`, divide by trial P90−P10, and clip stored plot values to [0,1]. The helper configuration sets baseline_window=15. Thus those conditions select times EARLIER than −15 s, not the approved [−15,0) interval. Their mean-raw-bout summaries, range division, clipping and plotted frame-time grids also differ from the present log-bout-median / half-second-bin recipe. They are useful historical context, not a suitable implementation to copy for the current requirement.

The plotting code skips empty/degenerate reference trials with `continue`, leaving their original data_plot entries in place. Those rows may consequently mix unscaled raw values with normalized rows. The corrected builder instead makes absent-reference rows explicitly NaN. No missing-reference trial occurs in this dataset, so the policy is covered by a regression test rather than claimed as an observed problem here.

The revised method retains the useful legacy separation of off-bout missingness and movement amplitude, and explicit trial-specific references. It rejects copying reference bounds and transformations that conflict with the current instructions.

## Verification

- Four regression tests pass: independent trial references and boundary exclusion; finite-only/missing-reference behavior; preservation of input/support and within-trial differences; rejection of duplicated bins/inconsistent source references.
- All 270 displayed trial baselines have median zero within 1e-12.
- All 21,600 bin support positions and contribution counts are unchanged.
- Every corrected finite bin differs from its original by exactly one constant for that trial. Within-trial contrasts and timing are preserved.
- G trial 93: previous baseline-bin median −0.271605, corrected median 0; 27 supported baseline bins, 13 negative, 13 positive and one zero. Its arithmetic baseline mean is +0.147749; this is compatible with a zero MEDIAN and was not artificially adjusted.
- Vector SVG/PDF and PNG outputs were rendered and visually inspected. A before/after PNG uses the same palette, limits, layout, fish and bins to isolate the reference change.

The old reference and the new reference are explicitly named in the data. Existing Panel E binned bars retain their old frame reference, so they no longer have the same numerical values as these corrected F heatmap bins. No Panel E change was made within this strictly F/G/H task. Any later E-to-F presentation must account for the reference change before claiming exact bin-value correspondence.

Selection/freeze and broader fish/population analysis remain deferred. This change is scoped to the requested F/G/H presentation and does not mutate shared preprocessing or other analysis routes.
