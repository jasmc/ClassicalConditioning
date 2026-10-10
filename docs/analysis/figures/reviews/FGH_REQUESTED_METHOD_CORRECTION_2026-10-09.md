# F/G/H requested method corrections

The user requested corrections to the actual analysis described in the HTML summary. These are processing changes, so the existing figures must not be relabelled as though they already implement them. Earlier exports and freeze manifests remain historical records.

## Bout medians

Current C/D sample versions group eligible log vigor by bout within the displayed [-20,+20) s crop for each trial. The requested method instead determines each bout's median from its eligible support over the entire trial, repeats that median on all eligible timepoints belonging to that bout, and only then crops values for the heatmap.

The protocol's Cycle.Beg/End describes the approximately 10 s CS pulse, not a full trial. Trial boundaries therefore need a defined convention. The user was asked whether to use CS-20 s to next-CS-20 s (last trial to recording end), or each complete detected bout regardless of the plotting crop.

Version 2 was previously explicitly defined without bout medians. The latest phrase "in all versions" conflicts with that definition. The user was asked whether Version 2 should now include bout-median replacement before binning or retain direct framewise log vigor.

## Baseline median

The requested baseline median must use every eligible timepoint of all bouts in [-15,0) s, rather than one scalar per bout or per bin. A lossless display run representing a bout is not one baseline vote: its constituent sample timepoints must all contribute.

Current C/D sample versions already use all eligible baseline timepoints after bout-median replacement. Their repeated values implicitly weight by eligible sample duration. Current Version 2 instead uses finite bin means, so its baseline median must move to the unbinned timepoint population. The user was asked whether its P10/P90 should also move to the timepoint population. This prevents silently defining an undocumented mixed sample/bin scale.

## Scaling critique

Let x be a display value, m the timepoint baseline median, and z=x-m. Neither current C nor current D is conventional min-max scaling:

- C: `clip(z / ((P90-P10)/2), -1,+1)` uses the central percentile range and preserves m at zero.
- D: `clip(z / max(m-P10,P90-m), -1,+1)` uses the larger baseline percentile distance, with one common denominator for both signs, and preserves m at zero.

Ordinary min-max scaling to [-1,+1], with extrema a and b from a chosen reference population, is:

`2 * (z - (a-m)) / ((b-m)-(a-m)) - 1 = 2 * (x-a)/(b-a) - 1`.

The median subtraction cancels. The baseline median maps to zero only if m=(a+b)/2. Calling this median-centred while interpreting zero as the baseline median would therefore be wrong. For example, if a=0, b=10 and m=2, the baseline median maps to -0.6.

A median-preserving alternative based on actual baseline extrema is `clip(z / max(m-a,b-m), -1,+1)`. It preserves baseline zero but only the more distant extreme necessarily reaches -1 or +1. It is symmetric scaling about the median, rather than ordinary min-max scaling. D is its percentile-based analogue.

Actual extrema are more sensitive to unusual samples than P10/P90. Using full-trial extrema would also let post-CS responses determine the normalization denominator, potentially reducing visible trial-to-trial response changes. Baseline-derived scaling avoids this particular dependence. These are substantive choices, not wording changes.

The user was asked to choose median-preserving baseline-extrema scaling, ordinary baseline min-max scaling, or the current percentile C/D scales. No new figures have been claimed to implement a scaling choice before that choice is resolved.

## Subsequent answers

The user confirmed that Version 2 must remain without bout medians. They also specified min-max scaling based on baseline P10 and P90, rather than actual extrema. For an output range [-1,+1], applying min-max to median-centred values with median-centred percentile anchors is:

`clip(2 * (z - (P10-m)) / (P90-P10) - 1, -1,+1)`, where `z=x-m`.

It maps P10 to -1 and P90 to +1. Algebraically it equals `clip(2*(x-P10)/(P90-P10)-1,-1,+1)`; subtracting m therefore does not make m the colour-zero anchor. Zero corresponds to the midpoint of the two percentiles. The HTML interpretation must reflect this if that scaling is used.

The remaining questions concern the timepoint versus bin population for P10/P90 and the meaning of the full-trial support for bout medians. Recalculation dependent on those choices remains pending; the previous figures have not been silently changed.

## Final clarified method and completed corrections

The user confirmed that P10/P90 use unbinned timepoints already carrying their corresponding bout median, and that complete detected bouts should be used beyond the plotting crop. The user then explicitly stated that they wanted the baseline median at zero. Ordinary percentile min-max does not meet that zero requirement; median-centred C/D scales were retained and described accurately.

All three corrected rows are in `reviews/fgh_full_bouts_baseline_samples_20261009`, with independent readback and source hashes. C/D sample versions use complete-bout medians. Version 2 keeps direct framewise-log means in 0.5 s cells. All three use the same P10/P50/P90 from eligible baseline timepoints carrying full-bout medians.

Full eligible bout support was saved. Selected bouts end at least 3.3 s inside the loaded segments, so their medians are not clipped by the read window. Eligibility and direct log values remain identical to the previous version. Extending the median support changes 8,196 displayed eligible sample medians for Delay, 11,237 for 3 s Trace and 5,076 for Control.

The C/D sample baseline medians are zero to a maximum absolute residual of approximately 1.2e-15. Version 2's displayed baseline-bin medians remain negative because its direct-log means use a different value population from the bout-median baseline reference; ranges are Delay [-1,-0.303], 3 s Trace [-1,-0.645], Control [-1,-0.511]. This distinction is explicit in the corrected HTML.

Some baseline bouts cross CS onset, so their full medians also include post-CS samples: 38 bout/trial occurrences for Delay, 66 for 3 s Trace and 13 for Control. This follows the requested complete-bout rule and is documented.

The HTML summary now contains the corrected rows. Its prior version is preserved as `reviews/fgh_latest_summary_20261009/index-before-full-bout-correction.html`. Registration uses `configs/paper-figures/selections/figure1-fgh-full-bout-correction-20261009.json`, preserving the earlier freeze records.

## Direct-bin blue-offset diagnosis

The user questioned the predominantly blue Version 2 display. An independent audit (`audit_direct_bin_offset.py`, `direct_bin_offset_audit.csv`, and `direct_bin_offset_summary.json` in the complete-bout review folder) quantifies the reference mismatch. Version 2 displays means of direct eligible log vigor but is centred/scaled using timepoints carrying complete-bout medians. Those value distributions differ substantially in these recordings.

Typical trial baseline-cell medians are -0.931 for Delay, -1 for 3 s Trace and -1 for Control. The median trial fraction of finite baseline cells clipped at -1 is approximately 47%, 70% and 100%, respectively. The direct-bin baseline medians are lower than the bout-median reference medians by about 0.334, 0.260 and 0.294 natural-log units, respectively. Dividing by the narrower bout-median percentile range amplifies this offset, and clipping makes it appear uniformly blue. The visible-zero-baseline goal is therefore not achieved for Version 2 by this reference choice.

Two alternatives were checked without changing the current plots:

- Baseline statistics from direct unbinned log-vigor timepoints reduce the offset, but the median of displayed baseline-bin means remains typically -0.209, -0.195 and -0.243. Averaging does not preserve the median under unequal sample/bin weights.
- Baseline statistics from finite direct baseline-bin means give displayed baseline-bin medians of zero (maximum numerical residual about 2.8e-15). This is the consistent population for the user's desired zero-median bin display, but conflicts with the earlier universal unbinned-timepoint baseline rule.

The user was asked which baseline population to use for Version 2. The current figures have not been silently changed while that choice is pending. C/D full-bout sample versions retain their consistent timepoint references and zero baseline medians.

## Approved direct-bin reference fix

The user selected "Use baseline-bin means; guarantee displayed median zero (recommended)". Version 2 now uses P10/P50/P90 from its finite unscaled baseline-cell means, with one vote per bin. Its direct log-vigor means and eligible frame counts are unchanged. The C/D sample versions remain unchanged, including complete-bout medians and unbinned timepoint baseline statistics.

The current Version 2 outputs are in `reviews/fgh_direct_bins_own_reference_20261009`. All 269 defined fish/trials have a displayed baseline median of zero to a maximum absolute residual of 2.8e-15. H trial 16 remains undefined. The HTML summary and current scoped Figure 1 registration now point to this row. The earlier blue row and its diagnosis remain preserved as history.

## Implementation constraints

- Fish remain F=20221115_07, G=20230310_08, H=20221115_09.
- Shared detector, eligibility, log transform, baseline interval [-15,0), display interval [-20,+20), and shared managua_r layout are retained unless explicitly changed.
- Missing/zero denominators produce undefined display values, not borrowed scales or filled zeroes.
- Corrected plots require new numeric exports, manifests and validation. The latest HTML must distinguish earlier figures from the corrected results.

## Additional Version 4: direct bin medians without scaling

The user requested a fourth row and answered all three clarification questions before implementation: use direct eligible framewise log vigor without bout-median substitution; subtract the trial baseline median without scaling; retain the existing eligible moving-sample mask.

Each CS-aligned 0.5 s bin supplies the median of its non-NaN eligible log samples. A single eligible sample is enough, regardless of how many NaNs the bin contains. All-NaN bins stay missing. The baseline reference is the median of finite bin medians in [-15,0) s, one scalar per bin. The displayed value is `bin_median - baseline_median`. No P10/P90, scaling or numerical clipping is used. The shared managua_r colour range is [-0.25,+0.25] natural-log units; values outside it use endpoint colours but retain their numeric values.

The three original rows remain unchanged. The fourth row is saved in `reviews/fgh_version4_bin_medians_20261009`, registered in the current scoped Figure 1 configuration and included in the latest HTML summary. Fish remain F=20221115_07, G=20230310_08 and H=20221115_09. Source tables are hash-verified, and the direct `log_vigor` column is used.

All 270 trial baselines are defined and median-zero within floating-point precision; all 18 bins with only one eligible sample are retained. H trial 16 becomes defined because no percentile-range denominator is required. Per-bin NaN-aware medians, exported subtraction results, baseline bin counts, SVG intervals and colours, PDF layout and HTML rendering are verified.

Assessment: the median reduces the influence of extreme eligible frame values. Equal bin weighting avoids duration weighting within the baseline, and subtraction from the same bin-median population guarantees its displayed baseline median is zero. Allowing one sample means sparse bins can carry the same baseline weight as dense bins, as explicitly requested. The fixed colour limits allow comparison in log units but saturate differences beyond ±0.25, so exact magnitudes beyond the colour limits must be read from the numeric exports. Missing bins indicate absence of eligible moving samples rather than a measured vigor of zero.

## Legacy CS boundaries and provisional panel E trial markers

The subsequent layout request applies to all four current rows. The supplied legacy SVG uses green CS boundary strokes at 0 and 10 s with width 2 pt and opacity 0.75. The updated rows use the same green (`#0d8136`) with width 2.4 pt and opacity 0.8 so the boundaries remain clear on the larger three-panel layout. White block separators are drawn above these strokes. Pre/Train/Test labels move to the left of F; the shared colourbar stays at the far right.

Panel G (20230310_08) has black left-pointing arrowheads outside its right spine on trials 9, 17, 63, 66 and 93. These span Pre, early/late Train and early/late Test and preserve the legacy five phase positions. Each selected trial has at least 79/80 finite displayed half-second bins and 29/30 finite baseline bins in Version 4. The arrows are provisional examples for a later panel E adaptation, not a selection of maximal responses; panel E has not been changed. Exact counts and source hashes are in `reviews/fgh_legacy_layout_20261009/example_trials.json`.

New figures and per-row layout validations are in `reviews/fgh_legacy_layout_20261009`. All existing numeric values and baseline calculations are retained and checked against their previous hashes. Earlier outputs are preserved. The HTML summary and scoped Figure 1 configuration refer to the new layout.

## Version 4: longer baseline, non-bout lower bound, then arithmetic means

The user requested a per-trial baseline balance audit, changed only Version 4 to [-20,0) s, and clarified that valid non-bout frames must contribute the minimum rather than be ignored. They explicitly selected an absolute lower bound (-infinity), then selected leaving the whole trial undefined when its baseline reference is -infinity.

The original eligible-only median version was audited: all 270 trials have exactly equal numbers of baseline bins above and below zero, excluding ties within 1e-12 log units. This property applies to the baseline population; it does not require the full trial to have the same sign distribution. Per-trial results are saved in `reviews/fgh_version4_baseline20_20261009/previous_v4_per_trial_balance.csv`.

Source validity masks were reconstructed independently from the original corrected frames, coverage, timing and detector. Detected bouts, eligible masks and direct log samples match the existing tables. Valid non-bout frames are separated from invalid tracking or ineligible frames inside bouts; only the former receive -infinity. Per-frame classifications and hashes are saved in `reviews/fgh_version4_baseline20_20261009`.

The intermediate median version uses all non-NaN bin medians, including -infinity, in the 40-bin baseline. It has 107 defined trials: 20 Delay, 87 Trace, 0 Control. All have 20 bins above and 20 below zero. The other 163 are undefined because their baseline median is -infinity. The supplied undefined-baseline rule is followed without replacing or borrowing a reference.

The user's subsequent instruction replaces medians with arithmetic means in Version 4. Both bin aggregation and the baseline summary now use arithmetic means, retaining the selected -infinity non-bout floor and undefined-reference rule. One -infinity sample makes its bin mean -infinity; one such baseline bin makes the baseline mean -infinity. All 270 current trials are therefore undefined. Mean centring targets zero mean when defined and does not guarantee equal sign counts.

Current mean-based outputs, CSVs and validations are in `reviews/fgh_version4_means_baseline20_20261009`. The first three versions and numeric inputs remain unchanged. The HTML summary and scoped configuration point to the mean-based fourth row; earlier fourth-row outputs remain preserved.

Assessment: assigning valid non-bout frames an absolute log lower bound incorporates their time occupancy, but causes reference subtraction to become unavailable for many median baselines and all current mean baselines. A finite floor would be required to retain these contributions while obtaining defined centred arithmetic means. No finite floor has been substituted without instruction. Black current rows denote undefined arithmetic, not measured zero vigor or missing original source data.

## Current Version 4: remove the lower bound and use 0.25 s means

The user reverted the negative-infinity replacement: non-bout samples stay NaN. They requested 0.25 s bins. The current fourth version retains direct eligible natural-log vigor, arithmetic means for both bin aggregation and baseline calculation, baseline [-20,0), no percentile scaling or numeric clipping, and shared managua_r colour limits [-0.25,+0.25]. The first three versions are unchanged.

There are 160 quarter-second bins per trial, with up to 80 baseline bins. Each bin averages finite eligible log samples, ignoring non-bout and invalid NaNs; one finite sample is sufficient and all-NaN bins remain missing. The baseline is the arithmetic mean of finite baseline-bin means, one vote per bin. It is subtracted from all trial bin means.

All 270 trial references are defined, and centred baseline means are zero to below 9.5e-16 log units. Eight trial baselines have equal above/below counts; the maximum count difference is 33, using a zero tolerance of 1e-12. This is compatible with mean-zero centring, which does not require a zero median or a 50/50 sign split. Finite display cells: Delay 10,743/14,400; Trace 13,654/14,400; Control 5,455/14,400. Seventy bins across the three fish have just one eligible sample and remain valid.

Current outputs, manifests, baseline/sign-count CSVs and verification are in `reviews/fgh_version4_quartersecond_means_20261009`. The latest HTML summary and scoped Figure 1 selection point to these outputs. Earlier negative-infinity experiments remain preserved. Bin means are independently verified against grouped frame values, source hashes and missing masks are checked, and SVG geometry and PDF/HTML layouts are reviewed.

Assessment: reducing the bin width improves temporal resolution and reduces samples per bin; sparse bins and missing bins become more common. Ignoring non-bout values means this view describes eligible moving vigor conditional on movement contributing to the bin; missing cells do not quantify immobility as zero. Equal weighting of finite baseline bins preserves a zero displayed baseline mean, without guaranteeing equal colour occupancy above and below zero. Finite log differences retain their units across trials and fish; the fixed colour limits saturate visually while numeric values remain unclipped.
