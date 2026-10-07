# Figure 1 single-fish heatmap review — provisional, no selection frozen

Current lettering: E = raw/signed vigor reference, F = Delay, G = 3 s Trace, H = Control. Historical E/G heatmap names map to current F/H by condition; historical C/D trace names map to current D/E by content. Panel E is included only to explain signal-to-bin correspondence; its design remains under separate review.

The v5 acquisition-clock builder was reused verbatim with a read-only trial callback. All 21,600 saved F/G/H bins reproduced within 1e-12, including identical NaNs. This is a narrow figure review, not confirmation of the full legacy preprocessing pipeline. No population reruns or freezes were performed.

## Saved alternative families and their semantics

| Family | Metric / clock | Reference and operation | Support / binning | Palette and clipping |
|---|---|---|---|---|
| Historical per-trial linear conditional vigor | Multiple metrics, including tail-length-weighted angular L1; corrected arrival-clock profiles | P10–P90 of covered conditional-intensity bins before −15 s; divide by trial range | Detector-conditional bin means, coverage threshold; historical bounds, not proposed here | Original palette or managua_r; stored 0–1 clipping |
| Historical frame-first log P10–P90 | Multiple metrics; corrected arrival clock | ln(mean raw vigor per bout), reference [−20,0); divide by P90−P10 per trial | Moving positive valid frames; scale and clip frames before 0.5 s averaging; empty bins NaN | managua_r, stored 0–1 clipping |
| Historical all-frame candidate | Multiple metrics; corrected arrival clock | Mean raw vigor per bin; P10–P90 from covered bins before −15 s | All valid frames, not bout-conditional; 90% coverage flag; continuous display requires finite bins | managua_r, stored 0–1 clipping |
| Historical baseline-window signed-log review | tail_length_weighted_angular_l1; corrected arrival clock | Median moving-frame ln(vigor) subtraction; saved [−20,0) and [−15,0) versions | Bout median logs repeated on eligible frames; finite-only 0.5 s means | managua_r, ±0.25 display clipping only |
| Assembly legacy-vigor v1/v2 | legacy_distal_angular_speed; corrected arrival clock | Per-trial moving-frame median log subtraction: v1 [−20,0), v2 [−15,0) | Detector-conditional bout medians; saved v1/v2 differ mathematically through reference bounds | managua_r, ±0.25 display clipping; NaN black |
| Current cadence v5 | legacy_distal_angular_speed, rad/ms; reconstructed presumed acquisition cadence | SAME [−15,0) per-trial reference | Median bout log repeated only on finite positive valid moving frames, bout_id>0; finite-frame-weighted 0.5 s means | managua_r, ±0.25 display clipping; NaN black |

Saved historical images are context, not controlled comparisons. Their clocks, metrics, references, order of operations and support can differ simultaneously. The complete file inventory includes paths and SHA-256 hashes; contact sheets preserve source filenames. A family with no located saved image is documented from its builder, not represented as a saved artifact.

## Controlled candidates

All candidates use the same reconstructed clock, legacy distal angular speed, detected-bout eligibility, trial windows [−20,20), and 80 half-second bins. Each bout median is calculated over its eligible portion within the trial window, then repeated on those same frames. Bin means weight bouts by their contributing eligible-frame counts. No unsupported value is filled with zero.

1. **Trial centred:** subtract the trial’s median eligible frame log in [−15,0). This is translation in log units, equivalent to an amplitude ratio reference; there is no P10–P90 division.
2. **Fixed fish centred:** subtract one median pooled from those SAME [−15,0) eligible frames across trials 5–94, separately per fish. Longer bouts and trials with more moving frames contribute more reference weight. This uses the whole displayed session retrospectively, not a pre-training-only reference.
3. **Uncentred:** retain median ln(vigor / (1 rad/ms)); the numerical unit reference is explicit. This and fixed fish centring differ by a constant per fish. Uncentred values permit direct numerical comparison across these fish; separate fish references do not preserve between-fish amplitude levels.

Columns 1–2 of candidate_gallery show IDENTICAL trial-centred stored values at ±0.25 and ±0.75. Columns 3 and 5 show IDENTICAL fixed-centred values and limits with managua_r versus RdBu_r. Column 4 uses viridis and a common 1st–99th percentile display range across all three fish. These quantiles set colour limits only; no data values are rescaled or clipped in storage. Black consistently means no eligible frame contribution, not zero vigor or proof of inactivity.

## Baseline and saturation audit

| Panel / fish | Baseline log range | Max/min geometric baseline ratio | Baseline frames min/median/max | Baseline bouts min/median/max | Current ±0.25 saturated | Trial ±0.75 saturated | Fixed ±0.75 saturated |
|---|---|---|---|---|---|---|---|
| F / 20221115_07 | -2.766 to -2.143 | 1.86× | [2131.0, 4132.0, 7819.0] | [14.0, 32.0, 60.0] | 32.8% | 3.3% | 1.2% |
| G / 20230307_12 | -2.980 to -2.443 | 1.71× | [1237.0, 5214.0, 6874.0] | [19.0, 31.0, 48.0] | 39.8% | 0.7% | 0.4% |
| H / 20221115_09 | -2.870 to -2.173 | 2.01× | [82.0, 2144.5, 4242.0] | [1.0, 11.0, 22.0] | 6.0% | 0.5% | 0.6% |

Saturation percentages count finite bin values outside the displayed limits. They do not count black bins and do not imply stored-value clipping. Thousands of frames are temporally correlated; baseline bout counts and leave-one-bout sensitivity are more informative than treating frame count as an independent sample size.

F: fixed reference -2.551645 ln units = 0.077953 rad/ms; missing baseline trials []; fewer than three baseline bouts []; largest leave-one-baseline-bout median shift 0.0632 ln units.
G: fixed reference -2.656402 ln units = 0.070200 rad/ms; missing baseline trials []; fewer than three baseline bouts []; largest leave-one-baseline-bout median shift 0.1840 ln units.
H: fixed reference -2.449552 ln units = 0.086332 rad/ms; missing baseline trials []; fewer than three baseline bouts [16]; largest leave-one-baseline-bout median shift 0.1910 ln units.

With no eligible trial baseline, the current function returns ALL 80 bins NaN even if movement occurs elsewhere. Fixed/uncentred retain supported bins under that condition. With a sparse or unusual baseline, per-trial subtraction shifts every supported bin by the same offset; the leave-one-bout statistic exposes dependence on individual bouts. We did not change missing-baseline policy for the current candidate.

## What centring removes or creates

For every supported bin, `trial_centred − fish_centred = fish_reference − trial_reference`. Thus centring does not change within-trial contrasts or timing, but it changes comparisons between trials. If the whole-trial movement amplitude rises with learning, trial centring removes the shared rise. If CS amplitude stays constant while baseline amplitude changes, trial-centred CS values change despite constant absolute CS amplitude. These are exact algebraic consequences; these example fish alone cannot establish whether the changes are caused by learning.

The next table compares mean supported-bin log amplitude before the paired US in early training (15–24) versus late training (55–64). Delay uses [0,9), Trace uses [0,13), Control uses [0,10) as a CS interval with no paired US. Equal trial weight, then equal finite-bin weight within each trial; it is descriptive, not an effect-size estimate or population inference.

| Panel | Late−early uncentred / fixed | Late−early trial centred | Late−early trial baseline |
|---|---|---|---|
| F | -0.2028 | -0.0527 | -0.1500 |
| G | +0.0585 | -0.0753 | +0.1339 |
| H | +0.0074 | -0.1469 | +0.1543 |

## Concrete bin examples

| Panel / trial / interval | Eligible frames | Uncentred | Trial centred | Fixed fish centred | Trial−fish offset |
|---|---|---|---|---|---|
| F / 9 / [0,0.5) | 113 | -2.08839 | +0.26649 | +0.46326 | -0.19677 |
| F / 17 / [0,0.5) | 266 | -2.23740 | +0.05638 | +0.31424 | -0.25786 |
| F / 63 / [0,0.5) | 134 | -3.11274 | -0.67675 | -0.56109 | -0.11566 |
| F / 66 / [0,0.5) | 131 | -3.03208 | -0.49084 | -0.48043 | -0.01041 |
| F / 93 / [0,0.5) | 265 | -2.95249 | -0.27258 | -0.40085 | +0.12827 |
| F / 8 / [0,0.5) | 152 | -2.01568 | +0.12765 | +0.53596 | -0.40831 |
| G / 9 / [0,0.5) | 17 | -2.65697 | +0.05174 | -0.00057 | +0.05231 |
| G / 17 / [0,0.5) | 56 | -3.27938 | -0.67369 | -0.62298 | -0.05071 |
| G / 63 / [0,0.5) | 83 | -3.04311 | -0.48626 | -0.38671 | -0.09954 |
| G / 66 / [0,0.5) | 297 | -2.50953 | +0.08720 | +0.14688 | -0.05968 |
| G / 93 / [0,0.5) | 142 | -2.86894 | -0.24464 | -0.21253 | -0.03211 |
| G / 86 / [0,0.5) | 94 | -3.16841 | -0.18828 | -0.51200 | +0.32372 |
| H / 9 / [0,0.5) | 97 | -2.54899 | +0.10412 | -0.09944 | +0.20356 |
| H / 17 / [-0.5,0) | 174 | -2.35530 | +0.05501 | +0.09425 | -0.03925 |
| H / 63 / [3.5,4) | 113 | -2.61542 | -0.18862 | -0.16587 | -0.02275 |
| H / 66 / [-0.5,0) | 289 | -2.49697 | +0.00574 | -0.04742 | +0.05317 |
| H / 93 / [1,1.5) | 132 | -2.56642 | -0.02267 | -0.11687 | +0.09420 |
| H / 70 / [-4.5,-4) | 190 | -2.76243 | +0.10750 | -0.31287 | +0.42038 |

The historical P10–P90 alternative adds a trial-dependent slope as well as an offset. The diagnostic column in candidate_bins deliberately applies that division/clipping to the CURRENT bout-median signal on the SAME baseline/support; it isolates the mathematical effect and does not claim to reproduce historical log-bout-mean data. Baseline P90−P10 widths and clipped-frame fractions are recorded per trial. Narrow ranges amplify noise; clipping before binning permanently loses amplitude distinctions.

## E–H visual review and provisional choice

Current E demonstrates unbinned raw vigor and the signed bout signal on matched support. Its orange signed traces are small at assembly scale, while F–H colour bins are heavily influenced by their narrow ±0.25 range. A heatmap cell summarizes supported movement within 0.5 s; it does not assert movement throughout that interval. E should explain this relationship in its caption; no E redesign was performed here.

Current F/G show broad patches of endpoint colours. Wider limits recover distinctions without changing the data. H has much more black support: its darker appearance partly reflects missing conditional movement samples, not simply weaker signed amplitude. All candidates retain actual chronological trial rows and phase boundaries; historical layouts include phase gutters, whereas the controlled gallery uses boundaries in a continuous grid. This layout difference is visual only.

For a figure intended to show amplitude changes across trials, the fixed fish reference is the strongest provisional candidate: it retains between-trial changes while giving a readable zero reference. Uncentred log vigor is the reference check and is preferable if between-fish absolute amplitude comparisons are central. Trial centring remains appropriate for a explicitly baseline-relative within-trial response question. The measured offsets and saturation should inform selection; the claim that all per-trial normalization is inherently bad is too broad.

Suggested selection for review: **fixed fish-centred bout median log vigor, ±0.75 shared display, explicit baseline/reference caption**. Compare managua_r and RdBu_r in the gallery; a diverging palette makes the zero reference clearer, while managua_r preserves the existing visual language. The wider range is a review choice, not an approved final range. No candidate has been frozen. The full preprocessing audit must be confirmed before broader reruns.

## Range-division diagnostic measurements

| Panel | P90−P10 ln range min/median/max | Maximum fraction of eligible bout-median frames clipped to 0/1 |
|---|---|---|
| F | 2.318 / 2.580 / 2.768 | 0.0% |
| G | 2.448 / 2.567 / 2.691 | 26.8% |
| H | 1.880 / 2.377 / 3.060 | 0.0% |

The diagnostic is generally unsaturated for bout medians: broad FRAME-log quantile ranges contain most BOUT medians. This does not validate historical mean-bout scaling or justify arbitrary range division. In G, one trial loses distinctions for 26.8% of eligible frames. H’s widths vary by a factor of 1.63, changing the relative gain between trials.

Control trial 16 has one baseline bout, so leave-one-bout sensitivity cannot be estimated there: removing that bout leaves no baseline. The reported maximum sensitivities concern trials with remaining baseline support.

## Saved v1 versus v2: measured mathematical differences

Sidecars identify v1 [−20,0) and v2 [−15,0) trial baselines on the same arrival-clock metric/movement inputs. Their support is unchanged; these are actual value differences, not palette-only alternatives. Those historical bounds were inventoried, not recreated or proposed for the new candidates.

F: changed-support bins 0; maximum finite-bin difference 0.272402 ln units.
G: changed-support bins 0; maximum finite-bin difference 0.116309 ln units.
H: changed-support bins 0; maximum finite-bin difference 0.189708 ln units.