# Frozen V12 definitions and population adaptation evidence

Consolidated 2026-10-10 from the dated handoff, preserving definitions, evidence and scoped authorizations. Historical instructions below do not supersede newer selections. Remaining implementation belongs to [Plan 09](../../../../Plans/09_FIGURES_AND_REPRODUCIBLE_REPORTING.md); unresolved choices remain unresolved.

Prepared 9 October 2026. Read repository `AGENTS.md` first. This handoff prepares
future work; Figure 2 has **not** been recalculated, selected or frozen here.
The user requested brighter endpoints, freezing V12, documenting other versions,
and a handoff to improve Figure 2 row 1 using the same analysis adapted to pooled
fish. Do not infer authorization to change other Figure 2 rows or cohorts.

## Author-selected source and immutable freeze

- Review and lossless freeze archive: `reviews/fgh_candidate_palette_versions_20261009/index.html`.
- New scoped selection: `configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json`.
- Existing scoped record now identifies V12 as the primary F/G/H choice:
  `configs/paper-figures/selections/figure1-fgh-full-bout-correction-20261009.json`.
- Renderer/extractor: `scripts/prepare_figure1_fgh_v12_freeze.py`.
- Freeze specification: `docs/analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md`
  and `configs/paper-figures/figure-elements.json`, version **1.1.1**.
- The explicit gate passed `--check-only`, then `--output`, for **159 elements**.
  A second check passed after extracting the packaged archive into a temporary
  directory. No unresolved issues. This is an **F/G/H row freeze**, not a
  whole-Figure-1 freeze. Final row size is 183 × 103.7 mm; whole-figure placement
  needs its own scale/assembly review.
- HTML script `version12-frozen-archive` stores exact gate input/output,
  scientific selection, original/styled SVG, renderer-property evidence and
  code/specification snapshots as SHA-256-bound entries. `freeze.json` is the
  unmodified gate-produced manifest. Recorded staging paths identify temporary
  materializations; `artifact_storage` maps them to their embedded bytes.
- Extract with `.venv-trace/Scripts/python.exe scripts/prepare_figure1_fgh_v12_freeze.py
  --extract <temporary-directory>`. For a portable recheck, run `python
  scripts/freeze_figure.py --candidate <temporary-directory>/portable-candidate.json
  --specification <temporary-directory>/specification-snapshot.json --check-only`.
  Do not create a new freeze of this historical artifact. Remove extraction
  files after verification. Existing source data and earlier freezes remain
  unchanged.

The common font/stroke rules are resolved at final size, condition headings use
configuration-derived colours, and G/H repeated trial labels are suppressed.
The CS offset is dashed, expected US dotted, and onset solid. The approved strong
CS width/opacity and custom palette are explicit scoped presentation exceptions.
Event lines are **nominal protocol guides**, including catch/omission trials;
they are not proof of actual stimulus delivery on every row.

## Exact frozen analysis

Example fish: F Delay `20221115_07`, G 3 s Trace `20230310_08`, H Control
`20221115_09`. Trials 5–94; display [-20,20) s; 80 half-open 0.5-s bins;
baseline [-15,0), with 30 possible bins. Phases: Pre 5–14, Train 15–64, Test
65–94; separators 14.5 and 64.5. G example arrows 9/17/63/66/93 are provisional
illustrations, not maximal responders or population selections.

1. Keep finite strictly positive eligible bout frames only. Raw metric is
   `legacy_distal_angular_speed_rad_per_ms`: absolute wrapped frame-to-frame
   change in summed corrected local tail angles divided by frame interval.
   Require valid adjacency and the authenticated movement/coverage mask.
   No-bout, invalid and discontinuous frames are NaN before all calculations.
   Do not add pseudocounts, smooth this analytical signal, interpolate gaps or
   substitute complete-bout medians.
2. Natural-log eligible **original frame vigor**.
3. Within each bin, median of finite log frames, `m[f,t,j]`. Any eligible finite
   frame suffices; empty bins stay NaN. Save eligible counts and endpoints.
4. `b[f,t] = median(finite m[f,t,j] during [-15,0))`. Each finite bin has one
   vote, regardless of frame count. This is not a frame-level baseline.
5. `d[f,t,j] = m[f,t,j] - b[f,t]`.
6. Fit `q10,q90 = quantile(finite baseline d, [0.1,0.9], method="linear")`;
   `s[f,t] = max(-q10,q90)`; `z[f,t,j] = 0.7*d[f,t,j]/s[f,t]`.
   One positive denominator serves both signs. Require at least 10 finite
   baseline bins and `s > 1e-12`. Otherwise the entire scaled fish/trial is NaN,
   with the physical values retained and reason recorded. This count is an
   eligibility screen, **not ten independent observations**.
7. Values remain unclipped. Colour saturation occurs only outside [-1,1].

For defined trials, baseline median stays zero; the wider P10/P90 side maps to
±0.7, the narrower stays nearer zero. Reciprocal ratios give equal opposite
scaled values. Equal colours describe fractions of each trial's baseline-spread
measure, not common physical ratios or additive rad/ms changes. Subtracting a
log baseline gives `ln(exp(m)/exp(b))`; doubling/halving are ±ln(2).
Exponentiating an even-count log median gives the geometric midpoint of the two
middle values, not their arithmetic midpoint. The raw signal reflects eligible
bout-frame intensity, not bout frequency or total activity.

Final palette: blue **#00bfff** at -1, dark managua centre **#383842** at zero,
red **#ff5252** at +1; 513 samples of piecewise straight CIELAB interpolation;
black **#000000** for NaN. Colourbar ticks -1/-0.7/0/+0.7/+1. Preserve the exact
frozen palette; do not resample another colormap by name. 257/270 example trials
are defined (F=90, G=90, H=77). The 13 H failures have <10 finite baseline bins.
The 0.7 factor reduces saturation but does not eliminate it.

All 270 physical trial/bin arrays and counts were independently recalculated
from the existing authenticated frame parquets before freezing. Their timing
uses the established **camera-cadence-reconstructed `time_s`**; the freeze did
not newly authenticate measured-AbsoluteTime reconstruction or revise movement
detection. Preserve and disclose this limitation rather than claiming a new
timing validation.

## Other versions and historical scope

The single review HTML retains the first eleven images unchanged, each recipe
section and comparison row, plus `version-history-catalogue`. Freeze selection
does not invent keep/discard decisions for other versions.

| Version | Definition and distinction from selected V12 |
|---|---|
| V1 C | Complete-bout log medians repeated on eligible frames; frame/timepoint baseline P50; denominator (P90-P10)/2; numeric clipping ±1; no 0.5-s binning. |
| V1 D | Same repeated complete-bout summaries; symmetric max(P50-P10,P90-P50) denominator; numeric clipping ±1; wider side reaches endpoint. |
| V2 C | Direct log-frame **means** in 0.5-s bins; baseline-bin P50; half P90-P10 width; clipped ±1. |
| V3 | No established recipe in this shortlist; do not invent one. |
| V4 history | Several incompatible earlier revisions. Current historical quarter-second direct means use equal-bin [-20,0) **mean** baseline and physical ±0.25 colours. Earlier bin-median/-infinity revisions remain historical. No guaranteed displayed median zero. |
| V5 history | Direct log means in 1-s bins; equal-bin [-20,0) mean baseline; physical ±0.25 colours; no guaranteed median zero. |
| V6 five / former softer High | Direct log means in 1-s bins; [-20,0) percentile bands P25/P45/P55/P75. These two retained views are identical under the common candidate palette. |
| Original frame-centred V7 | Bout-summary means with a frame baseline; excluded because the displayed bin-baseline median need not be zero. |
| V7 physical | 0.5-s means of repeated bout medians; finite-bin median baseline; physical ±0.25 colour limits. |
| V7 scaled | Same bout-summary bin values; separate P10/P90 stretches with median-preserving centre; anchors ±0.7; colour limits ±1. |
| V8 | V7 scaled divided by 0.7; both baseline percentile sides at ±1. |
| V9 | V7 scaled values classified at -0.5/-0.1/+0.1/+0.5; five colours. |
| V10 | V7/V9 values classified at ±0.1; three solid colours. |
| V11 original | Direct 0.5-s log-frame medians and bin-median baseline, followed by V7 separate-side scaling. |
| V11 current | Same direct medians; symmetric maximum-side denominator; wider baseline side at ±1; blue–charcoal–amber. |
| V12 frozen | Same physical direct-bin values; **0.7 symmetric scale**; continuous brighter blue–dark managua–red; registered specification-styled row. |

Earlier V12 two-endpoint, light-centre and three-band experiments were rejected
or superseded during this chat. They were display trials, not new physical
calculations. No duplicate historical exports were created. For earlier exact
paths/recipes, consult `docs/analysis/figures/reviews/FGH_COLOUR_AND_BINNING_HISTORY_2026-10-09.md`.

## Figure 2 row-1 current inventory

Resolve current scoped records again at the start of future work. At this
handoff, `configs/paper-figures/figure2-assembly.json` identifies:

- A: Delay, matched control; provisional cohort counts Delay 29 / control 28;
  cohort hash `9ef4b9297c939e0d34a99a4d606d0f9c8c2a42a0a7b9aee20d947823ef66f2d5`.
- B: 3 s Trace, matched control; provisional counts Trace 40 / control 19;
  cohort hash `63d6b8e25964210a0ecbdb562a2e652c3f96db544eac53cde64ebb9aad755580`.
- C: 10 s Trace placeholder, no authenticated population source; preserve the
  placeholder unless the required authenticated cohort/frame inputs exist.

A/B selected sources are under
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/sources/20261008T200908002798Z/`.
They are **equal-fish means of the older signed bout-summary bins**, physical
±0.25 managua colours, with upstream vigor/alignment review still open. This
inventory is from configuration/code inspection; it is not a new authentication
of the J-drive exports or their cohorts. Verify actual bytes against the record.

Entry points: `scripts/populate_figure2_available.py` (`verified_heatmap_paths`,
`fish_heatmaps`), `src/classical_conditioning/figures/signed_bout_heatmap.py`,
and `src/classical_conditioning/analysis/temporal_profiles.py`
(`_signed_bout_log_vigor`). The current helper calculates repeated bout medians,
frame-level baseline and bin means. Its caches **cannot recover direct log-bin
medians**. Recompute from authenticated frame-level metrics/movement/protocol,
with a separate candidate recipe and cache signature. Do not relabel old pooled
values as V12. Never substitute the three example fish for the population.

## Recommended pooling adaptation, with decisions still open

To preserve the selected interpretation, compute the complete V12 calculation
**within each fish and trial first**, then pool finite `z[f,t,j]` with one vote
per fish at each condition/trial/bin. Keep matched controls separate by assay.
No frame, bout-length, precision or movement-frequency weighting across fish.
Do not numerically clip individual values before pooling; apply saturation only
to the final colour mapping.

My recommended first candidate is the **equal-fish median** of normalized cells:
`Z[c,t,j] = median_finite_fish(z[f,t,j])`, with contributing fish counts alongside
every cell. It represents a typical fish relative to its own baseline spread;
it is not a common log ratio, and `exp(Z)` is not a physical fold change.
This estimator is a recommendation, **not an author-approved pooled selection**.
The existing population estimator is a mean; explain the change and, if useful,
compare an equal-fish mean using the same per-fish inputs within the one review
HTML. Mean and median of normalized fish values answer different questions.

Do not centre or rescale the pooled matrix automatically. Pooling is not
interchangeable with fitting a baseline reference: pooled raw frames overweight
fish with more eligible frames; scaling after pooling instead expresses spread
of the population aggregate rather than typical within-fish spread.

**Important baseline-centre exception:** even when every fish/trial has a zero
baseline median, the median over the pooled baseline bins need not be zero.
For a simple counterexample, three fish have five baseline-bin values:

```text
fish 1:  1  1  0 -1 -1    median = 0
fish 2:  0  1  1 -1 -1    median = 0
fish 3:  1  0  1 -1 -1    median = 0
pooled:  1  1  1 -1 -1    median = 1
```

Duplicating each bin gives ten bins per fish and the same counterexample. With
the V12 factor, the pooled median is +0.7. Changing contributors across time can
further shift it. Zero colour still means zero normalized deviation, but a
pooled baseline is not guaranteed to concentrate at the dark centre.

Ask a focused question **before selecting/freezing pooled results**: prioritize
the median/mean of individual baseline-relative fish deviations, or enforce
zero in the population baseline after pooling? If the latter is essential,
subtract the pooled finite-baseline median as an explicitly new operation and
label zero as the pooled reference. This removes the original interpretation
of zero as the aggregate unchanged-fish value; do not call it identical V12.
An alternative candidate can pool physical per-fish log deviations first and
then fit a pooled baseline/scale; that also changes the normalization target.

## Validation requirements retained from the source

1. Authenticate source manifests, cohort inclusion/condition mappings and
   one-to-one FrameID/AbsoluteTime joins. Resolve measured time versus cadence
   reconstruction explicitly; retain protocol-defined alignments and catch
   identities. Do not silently change metric or movement recipe across assays.
2. Calculate fish/trial direct-bin medians and eligible-frame counts. Save
   physical `m,d`, baseline reference, P10/P90, denominator, normalized `z`,
   exclusions, and source hashes as provenance within the single review artifact
   or existing scientific caches; avoid redundant data/figure exports.
3. Verify raw-log agreement, half-open boundaries, baseline median zero,
   ±0.7 wider-side anchors, reciprocal symmetry, NaN masks, unit invariance and
   response-independence of fitted scale. Include odd/even/sparse/empty/constant
   baselines and confirm no-bout values cannot influence results.
4. Pool each condition/trial/bin with one value per fish; reject duplicate
   fish/cells. Record numerator, total cohort denominator, eligible fish/trial
   counts and per-cell coverage. All-missing cells stay black, not zero.
   Do not invent a minimum contributing-fish threshold without discussing its
   scientific effect; show coverage and a sensitivity comparison if warranted.
5. Report pooled baseline medians, sign distributions, coverage change and
   saturation rates before deciding any second centring/scaling. The 0.7 factor
   provides headroom, not a guarantee of unsaturated responses.
6. Add review candidates to the **existing single HTML**, keeping the frozen
   V12 SVG/archive bytes and hashes unchanged. No separate PDFs/SVGs/PNGs or
   duplicated data copies. The capture-time whole-HTML hash will change when
   adding other content; preserve that historical hash and verify the immutable
   embedded freeze entries independently. Do not edit the archive or refreeze it.
7. For Figure 2 styles, resolve each row-1 panel at its actual assembly size.
   The current box width 540/1800 of 183 mm is 54.9 mm, but each A/B source may
   include conditioned/control subpanels and needs a fresh layout/legibility
   check. Do not copy the 183-mm example-row fontsize or provisional arrowheads
   into population panels. Keep time/trial/event/phase meanings explicit and
   follow the figure-element contract for any future freeze.
8. These heatmaps are descriptive. Do not transplant frozen D/E statistics,
   infer confidence intervals from across-fish spread, or add stars. Preserve
   frozen Figure 2 D/E and the selected G/H trajectories. C remains unavailable
   without authenticated 10 s Trace inputs.

Historical continuation wording: “I’ll preserve frozen V12 and the other Figure 2 rows,
authenticate the population inputs, and build direct-bin fish-level values.
Before choosing the pooled display, I’ll compare the equal-fish estimator and
check whether its baseline stays centred, so any extra centring is explicit.”
