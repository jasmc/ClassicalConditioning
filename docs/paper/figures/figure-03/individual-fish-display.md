# Figure 3 individual-fish illustrations and complete atlas

Author-requested design, 2026-10-15. The author agrees to deliberately choose a
few illustrative fish for the main paper and show all fish in supplementary
data. Exact examples, panel allocation, learner method and scientific choices
below remain proposed. This document specifies the display; [Plan 04](../../../../plans/04_FIGURE_3_LEARNER_ANALYSIS.md)
owns implementation and validation, and [Plan 02](../../../../plans/02_DISCARDING_AND_COHORT_IMPLEMENTATION.md)
owns cohort/exclusion decisions. The [Figure 3 overview](README.md) owns panel
composition. No data have been reprocessed and no learner labels approved by
this planning work. The subsequent [exploratory implementation](../../../running-analysis/figure3-individual-atlas.md)
provides all 57 available Delay/control fish and four candidate illustrations;
it does not approve labels, examples or a paper method.

## Purpose and boundary

Make it possible to trace each illustrative classification back through its
continuous evidence, exact trial summaries and measured behavior. Show where
the rule agrees with visible behavior, where the result is uncertain, and
where an apparently surprising classification follows from its definition.
Same-data agreement is an explanation of a decision, not independent validation.

Use only tail bend angular speed (`legacy_distal_angular_speed`) and the shared
bout detector. Movement occurrence, time moving and tracking coverage are
complementary outcomes of that detector, not alternative activity metrics.
Preserve the standing bout-only vigor policy: valid no-bout vigor is undefined,
not zero. Keep physical movement observation separate from vigor eligibility.

Figure 3 owns individual classification evidence and illustrations. Aggregate
Control–Learner, Control–Nonlearner and Learner–Nonlearner behavioral comparisons,
within and across experiments, belong in Figure 4 only. Controls can appear in
Figure 3 as calibration/reference evidence and individual audit examples.

## What the legacy source actually provides

The [source review](../../../../records/scientific-evidence/audits/figure3-individual-display-legacy-review-2026-10-15.json)
binds source hashes, functions, inspected archived outputs and limitations.

- Preprocessing `run_plot_individual_trials` supports CS/US-aligned raw and
  scaled per-fish heatmaps, tail traces and normalized-vigor trial summaries.
  In the checked source, raw-vigor plotting is enabled and scaled plotting is
  disabled; source availability does not prove a historical run used a flag.
- `3_FishGrouping.save_heatmap_grid` assembles pre-saved heatmaps and separates
  included/discarded fish by a reused text list. Its numerical pooling does
  not apply an equivalent exclusion. Grid membership cannot establish the
  model population.
- Nominal/new/improved/WIP learner scripts produce individual trial/block
  plots and model-feature summaries. New/improved/WIP also define heatmap grids;
  improved/WIP enable them in their checked source. The supported isolated
  worker disables those grids, including when scalar individual plots run.
- The heatmap grids iterate classified fish, match pre-saved images by the
  first two filename tokens, prefer PNG, and crop 2% from the images. They do
  not provide a complete, identity-authenticated cohort atlas.
- Saved legacy-metric WIP scalar individual plots and the all-Delay combined
  grid were hash-verified and visually inspected. Saved raw/scaled heatmap-grid
  images were not located in the checked inventories/mounted roots. Their
  source implementation is present; this review does not claim those images
  were inspected.

### Classifier differences the illustrations must explain

For the configured CS5–94 schedule, all variants use Pre5–14 and LateTest85–94.
Nominal uses middle trials 60–69; new/improved/WIP use 40–69. Thus the latter
middle epoch spans late training and early testing; it is not simply a final
training block. The shared middle observations enter acquisition and recovery
with opposite signs. Their relationship is not two independent measurements.

| Variant | Modeled behavioral quantity | Active decision settings in checked source |
|---|---|---|
| Nominal | `log(Mean CR + 1)` with `log(baseline + 1)` adjustment | Bootstrapped 75th control-distance percentile; point or conservative directional votes. The active conservative multiplier is **0.5 SE**; displayed intervals use **1.96 SE**. |
| New | Same response/baseline transformation; broader middle epoch | 75th control-distance percentile, 50 bootstrap draws; directional Gaussian score >0.60 for each selected feature under its primary probabilistic rule. |
| Improved | `log(Mean CR)` with `log(baseline)` adjustment | One-sided joint statistic, empirical 90th control percentile, all selected directions required; SE-aware scoring enabled. |
| WIP | `log(Normalized vigor)` without a baseline covariate | One-sided joint statistic, empirical 95th control percentile, all selected directions required; SE-aware scoring enabled. |

Do not infer active settings from dataclass defaults, names such as
“conservative”/“improved”, comments, or an output suffix. Record instantiated
configuration, units and exact source identity. Adding one before a logarithm
is unit-dependent; corrected rad/ms inputs and historical degree-based inputs
are not automatically equivalent.

The verified 2026-09-24 legacy-metric Delay comparison used 29 Delay/28 controls:
nominal/new/improved/WIP labeled 7/7/6/19 Delay fish and flagged 0/2/1/2 controls.
These are historical descriptive results preceding the 2026-10-09 bout-only
policy; the report has no recorded current vigor policy. They motivate an
agreement audit, not current paper labels or a method choice.

### Legacy display traps to avoid

1. WIP key middle markers summarize `Late Train + Early Test`, although its
   feature uses `Train 6` through `Early Test`. Mark the exact feature trial
   sets, not a convenient substitute.
2. Legacy preparation renumbers the retained trial sequence from one. Preserve
   canonical scheduled CS trial numbers in new displays and retain any model
   row-number mapping explicitly.
3. Old raw heatmaps choose fish-specific color quantiles. Scaled heatmaps replace
   within-bout values by bout means, use per-trial baseline quantiles and clip
   to [0,1]. Neither reproduces all classifier transformations. Register the
   display estimator separately and show the actual classifier input beside it.
4. Old scalar plots normalize baseline and response curves by their own Pre
   means; those curves are not on a common absolute intensity scale.
5. A cumulative two-feature BLUP line on an independently scaled secondary
   axis is a feature schematic, not a fitted trial-by-trial trajectory. Keep
   model features in a separate inset; do not overlay that line as a fit.
6. WIP labels use `T > quantile threshold` plus all directional votes. Its
   empirical tail rank is separately reported and is not the label's p-value
   cutoff. Gaussian directional scores are not validated probabilities that
   a fish learned. A reported interval crossing zero need not contradict a
   rule comparing features against a nonzero control reference.

## Three linked outputs

### 1. Complete author-review atlas

Before choosing main-paper fish, create one complete atlas for each experiment,
from a single authenticated inventory/cohort and learner manifest. Include
all conditions. The inventory must account for excluded, pending and missing
recordings as well as included fish; unavailable data receive an explicit
entry rather than a fabricated plot.

Every available fish receives one detailed card. Classification-ineligible
fish retain their observed data and reason codes. Technically excluded fish
occupy a clearly identified QC section, outside the included-cohort denominator.
Legacy behavioral-screen flags remain separate attributes; they do not
silently remove a fish or become learner labels.

Keep a stable experiment/condition/fish order for comparison across classifier
versions. Provide class- or score-ordered views only as derived review layouts,
with the same complete inventory. Matching a label or image by abbreviated
filename is insufficient; use experiment + condition + fish identity and hashes.

### 2. Main-paper illustrative selection

The author may deliberately choose clear illustrations after reviewing the
atlas. Call these selected illustrations, not a random/representative sample.
Publish every fish in the supplement and show all eligible classification
evidence in the Figure 3 decision summary; the examples do not establish
prevalence, effect magnitude or validation performance.

Start with a clear learner and an adequately observed nonlearner. Add a typical
learner and a borderline/otherwise informative nonlearner if space permits.
Consider one learner/nonlearner pair per experiment if all three assays have
such cases and the final-scale layout stays readable. Do not force an example
of a class that the selected method does not produce.

Particularly informative additional cases are persistent suppression that
fails a recovery-dependent rule, a large joint score with mixed directions,
or disagreement across methods. At least review these cases in the atlas;
they need not all occupy the main figure. Ineligible fish must not be used as
the nonlearner example.

Record each selected fish's identity, label/method, selection reason, eligibility,
continuous score/features, coverage, sensitivity flags and atlas page/card.
Keep an entry for any technically problematic candidate rejected as an
illustration. Selection cannot change primary cohort membership or a label.

### 3. Complete supplementary data

Publish the complete atlas, not only the clearly matching fish or those that
produced complete model features. Include a readable fish-status/coverage
table, excluded/pending inventory entries, and the cross-method agreement
matrix with explicit ineligible/failed states. Keep the chosen paper method
distinct from historical comparisons; do not create a majority-vote truth label.

Paginate instead of reducing dozens of fish to unreadable thumbnails. If full
cards are too extensive for journal figure pages, supply numbered supplementary
data pages plus machine-readable fish/trial summaries and an index linking the
main examples to them. All permanent media/numerical payloads remain external.

## The individual-fish card

| Component | Required content and purpose |
|---|---|
| Identity and status | Experiment, condition, stable fish ID, selected method/label, continuous evidence, and distinct technical/behavioral/eligibility status. |
| CS-aligned trial-by-time heatmap | All scheduled trials in canonical order; Pre/Train/Test boundaries; baseline, response and CS intervals; actual US whenever present, including independently timed US in controls; expected-US guides on omissions; catches 25,39,53,59,65 marked from configuration. Keep 65's catch and Test identities. |
| Movement and coverage | Separate valid no-bout behavior from missing/invalid tracking and unavailable intensity. Use an aligned movement raster/fraction display plus physical-coverage information. A blank vigor region alone cannot explain immobility. |
| Exact classifier input over trials | All individual trial points, declared summaries, original trial numbers, phase boundaries and exact feature-epoch membership. Show trials omitted from scoring with a reason; do not compress missing trials or connect lines across unsupported gaps. |
| Baseline/response context | Separate baseline and response measurements on a common declared scale; movement fractions alongside conditional vigor. Expose baseline drift, near-zero denominators, and no-bout windows. |
| Decision inset | Actual feature estimates, uncertainty status, control reference, directional checks and score threshold appropriate to the selected algorithm. State which criterion was not met for a nonlearner; keep insufficient information distinct. |

For a compact main-paper card, retain the heatmap, matched trial trajectory,
movement/coverage strip and a short decision explanation. Link it by ID/symbol
to the full decision summary and supplementary card. Put verbose diagnostics
and the complete feature table in the atlas rather than overprinting the data.

Heatmap time bins and transformations remain a scientific display choice.
A 1 s bin is a candidate for continuity with existing paper heatmaps; test its
ability to show the relevant timing before selection. Bind the exact frame-to-
bin estimator, baseline transform, mask and coverage policy. Match color limits
across illustrations that share a quantity/transform; do not use per-fish
autoscaling to manufacture comparable-looking suppression. Show saturation
explicitly. A displayed median-log heatmap is not numerically identical to a
mean-ratio or baseline-adjusted response model.

Keep condition colors consistent with the paper. Identify class using a badge,
symbol or other separate channel. Missingness, technical exclusions and class
must not share an ambiguous red/blue border meaning. Layout frames remain
invisible under the active 1.0 mm inset policy. Use shared guides/legends and
final-scale renderer evidence; generate from authenticated panel data rather
than cropping previously rendered images.

## Fish selection and uncertainty safeguards

The original preprocessing discard switch is disabled in the checked source,
but its rules include US bout responses and at least three bout-bearing trials
in selected baseline/CR blocks. Such response-dependent rules can remove valid
suppression; they are not automatically technical QC. Grouping/scaled-vigor
reuse discard lists by default, normalized-vigor does not, and its model
preparation has an ANY-block completeness rule. Learners require their own
constituent-block/epoch completeness and intersect available model features.
These stages must remain separate columns, not one unexplained discard flag.

The display inventory must reconcile expected recordings -> processed/QC
status -> reviewed primary membership -> trial/outcome eligibility -> learner
eligibility -> classified/uncertain evidence -> illustrated/supplementary entry.
Do not double-count repeated application of a list as new attrition. Report
coverage/counts in every required block; no-bout conditional vigor and missing
tracking must remain distinguishable throughout.

For approved paper labels, validate precision, stability and control calibration
on the actual design. Same-data heatmaps and curves are explanatory. All four
default legacy rules consume late-Test information, so their Test curves cannot
be called held-out validation. A genuine prediction analysis needs a separately
specified construction/evaluation recipe; withholding a panel from view does
not withhold its data from fitting, calibration or eligibility.

The 2026-10-15 [G review](../../../../records/scientific-evidence/analyses/figure-02-g-independent/implementation-and-review-2026-10-15.md)
reports poor individual spline/state-space coverage, omitted covariance-
parameter uncertainty and no assigned learner labels. It is not a validated
source of individual timing intervals for Figure 3. Failed calibration must
withhold inferential bands/claims while retaining observed data. Show raw
trial evidence first; add fitted curves or uncertainty only with an explicit,
adequately validated interpretation. A nonlearner label means the criterion
was not met, not that absence of learning was established.

## Review checklist before choosing examples

- [ ] One source-linked inventory reconciles every fish and all status counts.
- [ ] Every available fish has a supplementary card; unavailable cases have entries.
- [ ] Labels/scores/features and data belong to the same authenticated run.
- [ ] All plotted feature epochs match the selected classifier's actual trial sets.
- [ ] Every omission has a tracking, no-bout, transform, coverage or model reason.
- [ ] Shared scales expose baseline drift and magnitude without per-fish clipping.
- [ ] Main examples are explicitly illustrative and indexed to complete supplements.
- [ ] Surprising/borderline/disagreeing cases were reviewed and remain available.
- [ ] Same-data illustration is distinguished from independent/predictive validation.
- [ ] Main-paper cards pass final-scale visual review with the active layout policy.

Unresolved decisions are the learner recipe, numerical eligibility/coverage
rules, exact heatmap transformation/bin, number of main examples and whether
one pair per experiment fits. Resolve these from the full atlas and applicable
scientific review; this design does not invent approvals.
