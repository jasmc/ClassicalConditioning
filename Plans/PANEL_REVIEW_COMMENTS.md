# Panel review comments

Use this file to collect statistical, scientific and presentation comments to
address later, organized by figure and panel. It complements
[paper review considerations](./PAPER_REVIEW_CONSIDERATIONS.md) and
[analysis and statistics](./02_ANALYSIS_AND_STATISTICS.md). It is a review
agenda, not a second implementation-status board or an approval of a method.
Decisions belong in [DECISIONS.md](./DECISIONS.md), implementation requirements
in the relevant numbered plan, and progress only in the
[implementation index](./IMPLEMENTATION_STEP_INDEX.md).

Use stable figure/panel IDs from the
[paper specification](../docs/analysis/figures/02_PAPER_FIGURE_SPECIFICATION.md).
Add entries as panels are reviewed; an empty section does not mean a panel
passed review. Keep a comment's evidence and proposed remedy distinct from an
author-approved decision. A selected appearance does not approve its inference.

## Entry format for every main and supplementary panel

For each panel, record:

- **Panel and review date:** stable figure/panel ID and date.
- **Intended claim:** descriptive, within-fish change, between-group difference,
  differential change, onset, extinction, learner characterization, or another
  explicit question.
- **Outcome and inputs:** metric, units, cohort/source hashes, alignment,
  windows, trial blocks, aggregation and eligibility/missingness rules.
- **Comment ID and concern:** a specific criticism, with code/data evidence.
- **Required comparison or remedy:** test/estimand, pairing unit, effect
  direction, uncertainty, assumptions and correction family; use "no
  inferential test required" when the panel is descriptive.
- **Current handling:** what the selected candidate actually implements;
  distinguish the original legacy behavior from a corrected review recipe.
- **To resolve later:** scientific choice or validation still needed, its
  owning numbered plan and decision-register link.
- **Presentation decision:** selected style and whether annotations show
  significant results only; retain every computed result in the companion table.
- **Evidence:** exact versioned artifacts, not stars copied from a screenshot.

## Figure 1

Add comments under the individual panel ID as each panel is reviewed. Existing
F/G/H discussions remain in
[the step assessment](./FGH_ANALYSIS_STEP_ASSESSMENT_2026-10-08.md),
[the requested method correction](./FGH_REQUESTED_METHOD_CORRECTION_2026-10-09.md)
and [the version-1 style record](./FIGURE1_FGH_VERSION1_FREEZE_2026-10-09.md).
This Figure 2 review does not change those panels or their inference.

## Figure 2

### D — Delay/control block ratios; E — 3sTrace/control block ratios

**Review date:** 2026-10-09. **Frozen presentation:** the author-selected
3sTrace/control upload: conditioned group left, control right; one shared left
axis; no spines on the right subpanel; PT/ET/LT labels; straight statistical
lines; colored fish trajectories; connected black medians with fish IQR and
alpha 0.72. Main figures show significant comparisons only; full-test views and
tables include nonsignificant results. Whiskers are IQR, not confidence limits.
The author froze this style with a horizontal reference at **1.0 in each ratio
subpanel**, drawn **fully black (`#000000`), alpha 1.0, behind all fish lines,
points and median/IQR summaries**. This opacity requirement applies to the
reference line; the selected fish and summary transparency is retained.

**Author constraints:** retain the frozen paper metric
`legacy_distal_angular_speed`; exclude recording day entirely for D/E; use
fish-level rank tests rather than an LMM for this block-comparison review.
These are scoped D/E instructions, not a method decision for all other panels.

**Current outcome:** mean response activity / mean activity in the preceding
CS-aligned `[-15,0)` s baseline for each trial, then median eligible trial ratio
per fish/block. Window means include finite valid stationary frames. D response
is `[0,9)` s; E response is `[0,13)` s. PT = trials 10–14; ET = 65–69; LT =
90–94. D retains 29 Delay and 28 control fish throughout. E retains 40 Trace
fish throughout and 19/19/18 controls at PT/ET/LT. Its missing control LT value
is not imputed; comparisons involving LT use 18 control pairs, while PT–ET uses
19. Existing minimum eligible-trial rules remain unchanged (D: 3; E: 1).

#### Complete candidate comparison coverage

The earlier seven block tests per panel omitted PT–LT. The separate change
supplement also omitted LT−PT. The complete review now computes:

| Question | Comparisons in each assay | Test | Count per panel |
| --- | --- | --- | --- |
| Within-fish block differences | PT–ET, ET–LT, PT–LT, separately in conditioned and control fish | Two-sided paired Wilcoxon signed-rank, explicitly matched by fish ID | 6 |
| Between conditions at the same block | Conditioned versus control at PT, ET and LT | Two-sided Mann–Whitney U on fish/block ratios | 3 |
| Between-condition difference in changes | Compare fish-level ET−PT, LT−ET and LT−PT between conditioned and control groups | Two-sided Mann–Whitney U on matched within-fish changes | 3 |
| **Total** | **All three block pairs and named group contrasts** | **12 per panel; 24 across D/E** | **12** |

Block-value comparisons have lines on the paired plots. Direct change
comparisons have separate companion axes: a line between two raw block values
must not masquerade as a comparison of changes. Every expected comparison has
a result row, method, sample size and raw/adjusted p-value.

Holm across all 24 comparisons is the conservative **review** family used for
this complete candidate. It supersedes the earlier review Holm14/18 for this
version; original raw p-values reproduce exactly and previous adjusted values
remain available for audit. The paper's primary/secondary claim hierarchy and
families are unresolved. These additions are exploratory, not retrospectively
prespecified. Choose the eventual family by the questions and claims, not by
which adjustment produces stars. All comparisons need calculation and an
inspectable record; not every comparison needs a line on the clean main figure.

An omnibus test is not automatically a prerequisite for these named pairwise
questions. Cross-assay tests and one-sample tests against ratio 1 are not added
by default: they answer different questions and require a separately justified
estimand/family. More tests alone do not make an analysis complete.

#### Comments to address later

| ID | Concern and evidence | Current handling / later action |
| --- | --- | --- |
| Fig2-DE-01 | The legacy paired-line branch uses Mann–Whitney between measurements from the same fish (`legacy/scripts/5_NormalizedVigorPlotting.py:1000`), while the boxplot branch uses Wilcoxon (`:1386`). The critique applies to a specific branch, not all legacy statistics. | Current corrected review uses paired Wilcoxon with explicit fish-ID joins. Preserve literal legacy reproduction; do not silently edit it. Carry the corrected recipe identity into final panel data. |
| Fig2-DE-02 | The boxplot branch clips `Normalized vigor` at `:1249` before later tests, potentially changing ranks and paired differences. | Current tests use original values; only plotting sets axis limits. Final pipeline must maintain this analysis/rendering separation. |
| Fig2-DE-03 | Mann–Whitney is not automatically a median-difference test. | Methods/caption should describe a rank/distribution comparison. Review any stronger median-specific wording against distribution assumptions. |
| Fig2-DE-04 | Significance in conditioned fish and nonsignificance in controls does not demonstrate different changes. | All three direct between-group fish-change contrasts are computed separately. Later claims must reference those results, not compare star counts. A change difference alone does not identify a learning mechanism. |
| Fig2-DE-05 | The original review omitted PT–LT within both conditions, and the change supplement omitted LT−PT. | All three pairs are present in the complete 24-test candidate and checked in `comparison-coverage.json`. Later choose which comparisons are primary versus supporting. |
| Fig2-DE-06 | Holm14, a separate change family, and Holm18 gave different conclusions; the comprehensive set now uses Holm24. | Define the paper family and claim hierarchy before final inference. Do not select the correction after inspecting significance. Preserve raw and adjusted p-values and all earlier review versions. |
| Fig2-DE-07 | Paired Wilcoxon has assumptions about symmetric paired differences and independent fish. Exact p-values alone do not establish adequacy. | Inspect paired-change distributions, ties/zeros and influential fish. Record a justified remedy if assumptions materially fail, rather than choosing a test for a smaller p-value. No assumption-validation pass is claimed by this review. |
| Fig2-DE-08 | Missing values and different minimum-trial rules affect the population and precision. | Preserve available pairs per contrast and list fish/trial counts. Review D/E eligibility consistency and missingness before paper approval; do not silently harmonize thresholds or discard a fish because an unrelated block is missing. |
| Fig2-DE-09 | Black whiskers are fish IQR, not uncertainty about a group effect; stars do not report effect size. | Keep the IQR label. Later choose fish-level effect estimates and an appropriate interval for the intended claim; specify whether the target is a rank effect or a location effect. |
| Fig2-DE-10 | Using the frozen legacy metric does not mean exact equivalence to every old preprocessing/cohort/window/block recipe. | Carry metric/formula and authenticated outcome/cohort identities together. Retain the current 15 s baseline and named blocks; document any later scientific changes separately. |
| Fig2-DE-11 | A nonsignificant result cannot establish equivalence, no learning or extinction. | Use cautious wording; any equivalence margin or extinction estimand needs its own justification and analysis under Plans 02/03. |

**Evidence bundle:**
[complete comparisons, figures and result tables](<J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row2-block-ratio-review/complete-comparisons/20261009T082633371591Z/comparison.html>).
The bundle includes `rank-test-results.csv`, `matched-fish-pairs.csv`,
`fish-level-changes.csv`, `comparison-coverage.json`, rendering code, sidecars
and visual checks. No day field or LMM is used. All original 14 raw p-values
are reproduced; that full-comparison version reproduced the selected main
PNGs exactly. The subsequent presentation-only
[frozen style with opaque 1.0 reference](<J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row2-block-ratio-review/style-baseline-freeze/20261009T110531835455Z/comparison.html>)
adds the requested line while retaining byte-identical saved statistical
results and fish/block data. Difference companion plots keep their separate
zero reference; 1.0 is the reference for the response/baseline ratio panels.

Method references: [Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html),
[Mann–Whitney](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html),
and [Gelman and Stern](https://www.stat.columbia.edu/~gelman/research/published/signif4.pdf).

#### Ratio versus legacy log-median outcome review — 2026-10-09

**SUPERSEDED scientific definition:** this earlier comparison included stationary frames in A. The author's later bout-only instruction below replaces that definition and its recommendation. Its files remain historical evidence.

**Intended claim:** population conditioning/suppression through mean activity, with typical movement intensity as a distinct supporting question. **Author request:** create a separate legacy-log candidate, include updated statistics, criticize both versions and recommend the better primary outcome. No author selection of the new outcome is inferred.

**Current handling:** [side-by-side comparison](../reviews/figure2_ratio_vs_log_20261009/comparison.html) keeps the selected A ratios and adds B: median ln(positive valid moving vigor) in response minus baseline, followed by fish/block trial median. Both use the frozen metric, corrected measured-time sources, [-15,0) baseline, assay 9/13 s response windows, blocks 10–14/65–69/90–94 and minimum trial rules D=3/E=1. B implements the historical outcome; historical rolling smoothing/downsampling, interpolation, inclusive endpoints and detector are not restored. Reference is 0 for B and 1 for A; both are opaque black behind data. No day adjustment or LMM is used. Both have fresh 24 rank tests, separate Holm24, all-test tables and direct-change companions. The 24 A statistics reproduce the selected result. Extra B missingness: D control: 33/420 otherwise-eligible trials additionally undefined; D delay: 7/435 otherwise-eligible trials additionally undefined; E control: 0/280 otherwise-eligible trials additionally undefined; E trace: 7/600 otherwise-eligible trials additionally undefined.

| ID | Concern and evidence | Required remedy / current handling |
| --- | --- | --- |
| Fig2-DE-12 | A mixes movement frequency and intensity, can amplify a noisy small baseline, and uses frame-sensitive window means. B excludes stillness and uses robust positive moving-frame medians; this changes the biological target and can hide reduced movement probability. | **Recommendation, not decision:** A primary for overall conditioning/suppression; B supporting movement-intensity analysis. Choose by claim, not star count. Decompose duty/probability and intensity before making mechanism claims. |
| Fig2-DE-13 | B is undefined without positive movement in either window and may drop strongly suppressed trials/fish. Source cohorts are identical, eligible samples can differ. | Saved `trial_sample_flow.csv`, per-trial counts and `paired_version_comparison.csv` expose exclusions and trial availability. Review informative missingness and D/E thresholds under Plans02/07 before scientific freeze. No missing response is replaced with zero. |
| Fig2-DE-14 | Even a pure log transform can change paired Wilcoxon absolute-difference ranks and tests of between-group changes. B additionally changes masks/window aggregation. | Recompute all tests; pure-log-of-existing-block-ratio sensitivity saved separately. All same-block MW raw p-values remain invariant for that sensitivity. No borrowed legacy stars or old LME transformations. |
| Fig2-DE-15 | Outcome comparison adds analytic choices; separate Holm24 families do not authorize selecting the smaller adjusted p-value across versions. Pairwise changes show sample asymmetry; signed-rank adequacy remains unapproved. | Preserve every result, use paired sign-test/Holm12 only as a separately labeled direction sensitivity, and define paper primary/secondary estimands and families before final inference (Plans02; DECISIONS register when author selects). |
| Fig2-DE-16 | IQR is not effect uncertainty, and within-group significance does not demonstrate different changes. | Fresh direct-change tests and rank-biserial/difference-of-median-change estimates with 5,000 fish-bootstrap pointwise 95% intervals are saved. These descriptive intervals are not multiplicity-adjusted. There are 2 adjusted significant direct-change results across both versions; ns is not equivalence. |

**Presentation:** same paired style, connected black medians/fish IQR, significant-only main annotations and an all-test toggle. B is an unselected scientific candidate; no freeze command or current-selection overwrite. F remains placeholder/inconclusive. [Full critique, results and recommendation](../reviews/figure2_ratio_vs_log_20261009/critique.md) records exact sources, formulas, deviations and unresolved validation. Outstanding biological/coverage/assumption choices remain under Plans02/07 and require a decision entry if selected.

#### Updated D/E comparison: both versions exclude no-bout periods — 2026-10-09

**Author instruction:** “whenever there is no bout (meaning, no movement, meaning vigor is nan) those values must be ignored all the time and everywhere.” This is a scientific sampling rule, applied before means, medians, scaling and logarithms. A no-bout frame stays NaN; an empty baseline/response makes the trial undefined. Raw source measurements and historical/frozen files remain preserved.

**Two active choices:** A = response mean bout vigor / baseline mean bout vigor (reference 1); B = median ln(positive bout-frame vigor) in response minus baseline (reference 0). Both use trial medians within fish/block, original cohorts/metric, baseline [-15,0), response D [0,9)/E [0,13), PT10–14/ET65–69/LT90–94, and D minimum 3/E minimum 1 eligible trials. B implements the legacy outcome using current corrected preprocessing; historical smoothing/downsampling is not restored. The all-tests toggle is an annotation option, not a third scientific version.

[Updated comparison](../reviews/figure2_bout_only_ratio_vs_log_20261009/comparison.html) · [full critique](../reviews/figure2_bout_only_ratio_vs_log_20261009/critique.md) · [all 48 tests](../reviews/figure2_bout_only_ratio_vs_log_20261009/all_statistics.csv) · [fish counts and medians](../reviews/figure2_bout_only_ratio_vs_log_20261009/summary.csv). Recomputed Wilcoxon and Mann–Whitney tests, with Holm24 separately per outcome; direct conditioned/control changes and pointwise bootstrap effect intervals are included. Eligible selected trials: D815/855, E873/885 for both outcomes. Main whiskers are fish IQR.

**Direct change tests (Holm24):** D PT→ET A p=0.005161, B p=0.001630; D ET→LT B p=0.000740 (A not significant). Neither version establishes a Trace/control differential change after correction. Other block/within-condition tests are in the CSV; significance within one group alone does not establish a between-group change.

**Critique and recommendation:** both outcomes now describe intensity conditional on movement, and neither describes total activity or movement frequency. A retains peak sensitivity and unstable small-baseline ratios, but is intuitive and appropriate for average bout intensity. B reduces peak influence and represents proportional changes symmetrically, but excludes zero-valued bout samples and uses less intuitive log units. Recommend B for typical bout intensity and A as sensitivity; choose A if mean intensity including peaks is the intended target. Do not choose by p-values. Report exclusions and assess movement probability/frequency separately. No A/B author selection or new freeze is inferred.

**B statistical audit, 2026-10-09:** [review and processing steps](../reviews/figure2_bout_only_ratio_vs_log_20261009/B_STATISTICAL_REVIEW.md). All 24 existing B tests reproduce. Existing stars do not test the zero baseline reference. A separate exploratory Holm12 audit finds Delay ET below zero (27/29 fish; median B=-0.07521, about 7.2% reduction; Wilcoxon p=0.000007555, sign p=0.00001948); this also survives combining the 24 original and 12 zero-reference tests (Holm36 p=0.000022665). Trace ET is not significant against zero (Holm12 p=0.5298), and its conditioned/control change remains inconclusive (Holm24 p=0.8283). Existing stars remain unchanged. Paired-change skewness, informative no-bout exclusions, D/E minimum-trial differences, bout-frame weighting, experimental-unit/day structure, and exploratory estimator selection must be disclosed or assessed before final inference.

#### Accepted B: corrected statistics and D/E freezes — 2026-10-09

**Author instruction:** correct the statistical tests if needed, then freeze the selected version B and prepare a processing handoff. This decision supersedes the preceding unselected status, Wilcoxon/Mann–Whitney stars, Holm24 families and separate zero-reference audit. Historical results remain preserved.

**Correction:** exact two-sided sign tests replace signed-rank tests for paired fish changes and comparisons with zero, avoiding the symmetry assumption. Brunner–Munzel tests with t calibration replace independent-group rank tests for the unequal-shape rank/probability question. These independent tests do not specifically test median differences. One **Holm36 family** covers D/E: 12 zero-reference, 12 paired-change, 6 between-group block and 6 between-group change comparisons. Fish are the units; paired comparisons use fish eligible in both blocks. B values, masks, windows, cohorts and minimum-trial rules are unchanged.

**Corrected evidence:** Delay ET is below zero (Holm36 p=0.0000552); Delay/control differ at ET (p=0.000000153). Delay PT→ET decreases (p=0.000488) and ET→LT increases (p=0.00322); both changes differ from controls (p=0.000120 and p=0.0000131). No Trace comparison survives Holm36. Main whiskers remain fish IQR; pointwise bootstrap intervals are descriptive and are not multiplicity-adjusted.

**Freeze:** [D record](../reviews/figure2_B_freeze_20261009/D.freeze.json), [E record](../reviews/figure2_B_freeze_20261009/E.freeze.json), [corrected tests](../reviews/figure2_B_freeze_20261009/statistics.csv), [review](../reviews/figure2_B_freeze_20261009/comparison.html). Both passed the explicit freeze gate with specification 1.1.1 at 54.9 × 53.2 mm, with semantic elements, source/data/export hashes, renderer checks and visual review. [Scoped current selection](../configs/paper-figures/figure2-DE-B-freeze-20261009.json) is newer than the whole-figure assembly. A remains sensitivity evidence; F and the whole Figure 2 are not frozen by this decision.

**Remaining limits:** inference assumes independent fish and does not adjust day/tank. No-bout exclusions may be informative; D/E use different eligibility thresholds. B measures typical positive bout-frame intensity, weighted by frame duration within windows, and cannot establish overall activity suppression or movement frequency. Outcome/test revision after examining results is exploratory. These limits remain disclosed in the frozen methods.

**Reusable scaffold:** [small processing handoff for Figure 1 E–H and whole Figure 2](./HANDOFF_BOUT_LOG_VIGOR_PANELS_2026-10-09.md). Existing Figure 1 quantile-scaled freezes must not be silently replaced by B. Full-time panels require authenticated frame windows beyond the 15 cached D/E block trials. The future assembly must accommodate the frozen panel dimensions before its own freeze.

### F — 10sTrace/control block ratios

No authenticated cohort and processed source table are available for this
review. Keep the panel inconclusive; do not invent comparisons, counts or
annotations. Once inputs are authenticated, review its intended questions and
eligibility using the same entry format before choosing an inferential family.

### A–C and G–I — separate heatmap and trial-trajectory reviews

Do not transfer D/E rank tests or exclusions to these panels automatically.
The existing [Delay G LME critique](../docs/analysis/figures/FIGURE2_DELAY_LME_CRITIQUE_2026-10-09.md)
belongs to its trial-trajectory questions; its diagnostics and estimands remain
separate from this block review. Add individual entries here when reviewing
each remaining panel, linking the applicable analysis evidence.

## Figure 3

Add individual panel entries as learner-method and validation questions are
reviewed. Follow [the learner-method plan](./04_LEARNER_METHOD_REVIEW.md) and
[learner analysis](./05_LEARNER_ANALYSIS.md). No new panel-specific inference
is approved by this Figure 2 discussion.

## Figure 4

Add individual panel entries as CR profiles and their intended claims are
reviewed. Follow [integrated analysis and CR profiles](./07_INTEGRATED_ANALYSIS_AND_CR_PROFILES.md).
Keep learner-stratified descriptions distinct from independent inference.

## Supplementary figures

Add entries with the approved parent figure and stable content/panel ID. Do
not assign final supplementary numbers merely to organize comments. Follow
[supplementary figures](./10_SUPPLEMENTARY_FIGURES.md) and retain the same
claim, outcome, comparison, missingness and evidence fields used above.
