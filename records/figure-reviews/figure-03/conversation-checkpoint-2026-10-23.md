# Figure 3 conversation checkpoint — 23 October 2026

This is a substantive summary for use after deleting this chat, not a verbatim
transcript or a scientific approval. It extends the dated 16 October summary.
The standalone HTML embeds both summaries, the relevant saved outputs and a
coverage index. Figure presentation state remains separate from this record.

## Goal and author decisions

The author wants to understand the analyses well enough to review them without
AI, compare alternatives, and later choose the final analysis, figure and panels
to freeze. This chat did not choose a scientifically validated winning method or
request a scientific freeze.

1. Figure 3 should quantify/explain individual conditioned expression, its
   definition and supporting individual evidence. Its main-paper version must
   show only one selected analysis path, not a comparison of competing recipes.
2. Control/Learner/Nonlearner behavioral comparisons belong to Figure 4, both
   within each experiment and across experiments. Controls may appear as
   reference/calibration evidence in Figure 3.
3. Use only legacy distal tail-bend angular speed,
   `legacy_distal_angular_speed`. Alternative-metric Figure 3 work was removed
   under the earlier explicit author request; preserved legacy versions were
   not authorized for indiscriminate deletion.
4. Main trial plots use Figure 2 G's observed outcome: median ln positive eligible
   CR-bout vigor minus median ln positive eligible baseline-bout vigor. Keep
   exact legacy classifier inputs separately. The author explicitly confirmed
   this distinction. Do not apply a second log or confuse it with ln of a ratio
   of arithmetic bout means.
5. Heatmaps use native managua_r at -0.25/+0.25, undefined intensity black and
   numerical values unclipped. This was clarified against Figure 2 A's heatmap
   scale and G's condition colors: control #00aeef, Delay #ec008c. The heatmap
   estimator (direct 1 s frame-log medians, centred by median finite baseline-bin
   medians) differs from the G pooled-window trial estimator.
6. Select a few deliberately illustrative fish for the main paper, while showing
   all fish in supplementary material. This is explanation, not random sampling
   or independent validation. Preserve uncertain/unclassified cases.
7. Distinguish technical exclusion, trial/outcome availability, feature/model
   eligibility, uncertain evidence and criterion-not-met. Undefined no-bout
   conditional vigor is not zero. Show movement separately. Detector availability
   does not establish independent tracking completeness.
8. Compare methods outside the main figure before selecting its one path.
   Maximizing learner counts or retention alone does not establish accuracy.

## Existing data and exploratory outputs

The current authenticated Figure 3 pilot has 57 fish: 29 Delay and 28 controls,
90 scheduled CS trials each (5–94), 5,130 rows and 4,811 defined paired G outcomes.
Only Delay/control processed inputs were authenticated; Trace remains incomplete.
Historical 59-fish/53-fish 3sTrace drafts do not fill this gap.

The complete exploratory atlas includes all 57 fish, original classifier inputs,
four original rules, per-fish decisions and eligibility reasons, heatmaps,
G observations, movement/availability context and source/configuration evidence.
Its selected state is docs/paper/figures/figure-03/exploratory-state.json.
Its external root is:

/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure3-learner-exploration-g-style-20261015

The first selected illustrations were 20240618_08 (four-rule flag agreement),
20240612_08 (four-rule unflagged agreement), 20240702_01 (disagreement) and
20221123_02 (near the WIP threshold). Same-data agreement did not validate labels.

JOAQUIM was temporarily disconnected and the author offered to reconnect it.
It became available and authenticated numerical sources were used. No temporary
browser cache was substituted for missing scientific inputs.

## Four legacy rules and their output counts

All four original routes use mixed-model fish-specific epoch effects/BLUPs,
acquisition-direction suppression and later recovery-direction change. Their
final labels describe a particular conjunction, not a direct biological truth.
Nominal uses middle trials 60–69; new/improved/WIP use 40–69. All use Pre5–14
and LateTest85–94, so Test data enter the labels and cannot validate them as
held-out observations.

Nominal/new model log response arithmetic means with added one and baseline
adjustment; improved omits the added one; WIP uses log normalized arithmetic-mean
vigor. Outcomes, thresholds, directional rules and uncertainty treatments differ.
The active nominal directional shift is 0.5 SE, whereas displayed intervals use
1.96 SE. New uses a 0.60 directional Gaussian score; these are not validated
probabilities of learning. Improved/WIP thresholds use empirical 90th/95th
control percentiles, with score and directional conditions both required.

Current bout-only reruns on the identical input bundle:

| Rule | Delay flagged/eligible | Delay unclassified/all | Control flagged/eligible | Control unclassified/all |
|---|---:|---:|---:|---:|
| Nominal | 6/27 | 2/29 | 1/23 | 5/28 |
| New | 8/26 | 3/29 | 3/23 | 5/28 |
| Improved | 9/26 | 3/29 | 3/23 | 5/28 |
| WIP | 16/26 | 3/29 | 2/23 | 5/28 |

The earlier 24 September run found 7/7/6/19 Delay learners and 0/2/1/2 control
flags, with all 29/28 fish eligible. It preceded the documented current bout-only
policy. Keep these tables separate: their difference does not isolate one
statistical correction. WIP finds the most current legacy learners, but no
independent evidence established it as the most accurate method. Same-data
control flags do not independently demonstrate a claimed false-positive rate.

Saved historical scalar individual/grid examples were authenticated. Source
supports old raw/scaled heatmap grids, but those saved Heatmap_Grid files were
not located in the inspected archive. Old grid construction could crop images
and omit unclassified fish. Its cumulative BLUP overlay is a feature schematic,
not a fitted trial-by-trial trajectory. The full guide explains these traps.

## Discarding, retention and validation concerns

Legacy preprocessing rules include final-US viability, responses to training
US events, optional Re-Train checks and minimum bout-bearing trials in selected
baseline/CR blocks. The inspected discard switch is off, but its default does
not prove what a historical interactive session ran. Some scripts reuse discard
lists for plots without equivalent numerical/model filtering.

Normalized-vigor preparation's six-trial rule accepts ANY qualifying block.
Learner feature eligibility instead requires at least three usable trials in
EACH selected five-trial sub-block, six in each active epoch and complete selected
features. These are distinct from primary technical exclusions.

Projected rules retain control/Delay counts of 28/29 (ANY block), 25/29
(baseline coverage), 24/29 (CR/intersection), 26/27 (all training US), 24/27
(combined with a last-US proxy), and 23/27 (ALL-required-blocks alternative).
The source-only protocol sensitivity retains 28/28 by removing 20221122_03.
Those projections are not exact reconstructed historical cohorts: original
segmented Bout/700-FPS and annotated last-US-end inputs/rosters are incomplete.
Response-dependent filtering can inflate apparent effects. No recipe was
selected here because it yielded significance or more attractive learner counts.

Earlier Figure 2 G spline/state-space and other timing checks were consulted as
background. Their poor individual uncertainty coverage did not disprove all
BLUP approaches or select a Figure 3 method. Numerical tests, fit success and
saved figures do not establish inferential calibration.

## Main-paper iterations and corrections

The first new main-paper draft combined a G descriptive score, four-rule
comparison counts and legacy-based individual labels. The author rejected the
mixed-method composition: the main figure must contain only the best selected
analysis path. That draft remains preserved.

The selected single-path draft contains a: G outcome/contrast definition;
b: all-fish continuous scores; c: larger contrast (20240618_08);
d: intermediate contrast (20240618_02); e: near-zero contrast (20240612_08).
Its score is median G outcome in Pre5–14 minus median G outcome in late
Train40–64. Positive indicates more suppression. Recovery is separate.
Black bars show those epoch medians. Main G axes share -0.6/+0.6; heatmaps
remain at +/-0.25. All fish have a mathematical contrast, but Pre coverage ranges
4–10 and late-training coverage 5–25; this is not established adequate precision.

No binary G classifier, individual calibrated interval or learner count was
introduced. A count of zero would misrepresent an undefined classification.
Previously inspected Test data cannot retrospectively be called prospectively
untouched. Choosing near-zero examples does not establish absence of learning.

The author asked about BLUPs. The assistant acknowledged that it had simplified
too early: no completed comparison established the median contrast as superior
to a properly implemented BLUP analysis. The author requested a statistically
corrected BLUP candidate and a comparison of all versions before selection.

## BLUP work still unfinished in this chat

A prototype exists in
src/classical_conditioning/analysis/review/blup_comparison.py. It jointly fits
condition-by-epoch fixed means and correlated fish random epoch means, with
condition residual variances and optional within-epoch AR1 based on scheduled
gaps. It reports fixed-plus-random fish acquisition and a separate control-
relative contrast, accounting for shared reference and fixed/random dependence
in known-covariance prediction-error calculations. Refitted parametric bootstrap
and stress-simulation functions were written.

Inspection identified problems in the legacy WIP two-stage control anchoring:
a second model with random epoch slopes but no fixed epoch effect, and interval
terms combined without the full joint dependence/variance-parameter uncertainty.
Those findings motivated a candidate correction; they do not validate it.

A first model fit and five-refit smoke check ran, and eight implementation tests
passed. The full bootstrap calibration, independent simulation validation,
coverage/retention policy, recovery analysis, sensitivities and comparison plots
were not finished or published. This was explicitly acknowledged when the
author later asked whether the task had been completed. The external BLUP work
area still contains only the starting ticket at the 23 October inspection.
No corrected-BLUP learner count or validated candidate figure exists in the
recorded outputs. Do not treat the prototype as a completed statistically
validated method or a biological learning-probability model.

A future comparison should include unpooled mean and median contrasts on the
same G windows, to separate pooling/shrinkage from a mean-versus-median estimand
change. Legacy rules retain their distinct outcomes and recovery requirements.
Alternative control-rank, onset/change-point, functional/PCA, mixture and
state-space ideas were discussed, not all executed or validated.

## Independent-review HTML, reusable skill and link repair

The author selected plain language with expandable equations, assumptions and
code references. A 14-chapter standalone HTML was built with all 57 original
fish cards, 228 legacy decisions, 57 G scores, both main-paper drafts, historical
output examples, instantiated configurations/logs, code downloads, a toy BLUP
calculator and a worksheet. Figure/analysis selections were preserved.

A personal reusable skill, $analysis-review-guide, was created at:
/Users/joaquim/.codex/skills/analysis-review-guide/SKILL.md

It covers evidence inventories, source authentication, distinctions between
proposals/execution/validation, counts/exclusions, layered HTML explanations and
independent-review notes. It and its static checker were validated; the checker
was exercised on seven meaningful valid/failure scenarios. The current installed
skill may include later refinements from outside this chat; the preservation
package is explicitly a snapshot of the installed files, not evidence that every
later feature was authored here.

The author asked to ensure links work. The localhost server had stopped and
was restarted. Embedded data-URL source downloads did not trigger in the in-app
browser, so a guarded repair replaced their click handling with offline Blob
downloads. Scientific data, figures and note schema were unchanged. Duplicate
source basenames were disambiguated. Fourteen anchors, 17 payloads and three
external references were checked; real CSV and Python downloads were verified
against their bytes. The standalone file does not need a running server, but
external references need network access and the file needs accessible storage.

## What to preserve before deleting the chat

Keep the standalone HTML and, if needed, its companion source/evidence directory
on JOAQUIM. The HTML contains the substantive record and saved outputs; localhost
URLs are conveniences, not durable scientific authority. Browser-local review
notes are not automatically written into the HTML. Export them as JSON before
closing/removing the chat or changing browser/file origin. Preserve exported
notes separately; importing them restores the worksheet, not scientific approval.

The latest HTML contains a conversation-preservation chapter, coverage index,
this checkpoint and the earlier summary as offline downloads. It retains the
original cases, data, legends, repaired downloads and storage key. It is not a
verbatim chat transcript or a claim that every tool log is preserved.

Next scientific work remains: complete the corrected BLUP comparison and its
validation; review coverage, assumptions, control calibration and sensitivities;
then select one analysis path and final main/supplementary panel contents.
Resolve Trace inputs and cohort/threshold choices as needed. Only a later
explicit scoped freeze request invokes the scientific/presentation freeze gate.
Keep immutable source/frozen dependencies, historical versions, manual edits
and other figures. New paper artwork uses Nature's selected profile and active
1 mm invisible-frame policy; presentation checks do not approve science.
