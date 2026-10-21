# Fresh-chat handoff: 3sTrace learners and a better Figure 1 example

## Author request and expected result

Identify learners in the expanded fixed-3sTrace dataset, choose the best
illustrative fish in the fresh chat, and recommend a better candidate for
Figure 1's 3sTrace example. Carry out the analysis and make a concrete choice;
do not stop at proposing a workflow. Deliver a ranked shortlist, one preferred
fish with reasons and limitations, and matched visual comparisons against the
current example. The author authorized the new chat and this analysis.

The existing example is **20230307_12**. After the recorded control-first
reorder, the working Figure 1 row is F Control, G Delay, H 3sTrace. The older
freeze selection's `fish` dictionary retains pre-reorder panel labels; its
author-instruction text correctly identifies Delay 20221115_07, Trace
20230307_12 and Control 20221115_09. Resolve actual roles through current state
and the semantic registry, not that dictionary alone.

## Read first and preserve

- Repository AGENTS.md and `.agents/skills/paper-figures/SKILL.md`.
- `docs/paper/figures/shared/iteration-workflow.md` and `nature-style.md`.
- `docs/paper/figures/figure-01/current-state.json`, its current legend and
  selected scientific records. Resolve relocated paths; `/Volumes/JOAQUIM/`
  normally maps to J: on this Windows machine, but authenticate bytes.
- `docs/paper/methods/individual-learning.md`, Figure 3's README, current state,
  review, and `docs/running-analysis/figure3-main-paper-draft.md` and
  `figure3-individual-atlas.md` for existing methods and unresolved validation.

Keep Figure 1/2 current artwork, frozen originals and other uncommitted work
intact. This task authorizes candidate analysis and comparison; it does not
request freezing or silently replacing Figure 1's selected example. Make the
recommended replacement concrete and reviewable in a separate candidate.
Use guarded publication if a subsequent author instruction selects it.

## Cohort and authenticated numerical starting points

Processed project: `F:/Digested Data/all3sTrace-full-v1/`.
Raw sources are on D:. Treat F: and D: as read-only.
Permanent numerical results, media and review HTML belong on mounted JOAQUIM,
at a new explicit destination such as
`J:/ClassicalConditioning Outputs/current/reviews/figure-01-trace3-learner-selection-20261019/`.
Check for existing work before creating files; preserve it if present.

The expanded preparation processed **183 fish (91 Control, 92 Trace)**.
The selected common Figure 2 B/E/H roster is **179 fish (90 Control, 89 Trace)**.
Exclude these four from the candidate display cohort and retain explicit flags:

- 20230426_01
- 20230802_01
- 20230803_12
- 20230822_04

Their heatmap timing gates fail through buffer-capacity exceedance; exported
FrameID gaps are zero. Do not describe them as confirmed counts of missing
exposures. Increasing-trace recordings without fixed 13 s CS-to-US timing are
separately flagged and must not enter the fixed-3sTrace cohort. Fifteen held
intake identities remain held. Do not revive a historical 53/59-fish subset as
the complete expanded cohort. Controls provide calibration/reference; rank
conditioned Trace fish as candidate examples.

Authoritative analysis directory:
`J:/ClassicalConditioning Outputs/current/reviews/figure-02-3strace/`

- `plan.json`: expanded 183-fish input roster/provenance.
- `fish/<recording_id>.json`: authenticated source paths, per-fish output
  hashes, cadence/detector/timing evidence and flags.
- `fish/<recording_id>-bins.parquet`: individual 90×40 heatmap bins, eligible
  frame counts, absolute log medians, baseline and centred values.
- `fish/<recording_id>-windows.parquet`: measured-time log-median trial outcomes
  and positive moving-frame counts.
- `matched-delay-statistics/plan.json`: explicit selected 179-fish roster and
  four exclusions. Use this roster after authenticating inputs.
- `matched-delay-statistics/windows.parquet`: 179×90 scheduled trial rows;
  `legacy_log_median_difference`, baseline/response log medians and counts.
- `matched-delay-statistics/E-fish-blocks.parquet`: PT10–14, ET65–69, LT90–94.
  These test-phase blocks are not the late-training acquisition epoch.
- `matched-delay-statistics/ab-analysis-review/`: numerical A/B review including
  per-fish/phase summaries, independent pooling and fixed-timing checks.

Supported implementations to inspect/reuse:
`scripts/build_figure2_trace3_panels.py`,
`scripts/analyze_figure2_trace3_matched.py`,
`src/classical_conditioning/figures/pooled_log_heatmap.py`,
`scripts/render_figure1_best_learners_version08.py`,
`scripts/build_figure3_main_paper.py`,
`scripts/build_figure3_individual_atlas.py`.
Historical filenames and “best learner” labels are not current scientific
authority. Reuse verified numerical caches instead of regenerating large raw
tracking/frame data unless needed to resolve a specific discrepancy.

## Scientific definitions and learner-selection requirements

Use tail bend angular speed (`legacy_distal_angular_speed`), the standing metric
restriction. The current observed trial outcome is median(log positive valid
moving-frame vigor in response) minus the corresponding baseline median.
Baseline [-15,0) s; fixed-3sTrace response [0,13) s; intervals are left-closed,
right-open. Negative means lower typical positive bout vigor during response.
This is not bout probability, total movement, or a learner label by itself.

The existing Figure 3 descriptive acquisition ranking is median trial outcome
over **Pre5–14 minus late Train40–64**: positive means greater suppression.
Read its method review and validation limitations before transferring it.
Training is 15–64; Test is 65–94. Recovery/persistence, catch trials, actual US
responses, baseline drift, movement frequency and coverage are separate audit
properties. Do not rank by post-US vigor or extreme single trials. Do not use
the population mixed model or its group significance as individual evidence.

There is currently **no approved validated binary learner threshold**. The
author's request authorizes investigating individual learners, ranking and
selecting illustrative candidates. Implement an explicit, justified operational
rule if needed, retain continuous scores/coverage and distinguish provisional
classification from validated categories. Do not call every failure to classify
a nonlearner. Do not tune thresholds after seeing attractive fish without
disclosing the selection. Use controls for reference and sensitivity checks,
retain the full roster and exclusions, and assess stability across reasonable
epochs/coverage choices. Avoid claiming prospectively held-out validation from
previously inspected Test trials.

For candidate artwork, reproduce the selected Figure 1 heatmap method exactly:
cadence-reconstructed timing; finite positive valid adjacent detected-bout
original frames; natural log; direct 1 s median; subtract each fish/trial's
median finite baseline-bin medians [-15,0); no amplitude scaling; managua_r
limits ±0.25; NaN black; acquisition5–94 displayed1–90. This heatmap estimator
differs from the unbinned measured-time trial outcome. Show actual fixed-13 s
US timing, shared scales and matched visual dimensions for every candidate.
Use current Nature profile and shared theme, editable SVG and associated legend,
1 mm invisible frames, exact saved-render checks, and scoped guards.

## Findings from the preceding review

`records/figure-reviews/figure-02/heatmap-ab-analysis-review-20261019.md`
documents that B's paler background does not arise from a detected calculation
or colour-scale error. More contributing fish and cancellation of heterogeneous
responses reduce pooled variation. All 89 included Trace fish passed a fresh
recorded-event check against fixed13 s timing. Individual fish can show much
stronger patterns than the average. Do not infer individual absence of learning
from the pooled heatmap.

One remaining methodological asymmetry is explicitly documented: Figure 2 E
currently requires ≥1 eligible trial per block, while Delay D requires ≥3.
The prior author request to match Delay was not implemented literally in that
respect. Do not silently copy this threshold into learner identification or
claim all analysis settings are identical. The one-trial mathematical minimum
in Figure 3's draft is also not a validated scientific coverage rule.

## Deliverables for the fresh chat

1. Authenticate roster, source/output hashes, protocol timing and current
   Figure 1 example/selection. Record code, inputs, exclusions and limitations.
2. Compute transparent individual acquisition scores and coverage for the full
   included roster, audit the strongest conditioned fish, and compare ranking
   stability and controls. Separate intensity, movement availability and actual
   US responses. Justify any operational learner definition.
3. Rank a short list of the strongest credible Trace examples; compare each
   with 20230307_12 using identical heatmap and trial-outcome scales. Check
   tracking/timing quality and consistency across training/catch/test periods.
4. Choose one preferred Figure 1 candidate with an explicit quantitative and
   visual rationale, plus alternatives and honest uncertainty. If the current
   fish remains the strongest credible example, say so with evidence.
5. Save a readable handout/report, machine-readable full score/rank table,
   exclusion/coverage records, source hashes and editable candidate comparison;
   open the actual saved comparison for author review. Leave selected Figure 1/2
   unchanged until an explicit replacement instruction. No freeze is requested.

This handoff's companion `trace3-learner-handoff-20261019.json` binds the
current Figure 1/2 states and principal cached inputs at dispatch time. Re-read
current authority in the fresh chat and reconcile any changes instead of
assuming the dispatch snapshot remains current.
