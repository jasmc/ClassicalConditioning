# Fixed-10sTrace learners and Figure 1 candidate selection

## Author request and execution order

Author direction in this chat, 20 October 2026: after the fixed-10sTrace
processing and requested Figure 2 analysis, adapt and execute the 3sTrace
learner-selection request for **10sTrace**. Carry out the analysis and make a
concrete recommendation in this same chat; preparing this handoff alone does
not complete the request. Do not create another chat.

First complete and verify Figure 2 C/F/I and the authorized D/E corrections
described in `docs/paper/figures/figure-02/panels-c-f-i/README.md`. Then execute
the individual learner analysis below. Keep the existing 45-minute heartbeat
active through both tasks; disable it only after both sets of deliverables
are successfully produced and verified. This supersedes the older instruction
to disable the heartbeat immediately after Figure 2 completion.

## Read first and preserve

Read repository AGENTS.md, the paper-figures skill, the authoritative figure
workflow and Nature guidance. Read the original methodological handoff and
its companion JSON:

- `records/figure-reviews/figure-01/trace3-learner-handoff-20261019.md`
- `records/figure-reviews/figure-01/trace3-learner-handoff-20261019.json`

Those documents provide method context, not the 10sTrace roster or timing.
Re-read `docs/paper/methods/individual-learning.md`, Figure 3's current methods,
method review and validation limitations. Resolve current Figure 1/2 state,
selection records and scientific mappings again before comparison/publication.
Preserve all existing/uncommitted work and frozen originals. The author
authorizes analysis, provisional learner identification and candidate selection;
no Figure 1/2 replacement or freeze is authorized by this request.

## Cohort, timing and existing comparator

Processed project: `F:/Digested Data/all10sTrace-full-v1/`.
Use the authenticated fixed-10sTrace common roster and verified caches resulting
from the C/F/I work. The preserved processing inventory contains **63 acquisition
identities: 31 conditioned Trace and 32 associated controls**. This inventory
is not an automatically approved complete paper analysis roster. Reconcile all
63 with processing completion, protocol evidence, common C/F/I membership,
eligibility, exclusions and flags in a complete score/coverage/exclusion table.
Do not import the 179-fish 3sTrace roster or its four timing exclusions.
Increasing-trace recordings remain separate; determine any 10sTrace-specific
timing exclusions from authenticated evidence. Do not silently drop failures.

The 32 controls share acquisition identities with historical 3sTrace inputs.
Keep this provenance and never count them as independent new acquisitions
across assays. Use controls as reference; rank conditioned fixed-10sTrace fish.

Actual training **CS onset to US onset is approximately 20 s**, with a 10 s CS
and a nominal 10 s trace interval. Authenticate actual events for every
candidate and the full roster; use the recorded method's timing tolerance.
The prior raw audit found 46 training pairings per complete Trace recording,
approximately 19.971–20.063 s apart. Do not substitute the 3sTrace 13 s schedule.
Keep shortened recording **20250715_08** explicit: 53 CS/54 US, 39 observed
training CSs, 36 near-20 s pairings, missing later coverage. Successful
processing does not make this a complete protocol or candidate.

The provisional legacy-highlighted 10sTrace comparator is **20230307_04**,
not 20230307_12 (the selected 3sTrace example). It is a standalone candidate,
not a selected Figure 1 replacement or validated cohort-wide best learner.
Resolve it through
`docs/paper/figures/figure-01/10strace-example/current-state.json`.
Its authenticated individual 90×40 heatmap, bins and calculation report are at
`J:/ClassicalConditioning Outputs/current/reviews/figure-01-10strace-example-20261019/`.
Legacy explicit designation is in `legacy/scripts/5_NormalizedVigorPlotting.py`
and the example in `legacy/scripts/2_ExampleFishPlotting.py`.

## Adapted analysis request

Identify learners in the expanded fixed-10sTrace dataset and choose the
strongest credible illustrative conditioned fish as a Figure 1 candidate.
Use a transparent individual analysis, justified coverage and sensitivity
checks, controls as reference, and explicit distinctions between provisional
operational classification and validated biological categories.

Inspect the existing **Pre5–14 minus lateTrain40–64** acquisition contrast and
method review. Training is 15–64; Test is 65–94. Use the standing tail bend
angular speed metric (`legacy_distal_angular_speed`). The primary observed
trial outcome is the measured-time median(log positive valid moving-frame
vigor in response) minus the corresponding baseline median, baseline
**[-15,0)** s and 10sTrace response **[0,20)** s. Positive acquisition contrast
means greater suppression. This conditional-on-movement outcome is distinct
from bout probability, overall movement and missing tracking. Do not infer
individual learners from population-model significance or post-US responses.

Retain the complete continuous score/coverage/exclusion table. Justify the
primary coverage rule before inspecting ranked outcomes, with reasonable
epoch and coverage sensitivity analyses, trial/block deletion and ranking
stability. The D/E/F >=3-of-5 rule does not by itself validate a learner
classifier or coverage threshold over different epochs. Use eligible controls
for reference with explicit reference counts, comparability limits and
leave-one-control-out diagnostics where appropriate. Any operational learner
threshold must be transparent, separately justified and identified as
provisional; there is no approved validated binary learner threshold.
Distinguish insufficient data from uncertain conditioned-direction evidence.

Audit the strongest candidates for tracking/timing quality, baseline drift,
movement availability, catch-trial behavior, acquisition consistency and
later recovery/persistence. Keep these audit properties separate from the
ranking definition. Do not rank by an extreme single trial or post-US vigor.
Do not claim prospectively held-out validation from previously inspected Test
trials. Report limitations honestly if controls or coverage do not support
calibration or categorical conclusions.

Compare a shortlist with **20230307_04** using matched 1 s heatmaps and identical
trial-outcome scales. Reproduce the supported selected heatmap estimator:
cadence-reconstructed timing; finite positive valid adjacent detected-bout
original frames; natural log; direct 1 s medians; subtract each fish/trial's
median finite baseline-bin medians over [-15,0); equal scales, no amplitude
scaling; native managua_r +/-0.25; missing black; acquisition5–94 displayed1–90.
Show actual fixed-20 s US timing and shared dimensions. Keep the heatmap and
unbinned trial-outcome estimators distinct. Reuse authenticated caches from
C/F/I rather than regenerate large raw data unless a specific discrepancy
requires it.

Select one preferred candidate with quantitative and visual reasons, ranking
stability, limitations and alternatives. If 20230307_04 remains the strongest
credible choice, say so with evidence. Save a readable report, complete
machine-readable score/rank/coverage/exclusion table, classifier definitions,
sensitivity results and source/code hashes. Prepare an editable source-bound
candidate comparison with matching legend and minimal HTML; perform exact
saved-render/scientific checks and open the actual saved comparison for review.
Use nature-main, the shared theme, active invisible frames and scoped guards.

## Storage and protections

For this learner task, **D: and F: are read-only**. Permanent analysis payloads,
media, editable SVG and HTML belong at this author-requested new explicit
JOAQUIM destination:

`J:/ClassicalConditioning Outputs/current/reviews/figure-01-trace10-learner-selection-20261020/`

Authenticate JOAQUIM through the supported permission mechanism. Check for
existing work before creating files and preserve it. Capture state/artifact/
input hashes before edits; publish through scoped locks, evidence first and
state pointer last. Keep comparison scope separate from selected Figure 1/2.
No freeze or replacement without a subsequent explicit author instruction.

The companion JSON binds dispatch-time local authority and source plans;
re-authenticate the final roster/caches and reconcile newer selections when
execution begins. Neither the dispatch snapshot nor file age selects inputs.
