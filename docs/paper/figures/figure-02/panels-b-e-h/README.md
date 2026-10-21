# Figure 2 B/E/H: expanded fixed-3sTrace draft

## Later author correction requested, not yet published

The author subsequently directed this chat to use **separate statistics for
each panel** and **at least three eligible trials per fish/block for every
D/E/F panel**, explicitly requesting correction of E. This supersedes E's
one-trial minimum and the joint D/E Holm36 as targets for the next draft.
Recalculate E eligibility/data summaries and separate Holm18 D/E annotations;
preserve the historical outputs and frozen selections. The selected artwork
described below has not yet received this correction. The
[C/F/I handoff](../panels-c-f-i/README.md#resolved-author-directions) records the
explicit author replies and the guarded downstream work, queued after the
10sTrace processing batch.

## Current selected draft, 19 October 2026

The author's later instructions supersede the earlier no-inference build:
include B/E/H in Figure 2, match the current Delay column and apply its
statistical tests. Exclude the four timing-flagged fish from **all three**
3sTrace panels, retain their flags and use the same fish across panels.

The published draft now uses **179 fish: 90 Control, 89 3sTrace**. The exclusions
are `20230426_01`, `20230802_01`, `20230803_12`, `20230822_04`. This is one
shared roster; finite-frame/trial eligibility can still vary within it.
E contributing fish PT/ET/LT are Control 90/90/89 and 3sTrace 89/89/89.
H's model retains all 179 fish and 15,869 eligible trial observations.

Current artwork and legend resolve through the parent
[Figure 2 state](../current-state.json). New source data, diagnostics, code
snapshots, prior selected artwork and exact browser evidence are in
`J:/ClassicalConditioning Outputs/current/reviews/figure-02-3strace/matched-delay-statistics/`.
The guarded publisher updated the existing active SVG/HTML pair; the entire
figure and its frozen originals were not frozen or replaced as scientific
selections. A/C/F/G/I SVG subtrees were preserved unchanged in that assembly.
The subsequent author-requested axis refinement centers one shared x-axis title
under each A/B/C heatmap pair. D/E now display [-0.25,0.25], with the same native
data-rectangle size; observations outside this view remain in the analysis.
The earlier [-0.75,0.75] view is preserved in the pre-edit snapshot. Source values
and statistical results are unchanged. Exact title-centering and rendered-frame
evidence are under `matched-delay-statistics/axis-refinement/`.
H shares G's trial coordinates, [-0.21,0.16]
range, data-rectangle size and statistical-row centres. Final exact 600 dpi
painted/unclipped frame containment and recorded equal-spacing checks passed.

### Statistical methods and limitations

- E uses the selected D/E fish-level exact two-sided sign tests versus zero
  and on paired changes, plus Brunner–Munzel t tests between conditions at
  blocks and on paired changes. One Holm36 family combines the unchanged
  selected D values and updated E values. Adjusted D annotations therefore
  change with E; no frozen D result file was modified. E retains its accepted
  assay-specific minimum of one eligible trial per block (D uses three).
- H uses the selected G baseline-adjusted block interaction model (Holm8),
  local block condition differences (BH9), local slope differences (raw and
  BH9), and phase-aware trial LMM with fish random intercept/slope. Trial
  contrasts are control-minus-Trace change from average Pre5–14, BH90. No
  second log, pseudocount or smoothing is added. Observed medians/whole-fish
  bootstrap bands are distinct from fitted-model contrasts.
- Primary and Powell phase fits, random-intercept sensitivity, block model
  and all nine local fits converged and passed curvature/covariance checks.
  The two phase optimizers' maximum contrast difference is approximately
  1.18e-6; both identify the same 45 BH90-significant trials. These checks do
  not establish model adequacy: phase residual excess kurtosis is about 9.27
  and median within-fish adjacent residual correlation about 0.13. Full
  leave-one-fish-out influence remains a separate audit. Results remain
  exploratory and do not establish learning onset.
- No new freeze, complete production-font/PDF check or full Nature compliance
  is claimed. The final renderer source, preserved earlier implementations,
  plan and result hashes bind this draft. An exporter sidecar's output hash
  precedes final SVG provenance embedding; explicit final saved-byte hashes
  and embedded-sidecar linkage were checked for the staged panel inputs.

Reproduction: `scripts/analyze_figure2_trace3_matched.py` and
`scripts/assemble_figure2_trace3_matched.py`, followed by the scoped
`scripts/refine_figure2_axes.py` edit; preserve sealed prior outputs
and capture fresh state/source hashes before any further edit.

## Earlier 183-fish preparation (superseded within B/E/H)

Author request, 19 October 2026: make B, E and H without inferential statistics;
always match the current corresponding Delay panels. This authorizes H to use
G's log response-minus-baseline measure, replacing the earlier ratio draft.

All 183 per-fish calculations finished on 19 October 2026 at approximately
22:18 Europe/London. E/H descriptive summaries are complete. Their draft
`figure2-E-H.svg` and matching HTML/legend were rendered and visually inspected
in the built-in browser; they have not been published into Figure 2 state.
The existing assembly and scientific freezes remain authoritative until
guarded publication of the new scoped draft.

B has 179 successful per-fish heatmaps and four timing-gate flags:
`20230426_01`, `20230802_01`, `20230803_12`, `20230822_04`.
All four have zero missing exported FrameIDs, but their maximum arrival delays
exceed the supported 700-frame buffer-capacity criterion: approximately
1.058, 1.850, 1.836 and 1.988 s respectively (cadence approximately 1.423 ms).
This is frame-loss evidence under the selected method, not direct confirmation
of a missing exposure count. Full-cohort B rendering deliberately stopped;
no 179-fish substitute was silently pooled or published. The calculation JSON
records `B-review-required`. Timing/source review or an explicit cohort/method
decision is required before continuing B. E/H retain all 183 fish and do not
use the blocked constant-cadence reconstruction.

## Sources and definitions

The expanded processed roster has **183 fish: 91 Control and 92 3sTrace**.
It consists of the original 59 processed recordings plus 124 verified additions.
`20230803_12`, an original fish outside the 197-identity pickle intake, passed
full supported lineage verification before joining this roster. The 15 held
intake identities and separately recorded increasing-trace fish are not included.
No learner-based selection is applied.

- **B follows A:** positive detected-bout original-frame log vigor; direct 1 s
  medians; subtract the median finite baseline-bin medians over [−15,0) s;
  equal-fish arithmetic mean of centred cells. Use acquisition trials 5–94,
  displayed 1–90, native `managua_r` at ±0.25 and black missing cells. Keep
  contributing-fish counts. The supported constant-cadence reconstruction and
  frame-loss gate remain active. A failed fish blocks full-cohort B publication;
  it is recorded for review, not silently omitted.
- **E follows D's accepted outcome with E's scoped settings:** measured-time
  median log positive valid moving-frame vigor in response minus baseline;
  baseline [−15,0), response [0,13). Take per-fish medians in PT10–14, ET65–69,
  LT90–94 with the existing E minimum of one eligible trial. Show paired fish
  and across-fish median/IQR. Retain the accepted assay-specific E eligibility
  rather than silently replacing it with D's three-trial minimum.
- **H follows G:** the same measured-time window log-median difference as E,
  summarized across fish at each trial. Retain G's descriptive pointwise 95%
  percentile intervals from 5,000 whole-fish trajectory resamples, seed 10.
  Resampling a fish retains its complete trajectory and missing observations.

No hypothesis tests, fitted LMMs, p values or significance annotations are run.
The resulting expanded-cohort panels are scientific drafts, not new freezes.

## Build and completion

Use `.venv-trace/Scripts/python.exe scripts/build_figure2_trace3_panels.py`.
`--plan` seals inputs; `--calculate` creates authenticated per-fish caches and
numerical summaries; `--render` creates the editable SVG and matching minimal
SVG-and-legend HTML. A calculation lock excludes concurrent workers. Raw D: and
processed F: are read-only in the worker; J: writes are limited to this scope.
Uncommitted partial caches are preserved and require reconciliation.

Permanent files are on JOAQUIM at
`J:/ClassicalConditioning Outputs/current/reviews/figure-02-3strace/`.
`plan.json` and `calculation-launch.json` bind inputs, cohort and worker code;
`calculation-stdout.log` / `calculation-stderr.log` record progress.
The pre-edit ticket is
`records/figure-reviews/figure-02/trace3-panels-start-20261019.json`.
The previous shared heatmap source was saved there as
`pooled_log_heatmap-before.py` with its exact pre-edit hash. The live helper's
only change resolves untouched NoMK801 source names through the supported
canonical identity parser; it does not change the numerical heatmap calculation.

After calculation, inspect actual values, eligibility/coverage and the exact
saved rendering. Complete frame/axis/semantic mapping evidence and visual review
before guarded scoped publication with `figure_workspace.publish_state`.
Bind the starting ticket again under the publication lock and preserve the parent
Figure 2 assembly, frozen records and any manual edits. Resolve the Windows-only
LF/CRLF legend hash difference recorded by the ticket without replacing its text.
No successful display, publication, full Nature compliance or freeze is claimed
by starting the worker. Assembly integration remains a separate publication step.
