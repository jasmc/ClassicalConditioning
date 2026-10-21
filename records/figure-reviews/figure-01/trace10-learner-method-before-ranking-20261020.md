# 10sTrace individual ranking: analysis choices before inspecting scores

Recorded during the authenticated per-fish cache calculation, before computing
or inspecting the 10sTrace cohort's acquisition score distribution or shortlist.
This is a transparent descriptive analysis specification, not classifier
validation or a scientific freeze. It supplements the author-requested
`trace10-learner-handoff-20261020.md`.

## Outcome, ranking and eligibility

Use the verified measured-time median(log positive valid moving-frame tail
angular speed) response [0,20) minus baseline [-15,0), retaining frame counts
and finite-trial counts. Primary continuous acquisition score is median outcome
over Pre5–14 minus median over lateTrain40–64; positive means stronger
conditioned-direction suppression. Controls use the identical estimator.
No post-US behavior or population-model significance enters this score.

For primary feature eligibility require at least three finite outcomes in
**each constituent five-trial block**: Pre5–9 and 10–14, and lateTrain40–44,
45–49, 50–54, 55–59 and 60–64. This gives at least 6/10 Pre and 15/25 late
trials, distributed across each epoch rather than clustered in a few trials.
The 3/5 choice follows the author's common coverage rule, while its transfer
to learner features remains an explicit exploratory assumption. Retain raw
scores for ineligible fish separately, never silently delete their rows or
call them nonlearners. Technical/common-roster membership is distinct from
feature eligibility. The shared-control provenance remains explicit.

## Prespecified sensitivity and uncertainty description

- Recompute primary eligibility and rankings under 2/5 and 4/5 coverage in
  every constituent block. Report changing eligible counts and rank movement.
- Compare lateTrain40–64 against Train25–39 and Train50–64, with each epoch's
  corresponding block coverage. Train15–24 is an early-expression audit,
  not an interchangeable late-training endpoint.
- Delete one observed trial at a time and one five-trial block at a time
  within Pre and late Train. Retain score ranges, sign stability and shortlist
  ranking stability. These are robustness diagnostics, not independent-trial
  confidence intervals or biological ground truth.
- Compare pooled and acquisition-era control reference distributions when
  sample sizes permit. With small era-specific references, report coarse
  resolution and refrain from calibrated probability or significance claims.
- Report the descriptive upper-tail control rank
  `(1 + controls with score >= fish score)/(1 + eligible controls)` and
  leave-one-control-out reference diagnostics. This is reference extremeness,
  not a learner probability, validated p value or false-positive guarantee.
  Fish acquired in different experimental eras may not be exchangeable.

## Provisional operational flags and illustration choice

Keep the continuous score primary. A provisional **strong suppression
expression** flag requires primary feature eligibility, a positive acquisition
contrast, a negative lateTrain40–64 median, a descriptive control upper-tail
rank <=0.05 and positive scores in every five-trial-block deletion. This
explicit stringent operational phenotype is not a validated binary learner
category. Report each criterion separately and the number meeting the
conjunction. Failure of this rule does not establish failure to learn.
Do not relax it after seeing results; if no fish meet it, report that honestly.

Choose the strongest credible illustration after inspecting a bounded
shortlist: the top five primary eligible conditioned scores plus comparator
20230307_04 and any top-three alternative-window candidate outside those five.
Use the quantitative rank/sensitivity results and actual matched plots.
Inspect high Pre outcomes, baseline drift, positive-movement availability,
tracking coverage and timing, within-acquisition consistency, catch trials and
later Test recovery/persistence separately. A large acquisition score driven
primarily by an unusually elevated Pre value may be a poor illustration.
Explain any preference over the numerically top score; do not revise the
continuous score to make the preferred image rank first.

Use identical direct original-frame 1 s heatmaps (selected cadence/detector,
baseline-bin centering, managua_r +/-0.25) and identical trial-outcome limits
across the shortlist. Keep display clipping separate from retained numerical
values. Confirm actual approximately 20 s CS–US onset timing. Test and catch
trials are descriptive audits and have already been observed; no prospective
holdout validation is claimed. Actual US responses require a separately
defined post-US window if later computed and cannot change the acquisition
score. Missing post-US analysis must be reported, not invented.

The timing-flagged 20230315_04 and shortened 20250715_08 remain explicit in
the full table. A common61 roster excluding these two is currently awaiting
the author decision; do not imply it has been approved by this specification.
Any subsequent author revision to these assumptions must be dated and
distinguished from these pre-ranking choices.
