# Analysis Issues: Notes, Evidence, and Plan Traceability

## Purpose

This register connects the findings in
[Analysis findings](./01_ANALYSIS_FINDINGS.md) to implementation workstreams,
scientific gates, required evidence, and release consequences.

The audit is treated as a set of serious, evidence-backed notes. It is not a
complete data reanalysis and does not establish whether any biological
conclusion changes. Its recommended remedies are inputs to scientific
decisions, not automatic approvals.

## Issue lifecycle

Each issue moves through these states:

```text
reported
-> code-confirmed or not-reproduced
-> data impact unquantified
-> scientific decision approved
-> legacy behavior characterized
-> corrected implementation reference-tested
-> full data reprocessed
-> result and manuscript impact assessed
-> resolved in a named recipe and the final release record
```

An issue cannot be closed merely because code was refactored or a new function
was written.

## Traceability matrix

Numbered owners refer to the current implementation plans in `Plans/`;
archived step numbers from the original migration roadmap are not current
work assignments.

| ID | Audit note | Active owner(s) | Gate/decision | Required acceptance evidence | Release consequence |
| --- | --- | --- | --- | --- | --- |
| 1 | Executed point-15 vigor differs from manuscript sum of absolute segment speeds | Decisions P/T1; Plans 02, 03, 08, 10 | T0, P, T1, O, S | Exact legacy reproduction; documented manuscript reconciliation; validated tail-length-weighted angular L1 replacement; candidate comparison; full reprocessing; effect/claim comparison | Preserve legacy and candidate comparisons as evidence; final release states the selected metric and manuscript reconciliation |
| 2 | Spatial smoothing, temporal window, rolling bout metric, and second threshold differ from methods or are incomplete | Decisions P/T1; Plans 02, 08 | P, T1 | Legacy characterization; mathematical filter/detector specifications; impulse/step/edge and annotated-bout tests; threshold/smoothing sensitivity | Final release waits for frozen filter and detector versions |
| 3 | Exclusions are inconsistent across scripts and outputs | Plans 01, 02, 03, 05–08, 10 | C0, C1 | Stage-specific legacy cohort snapshots; immutable reviewed cohort; validated joins; sample-size reconciliation for every result/panel | Preserve legacy differences as evidence; final release uses the reviewed cohort ID/hash |
| 4 | Current scaled vigor differs from documented immediate-baseline proportional change | Plans 02, 03, 07, 08 | O, S | Legacy formula regression; explicit documented formula; bounded baseline test; display transforms separated; sensitivity comparison | Methods and figures identify analytical versus display scaling and version |
| 5 | Immobility is encoded as missing, conflating movement probability and conditional intensity | Plans 01, 02, 03, 07, 08 | C0, O, S | Rest/invalidity distinction tests; total activity, movement probability, fraction moving, and conditional intensity; two-part sensitivity | Record the legacy meaning; final release reports the approved complementary outcomes |
| 6 | Response-window movement inclusion may condition on the hypothesized outcome | Plans 01, 02, 03, 10 | C0, C1, S | Primary intention-to-analyze policy independent of response suppression; behavioral-engagement sensitivity cohort; flow and effect comparison | Final release reports both population definitions and impact |
| 7 | Minimum-trial logic may retain fish passing any block rather than all required blocks | Plan 01; archived legacy characterization | C0 (primary), C1 | Legacy regression; fish-by-required-block completeness table; tests where one block passes and another fails; corrected eligible counts under both OR and AND rules | Cohort and statistics artifacts record exact block-completeness rule; Gate C0 decides the corrected rule with paper-scale numbers |
| 8 | Bootstrap iteration defaults differ from manuscript and may be unstable | Plans 02, 03, 08, 10 | S, F | Legacy counts captured; frozen paper iteration count and seed; Monte Carlo stability check; methods/legend agreement | Record historical counts as evidence; final release records approved count and seed |
| 9 | Bootstrap may resample rows/trials rather than fish | Plans 02, 03, 08 | S, F | Fish-level or hierarchical resampler tests; equal-fish-weight property test; uncertainty comparison | Final intervals and legends state resampling hierarchy and unit |
| 10 | Mixed-effects inference has weak prespecification and incomplete diagnostics | Plans 02, 03, 10 | S | Frozen estimand/model/contrasts; convergence, boundary, singularity, Hessian and sample-count outputs; joint interaction test; robust sensitivity | Failed diagnostics block confirmatory success; final release includes model inputs and diagnostics |
| 11 | Response/baseline ratios are unstable when baseline is low or sparse | Plans 02, 03, 07, 08 | O, S | Zero/low-baseline tests; baseline-adjusted model; coverage input; ratio versus non-ratio sensitivity | Ratio is secondary unless explicitly approved; corrected primary estimand is named |
| 12 | Lost-frame detection may miss one or more missing frames | Decisions T0/P; Plans 01, 02 | T0, P | Direct frame-ID and timestamp-gap diagnostics; one/several/sustained-gap tests; legacy/corrected decision comparison; reprocessing impact | Gate P blocks corrected processing until policy is frozen |
| 13 | Manuscript, README, and executable methods have drifted apart | Plans 01–10 | All scientific gates | Claim inventory; authoritative analysis specification; methods-parameter table; claim-to-artifact index; manuscript diff review | Final release waits until prose, figures, samples, and executed recipe agree |
| 14 | Results lack immutable configuration, inputs, cohort, code, and environment identity | Governance; Plans 01–10 | G0 plus all artifact gates | Resolved config; run/artifact IDs; source/output file hashes; source/cohort/code/environment identity; explicit artifact selection; no first/latest loading | The final release contains a complete verifiable artifact manifest |
| 15 | Four learner classifiers and flexible thresholds prevent confirmatory use | Plans 04–08, 10 | L | Legacy variant equivalence; approved continuous or categorical representation; if categorical, frozen input metric/features/threshold and control false-positive/label-stability report; held-out/cross-fit mode; uncertainty export | Learner analysis is required; categorical labels and confirmatory learner claims remain gated until Gate L passes |

## Important interpretation notes

### Issue 1 adds a required benchmark

The three-metric candidate set (DECISIONS Gate T1) supersedes the manuscript
segment-speed sum with a tail-length-weighted angular L1 metric:

```text
tail_length_weighted_angular_l1
```

The full candidate set contains:

1. tail-length-weighted angular L1;
2. tail-length-normalized whole-tail XY mean speed;
3. legacy distal-point angular speed (historical benchmark on measured time).

The unweighted manuscript sum remains part of the historical reconciliation,
but not an active candidate: it changes with the number and spacing of tracked
tail segments, whereas the weighted L1 estimates the same absolute angular
activity per unit tail length.
All-segment angular RMS is also excluded because it is an L2 reweighting of the
same angular-speed field and emphasizes vigorous local events unnecessarily.

Gate T0 raw-angle semantics are decided (radians, local bends, pixels). Gate T1
selection remains open.

### Issues 5 and 6 are related but distinct

- Issue 5 concerns the outcome: rest is being treated as missing.
- Issue 6 concerns selection: response-window movement affects inclusion.

Correcting the outcome without correcting the cohort can leave post-treatment
selection bias. Correcting the cohort without decomposing the outcome can
still obscure whether suppression reflects fewer movements or weaker
movements.

### Issues 8 and 9 are related but distinct

Increasing iterations does not fix the wrong resampling unit. The hierarchy
must be correct first; iteration count then controls Monte Carlo stability.

### Issue 13 is the release-level integration test

No single code test resolves manuscript drift. Resolution requires one frozen
analysis specification and a release review that reconciles:

- equations and units;
- windows and thresholds;
- cohorts and sample sizes;
- models and uncertainty;
- results and figures;
- manuscript text and legends.

## Step review checklist

Before closing any implementation step:

- [ ] Identify which analysis issues it touches.
- [ ] Preserve the issue's legacy behavior in audit evidence where required.
- [ ] Link the approved scientific decision for any correction.
- [ ] Attach reference, integration, and regression evidence.
- [ ] State whether data impact remains unquantified.
- [ ] Identify affected cohorts, results, figures, and manuscript claims.
- [ ] Update this register's issue lifecycle state during implementation.

## Current state

All 15 issues remain open as scientific/reproducibility risks for
paper-authoritative release. Fixture-scoped engineering progress (local
two-fish intake, legacy equivalence, corrected candidate plumbing, cohort
freeze tools, model-input/LME/permutation/bootstrap scaffolds) does not close
any issue. Partial/plumbing progress is tracked in
[implementation status index](../../../Plans/IMPLEMENTATION_STEP_INDEX.md); this register
records scientific ownership and acceptance evidence only. No paper conclusion
is confirmed.
