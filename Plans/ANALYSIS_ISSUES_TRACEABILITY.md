# Analysis Issues: Notes, Evidence, and Plan Traceability

## Purpose

This register connects the findings in
[ANALYSIS_ISSUES.md](../docs/analysis/ANALYSIS_ISSUES.md) to implementation steps,
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
-> resolved in named recipe/release
```

An issue cannot be closed merely because code was refactored or a new function
was written.

## Traceability matrix

| ID | Audit note | Plan ownership | Gate/decision | Required acceptance evidence | Release consequence |
| --- | --- | --- | --- | --- | --- |
| 1 | Executed point-15 vigor differs from manuscript sum of absolute segment speeds | Steps 00, 05-07, 09-10, 13 | T0, P, T1, O, S | Exact legacy reproduction; validated `segment_absolute_angular_speed_sum`; candidate comparison; full reprocessing; effect/claim comparison | R1 labels legacy behavior; R2 compares both; R3/R4 state selected metric and manuscript reconciliation |
| 2 | Spatial smoothing, temporal window, rolling bout metric, and second threshold differ from methods or are incomplete | Steps 05-08 | P, T1 | Legacy characterization; mathematical filter/detector specifications; impulse/step/edge and annotated-bout tests; threshold/smoothing sensitivity | No corrected release until frozen filter and detector versions are recorded |
| 3 | Exclusions are inconsistent across scripts and outputs | Steps 00, 03, 08-10, 12-13 | C0, C1 | Stage-specific legacy cohort snapshots; immutable reviewed cohort; validated joins; sample-size reconciliation for every result/panel | R1 preserves and exposes legacy differences; R3/R4 use explicit cohort IDs/hashes |
| 4 | Current scaled vigor differs from documented immediate-baseline proportional change | Steps 03, 05, 07, 09-10, 13 | O, S | Legacy formula regression; explicit documented formula; bounded baseline test; display transforms separated; sensitivity comparison | Methods and figures identify analytical versus display scaling and version |
| 5 | Immobility is encoded as missing, conflating movement probability and conditional intensity | Steps 05, 07-10 | C0, O, S | Rest/invalidity distinction tests; total activity, movement probability, fraction moving, and conditional intensity; two-part sensitivity | R1 labels conditional legacy meaning; corrected release reports complementary outcomes |
| 6 | Response-window movement inclusion may condition on the hypothesized outcome | Steps 00, 08, 10, 13 | C0, C1, S | Primary intention-to-analyze policy independent of response suppression; behavioral-engagement sensitivity cohort; flow and effect comparison | Corrected release reports both population definitions and impact |
| 7 | Minimum-trial logic may retain fish passing any block rather than all required blocks | Steps 05, 09-11 | O, S, L | Legacy regression; fish-by-required-block completeness table; tests where one block passes and another fails; corrected eligible counts | Statistics/classification artifacts record exact block-completeness rule |
| 8 | Bootstrap iteration defaults differ from manuscript and may be unstable | Steps 00, 03, 10, 12-13 | S, F | Legacy counts captured; frozen paper iteration count and seed; Monte Carlo stability check; methods/legend agreement | R1 reports current values; R4 records approved value and seed |
| 9 | Bootstrap may resample rows/trials rather than fish | Steps 09-10, 12-13 | S, F | Fish-level or hierarchical resampler tests; equal-fish-weight property test; uncertainty comparison | Final intervals and legends state resampling hierarchy and unit |
| 10 | Mixed-effects inference has weak prespecification and incomplete diagnostics | Steps 00, 10, 12-13 | S | Frozen estimand/model/contrasts; convergence, boundary, singularity, Hessian and sample-count outputs; joint interaction test; robust sensitivity | Failed diagnostics block confirmatory success; R4 includes model inputs and diagnostics |
| 11 | Response/baseline ratios are unstable when baseline is low or sparse | Steps 07, 09-10 | O, S | Zero/low-baseline tests; baseline-adjusted model; coverage input; ratio versus non-ratio sensitivity | Ratio is secondary unless explicitly approved; corrected primary estimand is named |
| 12 | Lost-frame detection may miss one or more missing frames | Steps 04-06, 08, 13 | T0, P | Direct frame-ID and timestamp-gap diagnostics; one/several/sustained-gap tests; legacy/corrected decision comparison; reprocessing impact | Gate P blocks corrected processing until policy is frozen |
| 13 | Manuscript, README, and executable methods have drifted apart | Steps 00, 03, 05-10, 12-13 | All scientific gates | Claim inventory; authoritative analysis specification; methods-parameter table; claim-to-artifact index; manuscript diff review | R4 cannot be approved until prose, figures, samples, and executed recipe agree |
| 14 | Results lack immutable configuration, inputs, cohort, code, and environment identity | Steps 00-04, 08-13 | G0 plus all artifact gates | Resolved config; run/artifact IDs; logical hashes; source/cohort/code/environment identity; explicit resolver; no first/latest loading | Every release contains a complete verifiable artifact manifest |
| 15 | Four learner classifiers and flexible thresholds prevent confirmatory use | Steps 00, 03, 11-13 | L | Legacy variant equivalence; frozen input metric/features/threshold; control false-positive and label-stability report; held-out/cross-fit mode; uncertainty export | Learners remain optional/descriptive until Gate L passes |

## Important interpretation notes

### Issue 1 adds a required benchmark

The tail-dynamics plan proposes four alternatives to the current distal-point
metric. The audit separately identifies the manuscript-described measure:

```text
sum across segments(abs(segment angular speed))
```

This measure is not equivalent to all-point angular RMS. It must therefore be
implemented as its own metric identity:

```text
segment_absolute_angular_speed_sum
```

The full candidate set contains:

1. legacy distal-point angular speed;
2. manuscript segment-speed sum;
3. all-point angular RMS;
4. whole-tail XY RMS speed;
5. whole-tail XY mean speed;
6. curvature-change RMS.

Gate T0 must first establish source-angle semantics.

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
- [ ] Preserve the issue's legacy behavior in R1 where required.
- [ ] Link the approved scientific decision for any correction.
- [ ] Attach reference, integration, and regression evidence.
- [ ] State whether data impact remains unquantified.
- [ ] Identify affected cohorts, results, figures, and manuscript claims.
- [ ] Update this register's issue lifecycle state during implementation.

## Current state

At plan creation, all 15 issues remain open as scientific/reproducibility risks.
The plan assigns ownership and acceptance evidence; it does not claim that any
issue has been corrected or that any paper conclusion has been confirmed.
