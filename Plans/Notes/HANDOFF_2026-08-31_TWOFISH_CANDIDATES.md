# Handoff — two real fish for full pipeline tests (2026-08-31)

## Purpose

`20221115_04` and `20221116_12` are the **real local recordings used to test
the full refactor pipeline**, including cohort freeze/compare — not a
scientific mini-cohort, and **not** a substitute for paper-scale N.

**Synthetic fish (`20221115_SYNTH02` / Synthetic * trees) removed** from
`Paper data`. Unit-test numerical synthetics in `tests/` remain.

## Critical-path map

| Step | Status |
| --- | --- |
| 03–05 | Done for local fixtures (legacy frozen; pickle classified) |
| 06–07 | Corrected chain + compare/runner on both fish; Gate P/T1 open |
| 08 | `plan-batch` + `execute-batch`; fixture cohort frozen |
| 09/10 | Exploratory mixed-effects scaffold + **10.0 workshop** started |
| 11 / imaging | Deferred |

## Statistics note

Yes: after cohort/batch plumbing, the critical path is Step 09/10
mixed-effects **plumbing** — but Gate S is **not** frozen to LME yet.
Workshop record (critique of legacy + default LME, plus drastically
different families):
[STATISTICS_METHODOLOGY_WORKSHOP.md](./STATISTICS_METHODOLOGY_WORKSHOP.md).

Fixture two-fish LME runs are expected to be **singular**; that is evidence
for the workshop, not a paper result.

## Suggested next

1. Decide Gate S primary family after comparing LME vs fish-permutation on
   paper-scale N (not on two fixtures).
2. Gate P / paper-scale QC when ready.
3. Optional: planned contrasts once Gate S estimand is chosen.

## Inference plumbing on fixtures

| Recipe | Role |
| --- | --- |
| `candidate-model-input-v1` | Shared frozen table |
| `candidate-mixed-effects-v1` | Draft LME |
| `candidate-fish-permutation-v1` | Alternative family A |
| `candidate-fish-bootstrap-v1` | Fish-unit percentile CIs |

## Archive note

Steps 03–05 detailed plans and the pickle-timebase handoff now live under
`Plans/Archive/`. Active handoff remains this file.
