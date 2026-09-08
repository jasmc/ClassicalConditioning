# Handoff — two real fish for full pipeline tests (2026-08-31)

## Purpose

`20221115_04` and `20221116_12` are the **real local recordings used to test
the full refactor pipeline**, including cohort freeze/compare — not a
scientific mini-cohort, and **not** a substitute for paper-scale N.

**Synthetic fish (`20221115_SYNTH02` / Synthetic * trees) removed** from
`Paper data`. Unit-test numerical synthetics in `tests/` remain.

## Critical-path map

Status synced from
[IMPLEMENTATION_STEP_INDEX.md](./IMPLEMENTATION_STEP_INDEX.md)
(refresh this table when the index changes):

| Step | Status |
| --- | --- |
| 03–05 | Complete for local fixtures (archived plans) |
| 06–07 | **Done (fixtures):** corrected chain + compare/runner on both fish. **Open:** Gates P/T1 |
| 08 | **Done:** freeze-cohort; corrected runner; `plan-batch` + `execute-batch`. **Open:** paper-scale QC; Gates P/T1/C0 |
| 09 | **Done:** corrected temporal/trial + metric comparison on both fish. **Open:** cohort-scale / Gate O |
| 10 | **Done:** workshop record; model-input; LME; fish-permutation; fish-bootstrap. **Open:** Gate S; planned contrasts |
| 11 / imaging | Deferred |
| 12 | PNG + publication pilots; interactive HTML frozen |

## Statistics note

Gate S is **not** frozen. Workshop record (critique of legacy + default LME,
plus alternative families):
[STATISTICS_METHODOLOGY_WORKSHOP.md](./STATISTICS_METHODOLOGY_WORKSHOP.md).

Fixture two-fish LME runs are expected to be **singular**; that is evidence
for the workshop, not a paper result.

## Suggested next

1. Decide Gate S primary family after comparing families on paper-scale N
   (not on two fixtures alone).
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

Steps 03–05 detailed plans, the pickle-timebase handoff, and the LogMedian
note live under `Plans/Archive/`. Active handoff remains this file.
