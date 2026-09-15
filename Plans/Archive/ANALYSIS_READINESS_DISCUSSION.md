# Analysis readiness discussion memo

**Status:** Discussion required — no confirmatory inference authorized  
**Purpose:** Resolve the Gate O/S choices that determine a valid paper-scale
behavioral analysis. This memo is not a numbered implementation step and does
not itself freeze a scientific decision.

## Why this is the current decision point

The candidate pipeline now carries exactly three metrics, produces authenticated
outcomes, and has exploratory mixed-effects, fish-permutation, and fish-bootstrap
routes. That is enough to review the analysis design, but not enough to report a
confirmatory result. The figure inventory is maintained separately in
[`FIGURE_PIPELINES.md`](../../docs/analysis/figures/FIGURE_PIPELINES.md);
paper-figure implementation remains in
[`PAPER_FIGURE_AUTOMATION_PLAN.md`](./PAPER_FIGURE_AUTOMATION_PLAN.md).

## Current blockers

| Area | Current finding | Required resolution |
| --- | --- | --- |
| Condition estimand | The draft LME includes `log_baseline` and block, but not `condition_id`. | Define the conditioned-versus-control contrast and its time interaction before selecting a formula. |
| Alternative inference | Fish permutation/bootstrap pool conditions in their fish and population groupings. | Preserve condition in fish summaries and test the planned between-condition contrast. |
| Bout rate | Response bout rate is paired with baseline total activity. | Use a baseline bout-rate outcome or choose a non-baseline-adjusted bout estimand. |
| Effect direction | `log(baseline) - log(response)` increases with suppression, while a comment describes the reverse. | Establish one signed-effect convention in artifacts, tables, figures, and tests. |
| Outcome families | Positive activity, occupancy/probability, and rates use unlike distributions. | Select primary outcome and outcome-appropriate family/link; label all others secondary or sensitivity. |
| Cohort | Technical QC exists, but no reviewed paper cohort hash/completeness rule is frozen. | Decide C0/C1 using paper-scale sample flow before fitting confirmatory models. |
| Diagnostics | Singular/failure handling relies mainly on warning text and can publish coefficients. | Define diagnostics, fit-failure semantics, and publication-stopping criteria. |
| Validation | Current tests verify plumbing, not an estimand or contrast. | Add recovery and artifact-integrity tests before Gate S approval. |

## Gate O/S agenda and required record

The discussion must write the answers below into
[`DECISIONS.md`](../DECISIONS.md), including date, owner, and rationale.

1. **Primary claim and estimand.** Choose one: conditioned-versus-control
   difference in early-to-late change; condition-by-block/trial trajectory; or
   a CS-window contrast. State the direction and biological interpretation.
2. **Outcome hierarchy.** Name one primary outcome from total activity,
   movement probability/fraction moving, conditional intensity, or bout rate;
   define secondary and sensitivity outcomes, windows, coverage thresholds, and
   baseline handling.
3. **Cohort and validation.** Freeze technical inclusion, any behavioral
   sensitivity cohort, missingness policy, confirmation partition or
   cross-fitting method, and immutable cohort hash.
4. **Inference route.** Select one primary family and one prespecified
   sensitivity route; document fixed/random effects or cluster unit, factor
   references, planned contrasts, multiplicity family, and effect scale.
5. **Fit validity.** Specify convergence, singularity, rank, residual,
   influence, optimizer-sensitivity, and minimum-sample criteria; state which
   failures prevent results and figures from being emitted.

The existing methodology workshop is the source record for candidate model
families. It must be amended with these implementation findings, but it is not
approval to implement many competing models.

## Implementation only after decisions

1. Version the model-input contract with cohort ID/hash, condition identity,
   declared factor references, outcome denominator/provenance, and ordering.
2. Replace condition-pooled fish summaries with condition-aware effects and
   planned contrasts; correct bout-rate handling and signed-effect metadata.
3. Implement the approved primary model and exactly one sensitivity route.
   Failed or singular fits must be non-publishable and must not yield final
   coefficient or figure artifacts.
4. Publish cohort flow, contrast definitions, diagnostics, and analysis
   manifests for figure automation to consume.
5. Add synthetic recovery tests for condition-by-learning effects, denominator
   correctness, factor references, row-order invariance, all three metrics,
   multiplicity, and failure propagation; add integration tests for cohort and
   provenance joins.

## Documentation consolidation

- Active plans must say **three metrics**, not retired five-metric candidates
  or four implementations. Historical pilot evidence is retained under
  `Plans/Archive`, never discarded.
- The tail-dynamics annex must describe the active three-metric set and treat
  RMS/curvature work only as deferred mechanistic research, not an active
  candidate output.
- The active handoff is this memo until the Gate O/S record is frozen. The
  prior two-fish handoff is archived as historical fixture evidence.
- `REPOSITORY_MIGRATION_MAP.md` and dated status notes remain active only until
  their still-open migration/status facts are reconciled; they are not deleted.

## Exit condition

This memo is superseded only after Gate O and Gate S are explicitly recorded,
the approved implementation has evidence-backed tests, and the step index
distinguishes exploratory scaffolds from confirmatory analysis.
