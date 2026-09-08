# Implementation Step Index

## Scope decisions

The scattered per-step "gate" questions below were consolidated into one
short decision pass on 2026-08-30 to stop treating each gate as a recurring
blocking ritual. See [Plans/DECISIONS.md](./DECISIONS.md) for the actual
decisions, the current priority lane, and work deferred until later. Nothing
was removed from the original plans: all step plans and acceptance criteria
remain preserved and authoritative. The gate table further below remains as
reference for what each gate covers, but gates are not re-asked per step;
they are only revisited when new information changes a decision.

## Status legend

| Status | Meaning |
| --- | --- |
| Not started | No implementation work accepted |
| In progress | Work is active but the exit gate has not passed |
| In progress - prototype/fixture-only | Downstream components exist on synthetic or bounded pilot inputs, but upstream gates do not authorize paper-scope execution |
| Blocked | A named decision or dependency prevents progress |
| Review | Deliverables exist and are being validated |
| Complete | Exit gate passed and evidence is recorded |

This is a planning document. Update status only when implementation evidence
exists; writing the plan does not complete any implementation step.

## Master status

| Step | Status | Depends on | Can run in parallel with | Exit evidence |
| --- | --- | --- | --- | --- |
| 00. Governance and baseline | Not started | None | None initially | Baseline manifest and paper-scope inventory |
| 00A. Single-fish local intake | Complete | Minimal Step 00 decisions | Remaining baseline inventory | `fe02193`, `6e3c4dd`; 14 tests; 8,361,388 camera/tracking rows; raw hashes unchanged |
| 00B. Preexisting codebase characterization | Complete | Step 00A and source inventory | Package infrastructure only | 106 tests; legacy routes characterized; bounded-memory historical LogMedian pilot authenticated |
| 01. Package and environment | In progress | Step 00 decisions | Late Step 00 inventory work | Frozen clean install and 113 tests pass; representative numerical baseline awaits Step 00 scope |
| 02. Artifacts, schemas, provenance | In progress | Steps 00-01 | Early Step 03 | Byte hashing and transactional publication implemented; schema registry, logical hash, resolver, and controlled conversion pending |
| 03. Domain and configuration | Complete (allDelay) | Steps 00-01 | Early Step 02 | `allDelay` resolved config, trial map, FishKey, stage hashes, `resolve-config` CLI; other experiments deferred until needed |
| 04. Ingestion and raw validation | Complete (local fixtures) | Steps 02-03 | — | Readers, validate-raw, Gate T0, tracking audits for both local fish; cohort-breadth inventory when more raw arrive |
| 05. Legacy preprocessing equivalence | Complete (local fixtures) | Steps 02-04 | — | `legacy-paper-v1` + pickle reciprocal timebase classified; do not invert interpolate |
| 06. Corrected preprocessing, tail representation, and candidates | In progress | Steps 04-05 and Gate T0 | Exploratory figures | **Done (fixtures):** full corrected chain on both local fish. **Open:** Gate P interp/filter; corrected scaling |
| 07. Metric validation and selection | In progress | Step 06; Gate O before selection | — | **Done:** development + corrected movement/temporal/trial + two-fish corrected compare/runner. **Open:** Gate T1 selection/partitions |
| 08. Full reprocessing, QC, cohort | In progress | Steps 05–07 plumbing | — | **Done:** freeze-cohort; corrected runner; `plan-batch` + `execute-batch`. **Open:** paper-scale QC; Gates P/T1/C0 |
| 09. Outcomes and temporal profiles | In progress | Step 08 plumbing | Figures on fixtures | **Done:** development + corrected temporal/trial + corrected metric comparison. **Open:** cohort-scale canonicalization / Gate O |
| 10. Statistics and sensitivity | In progress | Step 09 and Gate S | Rendering from frozen synthetic results | **Done:** workshop; model-input; LME; fish-permutation; fish-bootstrap. **Open:** Gate S; planned contrasts |
| 11. Learner classification | Not started | Fully implemented non-learner refactor, Steps 09-10, and Gate L | Descriptive legacy reproduction | End-to-end methodology review plus versioned method and validation-mode report, if classification is retained |
| 12. Figures, CLI, notebooks | In progress | Steps 02-11 as applicable | Incrementally throughout | **Done:** candidate PNG, frozen interactive HTML, semantic SVG/PDF pilots. **Open:** Gate F dimensions/panels; notebooks; visual regression |
| 13. Releases and retirement | Not started | All required paper steps | None for final release | Immutable release and reproduction verification |

## Deferred optional imaging track

| Track | Status | Depends on | Does not block | Exit evidence |
| --- | --- | --- | --- | --- |
| I00-I12. Behavior and imaging integration | Deferred | I00 begins with source inventory; I01+ require shared Step 02 artifact contracts, Step 03 identity/config/trial map, and canonical behavior intake and recipe | Behavior-only Steps 00-13 and behavior releases | Behavior hashes invariant to imaging; validated synchronization/registration/imaging outcomes; immutable multimodal release |

Detailed stages, gates, and acceptance criteria are in
[the behavior and imaging integration plan](./BEHAVIOR_IMAGING_INTEGRATION_PLAN.md).
The inherited code risks that must not be copied are in
[the imaging pipeline critique](../docs/analysis/IMAGING_PIPELINE_CRITIQUE.md).

## Critical path

```text
00 baseline (required before paper-authoritative claims)
  -> 00A intake [complete]
  -> 01 package foundation completion
  -> 02 artifacts (current-path hashing; schema registry deferred)
  -> 03 configuration [complete, allDelay fixtures]
  -> 04 ingestion [complete, local fixtures]
  -> 05 legacy equivalence [complete, local fixtures]
  -> 06 candidate metrics (Gate P open)
  -> 07 metric selection (Gate T1 open)
  -> 08 corrected reprocessing and cohort (Gates P/T1/C0 open for paper)
  -> 09 outcomes (Gate O open)
  -> 10 inference (Gate S open)
  -> 11 learner analysis, if used (deferred)
  -> 12 final figures and interfaces (PNG + publication; HTML frozen)
  -> 13 paper release (lightweight interim OK)
```

## Scientific gates

Gate meanings below are reference only. Current answers and deferred items
live in [DECISIONS.md](./DECISIONS.md). Do not re-ask gates per step; revisit
only when new information changes a decision.

| Gate | Required decision | Blocks |
| --- | --- | --- |
| G0 | Paper experiments, claims, figures, owners, and current reference run | All paper-authoritative work |
| T0 | Raw angle semantics, X/Y availability, coordinate scale, segment spacing, confidence fields, and synchronized video availability | Tail representation, 2D metrics, and video-validation score |
| P | Corrected frame-loss, synchronization, interpolation, filtering, time handling, and gap policies | Corrected tail representation and full preprocessing |
| T1 | Selected activity metric, smoothing, weighting, movement detector, units, and confirmation design | Corrected full reprocessing |
| C0 | Technical-validity rules, behavioral-engagement definition, missingness policy, primary-population rule, and sensitivity-population rules | Corrected QC and cohort construction |
| C1 | Reviewed cohort instance and immutable cohort hash | Cohort-dependent outcomes and inference |
| O | Primary/secondary outcome definitions, windows, scaling, binning, and fish-aware aggregation | Biological metric selection and canonical outcome build |
| S | Estimand, validation partition/mode, model, random effects, contrasts, bootstrap, multiplicity, and diagnostics | Confirmatory inference |
| L | Learner estimand; decision whether to classify; candidate method, input metric, features, power, threshold, and validation mode | Learner-stratified claims |
| F | Figure dimensions, panel semantics, labels, sample-size display, and output formats | Step 12 completion and final paper figures |

## Release milestones

### R1 — Legacy-equivalent engineering release

Reproduces current outputs under explicit configuration and a pinned
environment, with canonical Parquet/JSON mirrors and no approved scientific
correction.

### R2 — Candidate-development release

Contains the shared tail representation, all candidate activity metrics,
technical validation, comparison outputs, and a documented selection result.
It is not the final paper release.

### R3 — Corrected paper-candidate release

Contains complete corrected preprocessing, frozen cohort, canonical outcomes,
diagnosed statistics, and regenerated figures. It is ready for scientific
owner review.

### R4 — Paper release

The approved, immutable reproduction bundle tied to manuscript methods,
results, tables, figures, environment, code, source inventory, cohort, and
artifact hashes.

## Updating a step

For each implementation pull request or change set:

1. Name the affected step.
2. Declare `behavior-preserving` or `scientific-correction`.
3. Name the stage implementation and parameter-set versions.
4. List upstream artifact IDs and expected invalidation.
5. Run the step's required tests and comparisons.
6. Attach exit evidence without marking unrelated steps complete.
7. After the exit gate passes, move the plan to `Plans/Archive/`
   (`git mv Plans/<STEPFILE>.md Plans/Archive/<STEPFILE>.md`) and
   update both README files and this index in an atomic documentation commit.

## Git implementation protocol

Before implementation:

1. confirm the base branch and new refactoring branch name;
2. create and switch to that new branch;
3. leave current unrelated worktree changes untouched.

During implementation:

- use atomic semantic commits such as `feat(ingestion): ...`,
  `test(ingestion): ...`, `refactor(preprocessing): ...`, and
  `docs(plans): archive completed intake step`;
- never mix behavior-preserving migration and scientific correction in one
  commit;
- run the smallest relevant validation before each commit;
- update plan status and user-facing README material when behavior changes;
- never push unless explicitly requested.
