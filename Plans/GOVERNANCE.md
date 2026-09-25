# Analysis Governance

## Purpose

This is the short active engineering constitution for the analysis migration.
It retains the durable rules from earlier migration and repository plans;
their obsolete roadmaps remain in Git history.

## Authority

When documents conflict, use this order:

1. an approved immutable release manifest;
2. [DECISIONS.md](./DECISIONS.md);
3. the active domain plan responsible for the work;
4. [IMPLEMENTATION_STEP_INDEX.md](./IMPLEMENTATION_STEP_INDEX.md) for status;
5. current documentation under `docs/analysis/`;
6. original code under `legacy/` and historical plans in Git history.

No plan authorizes a scientific correction by itself. Scientific decisions are
recorded in `DECISIONS.md`, implemented under a stable recipe identity, and validated by
the relevant exit gate.

## Two scientific lanes

- `legacy-paper` reproduces characterized historical behavior. It may expose
  known flaws but must not silently repair them.
- Corrected recipes implement explicitly approved scientific behavior. They
  must not overwrite or masquerade as legacy reproduction.

Comparisons always name the recipe, implementation version, parameter set,
cohort, and scientific status.

## Data boundaries

Keep these stages explicit:

```text
source inventory
  -> raw scientific tables
  -> corrected/legacy tail representation
  -> activity metrics + shared bout segmentation
  -> trial and temporal outcomes
  -> frozen cohort and model/classifier inputs
  -> statistics and learner results
  -> panel data
  -> figures
  -> final analysis release
```

Raw inputs are immutable. Derived outputs are published transactionally and
carry their upstream identity. Active output filenames are stable across reruns;
implementation and input hashes belong in manifests, not filename suffixes.
Plotting code does not define cohorts, analysis
windows, statistics, learner labels, or scientific transformations.

## Identity and versioning

Distinguish:

- metric identity and implementation version;
- detector identity and parameters;
- representation and schema versions;
- analysis recipe and parameter-set version;
- cohort ID and hash;
- outcome/model/classifier version;
- figure panel-data, renderer, layout, and theme versions.

A changed display theme should rerender figures only. A changed scientific
input must invalidate every dependent outcome, model, learner result, panel
table, and figure.

Byte hashes establish exact file identity for the planned analysis release.
Additional semantic identity, schema resolution, or legacy conversion requires
a concrete use case; the earlier provenance proposal remains in Git history,
not as an active requirement.

## Execution contract

Every supported stage should eventually be callable through the package API
and CLI with explicit inputs and outputs. Before execution, `plan`/`explain`
reports selected recipes, resolved input hashes, stage versions, cache state,
invalidation reasons, and planned outputs.

Required failure behavior:

- nonzero exit on stage failure;
- no success-shaped empty artifact;
- no silent row, fish, trial, or recording loss;
- atomic publication;
- an immutable final analysis release;
- explicit resumability based on matching identities, not filenames alone.

## Scientific invariants

- Fish are the biological population unit for paper-level inference.
- Bout segmentation is shared across all activity metrics.
- Condition identity is retained in population comparisons.
- Frame, trial, fish, condition, and experiment aggregation order is explicit.
- Missingness and coverage are data, not zeros unless the outcome explicitly
  defines them as zeros.
- Learner analysis is required for the paper, while categorical thresholds must
  be justified against continuous and model-based alternatives.
- Same-data learner stratification is descriptive unless held-out,
  cross-fitted, or independent validation is used.
- Failed or disallowed statistical fits cannot yield publication-ready results.
- Candidate methods are compared by prespecified criteria, not favorable
  significance alone.

## Repository migration rule

Do not move or retire a legacy executable merely because a replacement exists.
First require:

1. a written current-behavior contract;
2. characterization tests;
3. required legacy-equivalence evidence;
4. migrated consumers or a compatibility wrapper;
5. the relevant scientific decision and validation gate.

Legacy implementations may then move under a clearly named legacy location
while a tested wrapper preserves any supported entry point. Avoid creating
empty aspirational package trees before real behavior is migrated.

## Plan and documentation boundaries

- `Plans/` contains active governance, decisions, status, and implementation plans.
- `docs/analysis/` explains current behavior and architecture or records audits
  and legacy references.
- The implementation index is the only plan-status board.
- Dated snapshots are historical documentation, not live status authority.
- Historical plans in Git history do not imply completed work; the
  implementation index records the current status of each gate.

## Definition of paper-authoritative

A result is paper-authoritative only when its source inventory, configuration,
environment, cohort, outcomes, model or learner method, diagnostics, panel data,
figures, code commit, and artifact hashes are frozen and mutually traceable.
Exploratory or fixture-only results remain clearly labeled even when their
software path is complete.
