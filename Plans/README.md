# ClassicalConditioning Plans

## Purpose

This folder contains future work and the small set of governance files needed
to manage it. Descriptions of current architecture, behavior, audits, and legacy
routes live under [`docs/analysis/`](../docs/analysis/README.md).

All processing and artifacts remain local. No plan authorizes a scientific
correction by itself: decisions are recorded, implemented under stable recipe
names with authenticated inputs and settings, and validated before they become
paper-authoritative.

## Start here

- [Analysis governance](./GOVERNANCE.md) — durable engineering and scientific invariants
- [Decision register](./DECISIONS.md) — approved decisions and open gates
- [Implementation status index](./IMPLEMENTATION_STEP_INDEX.md) — the only live status board
- [Exclusion and selection inventory](./EXCLUSION_AND_SELECTION_INVENTORY.csv) — machine-readable register of every active, legacy, and deferred selection rule

## Active implementation plans

The numbers order work packages, not the stage execution graph. The
[scientific and figure order](../README.md#scientific-and-figure-order) runs
from raw data through Figures 1–4 and their supplements. Imaging integration
is a parallel track; design and fixture work may overlap. A final paper run
still depends on its relevant upstream decisions and outputs.

| Plan | Role |
| --- | --- |
| [01 Paper cohort completion](./01_COHORT_IMPLEMENTATION.md) | Gate C0/C1, population boundary, consumer migration, and reconciled counts |
| [02 Analysis and statistics](./02_ANALYSIS_AND_STATISTICS.md) | Claims, estimands, outcomes, model choice, diagnostics, uncertainty, and population robustness |
| [03 Learning-onset completion](./03_LEARNING_ONSET_IMPLEMENTATION.md) | Gate O/S, condition-aware models, onset calibration, any separately defined extinction claim, diagnostics, and final figure |
| [04 Learner method review](./04_LEARNER_METHOD_REVIEW.md) | Gate L discussion of continuous scores, trajectories, probabilistic methods, and validation |
| [05 Learner representation and analysis](./05_LEARNER_ANALYSIS.md) | Approved representation, manifest, validation, outputs, and non-circular inference |
| [06 Learner outputs and validation](./06_LEARNER_OUTPUTS_AND_VALIDATION.md) | Schemas, joins, temporal analyses, output inventory, and validation |
| [07 Integrated analysis and CR profiles](./07_INTEGRATED_ANALYSIS_AND_CR_PROFILES.md) | Catch/block estimands, sensitivity boundary, and final orchestration |
| [08 Figures and reproducible reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md) | Stage interfaces, panel data, figure registry, paper automation, and QC |
| [09 Tail mechanistic analyses](./09_TAIL_MECHANISTIC_ANALYSES.md) | After Figure 4's pooled-learner CR analysis, explore tail geometry, rhythmic power, and traveling waves |
| [10 Supplementary figures and data](./10_SUPPLEMENTARY_FIGURES.md) | Complete controls, coverage, validation, and timing analyses alongside Figures 1–4 |
| [11 Behavior and imaging integration](./11_BEHAVIOR_IMAGING_INTEGRATION.md) | Add optional imaging without changing canonical behavior results; stage work behind imaging gates |
| [12 Final analysis release](./12_FINAL_ANALYSIS_RELEASE.md) | Freeze the finished analysis and its evidence once |

Learner analysis is required for the paper. Whether its final representation is
continuous, probabilistic, categorical, or combined remains a scientific
decision, not an omission option.

## Documentation and evidence

- [Analysis documentation index](../docs/analysis/README.md)
- [Current pipeline guide](../docs/analysis/02_CURRENT_PIPELINE.md)
- [Proposed multimodal architecture](../docs/analysis/12_PROPOSED_MULTIMODAL_ARCHITECTURE.md)
- [Activity metrics, smoothing, and shared bouts](../docs/analysis/05_METRICS_AND_BOUTS.md)
- [Analysis findings](../docs/analysis/audits/01_ANALYSIS_FINDINGS.md)
- [Issue register](../docs/analysis/audits/02_ISSUE_REGISTER.md)

Documentation explains what exists or what was observed. It does not compete
with this folder as an implementation queue.

## Historical plans

Earlier plan versions remain in Git history. The numbered files here are the
active implementation plans; the index records current status and open gates.

## Lifecycle

- Add future implementation to the narrowest applicable active plan.
- Record cross-cutting scientific decisions in `DECISIONS.md`.
- Update status only in `IMPLEMENTATION_STEP_INDEX.md`.
- Move stable descriptions of implemented behavior to `docs/analysis/`.
- Keep completed or superseded plan versions in Git history and update links to
  the current authority.
- Keep user-facing commands and formats synchronized with the repository README.

## Working rule

Later stages may be prototyped on synthetic or local fixtures, but they do not
become paper-authoritative while their upstream scientific contracts, cohort,
validation mode, or release evidence remain open.
