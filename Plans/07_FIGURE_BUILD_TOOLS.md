# Reliable Figure Builds and Panel Replacement

## Purpose

Make the existing tools usable across Figures 1–4: resolve the correct selected panel, reproduce or replace it without changing unrelated panels or scientific definitions, and publish an auditable review through the existing workflow.

AGENTS.md defines review, storage and freeze rules. This plan covers their implementation across supported paper builds. Scientific choices belong to Plans 01–06; release packaging belongs to Plan 08. Unavailable or unapproved evidence must be reported explicitly.

## Implementation to reuse

The assembler now provides no-write source/geometry preflight, external-artifact resolution, original/candidate provenance, reproduction commands, baseline comparisons and locked transactional publication. It checks panel payloads, clipping and shared assembly context; protects bound inputs and accepted/managed artifacts; and rolls back failed publication. These are existing building blocks, not tasks to implement again.

Use the [preflight and replacement guide](../docs/paper/figures/shared/assembly-preflight.md). The [implementation audit](../records/scientific-evidence/audits/figure-build-integrity-2026-10-12.json) records 424 passing tests and 70 unchanged frozen records for that implementation. Those checks used synthetic assembly inputs; they do not establish complete paper routing, scientific identity or final visual quality for Figures 1–4.

Reuse [paper-panel orchestration](../src/classical_conditioning/figures/paper_panels.py), [selected-assembly wiring](../scripts/update_latest_figure_assemblies.py), semantic/theme validators and the existing guarded review/current-state workflow. Do not add a second selection registry, review generator or state writer.

## Remaining work

### 1. Connect current selections to supported builds

Connect registry entries, scoped selection records, presentation state and assembly configurations to exact source hashes and rendering routes. The generic assembler authenticates explicitly supplied inputs; it does not discover the scientifically current selection.

A newer scoped panel selection takes precedence over an older whole-figure assembly. Keep frozen selection, active draft and release readiness distinct. A review precursor must never silently substitute for a selected frozen panel.

**Acceptance:** Each supported route resolves the recorded current source or reports a clear missing, stale, blocked or unavailable result. Source selection is never based on filename age.

### 2. Complete available paper reproduction routes

Extend existing commands for approved, available panels, including Figure 3 and current-source wiring for the other figures. Connect the assembler preflight and provenance to these routes. Document exact inputs, settings, renderer and reproduction command; identify unresolved scientific dependencies rather than inventing defaults.

Exercise selected-panel preview, candidate replacement and reproduction on authenticated paper inputs when available. Check untouched panel identities and protected scientific payloads as well as intended geometry. The generic synthetic tests already cover publication failures and replacement safeguards; add route-specific checks where they establish something new.

**Acceptance:** Each available release panel/figure has a supported reproducible command and meaningful preflight/run record. Failures preserve prior accepted work; no build reports a partial result as complete. Missing scientific approvals remain owned by their scientific plans.

### 3. Connect final assembly checks

Apply existing semantic mappings, source identities, dimensions, effective fonts/strokes, clipping, legends and protected-payload checks at final assembly scale. Generic geometry and XML integrity checks are implemented; they do not verify scientific roles, visual appearance or effective styling across renderers.

Use real renderer/visual evidence and approved references where available. Report incomplete semantic coverage and unresolved measurements explicitly. Keep scientific corrections distinguishable from presentation edits.

**Acceptance:** Supported paper builds detect wrong/stale sources and unintended assembly changes, with evidence for relevant final-scale measurements and visual checks. The explicit freeze gate remains a separate author-requested operation.

### 4. Connect builds to the existing review workflow

Pass verified build artifacts and their provenance into the existing consolidated external HTML review and guarded working-copy workflow. Preserve requested versions, editable SVGs, legends, settings and reproduction information. Use the existing figure-state publication lock and snapshot checks when updating a current pointer; assembler-side publication alone does not publish figure state.

**Acceptance:** A reader can locate the exact reviewed artifact and reproduce or replace its panels. Saved artifacts/evidence are verified before atomic state publication; concurrent work is preserved. Missing external evidence remains explicit. Permanent media/data stay external.

## Scope and completion

This plan does not choose cohorts, learner methods, examples, trial matching, pooling, centring or presentation exceptions. It connects approved choices without recomputing or refreezing them. Historical frozen snapshots remain immutable; candidates remain separate.

Additional notebooks, inspection APIs and a general command framework are optional, not first-release prerequisites. Background ideas remain in the [preserved tooling design](../records/repository-history/tooling-designs/figure-build-tools.md).

Complete this plan when the four integration requirements above are demonstrated for supported Figures 1–4 release builds, with unavailable scopes and their owning dependencies explicitly recorded. Generic assembler implementation alone is not completion. Track progress only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md).
