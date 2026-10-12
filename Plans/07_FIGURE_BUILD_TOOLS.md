# Reliable Figure Builds and Panel Replacement

## Why keep this plan

Make the existing figure tools dependable for everyday work: identify the correct panel, replace or reproduce it without changing unrelated panels or science, and save an auditable assembly review. This is shared implementation work for Figures 1–4, not another scientific decision or approval process.

AGENTS.md defines how reviews, storage, styling and freezes must be handled. This plan covers the remaining integration that makes those rules practical. Scientific analysis and approvals belong to Plans 01–06; release packaging belongs to Plan 08. A blocked/unavailable panel is an explicit outcome, not permission to substitute old evidence.

## What already exists

Checked against the current code on 2026-10-12:

- [SVG composition](../scripts/assemble_svg_figure.py) supports standalone panel previews, source replacement, source/layout hashes and sidecars. [Assembly tests](../tests/test_svg_figure_assembly.py) verify that updating one source preserves the other panel's source hash and that strict mode rejects missing sources.
- [Paper-panel orchestration](../src/classical_conditioning/figures/paper_panels.py) declares commands, supports a no-write `plan_only` mode, and records step outcomes and blocked registry panels. It covers selected Figure 1, Figure 2 Delay and Figure 4 routes; several routes are explicitly review precursors, not reproduction of current frozen panels.
- [The selected-assembly updater](../scripts/update_latest_figure_assemblies.py) authenticates and embeds selected Figure 1/2 panel sources. Its figure-specific wiring is not a complete Figures 1–4 build route.
- Semantic export, theme tools, relocation resolution and the explicit [freeze gate](../scripts/freeze_figure.py) exist. Existing validators should be connected consistently, not duplicated.

The focused assembly implementation is documented in [preflight and replacement](../docs/analysis/figures/assembly-preflight-and-replacement.md): no-write source/geometry plans, archive resolution, original/candidate provenance, baseline protection and locked transactional publication. Reuse it when connecting the remaining paper routes; do not reimplement these checks. Final-scale visual/scientific review and current-state publication remain separate existing workflows.

## Five core integration contracts

### 1. Resolve current panel sources consistently

Connect the registry, scoped selections, external-artifact resolver and assembly configurations so each build identifies its exact source, hash, scientific status and renderer. A newer scoped panel takes precedence over an older whole-figure assembly. Preserve the distinction between a selected frozen panel, a candidate and paper-release readiness.

**Done when:** Supported builds select the recorded current source or give a clear missing/stale/blocked result; they do not choose a version by filename age or silently use a review precursor instead of frozen evidence.

### 2. Make replacement and reproduction reliable

Build on the existing single-panel/replacement functionality. Record the source override and reproduction call; authenticate saved panel data and scientific definitions before rendering. Verify that untouched panels retain their source identities and protected scientific payload, while intended assembly geometry remains reviewable. Changing layout must not silently rerun statistics or alter a cohort, window or learner definition.

**Done when:** One selected panel can be previewed, replaced and reproduced with traceable inputs, and unrelated panels/scientific results remain protected. Historical snapshots remain unchanged; new changes produce separate reviewable candidates.

### 3. Connect the supported build routes

Extend the existing entry points as approved panels become available, including the Figure 3 route and current-source wiring for the other figures. Reuse no-write planning and existing renderers; report which inputs are valid, stale or unavailable and which work will execute. Make interrupted/failed builds recoverable from matching hashes without publishing a success-shaped partial result or overwriting accepted artifacts.

**Done when:** A documented supported command can reproduce each available paper figure/panel from explicit inputs, with a meaningful preflight and run record. This does not require implementing every proposed CLI/API/notebook interface.

### 4. Apply existing QC consistently at assembly scale

Integrate source/panel identity, semantic element mappings, sidecars, dimensions, effective fonts/strokes, clipping, legend/colorbar meaning and protected-payload checks across supported renderers and final-scale assemblies. Use the existing export/theme system and freeze validator. Add approved visual references when available; keep intentional scientific corrections distinct from presentation changes.

**Done when:** Structural, provenance and relevant visual checks cover the supported paper builds and detect stale/wrong sources or unintended changes. Freezing remains a separate explicitly requested operation under AGENTS.md.

### 5. Save one verified review with usable provenance

Connect build output to the current review workflow: one consolidated external review containing requested versions, editable candidate geometry and source/settings/layout/reproduction provenance. Verify the saved artifact before updating its current pointer. Preserve unavailable panels explicitly and make external-path/hash failures clear.

**Done when:** Another researcher or agent can locate the exact reviewed artifact and reproduce or replace its panels without relying on a chat preview or undocumented manual edits. Permanent media/data remain external; definitions and exact frozen code remain in the repository.

## Scope limits and completion

Figure 1 example/trial selection, Figure 2 pooling/centring, cohorts, learner representations, unavailable assays and presentation exceptions require their owning scientific/presentation decisions. This plan supplies the wiring once those choices are approved; it does not choose them. Current scoped freezes remain authoritative within their original scope.

Notebooks, a broader inspection API and a new general-purpose command framework are optional design ideas, not release prerequisites. The [detailed tooling reference](../docs/analysis/figures/figure-build-tooling-design.md) preserves those ideas and background specifications without making them mandatory work.

Complete this plan when the five practical contracts above pass appropriate replacement, source-resolution, preflight/failure, provenance and assembly-QC checks for the supported release builds. Track progress only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md); review/approval rules remain in AGENTS.md.
