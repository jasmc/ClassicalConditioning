# Independent audit: Plans

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused review of the index, navigation and completed/mixed work in Plans 12–13;
Plan 07 was inspected only to preserve its outstanding completion boundaries.

## Ownership

Retain numbered future work, dependencies, acceptance criteria, README navigation
and the sole live implementation index. Exclude approved scientific decisions,
full studies and completed implementation narratives. Keep `Plans/` and existing
number identities; retired number 14 must not be silently reused.

## Findings and dispositions

- **P1 — Retire the completed Plan 13 body from future work.** [Plan 13](../../../Plans/13_SCRIPT_COMMAND_ORGANIZATION.md)
  lines 4–5 declares implementation complete; its remaining sections describe
  completed repairs, source recovery and old verification counts. Preserve its
  exact pre-edit bytes at
  `records/repository-history/2026-10-24-scripts-audit/completed-script-command-organization-plan.md`.
  Move only still-useful operating instructions to the existing lifecycle/tool
  guides, and put newly identified work in Plan 15. Update live navigation,
  rules and index references before removing the future-plan body; preserve
  historical references through exact relocation metadata, not mass rewriting.
- **P2 — Split mixed Plan 12 by lifecycle.** [Plan 12](../../../Plans/12_STUDY_ARTIFACT_STORAGE.md)
  lines 169–202 repeats implemented migration details; lines 204 onward retain
  actual reader-verification tasks. Preserve the original completed account at
  `records/repository-history/2026-10-23-study-source-migration-plan.md` before
  shortening the active plan to remaining QA, dependencies and acceptance.
  Central study policy keeps ownership and preservation rules. Do not retire
  Plan 12 or mark browser/print/notebook checks complete.
- **P3 — Reduce status drift.** The mutable index's Plan 12 row says the library
  has expanded to 63 entries; the current catalogue has 67. Prefer a catalogue
  link over another copied count. Explicitly dated 17/23/25 October statements
  are historical facts, not errors to replace with today's values. Plan 07's
  scientific and presentation gates remain open.
- **P4 — Preserve Git path case across hosts.** Git tracks this folder as
  `Plans/`, while this Mac's actual directory entry is `plans/` and
  `core.ignorecase=true`. Both spellings resolve locally, masking differences
  that matter on a case-sensitive checkout. Use the tracked `Plans/` spelling
  in new references and when recording new paths in Git. Do not rename the
  existing directory during this audit; any future case-only normalization
  needs an explicit, separately verified path change.

## Proposed instruction wording — not active

Owner: `Plans/README.md`, with progress rules retained in the existing index.

> Add a numbered plan only for remaining implementation work; do not reuse a
> retired number. Keep goals, dependencies, tasks and acceptance in their owning
> plan, with workstream progress only in the implementation index. When work is
> complete, retain its evidence in records and its operating rules in docs,
> updating active references before retiring the plan. Split mixed completed and
> future material without losing unresolved tasks. Dated historical statements
> do not serve as current status or scientific approval.

## Verification and limits

Before either proposed retirement, enumerate exact path/basename links, protected
bindings and relocation consumers; preserve source bytes and update mutable
navigation in one reviewed change. These moves were not performed. Other plans'
scientific tasks were not exhaustively compared with executable implementation;
this audit closes no scientific or reader-delivery gate.
