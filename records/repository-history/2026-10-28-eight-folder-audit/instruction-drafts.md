# Proposed repository instructions — not active

These drafts supplement the existing authority map; they do not supersede
standing author directions. Each folder report contains the exact proposed
folder wording and names its maintained owner. Apply A14 as a coherent guidance
change after reviewing the findings, not by treating this dated audit as policy.

## Central wording

Proposed addition to `docs/project-maintenance/structure.md`:

> Classify each item by its purpose, authority and lifecycle before choosing a
> home. Current guidance, future implementation, dated evidence, supported code,
> inherited reference source and generated artifacts have different owners.
> Age, filename similarity and absence of text references do not prove that
> material is obsolete. Keep one maintained home per topic and preserve required
> immutable snapshots and compatibility aliases separately. A rename, merge,
> move or retirement must identify the destination, affected consumers, exact
> source/recovery bindings and appropriate verification before publication.

Proposed addition to `docs/project-maintenance/script-lifecycle.md`:

> A catalogue entry establishes discoverability, not scientific approval or safe
> replay. Distinguish maintained operations, reproduction/audit tools and dated
> transaction drivers. Extract live shared helpers and inspect the dependency
> closure before reorganizing a driver. Keep source-bound historical versions
> recoverable even when a supported successor has a clearer name or location.

Proposed clarification to `docs/project-maintenance/storage.md`:

> Maintenance inventories and logs are control evidence, not figure versions.
> Give them an explicit external home and a compact local receipt. Before adding
> a new control-evidence subtree beneath a browser-inventoried root, verify its
> discovery contract. Do not hide it under another task's exclusion, weaken
> artwork completeness checks or regenerate selected work solely to accommodate
> maintenance output. A new exclusion requires an explicit narrow scope and
> tests that retained artwork remains inventoried.

## Folder wording and owners

| Proposed owner | Exact wording source |
|---|---|
| `configs/README.md` | [Configs draft](configs.md#proposed-instruction-wording--not-active) |
| `docs/project-maintenance/structure.md` | [Docs draft](docs.md#proposed-instruction-wording--not-active) |
| `legacy/README.md` | [Legacy draft](legacy.md#proposed-instruction-wording--not-active) |
| `Plans/README.md` | [Plans draft](plans.md#proposed-instruction-wording--not-active) |
| `records/README.md` | [Records draft](records.md#proposed-instruction-wording--not-active) |
| `docs/project-maintenance/script-lifecycle.md` | [Scripts draft](scripts.md#proposed-instruction-wording--not-active) |
| New `src/README.md` | [Src draft](src.md#proposed-instruction-wording--not-active) |
| New `tests/README.md` | [Tests draft](tests.md#proposed-instruction-wording--not-active) |

Existing folder indexes link to these central owners; root AGENTS already routes
through the authority map and needs no duplicated procedure. New Markdown/JSON
navigation names use lowercase hyphens; Python/importable command namespaces use
underscores. Keep established Plans naming and historical identities. Apply no
mass renaming merely to normalize conventions.

## Checks worth adding during implementation

- A subprocess regression proves importing the atlas helpers leaves caller
  plotting settings unchanged; a scoped rendering test checks its intended style.
- Focused tests prove extracted helpers preserve protocol, completion and partial
  recovery behavior, and source-loading tests restore interpreter state.
- Synthetic table fixtures prove the shared writer preserves schema, index and
  receipt contracts.
- A narrow browser-discovery fixture distinguishes registered control evidence
  from retained artwork; do not generalize the current two special-case prefixes
  without testing their historical inventory semantics.

Use existing structure/link/catalogue checks for naming, documentation placement
and command coverage. This audit proposes no schema migration, universal
unused-file detector, automatic deletion policy or new status registry.
