# Folder structure and ownership

Author-requested reorganization, 2026-10-13. Everyday paper work is figure-led; evidence is separate from current guidance.

| Location | Owns | Does not own |
|---|---|---|
| `docs/paper/figures/` | Figure purpose, selected panel navigation, legends, current state, review navigation and scoped concerns, figure-specific assembly | Duplicated shared methods, frozen code, generated media |
| `docs/paper/methods/` | Shared scientific definitions and contracts, including individual learning; full candidate-method study texts live on JOAQUIM under Plan 12 | A separate folder per scientific workstream, commands or progress |
| `docs/paper/decisions.md`, `review.md` | Current scoped choices and cross-paper concerns | Copies of all audit evidence |
| `docs/running-analysis/` | Running the supported code, parameters, interfaces, outputs, troubleshooting | Scientific approval or repository cleanup |
| `docs/project-maintenance/` | Current organization, governance and storage procedures | Historical inventories and completed cleanup reports |
| `records/frozen-analyses/` | Exact accepted records, scoped by figure/panel/date/role | Editable paper guidance |
| `records/scientific-evidence/` | Dated audits, approval support, conditional design evidence and legacy descriptions | Another live progress board |
| `records/figure-history/`, `figure-reviews/` | Past recipes and version-bound review evidence | Competing latest presentation pointers |
| `records/repository-archive/`, `repository-history/` | Archive/recovery control records and completed maintenance evidence | Archived media payloads |
| `Plans/` | Numbered future work and README/index | Decisions, audits, historical plans |
| `configs/` | Reusable executable settings and scientific selection pointers | Payloads or implicit scientific approval |
| `src/`, `scripts/`, `tests/` | Supported implementation, tools and tests | Superseded one-off candidate code |
| `legacy/` | Inherited reference/comparison source | Default supported execution |
| `.agents/skills/`, root `AGENTS.md` | Repository-specific assistant workflows and instructions | Personal/plugin skill installations or scientific authority |
| External storage | Permanent media, review HTML, recorded working SVGs, numerical payloads, superseded candidate code | New repository-local permanent exports |

One topic has one canonical home. Create a panel or joint-panel folder only when separate work warrants it; joint records are linked rather than duplicated. Learner analysis is a shared-method topic and Figure 3/4 concern, not a special documentation branch.

Author direction, 23 October 2026: full study/discussion/comparison manuscripts
belong with their HTML on JOAQUIM, including Markdown. The [study-artifact policy](review-artifacts.md)
distinguishes them from plans, approved definitions, instructions and operational
documentation. Ordinary study manuscripts now resolve through the external source registry.
Protected historical originals remain local compatibility evidence; reader,
notebook and print verification remains in the
[migration follow-up](../../Plans/12_STUDY_ARTIFACT_STORAGE.md). Preserved
byte-sensitive records are not moved or weakened automatically.

Scientific status (accepted/proposed/superseded), presentation status (frozen display/active draft) and storage role (local record/external archive) are independent. Age and filenames do not establish authority. `current-state.json` is the sole scoped presentation pointer. Other JSON evidence is version-bound. Existing indexes may be mutable; frozen records and published transfer checkpoints are not.

Superseded accepted freezes stay under their figure/panel scope's `history/`, retaining dated version and role folders. Current scientific selections stay under `configs/paper-figures/selections/`; superseded pointers use `history/`, supporting correction lineage uses `lineage/`. This author-requested migration does not authorize future automatic disposal after a freeze.

The original element specification and two linked source reviews retain their byte-sensitive paths under `docs/analysis/figures/`. The [maintained element explanation](../paper/figures/shared/iteration-workflow.md#scientific-element-identities) owns the later supported extension; exact historical hashes resolve preserved originals. Six brief compatibility pages preserve historical links; they are registered exceptions, not an active parallel hierarchy.

Use `python scripts/check_repository_structure.py` for read-only ownership, active-link, relocation and immutable-hash checks; add `--external` to verify current bound external artifacts and archive transactions. Do not replace missing storage with an older artifact.

The [migration record](../../records/repository-history/README.md#retired-maintenance-prose) records every source disposition, original hashes and preserved uncommitted state. Update supported consumers and navigation for new moves, preserving byte-sensitive aliases and old checkpoint bytes.

The supporting F/G/H correction pointer retains its original selection path because the immutable element specification links it directly. Its role is indexed under `selections/lineage/`; V12 remains preserved as superseded scoped evidence. Current selections resolve through the freeze index. The old artifact-index path is a redirect to records for the same reason.

The [implementation review](../../records/repository-history/README.md#retired-maintenance-prose) distinguishes completed repository separation from pending SSD version dispositions. Operational guides must not contain future command-framework or pilot-progress chapters; those belong in plans or preserved design evidence. The checker also verifies local Markdown heading fragments and preserved originals from subsequent migration records.

Author exception, 2026-10-13: `to_implement.md` is retained at the repository root as the author's working request notes. It is not the plan-status index or a scientific decision register. Its historical source remains in the archive records.

Live recovery indexes can evolve while their migration-original bytes remain
preserved under repository history. `mutable_index_originals` identifies those
exact originals; it never relaxes a frozen-record requirement. Published
transfer inventories remain unchanged. Read-only external audits recognize only
exact path/hash/size absences from authenticated completed author dispositions;
transfer verification and pruning remain strict. See the [reconciliation](../../records/repository-history/README.md#retired-maintenance-prose).

## Safe maintenance and folder guidance

Classify each item by purpose, authority and lifecycle before choosing its home.
Current guidance, future work, dated evidence, supported code, inherited source
and generated artifacts have different owners. Age, filename similarity and
absence of text references do not prove that material is obsolete.

Before replacing, shortening, merging or retiring anything, account for every
relevant requirement, behavior, dependency and historical identity. Record where
each survives, verify recovery and affected behavior, and retain unresolved
originals. A preserved file that cannot be found or used is not sufficient
completion evidence. Keep exact historical bytes and required compatibility
aliases separate from the maintained successor. Do not normalize protected
line endings or treat a byte-preserving relocation as scientific approval.

Classify documentation by purpose: maintained definitions and explicit decisions
belong in paper guidance, executable usage in running-analysis, and repository
procedures in project-maintenance. Keep future work in Plans and completed
evidence in records; full study manuscripts follow the external study policy.
Link to the canonical owner instead of repeating its rules or current status.
When updating a command example, verify its current path, dispatch stage and
defaults against source. Preserve registered historical/byte-sensitive pages.

Folder guides specialize this policy rather than copy shared procedures:
[configs](../../configs/README.md), [legacy](../../legacy/README.md),
[plans](../../Plans/README.md), [records](../../records/README.md),
[source](../../src/README.md) and [tests](../../tests/README.md).
Use the [script lifecycle](script-lifecycle.md) for commands. Preserve the tracked
`Plans/` spelling across hosts; a case-insensitive checkout can conceal its
lowercase filesystem spelling.

Retired Plan 13/15 accounts resolve through preserved Git context and the
existing archive aliases. Current navigation points to maintained verification
and requirements records; no old-path plan stubs are needed.

## Maintenance retention

Completed maintenance prose does not remain in the working tree by default.
Preserve its exact committed version, account for outstanding requirements and
verify recovery and consumers before retirement. Keep current authority and
operational dependencies accessible; Git-only history is not a runtime fallback.
Use one canonical owner and retain old-path pointers only for demonstrated
current readers. Scope figure-specific material by assembly, panel or joint scope;
shared material has one home. Routine communication belongs in chat.
