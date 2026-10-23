# Folder structure and ownership

Author-requested reorganization, 2026-10-13. Everyday paper work is figure-led; evidence is separate from current guidance.

| Location | Owns | Does not own |
|---|---|---|
| `docs/paper/figures/` | Figure purpose, selected panel navigation, legends, current state, review navigation and scoped concerns, figure-specific assembly | Duplicated shared methods, frozen code, generated media |
| `docs/paper/methods/` | Shared scientific definitions and contracts, including individual learning; full candidate-method study texts migrate under Plan 12 | A separate folder per scientific workstream, commands or progress |
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
documentation. Existing local narratives remain pending the dependency-aware
[migration plan](../../Plans/12_STUDY_ARTIFACT_STORAGE.md); preserved byte-sensitive
records are not moved or weakened automatically.

Scientific status (accepted/proposed/superseded), presentation status (frozen display/active draft) and storage role (local record/external archive) are independent. Age and filenames do not establish authority. `current-state.json` is the sole scoped presentation pointer. Other JSON evidence is version-bound. Existing indexes may be mutable; frozen records and published transfer checkpoints are not.

Superseded accepted freezes stay under their figure/panel scope's `history/`, retaining dated version and role folders. Current scientific selections stay under `configs/paper-figures/selections/`; superseded pointers use `history/`, supporting correction lineage uses `lineage/`. This author-requested migration does not authorize future automatic disposal after a freeze.

The original element specification and two linked source reviews retain their byte-sensitive paths under `docs/analysis/figures/`. The [maintained element explanation](../paper/figures/shared/figure-element-specification.md) owns the later supported extension; exact historical hashes resolve preserved originals. Six brief compatibility pages preserve historical links; they are registered exceptions, not an active parallel hierarchy.

Use `python scripts/check_repository_structure.py` for read-only ownership, active-link, relocation and immutable-hash checks; add `--external` to verify current bound external artifacts and archive transactions. Do not replace missing storage with an older artifact.

The [migration record](../../records/repository-history/2026-10-13-paper-structure/README.md) records every source disposition, original hashes and preserved uncommitted state. Update supported consumers and navigation for new moves, preserving byte-sensitive aliases and old checkpoint bytes.

The supporting F/G/H correction pointer retains its original selection path because the immutable element specification links it directly. Its role is indexed under `selections/lineage/`; V12 remains preserved as superseded scoped evidence. Current selections resolve through the freeze index. The old artifact-index path is a redirect to records for the same reason.

The [implementation review](../../records/repository-history/2026-10-13-paper-structure-review/README.md) distinguishes completed repository separation from pending SSD version dispositions. Operational guides must not contain future command-framework or pilot-progress chapters; those belong in plans or preserved design evidence. The checker also verifies local Markdown heading fragments and preserved originals from subsequent migration records.

Author exception, 2026-10-13: `to_implement.md` is retained at the repository root as the author's working request notes. It is not the plan-status index or a scientific decision register. Its historical source remains in the archive records.

Live recovery indexes can evolve while their migration-original bytes remain
preserved under repository history. `mutable_index_originals` identifies those
exact originals; it never relaxes a frozen-record requirement. Published
transfer inventories remain unchanged. Read-only external audits recognize only
exact path/hash/size absences from authenticated completed author dispositions;
transfer verification and pruning remain strict. See the [reconciliation](../../records/repository-history/2026-10-17-organization-reconciliation/README.md).
