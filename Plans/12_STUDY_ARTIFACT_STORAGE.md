# Study and discussion artifact storage

Author request, 23 October 2026: review the existing HTML-study skill and plan
how to organize Markdown/HTML used to learn, compare and make decisions with
Codex. Study/discussion Markdown can live entirely on JOAQUIM; numbered plans
remain in the repository.

The [storage policy](../docs/project-maintenance/review-artifacts.md) records the
new-work rule. The personal `analysis-review-guide` skill is adapted locally;
the migration of existing manuscripts and their consumers remains planned.
This plan changes neither scientific selection nor an existing guide's bytes.

## Ownership and mechanism

Use the existing personal skill for the reusable learning workflow, a short
root `AGENTS.md` instruction for routing, and the project policy for storage.
The existing catalogue/build/check tools provide discovery and integrity.
No additional custom agent or duplicate repository skill is needed.

| Keep in Git | Keep on JOAQUIM |
|---|---|
| Numbered plans and implementation index | Full study and comparison manuscripts |
| Instructions, repository skills and operating procedures | Preserved discussion/checkpoint text |
| Approved definitions and explicit scoped decisions | Unresolved questions and provisional preferences |
| Supported builders, tests and schemas | Published HTML, required sources/assets and retained revisions |
| Compact catalogue, provenance and exact-recovery records | Exported personal notes and full reading inventories |

The personal skill stays in its installed personal directory, currently
`/Users/joaquim/.codex/skills/analysis-review-guide/`. That is an observed local
installation, not a request to install or relocate it on another computer.
The repository's `.agents/skills/paper-figures/` retains its figure-work scope.

## Skill assessment and adaptation

The existing skill already covers evidence-grounded explanations, method status,
counts/exclusions, authentic figures, notebook import/export and browser checks.
Its missing distinction was ownership of the evolving Markdown manuscript and
its relationship to the published HTML and repository authority.

The narrow adaptation adds:

- Source/HTML co-location in the project's designated artifact store.
- One editable topic source, rather than repository and SSD manuscripts edited independently.
- A distinction between questions, recommendations, provisional preferences and explicit author choices.
- A route from an explicit choice to the existing decision/selection workflow.
- Dependency-aware relocation, preserved guide/section IDs and original-origin notes.
- Existing-catalogue reuse and honest static versus rendered verification.

Project drive paths remain in the project rule, not the reusable personal skill.
Its notebook helper and scientific content are unchanged. A personal skill edit
is local configuration; a repository commit does not distribute it to the PC.

## Target organization

Keep the SSD's `START-HERE.html` and `REVIEW-LIBRARY.md` entry points. For new
canonical study work, use:

```text
ClassicalConditioning Outputs/
  START-HERE.html
  REVIEW-LIBRARY.md
  studies/
    figure-02/panel-g/
    figure-02/panel-h/
    figure-02/panel-i/
    figure-03/
    figure-04/
```

A topic can contain `study.md`, `study.html`, the existing manifest format,
`sources/`, optional exported `notes/` and identifiable prior revisions. Reuse
an established generator/source format where replacing it would add duplication.
Separate dated conversation snapshots only when they preserve distinct evidence.
Do not create empty placeholder files or a second project progress board.

Existing dated HTML bundles under `current/reviews/` can remain where they are.
Catalogue their identities and migrate their editable source first. Moving every
HTML just to obtain uniform folders is unnecessary and risks notes and links.
Generated `review-library/<identity>/reading/` copies remain read-only snapshots;
they must not accidentally become a second canonical manuscript.

## Priority 1 — classify sources and preserve dependencies

Create a per-file disposition inventory with source hash, content role, consumers,
target path, existing copies, notebook origin where relevant and migration status.
The 23 October repository/output inventory is the starting evidence, not proof
that every Markdown file is movable.

| Existing material | Intended treatment |
|---|---|
| `docs/paper/figures/figure-02/panel-g/analysis-guide.md` | External canonical study source; keep its current HTML and bound source snapshots intact |
| `records/figure-reviews/figure-03/conversation-summary-2026-10-16.md` and `conversation-checkpoint-2026-10-23.md` | External dated conversation evidence; preserve exact bytes and source identities |
| `records/figure-reviews/figure-04/conversation-summary-2026-10-23.md` | External discussion source beside the existing planning HTML |
| `records/scientific-evidence/analyses/figure-02-g-independent/current-G-comparison-2026-10-15.md` and related explanatory narratives | External comparison material, with compact repository provenance retained |
| `docs/paper/methods/individual-learning.md` | Mixed candidate-method review: move teaching/comparison content; retain only necessary canonical definitions/boundaries in repository guidance |
| Figure `review.md` pages, shared review concerns and historical discussion Markdown | Classify content and bindings; move full discussion, retain concise scope/status/evidence pointers where needed |
| Plans, `AGENTS.md`, `SKILL.md`, current operational documentation, approved methods and frozen records | Retain their existing authority/ownership; do not blanket-move Markdown |

Do not overwrite an existing SSD copy unless its identity and role are established.
Preserve original bytes, manual edits and exact historical bindings. Frozen and
hash-protected paths are compatibility constraints, not a reason to relax checks.

Completion evidence: every proposed source has a disposition and consumer list;
no payload has been removed and no current pointer has changed.

## Priority 2 — teach readers to resolve external manuscripts

Update supported consumers before source relocation:

- `scripts/build_figure2_g_analysis_guide.py` has repository-bound `CONTENT` and reads comparison prose directly.
- `scripts/complete_figure2_g_analysis_guide.py` names the manuscript and methods in `SOURCES`.
- `scripts/build_figure3_explanation.py` reads and embeds repository discussion references directly.
- `scripts/preserve_figure3_guide_conversation.py` reads both conversation files from fixed local paths.
- `scripts/review_library.py` currently treats `storage=repo` reading sources as canonical and makes convenience copies.
- Catalogue entries, Markdown links, archive aliases and preservation checks must retain exact-version recovery.

Use existing `external_artifacts` resolution and project-native manifests rather
than another latest-version selector. Allow an explicit external source/manifest
at build time; authenticate it before reading. Keep rendering code and tests
local. A missing drive must report an unavailable source, not load an older copy.

Local/CI checks can validate discovery metadata without the SSD. Exact external
byte authentication remains an explicit connected-drive check; offline checks
must not claim to have verified unavailable bytes. Frozen local requirements
remain strict. Historical narrative relocation requires verified recovery aliases
and explicit preservation-aware checker support before removing its source path.

Completion evidence: representative builders and catalogue resolution read the
same exact source bytes through external storage, with missing/mismatched-source
failure checks. No scientific model is rerun to test this plumbing.

## Priority 3 — migrate the study Markdown in batches

Start with Figure 4 discussion, then Figure 3 checkpoints, then the G manuscript
and its comparison prose. Split mixed shared-method/review documents last.

For each batch: copy to an explicit topic destination; verify hashes; register
source and historical identities; update consumers, catalogue and live navigation;
test access from the repository and SSD; remove a redundant editable repository
body only after recovery and all consumers work. Keep a small pointer when useful.
Do not edit immutable source snapshots or published transfer checkpoints in place.

The author has chosen SSD ownership for this content. Execution still needs an
exact per-file disposition inventory and verification, not a blanket deletion
of Markdown. Historical/frozen compatibility requirements remain independent.

Completion evidence: one canonical editable source per topic, working source/HTML
links and unchanged selected/frozen science; no unregistered local full-text twin.

## Priority 4 — revisions, notes and decisions

Retain the old HTML paths and original browser origins while source-only migration
is sufficient. Before retiring an old reading route, preserve guide/storage IDs
and section meaning, export existing notes from their original context, and test
compatible import in a fresh context. Unrecovered notes keep the route-retirement
step open. Never clear or inspect the author's actual notebook for a test.

Subsequent study revisions can update understanding, objections, comparisons and
provisional preferences on the SSD. After an explicit author choice, add its
concise scope, reason and reviewed evidence identity to the owning repository
decision/selection record. Full discussion stays external. An agreement to study
or a provisional note is not scientific acceptance; freezing remains separate.

Completion evidence: compatible notes survive; prior guide/source versions remain
recoverable; publication binds the actual source and HTML; entry points lead to
the recorded reading revision. Unsupported preview/print checks remain explicit.

## Verification and current status

The immediate skill/rule/plan changes are documentation and configuration work.
Run the skill validator, active-link/structure checker and relevant existing
repository checks. Do not build a new study or move existing manuscripts merely
to validate this plan.

Current status belongs in [the implementation index](IMPLEMENTATION_STEP_INDEX.md).
[The adaptation record](../records/repository-history/2026-10-23-study-artifact-policy.json) identifies the local skill changes and exact SSD backups. Bulk migration and builder changes remain unimplemented. The four previously
identified Figure 1/2 external working-copy mismatches and the GitHub billing
blocker are separate from this storage plan.

The mechanism follows OpenAI's distinction between [reusable skills](https://learn.chatgpt.com/docs/build-skills)
and [project instructions](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
The SSD choice and migration scope above are the author's project directions.
