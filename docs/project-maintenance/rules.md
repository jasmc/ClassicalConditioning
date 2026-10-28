# Rules and authority map

Start here when deciding which instructions apply. Detailed rules stay with
their owners; this page is navigation, not a second scientific decision register.

| Concern | Owner |
|---|---|
| Assistant instructions and preservation protections | [AGENTS.md](../../AGENTS.md) |
| Repository-specific figure skill | [paper-figures](../../.agents/skills/paper-figures/SKILL.md) |
| File placement and exceptions | [Structure](structure.md) and [executable policy](structure-policy.json) |
| Scientific precedence, identities and release gates | [Governance](governance.md) |
| Accepted scoped scientific versions | [Freeze index](../../records/frozen-analyses/selections.md) and [selection records](../../configs/paper-figures/selections/) |
| Selected working artwork | Scoped `current-state.json` files linked from [paper figures](../paper/figures/README.md) |
| Figure procedures and guarded publication | [Iteration workflow](../paper/figures/shared/iteration-workflow.md) |
| Current presentation target | [Nature guidance](../paper/figures/shared/nature-style.md), [profile](../../configs/paper-figures/nature-main.json), [frame policy](../../configs/paper-figures/layout-frames.json) |
| Current element/role explanation | [Maintained specification](../paper/figures/shared/figure-element-specification.md); the [byte-sensitive original](../analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md) remains preserved |
| Shared definitions and scientific decisions | [Methods](../paper/methods/README.md), [decisions](../paper/decisions.md) |
| Running code, parameters and outputs | [Running-analysis](../running-analysis/README.md) |
| Future work and progress | [Plans](../../Plans/README.md), [sole status index](../../Plans/IMPLEMENTATION_STEP_INDEX.md) |
| Storage, exact recovery and historical disposition | [Storage](storage.md), [archive records](../../records/repository-archive/README.md) |
| Study/discussion Markdown and HTML; reusable guide workflow | [Study-artifact policy](review-artifacts.md), personal `analysis-review-guide`; [migration plan](../../Plans/12_STUDY_ARTIFACT_STORAGE.md) |
| Supported versus historical tools | [Script lifecycle](script-lifecycle.md), [tool catalogue](../../scripts/README.md) |

Scientific selection, presentation selection and storage location are separate.
A style edit does not choose a method; a panel freeze does not freeze its whole
assembly; an old filename or cached source does not select current work.

## Apply instructions efficiently

Use `AGENTS.md` for essential protections, the applicable skill for routing and
the owning document above for procedure. Read the requested figure/panel state
and exact bindings; expand into plans, history or dependency payloads only when
needed to resolve a question or perform a required check. Do not reload unchanged
instructions within a task unless edits or conflicting evidence require it.

For scientific work, establish the claim/estimand and affected input chain before
execution. Use shared definitions once, scoped approvals as recorded, and the
existing supported tools. A storage, HTML or styling task does not reopen method
selection or run an analysis merely to fill a missing explanation.

Keep durable rules with one maintained owner; route from skills/agent instructions
instead of duplicating procedures or copying current status into handoffs. Preserve
dated evidence. Put large inventories and verbose verification output in their
designated stores and inspect summaries/affected entries. Catalogue generation
maintains version links without printing the catalogue into model responses.
This reduces instruction/context overhead; output-token use depends on the
response and tool reporting, not the number of generated links alone.

## Repository and personal customization

Repository skills belong in `.agents/skills/`; this repository's `paper-figures`
skill stays here. Its procedures live in the figure workflow. Root `AGENTS.md`
supplies instructions; it is not a custom-agent definition.

Personal skills and agent configuration live outside this repository. In the
author's current installation, `analysis-review-guide` is discovered from
`~/.codex/skills/`. Do not copy personal or bundled/plugin skills into the paper
repository merely because they are available during a task. Custom agents, when
needed, have separate project or personal configuration; this maintenance does
not introduce any. The [dated reconciliation](../../records/repository-history/2026-10-17-organization-reconciliation/README.md)
records the inventory and implementation boundary.

## Routine checks

From the repository root:

```sh
.venv/bin/python scripts/check_repository_structure.py
.venv/bin/python scripts/check_repository_structure.py --external
.venv/bin/python -m unittest discover -s tests -p 'test_repository_*.py'
```

The first command checks local organization and direct current bindings. The
external check additionally needs JOAQUIM and authenticates archive payloads and
exact recovered versions. Tests include the actual local repository policy,
synthetic failures and strict transfer/prune protections. Passing these checks
does not approve scientific results or certify rendered figures.

Choose checks by the changed behavior: documentation/skill routing needs local
structure, link and skill validation; executable changes need focused consumer
and failure-path tests; storage/resolver/publication changes also need their
archive, external, working-state and review-library checks. Scientific changes
need the relevant scientific validation as well. These commands are a catalogue,
not a requirement to repeat the entire suite after every prose edit. Preserve
required guards and report unavailable external/render checks explicitly.

Author direction, 25 October 2026: this repository has no CI. Run checks locally
when relevant to a change; do not recreate automated hosted workflows. The
commands above remain supported without a GitHub Actions dependency.

`recorded_repository_roots` in the structure policy maps only known historical
checkout prefixes for local verification in another checkout; it never substitutes
external payloads or rewrites frozen paths. External authentication still
requires `--external` on the connected author machine.
