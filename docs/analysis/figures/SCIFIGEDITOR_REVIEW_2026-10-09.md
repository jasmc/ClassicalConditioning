# SciFigEditor source review for paper figures

Inspected the author's public [SciFigEditor repository](https://github.com/jasmc/SciFigEditor)
at commit `7c1de08f043aa9765060d50ee2285e4448f05671` on 2026-10-09 via
GitHub file access. No clone, installation or application runtime test was
performed. This is a source-code assessment, not a claim that the current
paper SVGs have been tested in the editor.

## Useful ideas and current adoption

| SciFigEditor idea | Use in this repository |
| --- | --- |
| Separate semantic role, classification confidence/evidence and protection class | Freeze elements carry scientific roles plus `classification_confidence`, `classification_evidence` and `protection`. Unknown classifications block a freeze instead of being guessed. |
| Semantic objects and relationships connect composite artists | Retain constituent IDs, `composite_id`/`composite_part`, and explicit colorbar `mappable_ids`; the gate rejects dangling mapped IDs. |
| Physical typography and geometry tokens | Measure fonts and strokes at the final assembly width, accounting for SVG transforms. Keep the paper's DejaVu Sans and agreed numerical defaults. |
| Immutable source geometry and restricted editing capabilities | Compare scientific/event/reference geometry, resources and ancestor transforms against the selected original SVG. Styling cannot silently move or replace the marks. |
| Proposals tied to source hash and working markup | Bind the reviewed candidate and its artifacts to hashes, recheck before publishing, and refuse overwriting an existing freeze manifest. |
| Integrated review records distinguish passed, failed and not-run channels | Require actual visual/scientific/structure review evidence. Missing reviews and unsupported evidence remain unresolved. |

The editor defines confidence and protection independently in its
[semantic types](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/svg/types.ts),
builds composite objects in
[semanticObjects.ts](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/svg/semanticObjects.ts),
and keeps geometry permissions distinct from ordinary style edits in
[roleCapabilities.ts](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/svg/roleCapabilities.ts).
Its [operation proposals](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/svg/operationProposals.ts)
bind `sourceHash` and `baseMarkup`; its
[integrated review](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/svg/integratedReview.ts)
compares protected geometry with the source and explicitly leaves model
review channels not-run until those reviews occur. This repository adopts
the concepts in its small Python freeze gate; it does not depend on the editor
or port its application/UI code.

## Direct integration limits

SciFigEditor's [semantic classifier](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/svg/semantics.ts)
recognizes its own role vocabulary and explicit `data-scifig-role` attributes.
Our `fig1__...` IDs and scientific roles such as `stimulus.us.expected` do not
automatically become those editor roles. A future adapter would need an
explicit mapping, retaining the original scientific role and sidecar, and
adding editor metadata separately. Generic data/reference-line roles cannot
replace the CS/US subroles or distinguish time zero from signal zero and
baseline equality. Such an adapter is not implemented by this change.

The editor's [default project tokens](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/project/types.ts)
use Arial, larger typography, 3.5 pt ticks, 1.5 pt primary lines and 0.8 pt
reference lines. Those values differ from this paper's agreed defaults.
Its generic reference-line styling in
[operations.ts](https://github.com/jasmc/SciFigEditor/blob/7c1de08f043aa9765060d50ee2285e4448f05671/src/svg/operations.ts)
also includes reference dash tokens. Applying that profile indiscriminately
would conflict with solid CS-onset/actual-US and dotted expected-US guides.
Keep the scientific subroles and their distinct patterns authoritative.

Font-file provenance, editor-side role adapters and an interactive exception
review UI could be useful later. They are separate work: this change provides
the repository contract and code-enforced freeze gate only. The author selected
checks **only at freeze time**; normal rendering and exploratory exports stay
outside that gate.

See the [paper element specification](FIGURE_ELEMENT_SPECIFICATION.md) and
[machine-readable roles/styles](../../../configs/paper-figures/figure-elements.json)
for the implemented contracts and agreed values.
