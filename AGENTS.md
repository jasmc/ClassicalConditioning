# Repository instructions

## Panel and figure freezes

Whenever the user asks to freeze a panel or a whole figure, apply the
[figure element specification](docs/analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md)
and its [versioned roles and styles](configs/paper-figures/figure-elements.json).
This is a standing author instruction, recorded on 2026-10-09.

1. Identify the exact selected source revision and its scientific definitions.
   Check scoped current-selection records; the full assembly can be older than
   the selected panel. Preserve existing frozen files and their hashes.
2. Assign or verify unique element IDs, scientific roles, Matplotlib/SVG types,
   data mappings, units, normalization, coordinate systems and composite groups.
   Record tick values. Never infer a scientific role from color or position alone.
3. Apply the common styles to a separate, reviewable candidate. Resolve fonts
   and stroke widths at the intended final assembly scale. Do not change data,
   cohort membership, transformations, event times or statistical results as a
   styling operation.
4. Check semantic IDs, scientific mappings, effective opacity, stacking order,
   legibility, clipping, shared axes and colorbar meaning at that scale.
5. Ask a targeted question only for an unresolved exception, such as a selected
   appearance conflicting with a default, a visibility requirement, or ambiguous
   event identity. Complete independent candidate work first. Existing explicit
   approvals remain authorized within their original scope; do not ask again.
   An observed appearance alone is not evidence of approval. If the user has
   already authorized replacing an older style, use the new instruction.
6. Record each approved exception's ID, element role, property, value, reason,
   panel/figure scope, and approval evidence. Keep scientific definitions
   separate from presentation exceptions. In a noninteractive command-line
   workflow, report unresolved exceptions and do not finalize the freeze or
   invent approval.
7. Use `python scripts/freeze_figure.py --candidate <candidate.json> --output
   <new-freeze.json>` (or `classical-conditioning freeze-figure`) for every new
   panel or whole-figure freeze. Run `--check-only` first; resolve every reported
   issue before publication. Populate confidence/evidence/protection records,
   renderer-property review evidence and scientific/structure/visual reviews.
   Freeze the accepted candidate with the specification version and SHA-256,
   resolved styles, scoped exceptions, source/data hashes, assembly scale,
   element registry, and export paths/hashes. Follow the freeze-record contract
   in the specification. User-authorized freezing needs no redundant approval;
   ask only about decisions that remain unresolved.

These checks run only through the explicit freeze command. Normal rendering,
exporting and exploratory reviews do not invoke them. The agent prepares and
styles the candidate using the existing theme/export tools and relevant renderer;
the command validates it and refuses unresolved cases. It does not infer roles,
restyle input files or invent review/approval evidence. Do not claim compliance
from a configuration pointer alone. Do not retroactively
restyle, refreeze, or overwrite historical artifacts without a user request.

## Single-file review artifacts

Standing author preference recorded on 2026-10-09: save one consolidated review
file containing all requested versions. Do not also save separate PDFs, SVGs,
images, alternate output formats, or redundant data copies unless the user
explicitly requests them. Embed the required figures and provenance in the
single file. Use in-memory rendering and verification where possible; remove
task-created temporary files after verification. Preserve existing scientific
source data and historical frozen artifacts.

## Repository and external artifact storage

Author-approved on 2026-10-10: `Plans/` contains numbered future-work plans plus README/index only. Decisions, governance, audits, review concerns and transfer guidance live under `docs/`. Supported source/tests/tools remain in the repository; exact frozen definitions/manifests/code live under `records/frozen-analyses/`. Permanent media, visual HTML reviews, numerical payloads and superseded candidate code belong on JOAQUIM. Use `scripts/repository_archive.py` and the archive indexes; never remove a pending original before its external copy is verified. Preserve historical freeze bytes and resolve relocated paths externally. New permanent exports must use explicit external destinations; temporary test/render files are allowed.

Frozen-record naming and structure: use `records/frozen-analyses/figure-NN/panel-x/YYYY-MM-DD-description/` for single panels and `panels-x-y-z/` for shared scopes. Use role subfolders (`manifests`, `settings`, `code`, `sources`, `reviews`, `notes`), lowercase hyphenated JSON/Markdown names and underscore-separated source-module names. Preserve original frozen bytes and path aliases when reorganizing; see `records/frozen-analyses/README.md`.
