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
7. Freeze the accepted candidate with the specification version and SHA-256,
   resolved styles, scoped exceptions, source/data hashes, assembly scale,
   element registry, and export paths/hashes. Follow the freeze-record contract
   in the specification. User-authorized freezing needs no redundant approval;
   ask only about decisions that remain unresolved.

The specification is currently a declarative contract, not an automatically
invoked renderer or CLI gate. Future freeze work must perform these steps
explicitly using the existing theme/export tools and the relevant renderer.
Do not claim compliance from a configuration pointer alone. Do not retroactively
restyle, refreeze, or overwrite historical artifacts without a user request.
