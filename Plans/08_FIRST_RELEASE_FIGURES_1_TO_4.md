# First release: Figures 1–4

## Goal and dependencies

Release the four main figures with the scientific evidence necessary for their claims. Complete the relevant baseline, cohort, outcomes/statistics, learner validation, Figure 4 analysis and figure-assembly requirements. An unresolved choice cannot be approved by releasing it.

Supplementary figure composition, supplementary independent response-timing analyses, tail mechanistic analyses and imaging are separate workstreams and do not block this release. Main-figure claims still require their applicable controls, source data, diagnostics and validation, even when no supplementary figure is published.

## Relationship to the Nature completion handoff

[Plan 07, section 6](07_FIGURE_BUILD_TOOLS.md#6-nature-completion-handoff-2026-10-17)
is the complete implementation handoff for element/scientific registration,
Nature styling, retained author rules, frames, accessibility, production export,
renderer checks and guarded publication. Its N/F checklist and H01–H10 packages
own that work; this plan consumes their evidence rather than duplicating style
constants or creating another validator/state system. The
[maintained Nature target](../docs/paper/figures/shared/nature-style.md) and
[authoritative workflow](../docs/paper/figures/shared/iteration-workflow.md) remain
the rule and procedure authorities. Progress stays in the
[implementation index](IMPLEMENTATION_STEP_INDEX.md).

The 2026-10-17 author clarification preserves the applied Nature changes and
limits the immediate visual edit to a black review-size box. Release planning
does not authorize further resizing, reflow, a return to old eLife styling or
putting the review guide into submission artwork. Current presentation selections
remain authoritative until explicitly scoped replacements pass publication.

Learner counts/classifications belong to Figure 3 and its owning scientific
plan only. Do not add them to Figures 1/2. Relevant fish/cohort sample sizes,
block-specific denominators and matched-control identities remain required.

## Work

- Reconcile the four selected assemblies with their newest scoped panel selections, definitions, counts, captions and manuscript claims.
- Verify code/environment, source inventory, configuration/recipe, metric/detector, cohort, outcomes, statistical or learner method, diagnostics and panel inputs by identity and hash.
- Check that unavailable/inconclusive assay inputs are represented honestly; do not invent 10sTrace data or approve a placeholder implicitly.
- Freeze new accepted panels/assemblies only through the explicit figure-freeze command when authorized; preserve existing freezes.
- Publish one immutable release manifest and tagged commit linking repository definitions/code to verified external numerical payloads and figure exports. Record reproduction commands and SSD/archive locations.

## Release preparation and acceptance gates

These gates define a future release; writing this plan, completing a style pass
or including a draft in HTML is not release or freeze approval. Prepare all
reviewable evidence before requesting an unresolved scientific decision or the
final release action. Reuse existing approvals within their exact scope. Report
each gate's artifact/evidence references and unresolved dependencies explicitly.

| Gate | Required result | Evidence/dependency |
|---|---|---|
| R01 — Scope and authority | Resolve Figures 1–4 and their exact panel selections, active presentation states, manuscript claims and declared release composition. Distinguish accepted, exploratory, pending and excluded scopes. | Current scoped states/selections and claim-to-panel matrix; Plans 01–06. Do not select sources by date or expand the ordered main collection implicitly. |
| R02 — Scientific validity | Every released claim has its applicable metric, cohort, outcomes, statistical/learner validation and diagnostics evidence. Unavailable/inconclusive inputs have an explicit disposition. | Scientific gates in the implementation index and owning plans. Figure 3 requirements stay with Figure 3; they do not create learner-count requirements for Figures 1/2. |
| R03 — Identity and reporting | Every visible result maps to its experiment, condition, matched control, source cohort/data, estimator and relevant uncertainty/contrast. Counts, exclusions and changing denominators agree across panels, captions, source tables and manuscript. | Plan 07 H02/H06 and approved source evidence. Shared colors or frames cannot establish shared controls or identical estimators. |
| R04 — Presentation and frames | All applicable Nature requirements and retained author rules have final-scale evidence or a justified scoped exception. Check physical fonts/strokes/ticks/gaps, ownership, related-panel alignments, meaningful grouping gaps and clipping. | Plan 07 H03–H07 and exact saved-SVG policy-bound evidence. Numeric containment and equal spacing alone do not establish final composition approval. |
| R05 — Accessibility and integrity | Color/contrast and non-color identification are reviewed. Native image/raster provenance, resolution where applicable, crops/composites and editable annotations remain authenticated. | Plan 07 H05; actual outputs and asset/source evidence. No artificial resolution, invented results or silent raster recoloring. |
| R06 — Caption and manuscript agreement | Final titles, panel references, units, symbols, uncertainty, sample units, exact relevant n and essential statistics match the selected data. Detailed definitions remain recoverable in canonical Methods. | Plan 07 H06 and a caption/Methods/manuscript cross-check, including source/display aliases. Review legends and submission captions are separate delivery components. |
| R07 — Production and source data | Final editable vector artwork, embedded valid fonts/resources, color space and physical dimensions pass checks on the actual exported files. Numerical/source tables and code/dependencies reproduce each result. | Plan 07 H08/H09. A browser HTML or print screenshot is not a production-font/editability check. Main/Extended Data requirements remain separate if both are actually needed. |
| R08 — Review and publication integrity | Selected standalone SVGs and collection embeddings agree within meaningful renderer tolerances; order, captions, isolation, self-containment, responsive fitting and every print page pass. Stale/manual/conflicting inputs and write failures preserve prior work. | Plan 07 H08–H10, exact HTML/artifact hashes, visual review and guarded-publication/failure evidence. Omit the black review box from release artwork. |
| R09 — Scoped acceptance/freezes | Any new scientific freeze is explicitly requested and passes the existing complete element/role/source gate. Existing accepted records stay immutable; unresolved decisions cannot be inferred from release packaging. | Exact scoped approval and `freeze_figure.py` check/output references where a new freeze is requested. A presentation draft can pass styling while remaining ineligible for a final scientific release. |
| R10 — Immutable release and recovery | Assemble, verify and review a complete manifest; only then publish the authorized immutable release/tag using the existing release/repository procedures. Verify external artifacts and reproduction/recovery references. | All applicable R01–R09 evidence; exact repository revision/environment and external payload hashes. No archive/deletion permission follows; later changes require a distinct release. |

### Known decisions that must remain explicit

- Figure 2 C/F/I are Pending in the older repository-selected draft. The later
  PC scientific draft exists on JOAQUIM and matches the current working files;
  [reconciliation](07_FIGURE_BUILD_TOOLS.md#immediate-completion-order-25-october-2026)
  must recover its state/caption/source bindings before adopting it. Final
  release still needs authenticated, scientifically accepted results or an
  explicit approved composition change; a later draft is not a freeze.
- Figure 2 A/B scientific estimators differ, and H ratios differ from D/E/G log
  differences. Preserve those distinctions until their owning scientific work
  explicitly resolves them. A styling/shared-colorbar check cannot do so.
- Retain Figure 1's current displayed/source aliases and D/E/Delay-heatmap data
  links. A manuscript/caption reference update must not reconnect examples to
  the wrong source panel or experiment.
- Figures 3/4 remain subject to their own current scientific and presentation
  state. This handoff does not select unfinished results, choose learner rules
  or transfer their scientific decisions into the figure style plan.
- A journal/editorial instruction can supersede generic Nature guidance for
  the submission. Record its scope/source and recheck affected outputs; do not
  silently rewrite historical freezes or apply Extended Data rules to main art.

### Required release manifest contents

Use or extend the existing manifest/release infrastructure. Record the following
by identity and hash rather than copying mutable payloads into another registry:

- Release identity/date, journal/submission stage, repository commit and pinned
  environment/dependencies; exact supported reproduction commands.
- Figure/panel selections, scientific freeze/approval references, experiment and
  matched-control identities, source numerical tables/payloads, cohort/exclusion
  records, estimator/configuration and the claim-to-panel mapping.
- Final SVG/vector-production files, captions, Methods/manuscript references,
  source/display aliases and any approved scoped exceptions or omissions.
- Profile/policy, role/relationship registry, fonts/resources, implementation
  hashes, numerical/frame/renderer/accessibility checks and actual human review.
- Ordered collection/review references where applicable, exact HTML verification
  and a statement that the black guide is a separate review derivative.
- Verified external locations, relocation/archive aliases, restoration steps and
  unresolved limitations. A final applicable gate cannot be hidden as an unnamed
  limitation; exploratory/conditional content must be explicitly scoped.

Keep permanent artwork, review HTML and numerical payloads on the prescribed
external storage; keep supported code/tests local and small evidence records in
their canonical records locations. Recheck hashes immediately before publication.
Missing storage, failed verification or conflicting manual edits leave the
current release/state untouched and produce a precise recovery report.

## Acceptance

Figures 1–4, their claims, relevant counts and analysis evidence agree. Every
applicable R01–R10 gate is satisfied with scoped evidence; every released artifact
is hash-verifiable and has a supported reproduction route. Complete Plan 07
presentation/production evidence is necessary but does not close scientific
gates. Later work has separate scope and cannot overwrite this release. Legacy
retirement and disposition of earlier versions remain separately governed.
