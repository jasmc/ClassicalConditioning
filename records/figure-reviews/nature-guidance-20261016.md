# Nature figure guidance comparison

Retrieved 2026-10-16. Scope: the journal **Nature**, main research figures;
not an assertion that all Nature Portfolio journals have identical rules.
Planning evidence only: Nature is now the selected target for the next
presentation revision; no figure, selection, active style or freeze changed.

## Official sources and requirements

- [Building and exporting figure panels](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/): 89/183 mm widths; maximum artwork height 170 mm; efficient alphabetical arrangement; embedded resources; editable vector main figures, with PDF/AI/EPS preferred and plain SVG accepted; approximately 50 MB file-size target. Accessible palettes and high-contrast text are advised; retain graphical keys.
- [Figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/): labelled axes with units in parentheses and visible axes/ticks; standard sans-serif fonts, preferably Arial/Helvetica; ordinary text 5–7 pt; lowercase, upright bold panel letters at 8 pt. Text remains editable, with embedded fonts rather than outlines. Use black key text with graphical color identifiers. Photographs need at least 300 dpi at final size; do not manufacture resolution by upsampling.
- [Final submission](https://www.nature.com/nature/for-authors/final-submission): one font family throughout figures; final strokes 0.25–1 pt; RGB recommended. It mentions a 247 mm page depth, which is not the current artwork-height allowance.
- [Formatting guide, figure legends](https://www.nature.com/nature/for-authors/formatting-guide): legends fewer than 300 words, with a brief title, panel/symbol descriptions and definitions of uncertainty/statistics. Detailed methods belong in Methods. Final legends are submitted in the text document separately from artwork.

The formatting guide also quotes 90/180 mm widths; the production figure guide
and final-submission page agree on 89/183 mm. Proposed production profile:
89/183 mm and the explicit 170 mm artwork cap. Record these source differences;
specific editorial instructions take precedence. Extended Data has separate
delivery requirements and needs a separate profile.

Additional official sources checked for the comprehensive plan:

- [Image integrity](https://research-figure-guide.nature.com/figures/image-integrity/): preserve original data and document composites, adjustments and reuse; no synthesized scientific image content.
- [Extended Data guide](https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/): dedicated 180 × 170 mm target and separate delivery route. Keep its rules distinct from main artwork; record conflicts with older page-size guidance.
- [Reporting standards](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards) and the [Reporting Summary](https://www.nature.com/documents/nr-reporting-summary-flat.pdf): audit sample units, exact n, inference, uncertainty, exclusions and reproducible data/code access against actual scientific evidence.

The specifications page itself gives both 300-dpi photographic advice and a
450-dpi export minimum. The plan uses native 450 dpi as a production target,
flags lower-resolution sources for resolution and prohibits upsampling as a
substitute for evidence. Discrete scientific heatmap matrices are not photographs.
Its explicit Symbol-glyph instruction also needs a resolved font/export check.
Review and production packaging have separate stages; no automatic submission
or change to numerical definitions is implied.

## Repository comparison

Read from current figure state and legend files on the retrieval date:

| Property | Current working rules/artifacts | Planned response |
|---|---|---|
| Width | Both 183 mm | Keep |
| Height | Figure 1: 215.4573 mm; Figure 2: 203.4216 mm | Recompose to the profile height; do not distort science |
| Type | DejaVu Sans; several roles 8 pt | Prefer Arial, with measured role sizes within the profile |
| Panel letters | Uppercase; contract default 10 pt | Lowercase display aliases, preserving scientific IDs |
| Tick stroke | 0.5 pt | Keep; audit other stroke roles |
| Caption length | Whitespace counts including draft headings: 774 and 794 words | Prepare concise captions and retain complete technical methods elsewhere |
| Condition headings | Colored text | Propose black text and explicit graphical keys |
| Review medium | Native SVG in self-contained HTML; 9 pt serif captions | Keep review format separate from submission packaging |

Measurements above are state-bound dimensions, not a fresh browser audit.
No new whole-figure compliance claim is made. Dense annotation typography still
needs full final-scale inventory, including retained statistical rows.

Our 1 mm invisible inset, two-inset panel spacing, equal-gap relationships,
3 pt geometric text gaps, 2 pt tick length, right-aligned keys, condition names
and experiment/control mappings are **author rules**, not Nature requirements.
They remain compatible project constraints unless a specific conflict is resolved.

## Comprehensive-plan update, 2026-10-16

[Plan 07](../../Plans/07_FIGURE_BUILD_TOOLS.md#5-nature-main-figure-presentation-profile)
now contains N01–N28 implementation checks and F01–F18 known fixes/decisions,
with explicit separation of journal rules, author rules and project defaults.
Further repository conflicts confirmed during planning include 1.2-pt
summary/median-IQR defaults and a condition-first `paired_block_ratio` family
that cannot describe the current control-first log-difference panels. The plan
also records incomplete full-element coverage, dense statistical annotations,
colorbar/estimator sharing, excess whitespace, unfinished panels and caption
length. These findings need version-bound implementation evidence rather than
an assertion of compliance from the rule file alone.

The external drive became available during planning. Read-only SHA-256 checks
matched both selected states, SVGs, legends and existing layout evidence against
the collection provenance; the selected collection HTML also matched. Actual SVG
root dimensions match the table above. Selected source SVGs:

- Figure 1: `e0f2db1f613c1cd149db5697f152054d570ea6dfa33d4efb10a27cc4648727bf`.
- Figure 2: `7decdf68b414f4a34cc6f01a345ec5a1d7381768b86787cdf6aeca73379903d3`.

No fresh complete browser, print, typography or scientific audit was performed
in this planning step. Offline cache snapshots remain historical sources and do
not supersede these selected drafts.

## Follow-up review of rule detail and repository integration

The initial plan separated journal and author rules, but the selected numerical
target remained in future-work planning and had no direct skill/AGENTS reminder.
The [maintained target](../../docs/paper/figures/shared/nature-style.md) now owns
those rules; AGENTS.md and the existing paper-figures skill route to it, and the
workflow owns procedures. Plan 07 retains N01–N28 and now F01–F22. No second
registry, exporter, skill or state pointer was introduced.

Additional details captured: explicit major/minor tick policy, spine sides and
offsets, label rotation/centering, numeric precision, multiplier/offset ownership,
and application to shared/secondary/twin/inset/colorbar axes. These are resolved
project choices/checks, not invented Nature numerical mandates. The cited
Nature pages require axes/ticks and legible labels but do not prescribe universal
tick direction, count, length or glyph gaps. Renderer padding is not a measured
glyph gap; current theme pad values alone cannot certify the author rule.

The final-submission page also permits 120–136 mm artwork where necessary and
requests essential layout instructions. Added both to the target/export tasks.
The official [common-delay checklist](https://research-figure-guide.nature.com/figures/top-10-ways-to-delay-your-paper/)
was cross-checked for missing assets, flattening, unreadable type and format risks.
The main specifications/production/formatting/final-submission pages were
rechecked. The statistics-policy URL could not be fetched; no new threshold or
plot prescription was inferred from other Nature Portfolio journals. Existing
reporting-summary evidence remains the reporting source used here.

Two repository issues were corrected: AGENTS.md's hard-coded Figure 1 v10 name
and workflow wording suggesting only Figure 1 owns presentation state. Both now
resolve scoped current state. The byte-sensitive original element specification
was left untouched, and implementation instructions now preserve accepted
contracts while adding compatible versioned profile resolution.

These are guidance/routing corrections only. The executable Nature profile,
generator integration, validators and artwork migration remain pending. Actual
selected figures were not restyled or re-certified by this review.

Validation of the follow-up:

- The skill-creator `quick_validate.py` reports `Skill is valid!`. PyYAML was
  loaded in a task-created temporary dependency directory, removed afterwards;
  the repository environment/dependency files were not changed.
- All new Nature routing links and the plan anchor resolve; N01–N28 and
  F01–F22 are complete and unique. Scoped `git diff --check` passed.
- All 13 protected specification/contract, state, legend, SVG, collection HTML
  and layout-evidence files captured before this review retain their starting
  hashes. This is preservation evidence, not a fresh visual compliance test.
- `scripts/check_repository_structure.py` fails with 19 issues in existing
  links, preserved hashes and fig-1E/F/G/H registry/state agreement. It checked
  262 protected files and two current states; external verification was not
  requested. The reported specification/archive files and scientific states
  were not changed by this review. The plan now names the relevant unresolved
  checks in F13/F21. No broad repair, hash reset or scientific approval was made.

## Coding-efficiency and full-coverage review

The remaining practical gap was the absence of supported code to load the
selected values. Implemented the versioned
[nature-main profile](../../configs/paper-figures/nature-main.json),
[read-only resolver](../../src/classical_conditioning/figures/presentation_profiles.py)
and explicit `get_theme(profile="nature-main")` adapter in the existing theme.
The resolver inherits the existing role catalog, colors and active frame policy,
checks policy consistency and returns the three input hashes. It rejects unknown
roles/properties, out-of-range style values, unresolved rule sources, unapproved
exception declarations and machine-dependent font alternatives. The resolved
contract view identifies its overlay and explicitly lacks freeze-gate integration.
No second semantic registry, state writer or exporter was introduced.

The maintained guide no longer duplicates a numerical project-role table.
Its coding entry point queries `summary`, `styles`, `typography`, `colors`,
`roles`, `rules`, `theme` or a single `--role` canonical identity; AGENTS.md, the existing skill, workflow and figure
index route there. Routine styling reads only the needed role/relationship;
plans/history are consulted for unresolved decisions or scope changes. Legacy
`get_theme()`/`apply_theme()` defaults retain DejaVu and prior sizes. The Nature
adapter explicitly chooses Arial, normal weight/style and the resolved role
sizes; no environment-dependent font-family substitution is allowed.

Coverage was checked again against the specifications, production, integrity,
Extended Data and formatting pages. Added explicit treatment of effective
stroke widths, caps/joins, dashes, marker/arrowhead/error-cap geometry, opacity
versus stroke width, composite colors, missingness, actual glyph/font embedding,
natural ratios and type-specific raster/export requirements. Pilot measurements
that Nature does not specify remain named as unresolved, rather than receiving
invented journal values. Conditional sequence, chemical, gel/blot and ED-table
rules are identified; they are not silently applied to unrelated artwork.

Also fetched the current [Nature Portfolio AI policy](https://www.nature.com/nature-portfolio/editorial-policies/ai).
Its verification/accountability framework differs from the broad imagery ban
on the figure-integrity page. The guide records this distinction and preserves
authenticated imagery, inspectable code and human scientific decisions. It does
not interpret a coding assistant as permission to fabricate imagery or results.
The final-submission page returned an error on this review's refetch; the earlier
same-day successful source record remains the basis for its quoted requirements.

Validation: 16 targeted unit tests pass (`test_presentation_profiles.py` and
`test_theme.py`), covering the opt-in/legacy boundary, inherited identities and
colors, original bytes, input-hash changes, invalid policy values, approval
references, font failures, color/typography reset and one-role lookup. Skill validation
also passes; temporary validation dependencies were removed. The real read-only CLI resolves
typography/colors and prints source hashes with render/migration limitations.
No new runtime dependency was added.

This implements profile lookup, policy checks and shared-theme adaptation only.
Per-renderer artist application, saved-SVG measurements, font embedding/export,
publication/freeze integration and current-artwork migration are still pending.
The repository-wide issues reported above are not silently marked resolved.
