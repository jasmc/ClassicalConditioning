# Script command organization

The author requested a whole-folder audit on 24 October, then clarified that
needed audit recommendations were to be implemented. Implementation is complete
for the maintenance findings. Scientific figure reconciliation and hosted CI
account access remain separate scopes.

The [implementation record](../records/repository-archive/script-organization-20261024/README.md)
owns source recovery, dispositions, current paths and verification. The original
[audit](../records/repository-history/2026-10-24-scripts-audit/README.md) remains
dated evidence; its full per-file Markdown/HTML lives on JOAQUIM. No new selection
registry, figure-state authority, custom agent or repository skill was introduced.

## Implemented repairs

The historical train/test metric producer and magnitude consumer share one
external workspace. Producer CSV hashes are recorded in a receipt and verified
by the consumer; an explicit SHA-256 supports older inputs. Both preserve saved
outputs and show CLI help without storage access. Scientific windows, formulas,
bootstrap settings and legacy reproduction styling were retained.

Browser SVG verification is the default. Separate `apply_axis_geometry.cjs`,
`apply_annotation_alignment.cjs` and `style_nature_presentation.cjs` routes make
candidate modification explicit. Verification rejects conflicting edit flags;
`--center-labels` is an explicit geometry-edit mode. Three collectors now accept
`--base-url`, preserving existing origins as defaults. No browser collector or
artwork operation was run during maintenance.

## Shared implementation

Reusable helpers now live under `src/classical_conditioning/`:

- `figures/review_html.py`: minimal scoped HTML and historical font compatibility.
- `figures/svg_scope.py`: scoped extraction and comparison IDs.
- `figures/figure_01/`: legacy display/layout and selected F/G/H integration.
- `figures/layout_primitives.py`: deterministic serialization/addressing.
- `analysis/review/bout_block_review.py`: unchanged D/E calculation/render helpers.
- `analysis/review/metric_comparison_storage.py`: external storage/input receipt.
- `reviews/figure2_g_sources.py`: source bindings and Markdown bundling.
- `reviews/embedded_downloads.py`: reusable offline download behavior.
- `figures/browser/command_options.cjs`: pure browser command/URL configuration.

Three script cycles were removed: legend/publisher, layout/relationships, and
block-log-median/bout-only review. Explicit parameters replace the D/E callback's
mutable `OUT` override. The G guide no longer imports a completed supplement
driver. New Figure 3 guides include the compatible Blob download handler and
disambiguated source names; existing HTML/notebook bytes were untouched.

`command_provenance.py` follows local source/import/resource dependencies without
executing them. Guarded command snapshots bind this closure, so helpers extracted
from a driver remain protected against concurrent changes. Existing scientific
input/state guards remain in place.

## Physical command organization

The [recursive catalogue](../scripts/README.md) lists 92 command/support files
under maintenance, workflows, reviews and paper figure/panel scopes. The top
level holds seven small compatibility facades plus its README. Importable Python
namespaces use underscores (`figure_01`, `panels_f_g_h`), while scientific source
and display identities remain distinct and continue to use their recorded roles.

Only real operations have directories. No empty Figure 4/H/I wrappers were added:
existing package/CLI operations continue to own those capabilities. Supported
historical computations remain explicitly under reproduction scopes; they are
not merged into a different accepted scientific analysis or new Nature styling.

Ordinary explicit imports, relocated-root bootstraps, JS resource discovery,
PowerShell roots, tests and maintained navigation were updated. The catalogue
checker scans recursively and distinguishes complete relative command paths.
The immutable element specification retains its required compatibility link.

The Windows interface has two protocol dispatchers, `delay.ps1` and `trace.ps1`,
with explicit stages. Original stage parameter/cohort/manifest contracts remain
separate. A common completion predicate removes duplicate platform plumbing.
PowerShell runtime execution was unavailable on this Mac; no real PC processing
was launched. The source contracts and paths were inspected and preserved.

## Verified retirements and storage organization

All 111 original tracked script-folder files were transferred and SHA-256
verified before edits. Their exact identities remain recoverable. Twenty-six
completed recipes were retired first; after helper extraction and fresh caller
checks, three more completed drivers and one redundant helper facade were retired.
Thus 29 completed drivers and one helper facade left the active command tree.
Published inventories are immutable; exact frozen source copies were not changed.

The [storage implementation](../records/repository-history/2026-10-24-storage-organization/implementation.json)
adds a drive-level README/HTML map. The unfinished 18 September intake's eight
files were verified, quarantined and removed from the processed-project workspace
after full hash authentication of the complete published recording and an open-
handle check. No bytes were discarded; same-drive quarantine saves workspace
confusion, not capacity.

The full Delay baseline, raw/provenance variants, required archives and dated
handoff exports were retained. Moving the PC/Mac packets merely for naming would
break still-bound Figure 2 G recovery; their roles are now explicit in the drive
map. The raw duplicates save little space and remain preserved acquisition names.

## Verification and limits

Maintenance/storage/relocation tests and targeted figure regressions were run
from the working tree and an exported Git tree. Actual counts and results are
in the implementation verification record. All 120 starting selected/frozen/state
file hashes remain unchanged, and the structure checker still enforces its 365
preserved requirements and seven scoped states. Existing review HTML and study
sources remain byte-identical.

Two legend-layout test failures were reproduced before refactoring against the
unchanged starting code and remain unchanged in the broader regression run.
They expect old Trace labels and protocol geometry. This maintenance did not
restyle scientific artwork to satisfy those stale assertions or weaken the tests.
The four earlier external Figure 1/2 working-copy mismatches and hosted GitHub
billing blocker remain explicitly separate; neither was concealed with new hashes
or policy exceptions.
