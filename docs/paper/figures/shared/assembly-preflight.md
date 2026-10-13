# Assembly preflight, replacement and verified publication

Implemented 2026-10-12 as the focused software work in [Plan 07](../../../../Plans/07_FIGURE_BUILD_TOOLS.md). This extends the existing assembler and reuses artifact resolution, publication transactions, snapshots and locks. It does not introduce another scientific-selection registry, HTML-review generator or current-state pointer.

## Scope

The generic assembler works with an **explicit layout** and its recorded sources. Preflight verifies that layout; it does not choose a newer scientific selection or turn a review precursor into current paper evidence. For the current accepted figure display/draft and its publication pointer, continue using [the figure workflow](iteration-workflow.md), `figure_working_copy.py` and the guarded review generator.

The existing declarative SVG assembler remains the renderer. Frozen source files and layouts are not edited by a replacement preview. Permanent destinations must be explicit and outside the repository; disconnected `/Volumes/...` destinations are not fabricated. Published archive roots, bound inputs, outputs matching accepted freeze hashes and current-state-managed artifacts cannot be overwritten by this generic tool. Temporary synthetic tests may use system temporary directories.

## No-write preflight

From the repository:

```sh
.venv/bin/python scripts/assemble_svg_figure.py configs/paper-figures/figure1-assembly.json --plan --strict
```

This inspects sources, hashes, supported source types and layout geometry without rendering or writing. Historical Windows/repository references use the external-artifact resolver. It reports exact source paths/hashes, ready/placeholder/blocked panels, missing storage, invalid source/hash/geometry and original-versus-candidate provenance. A blocked plan exits nonzero. Non-strict previews may retain explicit placeholders; required frozen sources and explicitly requested replacement sources cannot silently become placeholders.

`--plan --output <external.svg>` also checks the destination. `--plan --baseline-manifest <previous.svg.json>` checks the saved baseline and source/geometry protection. Embedded artwork comparison requires rendering; preflight does not claim visual inspection or scientific freeze compliance.

## Candidate build and replacement

Use a new external candidate filename and a sidecar from a previous build by this assembler:

```sh
.venv/bin/python scripts/assemble_svg_figure.py /path/to/reviewed-layout.json \
  --output '/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/candidates/figure-01-candidate-02.svg' \
  --baseline-manifest '/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/candidates/figure-01-candidate-01.svg.json' \
  --replace-panel-source 'E=/path/to/candidate-panel-e.svg' --strict
```

`--panel E` produces a standalone preview. A replacement strips repairs/crops and freeze/source-provenance claims belonging to the old file; it retains the old source identity separately and marks the override as an unfrozen candidate. An original frozen source must still authenticate. This does not approve the replacement's science or appearance.

The sidecar records layout/effective-layout hashes, renderer/helper/input hashes, an executable reproduction command, original and replacement source hashes, geometry and canonical embedded panel/clipping hashes and shared SVG context hashes. With a baseline, it records before/after changes and refuses unrequested panel/source/geometry/payload/clipping or shared styling changes. Baseline hashes and panel claims are checked against the saved SVG. Older sidecars lacking payload/context hashes support source/geometry comparison only; that limitation is explicit in the report. When a candidate is explicitly overwritten, prior panel identities/output hash are retained in the report; use a new destination when retaining the complete prior assembly is required.

## Publication and recovery

The SVG and its `.svg.json` sidecar are built in staging, published under a lock, and checked against the staged hashes. Captured sources, layout, renderer helpers, baseline and destination state are checked again before/after publication. On a publication or verification failure, the existing transaction helper restores the previous artifact set. Concurrent input/output edits stop publication and preserve the other editor's work. An existing incomplete pair or an edited output with stale provenance must be reconciled or preserved at a different destination.

The CLI refuses existing outputs unless `--overwrite` is explicit. That flag applies only to a verified, unmanaged candidate pair, with snapshot protection; it cannot replace accepted state-managed artifacts or archived originals. Watch mode may update the candidate it created during that invocation, but cannot take over an existing file without `--overwrite` or bypass hash conflicts.

No Inkscape process is needed for the default SVG build. Explicit `--export png pdf` uses the existing Inkscape exporter inside the same verified transaction. Export failure preserves the previous set. Previously requested formats not rebuilt are recorded as unmodified prior exports, not claimed as current renders. Separate media formats still require explicit author instruction under AGENTS.md.

The generic assembler does **not** update a current figure pointer. Feed verified candidates/provenance into the existing consolidated-review/guarded-state workflow, which publishes evidence before its atomic pointer. This prevents a competing current-state writer.

## What this does not finish

This is source/provenance/structural verification, not an assessment of biological claims or final-scale visual legibility. Approved element mappings, effective typography, clipping and visual review remain governed by the existing specification/workflow. Complete Figure 3/current-paper build routing and integration across all renderers remain work in Plan 07 as their authenticated scientific artifacts become available.

Regression coverage is in [assembly integrity tests](../../../../tests/test_assembly_integrity.py), [existing assembly tests](../../../../tests/test_svg_figure_assembly.py), and [artifact transaction tests](../../../../tests/test_artifacts.py). Fixtures are synthetic and temporary; real selected figures, source data, freeze records and current pointers are untouched.
