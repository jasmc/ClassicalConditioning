# Independent audit: src

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused static review of dependency direction, shared helper candidates, packaged
assets and import behavior; no scientific pipeline was run.

## Ownership

Keep maintained importable scientific/operational implementation, reusable
helpers and required packaged browser/review assets. Exclude one-off transaction
drivers, frozen source snapshots and installed/generated metadata. The local
`classical_conditioning.egg-info/` is ignored installation metadata, not tracked
source requiring archival or a new maintained home.

## Findings and dispositions

- **U1 — Scope plotting style changes.** [Individual atlas rendering](../../../src/classical_conditioning/figures/individual_atlas.py)
  line 28 updates `plt.rcParams` at import. An isolated probe set font size to 17,
  imported the module and observed 8 afterward. Tests also import its pure
  selection helpers, so unrelated plots/tests can inherit this style. Move
  style application into a scoped render boundary with one resolved profile.
  Preserve historical reproduction settings separately; this recommendation
  does not authorize re-rendering or restyling any saved atlas. Evaluate backend
  selection at the executable boundary as part of the same limited repair.
- **U2 — Consolidate identical table-writing plumbing.** Five identical
  `_write_parquet` functions occur in `analysis/cohort_outcomes.py:95` and
  `analysis/inference/{fish_bootstrap:171,fish_permutation:223,
  mixed_effects:250,model_input:341}. They use safe Arrow conversion, Zstandard,
  statistics and identical receipt fields. Proposed owner:
  `classical_conditioning.artifacts.write_parquet_table(path, frame)`.
  Keep local wrappers where compatibility/source bindings require them. Verify
  schemas, index handling, metadata and receipts on synthetic fixtures before
  any consolidation; do not infer numerical-payload equivalence from AST matches.
- **U3 — Retain scientific duplication pending a separate case.** Identical
  `_extract_arrays` implementations in two preprocessing routes are a candidate
  for later review, not an approved merge. Their route-specific contracts and
  frozen dependencies need scientific parity evidence. No static package import
  of `scripts.*` was found. The repository-relative panel orchestrator intentionally
  invokes scripts by subprocess; static import direction does not prove wheel
  installation portability.

## Proposed instruction wording — not active

Owner: new `src/README.md`, linking shared preservation/figure rules centrally.

> Put reusable implementation in the package and entry-point orchestration in
> scripts. Package imports must not execute a run, publish state or change global
> plotting style. Apply presentation settings inside a scoped render operation;
> keep scientific transformations separate from presentation edits. Declare
> packaged resources and explicit checkout-dependent execution requirements.
> Consolidate shared plumbing only after checking public behavior, receipts and
> source bindings. Similar scientific code is not sufficient evidence for a merge.

## Verification and limits

Evidence includes the import probe and AST duplicate candidates. Broad numerical
equivalence, installed-wheel execution and every module's import behavior were
not tested. No module was renamed or extracted. The two proposed future code
repairs need focused regression tests, while generated metadata needs no cleanup.
