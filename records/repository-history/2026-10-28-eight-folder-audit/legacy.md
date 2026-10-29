# Independent audit: legacy

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused static review of the worker, adapter, learner imports and parity tests.
No historical algorithm or comparison analysis was executed.

## Ownership

Retain inherited reference source, exact historical comparison implementations
and characterization oracles. New supported implementation belongs in `src/`;
new supported entry points belong in `scripts/`. Legacy presence does not
approve a classifier or provide a default execution route.

## Findings and dispositions

- **L1 — Document the second live dependency.** [The guide](../../../legacy/README.md)
  describes isolated learner comparisons but calls all other originals reference
  source. `tests/test_legacy_preprocessing_parity.py` lines 12–25 imports helpers
  and AST-loads selected functions from `legacy/modules/my_functions.py` as an
  independent oracle. Add this explicit characterization role to the guide.
  Retain these sources and the import-path isolation tests.
- **L2 — Preserve the worker's dependency closure.** [The worker](../../../src/classical_conditioning/analysis/legacy_learner_worker.py)
  lines 19–38 names four learner scripts and loads them in isolated processes.
  Their imports include `analysis_utils`, `file_utils`, `plotting_style`,
  `experiment_configuration` and, for three variants, `figure_saving`; helper
  dependencies extend beyond these direct imports. File-stem labels such as
  `WIP` and `new` identify compared variants and are not rename candidates.
  The adapter hashes their sources and tests rely on original identities.
- **L3 — Retain historical duplication.** Static AST matching found repeated
  learner functions and plotting filters. Matching implementations across
  historical variants are not redundant runtime modules to merge. A future
  archival proposal must include comparison, parity and provenance consumers;
  this audit establishes no safe removal candidate among the remaining originals.

## Proposed instruction wording — not active

Owner: `legacy/README.md`, linking to central script/storage lifecycle rules.

> Preserve original filenames and algorithm identities. Supported legacy use is
> limited to the isolated comparison adapter and explicitly identified
> characterization tests. Other originals are reference source, not new command
> entry points. Keep required helpers and test oracles with their dependency
> closure. Do not deduplicate historical variants or edit their science as a
> maintenance refactor; put new implementation in src and document its relation
> to the retained original.

## Verification and limits

Retain `tests/test_repository_organization.py` contracts protecting original
locations and isolation. A later move needs adapter/parity tests, source-hash
checks and exact recovery aliases before publication. All 20 tracked files were
identified, but only the supported/test dependency routes were investigated
deeply. Reference-only plotting behavior and transitive optional-package runtime
compatibility were not exercised.
