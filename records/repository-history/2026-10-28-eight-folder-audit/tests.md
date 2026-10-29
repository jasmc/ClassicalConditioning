# Independent audit: tests

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused review of discovery, source-loading helpers, isolation and compatibility
contracts. The selected baseline test result is recorded in the validation receipt.

## Ownership

Keep behavior-level regression/characterization tests, small necessary fixtures
and test-only helpers. Exclude production implementation, permanent rendered
evidence and duplicate scientific payloads. Preserve independent oracles and
intentional compatibility assertions even when they mention old names.

## Findings and dispositions

- **T1 — Isolate source loading.** Eleven test files contain `sys.path.insert`
  or `append`; some correctly restore it. In
  [test_trace10_recovery.py](../../../tests/test_trace10_recovery.py) line 8 the
  root scripts path remains inserted after module loading. Bare driver imports
  can therefore depend on discovery order. Use the scoped pattern already in
  `test_trace3_run_completion.py` or import extracted package helpers directly;
  also isolate newly loaded module names where necessary. Keep recovery cases
  for partial publication and missing upstream stages unchanged.
- **T2 — Make browser regressions discoverable.**
  `tests/check_non_scaling_strokes.cjs` is not discovered by Python unittest;
  it requires Playwright, an executable browser and a server at hardcoded port
  8766. `layout_frame_browser_cases.py` supplies separate renderer cases.
  Neither has a direct usage link in current docs/scripts search. Add a test
  guide specifying exact invocation, server/dependency prerequisites and an
  external evidence destination. A future optional runner may parameterize the
  URL; do not report Python test success as browser coverage.
- **T3 — Retain intentional historical contracts.**
  `test_repository_organization.py` preserves the three `my_*` module identities
  and default-import isolation; parity tests AST-load an original oracle.
  `test_command_organization.py` protects facade/import/provenance behavior.
  These are not obsolete tests to remove merely because their names or paths
  predate the current package. No test deletion is justified by this review.

## Proposed instruction wording — not active

Owner: new `tests/README.md`, linked from normal contributor/test navigation.

> Organize tests around observable behavior and preserved contracts. Keep
> characterization oracles independent of the implementation they check. Use
> temporary fixtures and explicit external-test prerequisites; restore modified
> import paths, environment and module state. Share test-only setup without
> hiding scientific assumptions or importing transaction drivers for convenience.
> Document checks outside normal unittest discovery and report their results
> separately. Retire a test only after identifying its replacement coverage or
> the explicitly retired behavior.

## Verification and limits

No full-suite or browser run is claimed. A future isolation repair needs fresh
subprocess tests of affected modules in both orders; a style-import regression
belongs with the src repair. Fixture equivalence and all apparent coverage
overlaps were not audited. Physical browser regression remains unexercised;
its future receipt must identify actual collector/browser/artifact versions.
