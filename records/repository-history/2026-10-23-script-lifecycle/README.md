# Script lifecycle and CI follow-up — 23 October 2026

Continues the author-requested organization work after a fast-forward pull of
`develop` at `ac43b76`; the checkout was clean and already current.

The [assessment](assessment.json) resolves all 14 entries previously marked
`retained-pending-classification`, reviews the five historical candidates and
classifies the later download-repair helper. It inspects entry points, source
dependencies, direct AST importers and scientific/review evidence references.
Catalogue mentions alone do not establish a consumer. The original 17 October
inventory remains unchanged.

[Disposition recommendations](disposition-proposals.json) retain 19 assessed
sources locally. In particular, all five historical presentation recipes are
kept for their evidence/compatibility role. The completed single-fish archive
driver has a [concrete archive proposal](proposed-archive.json), with exact
source bytes, hash and destination. It remains local pending author disposition;
no transfer, prune or deletion was executed.

The scripts catalogue now names the 18 October guide repair. The author confirmed concurrent guide/comparison edits complete. Their tools
have separate live catalogue entries; their scientific and review work remains
separate from this maintenance change. Seven bundle-relative links in the new
G manuscript were reconciled to authenticated external saved sources, with
recipe link mapping retained. The published guide and its bound manuscript
snapshot remain unchanged; see the [link reconciliation](guide-link-reconciliation.json).

The [CI workflow](../../../.github/workflows/repository-organization.yml) runs
the local structure check and repository/archive tests on pushes to `develop`,
pull requests and manual dispatch. It installs Python 3.12 and the frozen lock
environment. Actions are pinned to commits; uv is pinned to an explicit version.
It does not connect to JOAQUIM, run scientific analyses or authenticate external
payloads. The external check remains an explicit local command.

The checker now resolves only explicitly listed historical repository-root
aliases to the actual checkout before checking local links and bound hashes.
It leaves unknown absolute paths and external payloads external. Frozen files,
recorded source paths and scientific selections are not rewritten. Regression
tests verify local corruption still fails in a relocated checkout.

[Starting hashes](starting-state.json) preserve the starting source/protection
identities. Verification is recorded separately after the checks complete.

The external audit detected [four current working-copy hash mismatches](external-working-copy-mismatches.json) in the Figure 1/2 HTML/SVG pairs. All recorded states and observed files were preserved. This finding remains open for scoped figure reconciliation; it is not an archive-transaction failure or an organization-rule exception.

The [first GitHub Actions run](https://github.com/jasmc/ClassicalConditioning/actions/runs/39988200887) was created after push, but no job steps started: GitHub reports the account locked due to a billing issue. The isolated local locked-environment checks passed. The repository owner must resolve that account issue before hosted execution can be verified.
