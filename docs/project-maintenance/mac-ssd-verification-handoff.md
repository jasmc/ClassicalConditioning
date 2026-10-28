# Mac verification handoff: organized JOAQUIM SSD

Continue the Windows implementation on the Mac. The author authorized its commit
and push to `origin/develop` (`jasmc/ClassicalConditioning`), then requested Mac
verification in a separate chat. Use the commit containing this handoff; verify
the actual fetched commit and remote before updating a checkout.

Windows Python 3.13.14 verification completed: 208 tests run, 205 passed and
three symlink cases skipped because the account lacked creation privilege.
Eleven additional metric command/relocation tests passed, for 216 passes and
three skips across the final suites. Disposable Git checkouts exercised 452 protected paths under Windows and Unix
line-ending settings. The SSD audit passed: 365 protected files, 10 registered
figure states, 14 archive transactions and 2,214 payload files; ten absences
matched exact, previously authorized disposition records. These are Windows
results. Native Mac execution and shared-filesystem locking remain pending.

Read this repository's `AGENTS.md` and, for figure-state verification, the
paper-figures skill and authoritative workflow. The
[source reconciliation](../../records/repository-history/2026-10-28-ssd-portability/source-reconciliation.json)
authenticates the captured 848 PC and 1,351 Mac files, preserved Git/source
snapshots, consolidation consumers and recovered implementation. Any later live
Mac edits have not been inspected: SSH to `joaquim@172.27.1.111` timed out from
the PC. The PC implementation restored missing Mac safeguards/tests while
retaining PC additions; do not replace it wholesale with a historical snapshot.

## Preserve and reconcile the live Mac checkout

Find the actual checkout. Previously recorded candidates are
`~/Documents/ClassicalConditioning` and
`~/Documents/Codex/2026-09-15/clo/ClassicalConditioning`; neither was reachable
for this audit. Inspect its status, branch, HEAD, remotes, untracked files and
local commits before changing it. Preserve independent edits and compare them
with the pushed implementation. A newer filename or timestamp does not select
scientific authority.

If the checkout is clean and compatible, update by an ordinary fast-forward.
Otherwise preserve its working tree and use a disposable clean checkout of the
pushed commit for native tests while reconciling the independent changes. Do
not reset, clean, stash away or overwrite unreviewed Mac work. Do not copy the
Windows virtual environment to the Mac.

## Run native verification

Use Python 3.13 for parity. If an environment needs preparation, run
`uv sync --locked --python 3.13 --extra analysis` in the verification checkout.
Confirm `.venv/bin/python --version`. Run from that checkout's root:

```sh
.venv/bin/python -m unittest tests.test_external_volume_portability tests.test_repository_archive tests.test_repository_structure tests.test_study_sources tests.test_storage_migration tests.test_figure_workspace tests.test_panel_g_review_repairs tests.test_main_paper_figures tests.test_command_organization tests.test_figure_version_browser tests.test_review_library tests.test_svg_figure_assembly tests.test_assembly_integrity tests.test_figure1_panel_b tests.test_figure1_panel_c tests.test_aligned_example_pair tests.test_individual_atlas tests.test_metric_comparison tests.test_metric_comparison_figure -q
```

Report each skip and its reason. The detected Windows drive-letter case is
Windows-only; the three symlink cases must actually run where supported. The
suite includes both Git checkout settings, source-index preservation,
subprocess lock contention and release, resolver failures, study ownership and
figure publication safeguards. Inspect any genuine Mac path or locking failure.

When the actual SSD is mounted, use its existing root:

```sh
export CLASSICAL_CONDITIONING_JOAQUIM_ROOT=/Volumes/JOAQUIM
.venv/bin/python scripts/check_repository_structure.py --external
.venv/bin/python scripts/figure_working_copy.py verify
.venv/bin/python scripts/main_paper_figures.py display-target
```

The collection must resolve ordered `fig1`/`fig2` and HTML SHA-256
`099740a392a20f0a9b22c5059d709de9989c2a530bae9a86cba4e594ae5ea724`.
Its final saved wrapper was recovered through the completion receipt, original
rendering functions and exact selected SVG/legend identities. Keep it intact;
these verification steps require no build, scientific recomputation, new freeze,
SSD reorganization or regeneration. If the SSD is absent, continue synthetic
tests and record external acceptance as pending. Never create a fake mount.

## Close the remaining acceptance items

Record the fetched commit, native macOS/Python/Git versions, checkout path,
test totals/skips, actual mount and external audit outcome in the existing
[verification record](../../records/figure-reviews/authority-reconciliation-20261025/ssd-portability-reconciliation-20261028.json).
Record the disposition of later independent Mac edits. Keep scoped scientific
selection hashes and freeze status unchanged unless separately authorized.

Local process locking is implemented on both OS families. Claim simultaneous
cross-machine exclusion only after both hosts access the same actual filesystem
through the intended arrangement and the existing independent run lock is
tested concurrently in a new isolated temporary directory. Test contention in
both directions, non-blocking rejection and release after an error. Separate
local copies or moving the SSD between machines cannot establish this result.
Leave this item explicitly pending if that shared access is unavailable.

The [storage guide](storage.md#host-configuration-and-verification) owns maintained
configuration and limitations; the
[reconciliation index](../../records/repository-history/2026-10-28-ssd-portability/README.md)
links the preservation evidence. Keep this handoff operational and compact;
study narratives and permanent payloads stay on JOAQUIM.
