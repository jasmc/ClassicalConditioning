# Repository maintenance guide

This guide separates supported code, historical reference, local tooling, and
generated files. It does not authorize deletion of raw scientific data.

| Path | Role | Safe maintenance action |
| --- | --- | --- |
| `src/` | Supported installable package | Review, test, and change through normal development workflow. |
| `tests/` | Active automated tests | Keep aligned with supported source. |
| `configs/` | Example/reusable run configurations | Copy before editing for a new run. |
| `docs/analysis/` | Current workflow, proposed architecture, audit, and reference docs | Keep links current when code/documentation changes. |
| `Plans/` | Migration plans, decisions, and status records | Preserve decision/history context. |
| `Archive/` | Historical packages, scripts, helpers, tests, and references | Treat as non-runnable source history; do not import into active workflow. |
| `.venv/` | Local dependencies | Safe to recreate through the documented environment setup. |
| `__pycache__/`, `*.egg-info/`, `.pytest_cache/` | Generated local state | Disposable; regenerates. |
| `.git/` | Repository history/object database | Never manually delete or edit. |

## Raw data and generated analysis projects

The normal raw data and `save_dir` output projects are configured outside this
repository. Raw data are immutable input. Generated output belongs in a
dedicated save directory; see [output and provenance](../analysis/OUTPUT_AND_PROVENANCE.md).
Do not store a working `Paper data` project tree inside the repository unless
that is an intentional, separately managed data release.

## Documentation maintenance

The root README is the landing page. Specialist guidance belongs below `docs/`:

- workflow, output, figures, troubleshooting, and terminology in `docs/analysis/`;
- decisions and implementation status in `Plans/`;
- historical explanation in `Archive/` and the marked legacy documentation.

When moving documentation, preserve its original content first, add a redirect
link, update the README TOC and `docs/analysis/README.md`, and verify Markdown
links. This prevents navigation improvements from erasing provenance.

## Existing cleanup notes retained from the root README

Generated `__pycache__/` folders and `src/classical_conditioning.egg-info/`
are disposable. `push.log` is an ignored local log. Archived numbered scripts
and helper modules are historical source reference; they are not part of the
supported package workflow.

Earlier cleanup notes named `README2.md` and
`MY___PLANS. we need to add some kinda loading ba`. Neither file is present in
this checkout; no current maintenance action is needed for them.
