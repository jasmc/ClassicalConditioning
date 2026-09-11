# Historical source archive

This directory preserves retired implementation history for review and
reproducibility. It is not an installable Python package, has no supported CLI
entry points, and is excluded from normal test discovery.

| Directory | Contents |
| --- | --- |
| `package/` | Retired package-route source and its legacy-only tests. |
| `modules/` | Original helper modules retained as historical references. |
| `historical-scripts/` | Numbered end-to-end analysis and learner scripts. |
| `historical-helpers/` | Root-level helper modules required by the numbered scripts. |
| `tools/` | Historical inspection utilities. |

The active, supported implementation is under `../src/classical_conditioning/`.
Do not treat archived source as a runnable workflow without an explicit,
separately maintained compatibility environment.
