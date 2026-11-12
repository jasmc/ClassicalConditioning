# Tests and separate verification routes

Organize tests around observable behavior and preserved contracts. Keep
characterization oracles independent of the implementation they check. Use
temporary fixtures and explicit external-test prerequisites; restore modified
import paths, environment and module state. Share test-only setup without
hiding scientific assumptions or importing transaction drivers for convenience.
Document checks outside normal unittest discovery and report their results
separately. Retire a test only after identifying its replacement coverage or
the explicitly retired behavior.

## Local Python checks

Run from the repository root with the declared environment, for example:

```sh
.venv/bin/python -m unittest tests.test_repository_organization tests.test_repository_structure
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Choose focused tests for changed behavior before the full suite. External tests
require the real mounted JOAQUIM root and authenticated inputs. Never create a
fake mount or turn a missing external check into a pass. Current commands and
preservation checks are owned by the [rules map](../docs/project-maintenance/rules.md).

## Browser checks outside unittest discovery

[Physical stroke calibration](check_non_scaling_strokes.cjs) is a standalone
Playwright regression. It receives repository root, browser executable and a new
external JSON evidence path as positional arguments. It uses a disposable local
page served at `http://127.0.0.1:8766/`; no selected artwork is needed. Start a
server from an empty temporary fixture directory, then run in a second terminal:

```sh
.venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory /absolute/temporary-fixture-directory
```

```sh
node tests/check_non_scaling_strokes.cjs /absolute/repository /absolute/browser-executable /absolute/external-evidence/stroke-calibration.json
```

Node must resolve Playwright (set `NODE_PATH` to the installed dependency root
when needed); the browser executable and evidence parent must exist. Use a new
evidence destination and stop the fixture server afterward.
[Renderer cases](layout_frame_browser_cases.py) supply deterministic SVG cases
for a separate collector review; importing them is not a browser test.
Python/unit, actual browser, native platform and exact-source recovery results
must be recorded separately. Do not claim browser/print QA from this guide.

Figure 2 command/presentation regressions live under
`tests/commands/paper/figure_02/`; package markers preserve recursive discovery.
For example, run `python -m unittest tests.commands.paper.figure_02.test_figure2_phase_guides`.
The repository-retention regression checks literal executable dependencies and
the compatibility contracts of extracted analysis helpers.
