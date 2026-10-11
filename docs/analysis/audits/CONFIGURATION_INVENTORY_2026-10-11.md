# Configuration assessment — 2026-10-11

**Implemented outcome:** see the [cleanup record](../../maintenance/archive/configuration-cleanup-20261011/README.md). The findings below document the initial assessment; the registry now has 27 panels, misplaced records have moved, V5/cohort payloads are verified externally, and all eight listed expected-hash variants resolve exactly. The original machine inventory remains the baseline evidence.

Assessed all 21 existing files recursively: 20 valid JSON files and one valid BOM-aware CSV. Three run configurations pass the current strict loader. No configuration, selection record, metric, approval or scientific definition was changed; this is an assessment and navigation guide.

## Findings and useful boundaries

- True settings specify run paths/metric choices, artwork assembly, schematic parameters and the explicit freeze validation contract.
- Four root files are records/payloads rather than settings: the preflight report, increasing-trace protocol audit, past raw-file move manifest and technical cohort CSV. They retain useful timing, classification and population provenance; misplaced does not mean disposable.
- `figure1-cadence-review-v5.json` is a superseded dated comparison. Its fixed builder is already archived; no active package/test/builder consumes this particular configuration. Archive it with its version evidence rather than silently forgetting the recipe.
- `behavior-paper.json` is active code input but scientifically out of date as a panel map. It has 26 entries and Figure 1 A-G, rather than the current 27 main panels with Figure 1 A-H. Its C/D/E/F/G roles use old lettering; D/E/G Figure 2 descriptions still name earlier CR ratios. `_paper_panel_statuses()` also expects the old set. Coordinate registry, readers and tests together. A blocked paper-release gate is distinct from an accepted scoped panel freeze; do not mark release readiness completed solely because a panel was frozen.
- Figure 1/2 assembly JSONs remain used layout scaffolds. Their newer E/F-G-H and D/E/G selections are supplied by scoped overrides in `update_latest_figure_assemblies.py`; direct older assembly recipes do not by themselves reproduce every current choice.
- `review-variants.json` is a useful historical catalogue of 25 recipes. Some renderer paths now resolve externally and two September 24 tags still say `selected_review_method`. Keep those histories separately from current scientific authority.
- Seven selection JSONs preserve current and historical accepted scope. Old accepted freezes are not stale garbage. Their bytes must remain immutable.

## Hash-reference audit

Eight `{path, sha256}` references failed against the copies reached by the current resolver: one V12-selection back-reference, five D/E B references (statistics, two manifests, two SVGs), and two G catalogue/cleanup references. These are **pre-existing version/location or byte-convention discrepancies**, not archive copy corruption: all actual bytes match the starting inventory and/or the immutable transfer checkpoint. Seven exact expected copies were found and SHA-256 verified under the prior `ClassicalConditioning-Mac-20261009/ClassicalConditioning` SSD handoff. The statistics CSV’s expected hash matches its CRLF representation; the archived copy uses LF. Original CSV rows/calculations were not changed or regenerated.

See [exact discrepancy and recovery paths](../../maintenance/archive/configuration-pointer-discrepancies-20261011.json). Matching files were located, but resolver precedence and expected-version location records still need a coordinated repair. Until then, strict readers correctly reject these mismatched references. Preserve both variants and update external hash-addressed location mappings; do not rewrite frozen hashes to accept different bytes.

The current recursive hash-dictionary check does not claim complete validation of every embedded archive entry, scalar historical path/hash pair, raw dataset or full scientific freeze. Earlier archive verification remains separate.

## Per-file inventory

| File | What it is / why useful | Disposition |
|---|---|---|
| [all3sTrace-full-windows.json](../../../configs/all3sTrace-full-windows.json) | Technical fixed-trace run: 59 complete fish, legacy distal angular speed | Retain; recording IDs agree with the saved preflight |
| [allDelay-full-v1-cohort.csv](technical-cohort-column-schema.md) | 57 fish (29 Delay, 28 control), technically approved by Codex-technical-all-complete | Archive full numerical/cohort payload with provenance; retain column-schema documentation; not final paper selection |
| [allDelay-full-windows.json](../../../configs/allDelay-full-windows.json) | Technical allDelay Windows run and resume recipe | Retain; machine-specific paths; not a paper cohort approval |
| [example-run.json](../../../configs/example-run.json) | Strict run-pipeline example with placeholder Windows paths | Retain; edit paths in a copy before use |
| [incTrace-stim-audit.json](inc-trace-stim-audit.json) | Hash-bound protocol audit classifying 14 increasing-trace recordings | Relocate to docs/analysis/audits; update audit writer default; retain unique evidence |
| [increasing-trace-move-manifest.json](../../maintenance/transfers/increasing-trace-move-manifest.json) | September 24 movement/renaming provenance for 14 recordings / 42 raw files | Relocate to docs/maintenance/transfers; retain raw identity/recovery evidence |
| [paper-figures/behavior-paper.json](../../../configs/paper-figures/behavior-paper.json) | Proposed panel readiness/artifact registry read by pipeline and paper-panel command | Revise together with readers/tests: 26 entries and old Fig1 labels / ratios; preserve separate freeze versus release readiness |
| [paper-figures/figure-elements.json](../../../configs/paper-figures/figure-elements.json) | Versioned semantic roles, styles, exceptions and freeze validation contract | Retain; explicit freeze gate and preparers consume it |
| [paper-figures/figure1-assembly.json](../../../configs/paper-figures/figure1-assembly.json) | Historical A-D selections and older E/F/G/H layout sources | Retain pending coordinated assembly update; scoped selections override its older panel choices |
| [paper-figures/figure1-cadence-review-v5.json](../../maintenance/archive/configuration-cleanup-20261011/transfer-inventory.json) | October 7 presumed-acquisition-cadence comparison layout | Archive with its named version/provenance; no active package/test/fixed builder references remain |
| [paper-figures/figure1-session-protocol.json](../../../configs/paper-figures/figure1-session-protocol.json) | Figure 1C session phases, CS/US timing and stimulus counts | Retain; builder/test consume it; marked exploratory, not an approval record |
| [paper-figures/figure2-assembly.json](../../../configs/paper-figures/figure2-assembly.json) | Nine-panel layout with older D/E/G sources and C/F/I placeholders | Retain pending coordinated assembly update; current selected D/E B and G overlay it |
| [paper-figures/review-variants.json](../figures/reviews/review-variants.json) | 25 alternative version recipes, renderers and old panel-letter mappings | Keep scientific/version history; move to docs or explicitly label as historical; old selected_review_method tags are not current freeze authority |
| [paper-figures/selections/figure1-fgh-full-bout-correction-20261009.json](../../../configs/paper-figures/selections/figure1-fgh-full-bout-correction-20261009.json) | Scoped accepted choice or accepted historical freeze; ties panels to exact manifests/exports/definitions | Preserve bytes; resolve path/hash inconsistencies externally, do not delete or rewrite as cleanup |
| [paper-figures/selections/figure1-fgh-version1-freeze-20261009.json](../../../configs/paper-figures/selections/figure1-fgh-version1-freeze-20261009.json) | Scoped accepted choice or accepted historical freeze; ties panels to exact manifests/exports/definitions | Preserve bytes; resolve path/hash inconsistencies externally, do not delete or rewrite as cleanup |
| [paper-figures/selections/figure1-fgh-version12-freeze-20261009.json](../../../configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json) | Scoped accepted choice or accepted historical freeze; ties panels to exact manifests/exports/definitions | Preserve bytes; resolve path/hash inconsistencies externally, do not delete or rewrite as cleanup |
| [paper-figures/selections/figure1-freeze.json](../../../configs/paper-figures/selections/figure1-freeze.json) | Scoped accepted choice or accepted historical freeze; ties panels to exact manifests/exports/definitions | Preserve bytes; resolve path/hash inconsistencies externally, do not delete or rewrite as cleanup |
| [paper-figures/selections/figure1-panel-e-heatmap-rows-freeze-20261009.json](../../../configs/paper-figures/selections/figure1-panel-e-heatmap-rows-freeze-20261009.json) | Scoped accepted choice or accepted historical freeze; ties panels to exact manifests/exports/definitions | Preserve bytes; resolve path/hash inconsistencies externally, do not delete or rewrite as cleanup |
| [paper-figures/selections/figure2-DE-B-freeze-20261009.json](../../../configs/paper-figures/selections/figure2-DE-B-freeze-20261009.json) | Scoped accepted choice or accepted historical freeze; ties panels to exact manifests/exports/definitions | Preserve bytes; resolve path/hash inconsistencies externally, do not delete or rewrite as cleanup |
| [paper-figures/selections/figure2-G-logmedian-freeze-20261009.json](../../../configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json) | Scoped accepted choice or accepted historical freeze; ties panels to exact manifests/exports/definitions | Preserve bytes; resolve path/hash inconsistencies externally, do not delete or rewrite as cleanup |
| [trace-preflight.json](trace-preflight.json) | September 25 generated raw-protocol preflight; 40 trace + 19 control selected, four incomplete records | Relocate to docs/analysis/audits; update generator/test/document references first; retain timing evidence |

## Checks and next work

The fixed-trace run and saved preflight agree on all 59 recording IDs. The preflight is dated September 25 and records four incomplete groups; its timing evidence remains useful but it is not a fresh scan of today’s raw files. The Delay CSV contains 57 unique fish (29 Delay / 28 control), labelled technical all-complete approval rather than author-reviewed publication membership. Windows paths are expected in the Windows launchers; on macOS use an appropriate path adapter. The example’s D:/E: paths are intentional placeholders, not missing production data.

Schema checks establish parseability and accepted options, not successful execution or scientific approval. Targeted configuration, trace-preparation and session-schematic tests are recorded in the machine-readable inventory after completion. No data processing, rendering of frozen figures, refreezing, deletions or configuration relocation was performed.

Final checks: all 10 targeted tests passed; all 21 original configuration files retain their pre-assessment SHA-256. New guide/report links resolve.
