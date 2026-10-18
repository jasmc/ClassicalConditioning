# Configuration guide

This folder contains supported settings and immutable scoped selection pointers. The [configuration cleanup record](../records/repository-archive/configuration-cleanup-20261011/README.md) records relocations, verified external copies and hash repairs.

| Files | Purpose and authority |
|---|---|
| `example-run.json` | Strict pipeline example; edit placeholder paths in a copy |
| `allDelay-full-windows.json`, `all3sTrace-full-windows.json` | Supported Windows technical run settings; these do not approve paper cohorts |
| `paper-figures/figure-elements.json` | Versioned semantic role/style contract for explicit new freezes |
| `paper-figures/figure1-session-protocol.json` | Figure 1C session schematic settings; exploratory status retained |
| `paper-figures/figure1-assembly.json`, `figure2-assembly.json` | Historical base layouts with explicit newer scoped overlays; use `scripts/update_latest_figure_assemblies.py` for current assemblies |
| `paper-figures/behavior-paper.json` | Current 27-panel proposed registry for Figures 1–4; scoped freeze status is separate from first-release readiness |
| `paper-figures/selections/` | Seven byte-preserved current/historical pointers; see the [freeze index](../records/frozen-analyses/selections.md) for scope |

Preflight and protocol evidence live in [audits](../records/scientific-evidence/audits); past transfer records live in [transfers](../records/repository-history/transfers). The [historical review catalogue](../records/figure-history/shared/review-variants.json) preserves its original version labels and recipes; its old selected-method tags do not override current scoped freezes.

The superseded V5 candidate and full technical cohort CSV are indexed on JOAQUIM. Read them through `resolve_artifact()` using their historical paths. Eight pre-existing hash discrepancies now resolve through [verified expected-version locations](../records/repository-archive/verified-artifact-locations.json); both versions remain preserved. Missing or corrupt selected bytes fail explicitly. Permanent payloads are external; settings and scientific definitions remain local.
