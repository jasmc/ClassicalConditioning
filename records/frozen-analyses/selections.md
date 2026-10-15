# Scoped figure freeze index

The accepted frozen versions define current. Start with [Current figures](../../docs/paper/figures/README.md) for the readable summary; this index also retains superseded accepted freezes as history.

Selection JSON files under `configs/paper-figures/selections/` are unchanged byte records. Exact locally available manifests/settings/code have been preserved under `records/frozen-analyses/`; [extraction provenance](../repository-archive/relocations.json) lists source and SHA-256. Numerical data, figures and embedded HTML archives are stored on JOAQUIM after verified transfer; historical paths remain recorded and are resolved through the archive map, not edited inside freezes.

| Scope | Selection record | Scientific definition and evidence |
|---|---|---|
| Figure 1 A–D current retained | [figure1-freeze.json](../../configs/paper-figures/selections/figure1-freeze.json) | Accepted A–D source panels; no later replacement is recorded for those scopes. Older E/F–H references inside this record do not override newer selections. |
| Figure 1 F–H V1 historical | [figure1-fgh-version1-freeze-20261009.json](../../configs/paper-figures/selections/history/figure1-fgh-version1-freeze-20261009.json) | Repeated bout-log medians and quantile scale; historical fish replacement recorded |
| Figure 1 F–H corrections | [figure1-fgh-full-bout-correction-20261009.json](../../configs/paper-figures/selections/figure1-fgh-full-bout-correction-20261009.json) | Correction history; subsequently selects V12 |
| Figure 1 F–H V12 | [figure1-fgh-version12-freeze-20261009.json](../../configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json) | Direct half-second log medians, bin baseline [-15,0), symmetric 0.7 scaling; 183 × 103.7 mm row; embedded manifest/code extracted unchanged |
| Figure 1 E historical | [figure1-panel-e-heatmap-rows-freeze-20261009.json](../../configs/paper-figures/selections/figure1-panel-e-heatmap-rows-freeze-20261009.json) | Raw traces and exact V12 rows; external container hash verified on newly mounted SSD; ten exact definition/code entries extracted |
| Figure 2 D/E B | [figure2-DE-B-freeze-20261009.json](../../configs/paper-figures/selections/figure2-DE-B-freeze-20261009.json) | Bout-log response minus baseline; sign/Brunner–Munzel t, Holm36; 54.9 × 53.2 mm; exact manifests/settings/code retained |
| Figure 2 G | [figure2-G-logmedian-freeze-20261009.json](../../configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json) | Native Historical LogMedian, phase-aware LMM; 183 × 98 mm; external manifest and scientific-selection hash verified on newly mounted SSD; exact records retained |
| Figures 3–4 | Not frozen by these records | Learner/validation and paper analysis gates remain open |

Authorizations, specification version/hash, exact source/data/export hashes and detailed dimensions are in each unchanged selection and manifest. No whole-figure approval is inferred from a panel freeze. Never restyle/refreeze historical artifacts as cleanup. Current assemblies must resolve newest scoped choices explicitly.

## Organized record locations

The [records catalogue](README.md) groups exact records by figure, panel scope and dated version:

- [Figure 1 E](figure-01/panel-e/README.md)
- [Figure 1 F/G/H current and historical](figure-01/panels-f-g-h/README.md)
- [Figure 2 D](figure-02/panel-d/README.md), [E](figure-02/panel-e/README.md), and [shared D/E records](figure-02/panels-d-e/README.md)
- [Figure 2 G](figure-02/panel-g/README.md)

All 70 files retained their original SHA-256 during reorganization. Names and directories are navigation only; immutable freeze IDs, definitions, approvals, embedded paths and original code are unchanged. The repository relocation map resolves prior record paths.

Current F/G/H selection: [1 s direct log-frame medians, managua_r](../../configs/paper-figures/selections/figure1-fgh-historical-one-second-managua-freeze-20261013.json). The 2026-10-09 V12 source is superseded within F/G/H and remains preserved; no whole-figure freeze is inferred.

Current E selection: [unbinned raw-baseline-scaled black traces, 0–5, exact current F bands](../../configs/paper-figures/selections/figure1-panel-e-raw-frame-freeze-20261013.json). The earlier V12-row E freeze remains historical; E only is frozen, not the whole assembly.

Earlier single-fish heatmap reviews, input cache and candidate code were archived on 2026-10-13. [Archive and recovery record](/Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/records/repository-archive/singlefish-heatmaps-20261013/README.md). Current Figure 1 and its selected F/G/H freeze remain active.

Current Figure 2 A: [pooled one-second direct log-frame medians, compact six-heatmap row](../../configs/paper-figures/selections/figure2-A-pooled-freeze-20261014.json). A only; previous wider A freeze remains preserved. B retains earlier data; B/C pipeline adoption remains pending.
