# Scoped figure freeze index

Selection JSON files under `configs/paper-figures/selections/` are unchanged byte records. Exact locally available manifests/settings/code have been preserved under `records/frozen-analyses/`; [extraction provenance](../../../maintenance/archive/relocations.json) lists source and SHA-256. Numerical data, figures and embedded HTML archives are stored on JOAQUIM after verified transfer; historical paths remain recorded and are resolved through the archive map, not edited inside freezes.

| Scope | Selection record | Scientific definition and evidence |
|---|---|---|
| Figure 1 A–D historical | `figure1-freeze.json` | Historical assembly/panel choices; not authority over newer E/F–H |
| Figure 1 F–H V1 historical | `figure1-fgh-version1-freeze-20261009.json` | Repeated bout-log medians and quantile scale; historical fish replacement recorded |
| Figure 1 F–H corrections | `figure1-fgh-full-bout-correction-20261009.json` | Correction history; subsequently selects V12 |
| Figure 1 F–H V12 | `figure1-fgh-version12-freeze-20261009.json` | Direct half-second log medians, bin baseline [-15,0), symmetric 0.7 scaling; 183 × 103.7 mm row; embedded manifest/code extracted unchanged |
| Figure 1 E | `figure1-panel-e-heatmap-rows-freeze-20261009.json` | Raw traces and exact V12 rows; external container hash verified on newly mounted SSD; ten exact definition/code entries extracted |
| Figure 2 D/E B | `figure2-DE-B-freeze-20261009.json` | Bout-log response minus baseline; sign/Brunner–Munzel t, Holm36; 54.9 × 53.2 mm; exact manifests/settings/code retained |
| Figure 2 G | `figure2-G-logmedian-freeze-20261009.json` | Native Historical LogMedian, phase-aware LMM; 183 × 98 mm; external manifest and scientific-selection hash verified on newly mounted SSD; exact records retained |
| Figures 3–4 | Not frozen by these records | Learner/validation and paper analysis gates remain open |

Authorizations, specification version/hash, exact source/data/export hashes and detailed dimensions are in each unchanged selection and manifest. No whole-figure approval is inferred from a panel freeze. Never restyle/refreeze historical artifacts as cleanup. Current assemblies must resolve newest scoped choices explicitly.

## Organized record locations

The [records catalogue](../../../../records/frozen-analyses/README.md) groups exact records by figure, panel scope and dated version:

- [Figure 1 E](../../../../records/frozen-analyses/figure-01/panel-e/README.md)
- [Figure 1 F/G/H current and historical](../../../../records/frozen-analyses/figure-01/panels-f-g-h/README.md)
- [Figure 2 D](../../../../records/frozen-analyses/figure-02/panel-d/README.md), [E](../../../../records/frozen-analyses/figure-02/panel-e/README.md), and [shared D/E records](../../../../records/frozen-analyses/figure-02/panels-d-e/README.md)
- [Figure 2 G](../../../../records/frozen-analyses/figure-02/panel-g/README.md)

All 70 files retained their original SHA-256 during reorganization. Names and directories are navigation only; immutable freeze IDs, definitions, approvals, embedded paths and original code are unchanged. The repository relocation map resolves prior record paths.
