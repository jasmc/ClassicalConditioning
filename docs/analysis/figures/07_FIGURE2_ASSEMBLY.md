# Figure 2 modular assembly review

The complete A-I main layout is configured in
`configs/paper-figures/figure2-assembly.json`. It follows the current paper
`Helpers/List of Figures.md` and `Helpers/Paper_Scaffold.md`: columns are
Delay/control, 3sTrace/control and 10sTrace/control; rows are population
heatmaps (A-C), selected-block CR ratios (D-F), and trial ratios (G-I).
10sTrace remains **inconclusive**. This is a composition review, not a
scientific freeze. Registry approvals have not been changed.

## Available-panel recovery after reviewing other chats

The initial all-placeholder layout was too restrictive: a missing SVG export
does not mean a saved plot cannot be reconstructed as vector artwork. The user
then supplied `C:/Users/joaquim/Desktop/Asset 7.svg`, a complete historical
Figure 2 A-I assembly. It differs from `J:/Asset 7.svg`, the older Figure 1 scheme
found during the first inventory. The uploaded reference is preserved byte for
byte under SSD `reference-20261008/`; its stars and inference rows are reference
artwork, not accepted model results.

Read-only retrieval from **Data processing**
(`01a0d4b6-87d9-7703-a771-e6e91c1c896a`) found the 2026-10-06 59-fish 3sTrace
E/H display versions: boxes, median/IQR, paired fish, bootstrap curves and
separate condition curves. **Compare panel layouts in Figures 1–2**
(`01a0d967-3f9c-7bc3-b97b-0451f556de3d`) identified the signed pooled heatmaps,
baseline comparisons, descriptive Delay D/G and exploratory inference variants.
These chats were read; no messages were sent to other chats.

Reconstruct the currently available provisional sources:

```powershell
.venv-trace/Scripts/python.exe scripts/populate_figure2_available.py --heatmaps
.venv-trace/Scripts/python.exe scripts/build_figure2_assembly.py
.venv-trace/Scripts/python.exe scripts/catalog_figure2_versions.py
```

The reconstruction uses authenticated existing outcomes for D/E/G/H and the
existing shared signed-bout calculation over processed frames for A/B. It does
not repeat raw tracking, recompute the activity metric, change learner labels,
or rerun inference. A/B remain provisional because the upstream vigor/alignment
audit is open. Each source, plotted-data table, dependency hash and scientific
sidecar is stored in a unique SSD `sources/<UTC stamp>/` directory. Per-fish
heatmap intermediates are on SSD `heatmap-cache/` and bind input hashes,
baseline, metric and calculation code hashes.

| Main source | Exact current recovery |
| --- | --- |
| A/D/G | Existing matched Delay cohort: 29 Delay, 28 controls; legacy distal metric. A uses [-15,0) signed bins; D/G authenticate each saved trial-summary config with [-15,0) baseline and [0,9) response. |
| B/E/H | Existing full exploratory 3sTrace cohort: 40 trace, 19 controls. B uses [-15,0) signed bins; E/H authenticate all saved trial outcomes with [-15,0) baseline and [0,13) response. Ratio contribution depends on finite positive denominators. |
| C/F/I | No authenticated current 10sTrace processed-data/cohort source was recovered; placeholders remain, with the inconclusive interpretation explicit. |

The old 3sTrace **5-9** block versions remain available separately. Main E is
explicitly rebuilt for **10-14**, then 65-69 and 90-94; this is recorded as a
block-selection revision rather than a display-only alternative. Existing
ratio eligibility is retained (minimum three trials/block for Delay; minimum
one for the earlier 3sTrace review). Historical 59-fish E/H ratios are checked
against the saved CSVs numerically before the source selection is written.

The plotted ratio is saved `response_total_activity / baseline_total_activity`;
the stored total-activity values are means over valid frames in their respective
windows. Fish/trial ratios and their medians differ from signed heatmap colours.
Main ratio sources show fish spread (IQR), not confidence intervals. Alternative
bootstrap bands show uncertainty of the median (100 resamples, seed 10).
Source records keep these semantics, cohort counts and response windows explicit.
All current sources have no significance annotations. The originals retain
their annotations only in the historical comparison gallery.

`available-versions.html` on SSD compares the reconstructed SVGs and unchanged
earlier PNGs. Original A angular-L1 baseline-review and B 53-fish/pre20 heatmaps
are labelled as historical alternatives; they do not substitute for current
legacy-metric, frozen-baseline sources. The initial placeholder versions remain
under `versions/`.

## Rebuild

From the repository root in PowerShell:

```powershell
.venv-trace/Scripts/python.exe scripts/build_figure2_assembly.py --inventory
```

Omit `--inventory` to reuse the saved inventory. The builder uses the existing
`scripts/assemble_svg_figure.py` composition engine and Inkscape for the PNG
inspection preview. SVG is the editable primary figure. The builder rejects
raster artwork in the assembly. DejaVu Sans regular and bold are copied from
the plotting runtime into the SSD font directory.

All figures, standalone panel SVGs, inventory, fonts, layout snapshots, hashes,
and previews are outside the repo:

`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/`

The stable entry point is `figure2-main.svg` with `figure2-main.png` and
`figure2-main.svg.json`. Each build has a unique UTC directory under `versions/`;
previous stable outputs are copied into the new version's `previous-main/`
before replacement. A version includes `panels/Fig2_PanelA.svg` through I,
the layout snapshot, inventory, and `build.json`. Existing scientific review
files and Figure 1 panels are never rewritten by this command.

## Inventory and initial selection (2026-10-08)

Read-only inventory covered the repo and paper repo, J: outputs, J: digested
and raw data, F: digested project figure folders, and unlabelled `J:/Asset *.svg`.
The initial sandbox access failure was resolved through filesystem approval.
No existing Figure 2 assembly or standalone Figure 2 SVG was found in these
inventoried locations. The inventory records root completeness, paths, file
hashes and SVG text excerpts. This conclusion is limited to those locations.
The J: root assets are Figure 1 preparation/protocol schemes.

All A-I panels initially remained explicit placeholders because no reliable,
current vector panel source was available. The inventory retains the historical
PNG/data/sidecar candidates; a placeholder is not fabricated or zero-valued data.

| Panels | Pending source and important alternate-version constraints |
| --- | --- |
| A | Matched Delay population SVG using the frozen baseline and paper metric. Upstream vigor and heatmap alignment remain under audit. |
| B | Matched 3sTrace population SVG. Historical signed review used [-20,0) s; 53-fish and 59-fish reviews remain distinct. |
| C | Authenticated 10sTrace population data and matched cohort; retain inconclusive interpretation. |
| D | Verified [-15,0) ratio inputs and current block SVG. The `baseline-window-review/figure2-pre15` outputs use angular L1, not the frozen paper metric; its filename does not verify ratio denominators. |
| E | Matched 3sTrace block SVG and frozen-baseline ratio inputs. The saved 20261006 descriptive review explicitly uses Pre-Train 5-9; the current plan requires 10-14. |
| F | Reviewed 10sTrace matched cohort, ratio inputs and selected blocks; inconclusive. |
| G | Verified Delay frozen-baseline ratio inputs and descriptive SVG. Historical LME influence failed and onset was not localized. |
| H | Matched 3sTrace trial SVG and verified frozen-baseline denominators; do not combine 53-fish/59-fish cohorts. |
| I | Reviewed 10sTrace matched cohort and authenticated trial-ratio data; inconclusive. |

During the initial assembly, no new scientific panels were calculated and no significance marks were added.
Historical review files remain in their original locations and are indexed,
rather than copied into current panel sources. Old labels do not determine
the new panel assignment; current panel roles and scientific provenance do.

## Frozen and pending definitions

The user-frozen pre-CS baseline is **[-15,0) s relative to CS onset**.
This overrides older specification/renderer defaults for newly produced
Figure 2 baseline-dependent panels. Historical [-20,0) panels are not corrected
by changing their labels or by re-exporting their PNGs.

The paper metric is tail bend angular speed (`legacy_distal_angular_speed`,
rad/ms). The layout requires selected blocks 10-14, 65-69 and 90-94 inclusive.
Response-window proposals are 0-9 s Delay, 0-13 s 3sTrace and 0-20 s 10sTrace;
they remain proposals in this manifest. Ratio formula, averaging versus totals,
movement masking, heatmap normalization, matched cohorts, response windows,
protocol alignment and inference still require scientific review. Frozen
baseline selection does not resolve the upstream vigor/alignment audit.
Figure 1 A-D remain frozen; Figure 1 E is not authenticated by this assembly.

## Replace one panel

Store the revised SVG, plotted data and scientific sidecar on SSD. Update only
that panel's `source`, `selection_status`, and `source_provenance` in the layout.
Clear its pending inputs only when the relevant decisions are documented.
The builder requires this provenance contract for each populated source:

```json
{
  "baseline_s": [-15, 0],
  "baseline_interval": "[-15,0)",
  "metric_id": "legacy_distal_angular_speed",
  "svg_sha256": "<actual source hash>",
  "cohort_hash": "<cohort identity>",
  "panel_data": "sources/<panel>/panel-data.parquet",
  "panel_data_sha256": "<actual plotted data hash>",
  "sidecar": "sources/<panel>/panel.figure.json",
  "sidecar_sha256": "<actual scientific sidecar hash>",
  "significance_marks": "none"
}
```

D-F additionally require `block_trials: [[10,14],[65,69],[90,94]]`.
Paths resolve against the SSD storage root; absolute SSD paths are also accepted.
These checks bind the declared provenance to files; they do not independently
prove a scientific method correct or confer approval. Baseline evidence must
come from the generating recipe and saved panel data, not from a title.
Rebuild with the command above to preserve versions and rerun vector checks.
The generic assembler's `--replace-panel-source` is useful for exploratory
layout checks but bypasses these Figure 2-specific provenance checks, so it
must not publish the stable main figure.
