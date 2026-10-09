# F/G/H colour and binning review, 9 October 2026

Open `index.html` for a portable nine-row comparison. All PNG, SVG and PDF
downloads are embedded. The original four-row summary and selected figure
records are preserved.

Current candidates:

- Version 5: frame-recomputed 1 s arithmetic-mean bins, matching equal-bin
  baseline mean in [-20,0) s, linear managua_r at +/-0.25 log units.
- Version 4 contrast and Version 5 contrast: signed square-root colour mapping
  only, with original numeric values and fixed endpoint saturation.
- Version 6: 0.25 s bin means; each trial's finite pre-CS baseline bins define
  P25/P45/P50/P55/P75 with linear interpolation. Subtract P50. Boundaries are
  P25/P45/P55/P75, making a small P45-P55 band containing median-centred zero.
  Five actual managua_r colours at 0/.25/.5/.75/1; centre is the dark purple
  midpoint, distinct from missing black. Scores [-1,-.5,0,.5,1] are ordinal
  display classes rather than normalized physical amplitudes.

Exact threshold ties enter the upper class. Missing values never enter a band.
Median-centred log values and all original Version 4 columns are exported.
The earlier four-colour Version 6 remains as superseded history.

Scripts use the repository `.venv-trace/Scripts/python.exe`:

1. `../fgh_version5_onesecond_means_20261009/build_version5.py`
2. `build_candidates.py`
3. `build_version6_centralband.py`
4. `build_summary.py`
5. `node check_summary.cjs`

`render_candidates_fourband.py` records the source revision used for the initial
four-colour candidate and continuous exports. `render_candidates.py` supports
the five-band candidate. Validation records bind renderer revisions, source
tables, source manifests and outputs to SHA-256 hashes.

Numeric checks independently verify frame means, exported equal-bin baselines,
mean/median centring, sample-count-weighted reconstruction, missing masks,
per-trial class intervals and preservation of source CSV fields. SVG checks
verify every finite cell's position, width, row and colour, CS boundaries, phase
labels and G arrows. PDF readback PNGs are under `pdf_readback/`.

Preservation inventory covers 187 pre-existing files. All figure exports,
data tables, freeze records and scoped selections retain their hashes. One
shared style configuration (`configs/paper-figures/figure-elements.json`)
changed during this run; none of these scripts writes it. Its before/after
hashes are recorded as a concurrent workspace change and it was not restored.
No panel E or frozen panel was modified. These are candidates, not freezes.
