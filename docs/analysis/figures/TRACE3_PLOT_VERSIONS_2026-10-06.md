# 3sTrace Figure 2E/H and Figure 4 plotting versions

These exploratory plotting alternatives use the existing 59-fish cohort
`all3sTrace-full-exploratory` (40 trace, 19 controls),
`legacy_distal_angular_speed`, and corrected 0–13 s response outcomes.
Figure 4 reuses the authenticated analysis
`figure4-3strace-window13-legacy-59fish` and its `legacy-wip` classifier:
16 trace learners, 24 trace nonlearners, one flagged control, 17 control
nonlearners and one unclassified control. No classifier or raw processing
is rerun to compare layouts.

Open `outputs/trace-plot-versions/20261006/comparison.html` to compare the PNGs.
`plot-versions.json` records input and output hashes. New rendered alternatives
have adjacent `.figure.json` files. The existing stacked Figure 4 is copied
for comparison; its original sidecar remains in the source analysis folder.

| Panel | Version | Meaning |
| --- | --- | --- |
| 2E | v1 legacy grouped boxes | Fish medians per five-trial block; box IQR and whiskers. |
| 2E | v2 descriptive median/IQR | Fish points and condition median/IQR, adapted from `fig2-dg-descriptive`. |
| 2E | v3 paired fish | One line per fish, black condition median; adapted from the block layout of `fig2-dg-legacy-stars-and-lme`. |
| 2H | v1 legacy bootstrap | Condition median with 95% bootstrap CI (100 resamples, seed 10). |
| 2H | v2 descriptive median/IQR | Condition median with fish IQR, adapted from `fig2-dg-descriptive`. |
| 2H | v3 separate conditions | The same bootstrap summary as v1, displayed on separate axes. |
| 4 | v1 stacked | Existing ten-row signed median/IQR profiles. |
| 4 | v2 compact grid | The same ten groups in a 5×2 grid. |
| 4 | v3 separate conditions | Separate 5×2 grids for trace and controls, retaining the common y scale. |

Figure 4 layouts have separate stable entries in the
[variant registry](../../../configs/paper-figures/review-variants.json):

- `fig4b-3strace-legacy-59fish-v1-stacked`
- `fig4b-3strace-legacy-59fish-v2-compact-grid`
- `fig4b-3strace-legacy-59fish-v3-separate-conditions`

Each entry links its exact PNG output(s), provenance, shared manifest,
analysis, cohort, metric and classifier execution. Version v3 comprises
both trace and control files. All are exploratory; no layout is frozen
as the manuscript choice.

The 3sTrace pre-training block is PTr trials 5–9, followed by ETe 65–69
and LTe 90–94, matching the archived 9-block protocol. Figure 2H is not
smoothed. IQR represents fish spread; bootstrap CI represents uncertainty
in the median, so those bands have different interpretations.

The v3 Figure 2E adaptation recomputes the legacy test families on current
fish ratios: three between-condition Mann–Whitney tests and four paired
within-condition Wilcoxon tests. Holm correction is applied separately to
the between and within families. Finite pairs and their sample counts are
recorded in `figure-2E_legacy-holm-tests.csv`; plots show the paired p-values.
No saved Delay LME marks or learning-onset claims are transferred. The
permutation and LME analysis versions in the historical guide require their
own 3sTrace analyses and are not labeled as completed plotting alternatives.

Figure 4 grid layouts are new display alternatives, not historical variants.
They preserve the stored group medians, IQR, missing values, classifier
labels, CS interval and protocol-verified expected US at approximately 13 s.
The blue flagged-control curve contains one fish and has no fish-IQR band.

Reproduce with:

```powershell
.venv-trace\Scripts\python.exe scripts\render_3strace_plot_versions.py --project-dir 'F:\Digested Data\all3sTrace-full-v1' --output-dir outputs\trace-plot-versions\20261006
```
