# Version 4: quarter-second means, non-bout NaNs, 20 s baseline

The user reverted the negative-infinity replacement and requested 0.25 s bins. Arithmetic means for bins and the baseline are retained.

- Read the unchanged hash-verified eligible framewise natural-log vigor for F=20221115_07, G=20230310_08 and H=20221115_09.
- Keep non-bout and invalid/ineligible frames as NaN. No floor or zero replacement is applied.
- Take the arithmetic mean of eligible log samples in each CS-aligned 0.25 s bin. Ignore NaNs; one finite sample is sufficient. All-NaN bins remain missing.
- Use 160 bins over [-20,20) s per trial, trials 5-94.
- The per-trial baseline is the mean of finite unscaled quarter-second bin means in [-20,0) s. Each finite bin supplies one vote; all-NaN bins are ignored.
- Subtract that baseline mean. Apply no P10/P90, scaling or numerical clipping.
- Keep shared managua_r colour limits at [-0.25,+0.25] log units, with endpoint colours for values outside the limits.
- Preserve the strong green CS boundaries, block names left of F, and G arrows at trials 9,17,63,66,93.

All 270 references are defined. Centred baseline means are zero to a maximum calculation residual below 9.5e-16 log units. Mean centring does not guarantee median zero or equal above/below-zero counts: only eight current trial baselines have equal strict sign counts. Black cells indicate no eligible bin contribution.

Bin means were independently checked against pandas grouping, and counts, all-NaN masks, baseline references and SVG cell intervals/colours are verified. PDF and HTML renders were visually inspected. The first three versions are unchanged; previous Version 4 experiments remain preserved in their review folders.
