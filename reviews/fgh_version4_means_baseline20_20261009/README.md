# Version 4: arithmetic means, non-bout lower bound, 20 s baseline

Latest user instruction: replace the medians in Version 4 with means. Both the 0.5 s temporal summary and the trial baseline summary use arithmetic means. The previously selected absolute non-bout lower bound (-infinity) and whole-trial undefined-reference rule are retained.

- Direct eligible bout samples keep their natural-log raw vigor.
- Valid non-bout frames contribute -infinity before binning.
- Invalid tracking and other ineligible contributions remain NaN and are ignored. They are not replaced by non-bout values.
- CS-aligned half-second bins use the arithmetic mean of all non-NaN input values, including -infinity. An all-NaN bin would remain missing.
- Each trial's baseline is the arithmetic mean of non-NaN bin means in [-20, 0) s, including -infinity. Each bin has one vote.
- Subtract that reference only if it is finite. Otherwise leave the entire trial undefined.
- No P10/P90, percentile scaling, or numerical clipping is applied. Shared managua_r display limits remain -0.25 to +0.25 log units.
- Display trials 5-94 over [-20, 20) s, using the retained legacy CS boundaries, left block names and G arrowheads.

All 270 trials have a baseline mean of -infinity and are undefined. A single -infinity contribution makes an arithmetic mean -infinity. Consequently no above/below-zero display count is available in the current version. Mean centring, when defined, guarantees zero mean rather than equal counts above and below zero.

The initial eligible-only median Version 4 balance audit found exactly equal above/below counts in all 270 trials, excluding numerical zero ties. The intermediate 20 s median version with non-bout lower bounds defined 107 trials (20 Delay, 87 Trace, 0 Control); every defined baseline contained 20 bins above and 20 below zero. Both audits are preserved in `../fgh_version4_baseline20_20261009`.

Source validity classifications were reconstructed from the original fish data, with source hashes checked and detector bout membership and eligible log samples matched to the earlier tables. Valid non-bout assignments, mean propagation, baseline references, exported trial flags and SVG geometry/colours are verified. The rendered PDF was visually inspected. The first three versions are unchanged.
