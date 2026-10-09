# Version 4: direct half-second medians, centred log units

Confirmed by the user on 9 October 2026:

- Use direct eligible framewise natural-log vigor, without bout-median substitution.
- Retain the existing moving-sample eligibility mask. Invalid and outside-bout samples remain NaN.
- In each CS-aligned 0.5 s bin, take the median of non-NaN samples. Even one eligible sample is sufficient; there is no minimum coverage rule.
- Keep all-NaN bins missing.
- For each trial, take the median of finite unscaled bin medians in [-15, 0) s. Each finite bin supplies one scalar, irrespective of its eligible sample count.
- Subtract that trial reference from every bin median. Do not compute P10/P90, scale or numerically clip.
- Display trials 5-94 in [-20, 20) s with 80 bins per trial.
- Use shared managua_r colours over [-0.25, +0.25] natural-log units. Values beyond those limits use endpoint colours; exported values remain unchanged.
- Fish: F Delay 20221115_07; G 3 s Trace 20230310_08; H Control 20221115_09.

`build_version4.py` verifies source hashes, reads the direct log-vigor column, independently checks each bin median with NumPy's NaN-aware median, and checks SVG cell geometry and colours. `verify_and_register.py` checks the exported CSV values and baselines and adds this row to the existing scoped Figure 1 registration. The first three versions are retained.

All 270 trial references are defined. Eighteen bins containing one eligible sample are retained. Baseline-bin medians are zero to floating-point precision after subtraction. Control trial 16 is defined here because there is no scaling denominator.

Numeric CSV files preserve bin counts, unscaled medians, baseline references and centred log values. `data_manifest.json`, `numeric_verification.json` and `validation.json` record the source hashes, numeric checks and figure checks. PDF rendering was visually inspected; the portable HTML summary contains this as its fourth row.
