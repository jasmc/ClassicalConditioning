# Corrected BLUP comparison

The [study](../paper/figures/figure-03/blup-comparison.md) owns the full explanation,
all-fish comparisons and questions. Its editable Markdown and HTML live together
on JOAQUIM. The completed numerical run remains immutable at
`ClassicalConditioning Outputs/studies/figure-03/blup-comparison-20261025/`.
The selected Figure 3 remains the separate descriptive single-path draft.

## Numerical run

Use `scripts/paper/figure_03/run_blup_comparison.py` with an authenticated source
ticket and a **new** external output directory. Defaults are 999 calibration
refits and 499 independent datasets in each of matched, heavy-tail, within-epoch
trend and null scenarios, with explicit disjoint random seeds. The command binds
source files and figure states, preserves all estimable input fish, records
failures/boundaries, snapshots the model implementation, then verifies its
completed artifact manifest. It never selects a cohort, method or figure.

The 25 October run completed every refit. It uses the 57-fish Delay/control pilot,
G log-median trial differences and Pre 5–14 versus late-Train 40–64. Recovery is
reported separately as late-Test 85–94 minus late-Train 40–64; it is descriptive
and does not inherit the acquisition bootstrap calibration. Legacy decisions are
copied from their saved source, with their different estimands/epochs explicit.

Tests in `tests/test_blup_comparison.py` retain the independent dense-matrix
checks. Simulation results are evidence under finite scenarios, not proof of a
biological classifier, missingness mechanism or full-Trace validity.

## Study build and revision

`scripts/paper/figure_03/build_blup_study.py` accepts `--analysis-dir`, a new
`--output-dir`, and the personal review notebook script plus its exact hash.
The notebook remains a personal skill asset; it is inlined into the external
HTML and is not vendored into this repository. The builder verifies the complete
numerical manifest, includes all fish and source identities, and preserves state.

For a reviewed manuscript revision, pass `--manuscript`,
`--manuscript-sha256` and an explicit `--guide-version`. Keep the stable section
meanings and notebook identity. Prepare a new edition; preserve the prior
source/HTML, verify it, and publish through the existing artifact transaction
before updating the reading catalogue and study-source identities. Never
regenerate over manual source edits or discard notes. The current study retains
its first edition under `history/`; browser notebook behavior and print layout
were not exercised in this task, and those limits remain in its provenance.
