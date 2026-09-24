# Reproducible historical learner comparison

The `compare-legacy-learners` command executes the four unchanged scripts in
`Archive/historical-scripts`, each in a fresh Python process. It is a
**descriptive historical-rule comparison**, not the Gate L learner decision or
independent validation of individual learning.

Install the optional dependencies with `pip install -e '.[legacy-learners]'`.
For an authenticated allDelay cohort, run:

```sh
classical-conditioning compare-legacy-learners \
  --cohort '/path/to/Processed data/Cohorts/<cohort-id>/cohort-manifest-v1.parquet' \
  --trial-outcomes '/path/to/Processed data/Cohorts/<cohort-id>/cohort-trial-outcomes.parquet' \
  --metric legacy_distal_angular_speed \
  --output-dir '/path/to/new/comparison-directory'
```

The output directory must be new or empty. It contains `comparison.json`,
`fish-comparison.csv`, the exact translated trial input, one evidence-rich
Parquet result, settings JSON, and one log per variant. Every primary-cohort fish remains in
the comparison, with `unclassified` status if a variant cannot score it. A
failed variant makes the command fail after it writes its diagnostic report.

The adapter maps `response_total_activity` to historical `Mean CR`,
`baseline_total_activity` to `Mean 9s before`, and their ratio to
`Normalized vigor`. These corrected activity values have different measurement
semantics and units from the original stage-5 pooled normalized-vigor files.
Consequently this route preserves **algorithmic rules**, not numerical
equivalence with historical published labels. In particular, the nominal,
new, and improved scripts transform the unscaled activity, so changing its
units can change their labels. Use the same translated input to compare the
four rules; do not interpret agreement as a biological ground truth.

The four rules all use acquisition and later recovery data to form labels.
Analyses of those same data by label are descriptive. A paper-level learner
claim still needs the representation choice, continuous-method comparison,
provenance, and non-circular validation required by Plans 04–06.

To render the archived scripts' summary trajectory, feature-space, BLUP
overlay, and available caterpillar plots after comparison:

```sh
classical-conditioning render-legacy-learner-figures \
  --comparison-dir '/path/to/comparison-directory' \
  --include-individuals
```

The renderer recomputes and checks each result against the saved comparison.
It writes variant-isolated PNGs and `figures/figures.json`. Individual
composite plots can be generated from the translated trial data. Historical
heatmap grids expect separately pre-rendered stage-5 images, which are not
part of the corrected cohort outcome bundle.

To inspect control fish flagged by the historical rules, run:

```sh
classical-conditioning compare-legacy-control-flags \
  --comparison-dir '/path/to/comparison-directory'
```

This saves a condition comparison chart, exact fish-level control flags, and
variant-specific counts. Its control fractions are observed **in sample**:
these same controls informed each rule's reference and threshold, so the
fractions are not independently calibrated false-positive rates.
