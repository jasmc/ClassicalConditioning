# Figure 3 main-paper draft

The [scoped Figure 3 state](../paper/figures/figure-03/current-state.json)
selects the editable SVG, self-contained HTML and matching legend. This is an
unfrozen Delay/control pilot, not a selected learner method. The author requires one recommended analysis path in the main
figure: G outcomes → acquisition contrast → individual illustrations. The separate
[57-fish atlas](figure3-individual-atlas.md) remains available without modification.

## Data and estimands

`scripts/paper/figure_03/build_main_draft.py` consumes the authenticated selected atlas:
trial-data.parquet, g-logmedian-outcomes.csv, fish-inventory.csv,
heatmaps.npz and reference-style.json. It verifies
source hashes and the exported G values. No raw-data preprocessing or classifier fitting is performed. All 57 fish and 5,130 scheduled
trial rows remain in the inventory.

The observed outcome is response_logmedian minus baseline_logmedian, as selected
for Figure 2 G. The new **descriptive acquisition contrast** is median of that
outcome over Pre 5–14 minus median over late Train 40–64. Positive means greater
suppression. Test 65–94 does not enter this calculation; prior Test inspection
prevents calling it prospectively held out. Recovery remains a separate property.

Each score needs at least one finite outcome in each epoch mathematically.
This is not an approved scientific coverage rule. All 57 are estimable; observed
coverage ranges are 4–10 Pre and 5–25 late-training trials. The exported table
retains all epoch counts, undefined cases and reasons. No binary G labels,
individual intervals or new exclusion criteria are introduced.

The main builder does not consume classifier decisions, compare algorithms,
or display legacy learner flags. The separate exploratory atlas preserves those
methods and exact inputs unchanged. The continuous acquisition path is preferred
for this descriptive draft because it is directly auditable, does not require
recovery, and avoids individual model intervals whose coverage failed.
It is not established as a validated binary learner classifier.

Main C–E deliberately illustrate larger (20240618_08), intermediate
(20240618_02) and near-zero (20240612_08) observed contrasts. Selection uses
observed magnitudes and these are explanatory examples, not an unbiased sample
or validation. Black horizontal epoch-median bars connect each trial plot to
the exact score; pink bars retain descriptive five-trial medians. The complete
atlas remains available, including fish with undefined intensity or legacy
classification failures.

Heatmap values are direct 1 s median eligible positive frame-log vigor centred
by median finite baseline-bin medians; this differs from the trial estimator.
Colors use native managua_r ±0.25, undefined black, without clipping numbers.
Move B/CR strips represent duration-weighted fraction moving among detector-
eligible time; availability is not independent tracking completeness. Catch
IDs 25,39,53,59,65 follow the atlas protocol configuration: the source is_catch
column is uniformly false and is not used to infer catches. Actual US ticks use
source actual_us_s. Main G trial axes share −0.6 to +0.6 and retain every
illustrated observation; five-trial median bars are descriptive.

## Reproduce and verify

Use a new existing directory on mounted JOAQUIM, with a start-ticket.json
binding the selected atlas state and consumed inputs, previous Figure 3 state,
protected other-figure pointers, intended outputs, previous working-copy hashes
and analysis choices. The corrected single-path draft preserves the previous
assembly and its state in external history; it does not overwrite that artwork. Existing
published artwork must be preserved rather than overwritten.

```sh
.venv/bin/python scripts/paper/figure_03/build_main_draft.py build --output-dir '<external-directory>'
.venv/bin/python -m unittest tests.test_figure3_main_paper tests.test_individual_atlas tests.test_legacy_learner_adapter
```

`scripts/paper/figure_03/verify_main_draft.cjs` uses bundled Playwright and a local review
server at the supplied base URL to inspect the exact saved SVG and collect standard 600 dpi
frame measurements. Assess its layout-browser.json with layout_frames.assess;
write layout-evidence.json and an honest hash-bound visual-review.json after
inspection. The publisher checks bound inputs/outputs and previous working-copy hashes, exact-render evidence
and visual review before atomically updating only the Figure 3 state:

```sh
.venv/bin/python scripts/paper/figure_03/build_main_draft.py publish --output-dir '<external-directory>'
```

The native composition is 183 × 158 mm with 1 mm invisible frame insets. It
uses editable Arial text and 6 pt ordinary plot text; this is a draft visual
check, not a claim of complete Nature production compliance. No scientific
freeze or global main-paper collection inclusion follows from publication.

## Remaining scientific decisions

Select and validate the final learner estimand, precision/coverage requirements,
threshold if any, calibration and stability before interpreting categories as
paper results. Existing individual model coverage failures preclude inferred
bands. Resolve Trace data availability before expanding across experiments.
Keep aggregate Control/Learner/Nonlearner behavioral comparisons in Figure 4.
