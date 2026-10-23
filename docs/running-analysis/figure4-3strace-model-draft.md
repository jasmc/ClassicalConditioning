# 3sTrace response-profile model draft

This exploratory implementation follows the [Delay pilot](figure4-delay-model-draft.md)
under [Plan 5.1](../../Plans/05.1_FIGURE_4_BOUT_AVAILABILITY_AND_POOLING.md).
The separate [scoped state](../paper/figures/figure-04/3strace-model-draft-state.json)
binds its editable SVG, minimal self-contained HTML, legend, numerical analysis,
draft registry and exact saved-artifact layout evidence. It selects no learner
method, primary paper cohort or scientific freeze. Progress is maintained only
in the [implementation index](../../Plans/IMPLEMENTATION_STEP_INDEX.md).

## Authenticated inputs and membership

The [starting ticket](../../records/figure-reviews/figure-04/3strace-model-draft-start-20261022.json)
captures previous authority, working artifacts and input hashes. The source is
the authenticated 2026-10-20 3sTrace assay review and its saved preliminary
classification mapping. Its current roster contains 179 fish: 89 trace fish
and 90 matched controls. The original processed project is still at its recorded
`F:\Digested Data\all3sTrace-full-v1` location on this Windows host; it is absent
from `J:\Digested Data`. No paths or drives are substituted. JOAQUIM contains
the authenticated review cache and all new permanent outputs:
`J:\ClassicalConditioning Outputs\current\reviews\figure4-3strace-model-draft-20261022`.

Original corrected metric, movement and temporal files are authenticated against
the saved per-fish source hashes. Temporal completion, upstream identity and
recorded CS/US timing are checked. Cached supported-reader profile values are
authenticated, joined by fish/trial/time and independently reproduced from the
original data for the first fish. The stored temporal signed-value column, which
uses another baseline, is not substituted for the supported signed estimator.

The saved classifier execution is
`legacy-wip-3strace-distal-window13-59fish`, using
`legacy_distal_angular_speed`. The saved table, manifest, assessment and source
comparison/cohort/outcome identities remain distinct from the current roster.
Only its authenticated overlap supplies labels: 16 learner-flagged and 23
nonlearner-flagged trace fish. Fifty trace fish remain explicitly unclassified
and outside these modeled experimental groups. All 90 controls are pooled,
including controls without a usable saved label. Overall, 122 of the 179 fish
lack a usable historical classification. The timing-flagged historical trace
fish `20230803_12` remains outside the current roster. Missing labels never
become nonlearner labels. These acquisition/recovery-based flags overlap the
evaluation data and support retrospective characterization only.

## Assay-specific model and outputs

The outcome matches Delay: signed conditional bout ln vigor, half-second bins,
each trial's [−20,0) s baseline, coverage ≥0.9, configured catches summarized by
median within fish and equal fish contributions. The 3sTrace configuration
independently resolves catches 25, 39, 53, 59 and 65. No-bout intensity remains
missing. Observed movement fraction is separate; technically inadequate
coverage stays unavailable in both outcomes. Per-fish trial counts and group
contributor counts are saved with the numerical data.

Recorded CS durations agree with the configured 10 s. Paired-training US events
agree with the configured 13 s expectation, leaving a 3 s trace interval.
The fitted window is −5 to 16 s, preserving Delay's 5 s pre-CS and 3 s post-US
context. The pre-US extrema grid is 0.25–12.75 s: 26 half-second bin centers,
versus Delay's 18 centers through 8.75 s. The spline retains six degrees of
freedom per group; basis spacing therefore changes with the assay window.
The random time covariate scales by 16 s rather than 12 s, with the same
unrestricted intercept/slope structure and diagnostic acceptance rule.

The Gaussian ML model tries fish intercept/time slopes first. An original-fit
failure or fewer than 95% successful whole-fish refits rejects that structure.
The complete bootstrap then restarts with fish intercepts, seed 20261022 and
399 requested refits. Draws from different structures are never mixed, and
each attempted structure retains its original-fit and refit diagnostics.
Population means have pointwise percentile ranges; direct suppression contrasts
have nominal joint bands over three contrasts and all 26 pre-US grid centers.
Extrema are recomputed in every retained refit. Search-boundary frequencies,
the complete timing histogram and observed contributor support are saved.

The saved run fits 5,287 observed bins from 129 fish. The original
intercept/time-slope fit passed, and 397/399 refits passed (99.5%), so the
intercept-only fallback was not needed. All 89 trace fish supplied 46 verified
paired-training US events each; their per-recording median latencies span
12.9945–13.021 s. Recorded CS durations span 9.906–10.054 s, within the declared
0.1 s timing tolerance.

| Direct contrast | Fitted maximum (ln vigor) | Grid time (s) | Descriptive timing range (s) | Bootstrap maxima at a search boundary |
|---|---:|---:|---:|---:|
| Control − learner-flagged | 0.0845 | 11.75 | 1.75–12.75 | 18.6% |
| Control − nonlearner-flagged | 0.0318 | 12.75 | 0.25–12.75 | 82.9% |
| Nonlearner − learner-flagged | 0.0984 | 7.25 | 2.25–11.75 | 1.8% |

At the respective fitted maxima, observed contributors are 87 controls/15
learner-flagged fish, 89 controls/23 nonlearner-flagged fish, and 23
nonlearner-flagged/16 learner-flagged fish. Minimum pre-US contributors are
85, 14 and 21 for control, learner-flagged and nonlearner-flagged groups.
Timing histograms and boundary mass preclude precise biological localization;
the extrema cannot independently validate these same-data labels.

The six panels show pooled catch response, direct suppression contrasts,
contrast maxima, observed movement, illustrative conditional fish fits and
maxima timing. Illustrations use median observed-bin support within each
experimental stratum, rather than selecting on peak magnitude or timing.
Conditional predictions borrow group shape and fish effects; they do not
reconstruct observed movement in silent periods. Bootstrap extrema ranges are
descriptive and are not calibrated biological peak-time confidence intervals.
No p-values, independently validated binary learner comparisons or localized
biological peak claims are supplied. Penalized GAMM comparison, informative
availability and temporal residual-dependence calibration remain unresolved.

## Build and review

The [builder](../../scripts/build_figure4_3strace_model_draft.py) runs sequential
`extract`, `model`, `render` and `publish` stages. The ticket belongs to this run;
it is not authority to overwrite a manually edited or subsequently selected
draft. The Delay builder, adapter, state and external artifacts are protected.
The [assay adapter](../../src/classical_conditioning/analysis/review/figure4_assay_profiles.py)
opts into explicit bounds while reusing Delay's contrast, prediction-design,
peak, resampling and Gaussian likelihood helpers. The original Delay adapter
remains unchanged so its bound implementation hashes remain valid.

The [browser checker](../../scripts/check_figure4_3strace_model_draft.cjs) accepts
the external directory and a temporary PNG path. It collects the exact SVG's
600-dpi layout-frame measurements and checks the minimal HTML scope. The build
uses one resolved `nature-main` snapshot, scoped rc settings, a native
183 × 135 mm canvas and the active 1 mm invisible frame policy. Automatic tight
cropping is disabled. Layout evidence and an actual SVG/legend inspection are
required before guarded publication; artifacts and evidence precede the atomic
state pointer. The publication lock rechecks all bound original/input/output
hashes and preserves concurrent changes. Large immutable Parquet originals are
hashed afresh in each locked publication attempt; repeated checks within that
same transaction reject changes to their identity/size/timestamp fingerprints,
following the repository's verified-artifact-state approach. Editable outputs,
states, code and other inputs rehash every time. No authentication cache crosses
attempts. Reads use 256 KiB blocks and bounded delayed retries for transient
storage errors; changed hashes always fail. Identical saved review evidence can
be reused after an interrupted publication, while differing evidence fails for
reconciliation. No freeze gate is invoked.

The exact saved SVG passed the 600-dpi assessment with no issues. The SVG and
its 241-word legend were inspected in Edge and opened in the built-in browser.
The [version-bound final audit](../../records/figure-reviews/figure-04/3strace-model-draft-verification-20261022.json)
verifies the published state, exact HTML/SVG/legend pairing, independent catch
aggregation and 24 prior protected records, including Delay. It records zero
3sTrace-scope maintenance issues separately from the repository checker's
existing archive/history, older-figure and catalogue failures.

Eight known-answer checks ([four assay checks](../../tests/test_figure4_assay_profile_models.py)
and [four unchanged Delay checks](../../tests/test_figure4_profile_models.py))
verify Delay equivalence, the extended 3sTrace grid, dense Gaussian slope
predictions, missing intensity and complete bootstrap restart without mixed
draws. This numerical and layout scope does not establish complete
Nature production compliance or statistical calibration.
Two [publication guard checks](../../tests/test_figure4_model_publication.py)
cover source changes, fresh authentication on a new attempt and manual output
edits with restored timestamps.
