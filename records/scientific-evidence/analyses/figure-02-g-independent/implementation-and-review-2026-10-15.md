# Figure 2 G independent implementation and review — 2026-10-15

The exploratory implementation is complete and has been run. The scientific
analysis is not approved for paper inference: the primary baseline treatment,
meaningful acquisition threshold, technical cohort policy and validated timing
uncertainty remain open. The selected G, its five protected originals and the
primary 28-control/29-Delay cohort are unchanged. This record supersedes the
implementation status, but preserves the findings and evidence, of the
[first review](review-2026-10-14.md).

## Bound result and preservation

External run:
`/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-G-independent-review/20261014-analysis-02`.
Resolve the current stage through its `analysis-state.json`; at this review it
is `stages/report-7bb6bd520060`. The self-contained `review.html`, matching
`population-candidates.svg` and `legend.md` are independent candidates. They do
not replace Figure 2's selected working or frozen artifacts. The companion
`implementation-evidence-2026-10-15.json` binds this stage, its full manifest,
the technical supplement, local supported sources and visual QA captures.

The first run, `20261014-analysis-01`, all preceding published stages and failed
unpublished attempts remain preserved. Each successful stage snapshots source,
records dependencies, checks originals and publishes its manifest before an
atomic state-pointer update under an exclusive lock. Resume authenticates the
selected inventory and hashes. Failed candidates cannot select themselves.
The imported reconstruction was checked against all 401 bound inputs and all
4,811 selected G outcomes; maximum numerical discrepancy was 4.44e-16.

## Implementation review findings

The original review's estimator mismatch, incomplete numerical retries,
publication/resume defects, failed-target stabilization crash and hardcoded
simulation description were repaired. Training-only individual fitting,
leave-one-control-out validation and simulation now use the same scheduled-grid
timing rule. Observed supported runs are separate; three available observations
cannot bridge missing scheduled trials or the training/testing transition.
All 57 leave-one-fish-out fits pass for each population target; random-intercept
and US-overlap sensitivities now produce accepted fits, with attempts retained.

The Gaussian ML accelerator preserves the phase-specific fixed effects,
fish intercept/slope structure and full observed-information fixed-effect
covariance. Comparison with statsmodels on the full G data found contrast
differences below 2e-7 and relative SE differences below 0.004%. Cached
integer-multiplicity bootstrap fits were also compared with explicit copies
having distinct fish IDs. These establish numerical equivalence, not model
adequacy. Earlier stages retain their own source snapshots.

A further report defect was found and repaired: an unsmoothed calibration
summary filtered fit acceptance using a timing-localization status and counted
zero accepted experiments. The 1,100 saved replicate results were intact.
`validation-review` rebuilt the summary from them; the superseded report is
preserved. A regression test distinguishes band acceptance from onset status.

The statistical concerns remain material. All 319 unavailable paired vigor
outcomes involve no detected valid bout in at least one window: 10.3% of
control rows and 2.3% of Delay rows. This is undefined conditional vigor, not
zero vigor or a fish-level technical diagnosis. New simulations stress
availability depending on baseline/response, phase and tracking coverage;
descriptive movement-occurrence/time-moving context is saved separately.
These checks do not prove identification under the real availability mechanism.

## Condition difference and acquisition are separate results

The candidate contrast is control-minus-Delay change relative to average Pre
trials 5–14. Positive values indicate conditioned-direction suppression.
Each target's bootstrap-t family combines 90 scheduled trial contrasts and
three planned phase averages. Main population runs request 999 whole-fish
resamples; 989 difference and 999 adjusted-response refits pass.

| Target | Late training 55–64 | Early testing 65–74 | Late testing 85–94 |
|---|---|---|---|
| Response-minus-baseline LogMedian | 0.103 [0.039, 0.167] | 0.074 [0.006, 0.141] | 0.014 [−0.059, 0.087] |
| Response LogMedian adjusted for baseline | 0.095 [0.012, 0.178] | 0.065 [−0.027, 0.157] | 0.026 [−0.085, 0.137] |

These are exploratory simultaneous model intervals in log units. Baseline
treatment changes the early-testing conclusion. A late-testing interval that
includes zero does not establish extinction; no extinction claim is tested here.

At the unapproved 10% threshold with three consecutive scheduled training
trials, main-model point crossings are 48 and 50; neither has a finite latest
acquisition bound. Different exploratory thresholds yield different answers.
Point crossing, earliest compatible onset and first supported run/latest bound
are explicitly distinguished. No onset is localized by choosing a favorable
cutoff or by compressing missing trials.

The unsmoothed comparator anchors each fish to its own available Pre mean and
averages contributing fish changes. It has a different estimand and broader
uncertainty: late training 0.098 [−0.048, 0.243], early testing 0.085
[0.001, 0.168], late testing 0.016 [−0.074, 0.106]. Only 27 controls contribute
to its late-training phase summary; all fish remain members. Similar point
directions do not establish model-independent significance.

## Historical exclusion experiments

Rules were read from Git revisions
`2f63ef48361d6e80d1b8e1393894d7e5a3dedf55` (February 14) and
`bf46bf7b02baaf6d8138d9772f255881f86c3c78` (March 24).
Five relevant files per revision are snapshotted and hashed. Historical
`RUN_DISCARD`/`APPLY_FISH_DISCARD` defaults are false; defaults do not prove
what an interactive historical session actually ran. No actual saved discard
roster was found in those Git trees. The six-trial filter accepts ANY block.

Rules were registered before cohort-effect fitting. Inclusive historical
baseline/CR/US bout windows are evaluated separately from the current outcome
windows. Selected baseline/CR QC blocks are 10–14, 65–69 and 90–94; training-US
checks cover ordinals 18–63. Original segmented Bout/700 FPS pickle data are
unavailable, so these are labelled projections on corrected authenticated data,
not exact historical cohorts. The combined recipe uses an explicit raw-US
duration proxy for the unevaluable frame-annotated last-US end.

| Recipe | Controls | Delay | Late-training difference / adjusted estimate |
|---|---:|---:|---:|
| All selected G; historical ANY-block six-trial filter | 28 | 29 | 0.103 / 0.095 |
| Baseline ≥3 bout trials per selected block | 25 | 29 | 0.110 / 0.102 |
| CR ≥3; baseline-and-CR intersection | 24 | 29 | 0.121 / 0.116 |
| All training-US responses | 26 | 27 | 0.126 / 0.123 |
| Combined projection with last-US proxy | 24 | 27 | 0.124 / 0.120 |
| ALL-required-blocks six-trial alternative | 23 | 27 | 0.114 / 0.108 |

The combined projection removes controls `20240611_05`, `20240620_06`,
`20240624_06`, `20240703_07` and Delay fish `20240610_07`, `20240618_02`.
The report contains each failed rule, overlap, cumulative attrition and roster.
All recipes have positive nominal late-training LMM intervals with 199
requested resamples per target; none supplies a finite latest 10% acquisition
bound. Behavioral filtering increases apparent effects and changes the target
population. It is not evidence that these fish are technically bad, and no
recipe was selected for its significance.

## Independent technical follow-up

All 57 authenticated acquisition/preprocessing summaries have zero camera or
tracking FrameID gaps, duplicates or reversals and coordinate-valid fraction
1.0. Small stream-boundary nonoverlaps of 0–5 frames are not automatically
exclusions. These limited metrics are not a complete tracking-quality approval.

Delay fish `20221122_03` has one invalid protocol duration and an event outside
acquisition. Inspection of its SHA-authenticated raw protocol confirms timestamp
precision loss in the first two reinforcer records. Intended timestamps are
unknown and were not reconstructed. The selected CS5–94 outcome windows are
available; that does not prove the protocol corruption is scientifically
irrelevant. A prespecified source-only sensitivity removes any fish with this
protocol flag, independently of vigor or learner status, and removes this fish
alone. It retains 28 controls/28 Delay fish and 4,721 outcomes.

`technical-followup-02` gives late-training difference 0.096 [0.032, 0.160]
and adjusted response 0.083 [0.006, 0.159], from 193/199 and 199/199 accepted
resamples. Neither resolves the 10% timing upper bound. Its complete manifest
and the raw trace are explicitly bound by `review-supplements.json` and copied
into the selected report. This is a conservative sensitivity, not automatic
approval of a whole-fish exclusion in the primary analysis.

## Validation limits and verification

The expanded LMM study comprises 110 independent datasets, each analyzed with
both targets: 220 model cases, not 220 independent experiments. Ten datasets
per scenario and 99 resamples expose full-curve coverage of 0/10 for both abrupt
and transient truth for both targets. Phase-average coverage in these examples
is 10/10; the abrupt examples' timing bounds contain the true onset. Neither
observation rescues general curve/timing validity. Zero-random-effect scenarios
also expose fit and bootstrap failures, which withhold inference.

The unsmoothed study uses 100 independent experiments for each of 11 scenarios
and 499 resamples. Full-curve coverage ranges from 92% to 98%; binomial intervals
are saved. Onset intervals contain truth in 100/100 per scenario, often because
they are broad/unbounded; no false 10% supported run occurred. This is promising
comparison evidence, not demonstrated timing precision or validation of the
LMM. Individual spline/state-space pilot coverage is poor for abrupt/transient
truth and omits covariance-parameter uncertainty. Variance ×0.5/2 and AR1 ±0.1
sensitivities do not integrate that uncertainty. No learner labels are assigned.

Twenty-two scoped unit tests pass. They cover timing gaps and phase boundaries,
held-out fitting, undefined outcomes, numerical equivalence, integer fish
multiplicities, failed-target propagation, lock/transaction/hash safeguards,
legacy endpoints/ANY-block semantics, calibration accepted counts and technical
flagging. The selected run and technical manifests, inventories and five
protected original hashes pass verification. Browser inspection confirms the
matching report, historical cohort counts, selected-fish curves and uncertainty
bands; screenshots are external under `qa/`. `git diff --check` passes.

The next scientific decisions are the conditional-vigor estimand and baseline
treatment, technical cohort policy, meaningful threshold/persistence and a
trajectory/uncertainty method that passes the chosen calibration targets.
The exploratory pipeline has completed these comparisons; Plans 02/03/04/06
and their paper-wide scientific gates remain open. Nothing has been frozen.
