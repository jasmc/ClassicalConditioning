# Consolidated scientific and scope decisions

**2026-10-09 standing author update:** [all analytical vigor is bout-only](BOUT_ONLY_VIGOR_POLICY_2026-10-09.md). No-bout periods are NaN and ignored before every vigor calculation. This supersedes earlier all-frame vigor definitions; historical/frozen artifacts remain preserved. The D/E mean-ratio and log-median comparison has been rebuilt under this shared rule; an estimator selection remains open.

This document exists to stop the analysis from feeling open-ended. It
collapses the scattered per-step "gate" checklists into one place, records
the actual decisions made, and sets implementation priorities so the full
roadmap is not silently re-litigated per step.

Earlier step plans and their acceptance criteria remain in Git history. Items
outside the immediate priority lane are deferred, not silently declared done,
and can be resumed when the corrected paper-analysis path is further advanced.

Update this file when a decision changes. Do not duplicate its content back
into the individual step plans beyond a short pointer.

## Decisions made (2026-08-30)

| Gate | Decision |
| --- | --- |
| G0 (scope) | Same core claim and same fish/experiments as the original paper. No scope expansion. |
| T0 (raw tracking semantics) | Raw `angleN` columns are **radians**. Legacy analysis converts them to degrees with `* (180/pi)` before vigor (`data_io.read_tail_tracking_data`, legacy `my_functions`). Candidate metrics keep radians. `angle1..angle14` behave as local intersegment bends (agree with XY-derived segment orientation changes on the pilot); `angle15` is a terminal placeholder. Measured `xN`/`yN` are present and treated as **tracking-image pixels**; absolute µm calibration is **not required** for relative activity metrics. Confidence and independent body-axis fields are absent. Synchronized video remains optional/deferred for blinded validation only. |
| T1 (activity metric) | **Metric frozen by the author on 2026-10-06: tail bend angular speed** (`legacy_distal_angular_speed`, rad/ms), using the existing corrected measured-time formula. All paper activity analyses, assessments and panels follow this choice. Angular L1 and whole-tail XY remain explicit comparison/sensitivity metrics. Evidence, name, formula, limitations and downstream consequences are in [the selection record](../docs/analysis/METRIC_SELECTION_2026-10-06.md). Shared bout segmentation remains unchanged; detector/smoothing policy approval remains separate. |
| C0 (cohort inclusion) | **Primary paper policy remains open; boundary decided.** One `assess-discarding` command runs technical assessment, then exploratory source-linked preprocessing checks and a merged learner-input prerequisite. Primary membership may use only approved label-independent technical criteria. Response-window movement, CR strength, and learner status cannot determine it. The exploratory combined status is a projection on the selected metric, not a historical or primary cohort. Normalized-vigor trial/block filtering is deliberately outside this command; LME trial eligibility and learner model-derived feature failures remain explicit in their respective later analyses. See the [current assessment behavior](../docs/analysis/04_DISCARDING_AND_SELECTION.md) and [active cohort work](./01_COHORT_IMPLEMENTATION.md). |
| S (statistics) | **Open in the [analysis and statistics design](./02_ANALYSIS_AND_STATISTICS.md).** The existing fish-grouped LME, fish permutation, and fish bootstrap are exploratory engineering scaffolds, not the approved strategy. The primary estimand, outcome family, condition contrast, random-effects structure, diagnostics, uncertainty, multiplicity, and validation status must be decided together. The LME may be improved, demoted to sensitivity, or replaced. The legacy statistical route remains frozen as `legacy-paper` reproduction only—not extended or “fixed.” |
| L (learner analysis) | **Required paper workstream; method open.** Complete the [learner classification and stratified analysis plan](./05_LEARNER_ANALYSIS.md). Begin with continuous fish-level effects and heterogeneity and compare continuous, longitudinal, probabilistic, and categorical representations. A binary or multiclass classifier is used only if it adds defensible meaning and avoids circular confirmation. If a hard classifier is rejected, the paper still reports the approved continuous or model-based learner result and the reason classes were not imposed. |
| F (figures) | Only two figure modes are actively maintained going forward: static PNG and publication SVG/PDF. Interactive local HTML figures are frozen as-is (already implemented, not broken, not deleted) but receive no further investment. |

### Addendum (2026-08-31) — historical pickle timebase

Stage-1 gzip pickles for `20221115_04` and `20221116_12` advance
`Original frame number` within trials at `expected/predicted` (reciprocal of
current interpolate). Package Parquet matches current
`FrameID *= expected/predicted` physics (`predicted/expected` Original slope).
CS onset still agrees; edge misalignment ~0.33 s. Do **not** invert
`legacy-paper` interpolate to match pickles. The underlying finding is also in
[the analysis audit](../docs/analysis/audits/01_ANALYSIS_FINDINGS.md#h1-historical-per-fish-pickles-warp-within-trial-time-via-reciprocal-original-frame-rate).

## Panel-specific author instructions (2026-10-09)

For **Figure 2 D/E block-ratio panels**, the author excludes recording day
entirely and selects the conventional fish-level rank-test review rather than
an LMM: paired Wilcoxon within fish and Mann–Whitney between independent groups,
including direct between-group comparisons of matched fish changes. The frozen
`legacy_distal_angular_speed` metric and CS baseline `[-15,0)` s remain in use.
The author selected the shared-left-axis PT/ET/LT paired-plot appearance with
straight statistical lines and translucent black median/IQR markers.
The author subsequently **froze this D/E style on 2026-10-09**, adding a
horizontal **1.0 reference in each ratio subpanel: fully black, alpha 1.0,
drawn behind all data**. Fish and median/IQR transparency remain as selected.
This is a presentation-only change; saved data, tests and correction are
unchanged. The exact frozen artifacts are linked from the panel comments.

The request to ensure complete comparisons is implemented as an exploratory
candidate with all three block pairs: 6 within-condition, 3 same-block
between-condition and 3 between-group change comparisons per available assay.
Holm24 is its conservative review family, not a final paper-family decision.
The primary/secondary claim hierarchy, assumption review, uncertainty and
scientific approval remain open. The author requested a durable place for
deferred comments per panel; these live in
[panel review comments](./PANEL_REVIEW_COMMENTS.md). The D/E instructions do
not change other panels, the separate onset/trajectory review, or literal
`legacy-paper` reproduction. F remains inconclusive without authenticated data.

## Figure element identities and future freezes (2026-10-09)

The author selected a uniform paper style and requested a durable repository
contract for **every panel or whole-figure freeze**. Its scientific role catalog,
Matplotlib/SVG identities, current Figure 1-2 source inventory and shared styles
are in the [element specification](../docs/analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md)
and [versioned machine specification](../configs/paper-figures/figure-elements.json).
The [root repository instructions](../AGENTS.md) require applying these rules
to a reviewable candidate, checking final assembly scale, and asking targeted
questions only about unresolved exceptions. Explicit prior approvals remain
authorized within their recorded scope; do not ask again or silently broaden
them. Record approved departures separately from scientific definitions.

Freeze records must bind the specification version/hash, element registry,
resolved styles, scoped exceptions, source/data hashes and final exports.
Existing frozen figures and data remain unchanged. The author subsequently
selected **code-enforced checks only at freeze time**. Version 1.1.0 adds the
[explicit freeze command](../scripts/freeze_figure.py): it checks candidate
SVG presentation at final size, identity coverage, scoped exceptions, protected
geometry and hashes, then publishes a new manifest without overwriting a freeze.
Scientific mappings, plot-family structure and unmeasurable renderer properties
require recorded completed review evidence. Ordinary rendering/export does not
invoke the gate. The [SciFigEditor review](../docs/analysis/figures/SCIFIGEDITOR_REVIEW_2026-10-09.md)
records the useful confidence/protection/physical-scale ideas and integration
limits. This is a concrete figure-element use case, not a
reactivation of the deferred general artifact-schema registry below.

## What is deferred or minimized in the current priority lane

- **Schema registry / logical-content hashing / typed artifact metadata**: the optional design remains in Git history, with no active implementation step. Current SHA-256 byte hashing and atomic transactional publication are sufficient for the planned single analysis release. Reopen only a specific part if a concrete need appears.
- **Interactive HTML figures**: frozen, no further polish (see Gate F above).
- **The 10-gate formalism as a recurring per-step ritual**: replaced by this single document. Gates are not re-asked per step; they are only revisited if new information changes a decision (for example, real cohort numbers when deciding Gate C0).
- **Categorical learner implementation**: final implementation waits for stable
  preprocessing, cohort, and outcomes, but the learner-method review and design
  are active and required (see Gate L).
- **Behavior-dependent legacy discard as primary QC**: rejected. Its exact
  checks remain available only as a named sensitivity analysis under the
  [integrated cohort/CR-profile plan](./07_INTEGRATED_ANALYSIS_AND_CR_PROFILES.md).
- **Final analysis release**: make one release when the code, scientific decisions, paper analysis, and figures are ready. Preserve a tagged commit and frozen manifest linking inputs, cohort, configuration, results, and figures; candidate and legacy runs are analysis evidence, not interim releases. See [the final release plan](./12_FINAL_ANALYSIS_RELEASE.md).
- **Expansion of legacy characterization**: lower priority because the existing characterization (see [the codebase behavior map](../docs/analysis/legacy/02_CODEBASE_BEHAVIOR_MAP.md) and [analysis findings](../docs/analysis/audits/01_ANALYSIS_FINDINGS.md)) is sufficient for the immediate path. Additional characterization remains in scope when required by equivalence testing or a concrete migration risk.

## Figure 2 B D/E selection and freeze — 2026-10-09

Joaquim authorized correcting the statistical tests and freezing the selected version B. D/E now use positive valid bout-frame median natural-log response minus baseline, followed by a median of eligible trials per fish/block. No-bout and invalid frames are excluded before every calculation; empty windows remain undefined. Scientific windows, cohorts and eligibility thresholds are unchanged.

Exact two-sided sign tests handle paired fish changes and zero-reference comparisons; Brunner–Munzel t tests handle independent conditioned/control rank/probability comparisons without requiring equal shapes. One Holm36 family covers all D/E comparisons. Fish independence is assumed; day/tank adjustment and exploratory selection remain explicit limitations. Delay effects survive correction; no Trace comparison survives. Earlier Wilcoxon/Mann–Whitney results are superseded, preserved historical evidence.

[D freeze](../reviews/figure2_B_freeze_20261009/D.freeze.json) and [E freeze](../reviews/figure2_B_freeze_20261009/E.freeze.json) passed the explicit specification 1.1.1 gate. [Scoped selection](../configs/paper-figures/figure2-DE-B-freeze-20261009.json) records the authoritative panel exports and hashes. This scope does not freeze F or the whole assembly. Existing whole-figure assembly and historical freezes remain preserved. Future assembly must accommodate 54.9 × 53.2 mm per panel to retain effective fonts/strokes.

[Processing handoff](./HANDOFF_BOUT_LOG_VIGOR_PANELS_2026-10-09.md) supplies the shared scaffold for Figure 1 E–H and whole Figure 2. Other panels require their own estimator, source coverage, statistical scope and freeze; existing Figure 1 quantile-scaled definitions are not replaced by this selection.

## Practical shortened critical path

Given the decisions above, the remaining path to a corrected, defensible
result is (fixture-scoped Steps 03–05 are already complete; see
[IMPLEMENTATION_STEP_INDEX.md](./IMPLEMENTATION_STEP_INDEX.md)):

1. Complete the paper baseline evidence that still matters for release;
   the archived schema/logical-hash proposal is outside this analysis path.
2. Approve corrected preprocessing and the three-metric candidate set —
   keep the pipeline generic so all three run the same way; local fish are
   debug fixtures, not selection/confirmation cohorts. The activity metric is frozen; preprocessing and detector policy
   remain open.
3. Cohort assembly — revisit Gate C0 per-block inclusion with
   real paper-scale numbers when available.
4. Outcomes and statistics — run identically across the three-metric set;
   freeze Gate O/S after the design record is decided.
5. Complete required learner analysis and its validation mode.
6. Build PNG + publication figures only; interactive HTML remains frozen.
7. After Figure 4's pooled-learner CR analysis is completed and reviewed,
   carry out the tail mechanistic analyses in
   [step 09](./09_TAIL_MECHANISTIC_ANALYSES.md).
8. Create the one final analysis release with a tagged commit and frozen
   manifest after the paper analysis and its figures are verified.

Local recordings currently on disk (`20221115_04`, `20221116_12`) are the
**real fixtures for end-to-end pipeline tests**, including cohort freeze and
multi-recording runners. They are not paper-scale N and do not authorize
Gate T1/C0 scientific decisions. Synthetic clone fish were removed once these
two recordings were available.

Step 00 governance evidence remains required before final paper-authoritative
claims and release. It is not required to block every engineering increment.
Earlier learner plans remain in Git history. Their operational content
is in the active learner plan, and learner analysis is required for
the paper even though the categorical-versus-continuous representation remains
open.


### 2026-10-09 — G selected Historical LogMedian freeze
Author explicitly selected native bout-only Historical LogMedian with D/M/R and phase-aware LMM trial statistics. See configs/paper-figures/figure2-G-logmedian-freeze-20261009.json. No second log or smoothing. Test3 local M/R n/a; exploratory, no onset claim. All alternatives consolidated in reviews/figure2_delay_all_versions_20261009.html; verified stale Delay copies removed. Whole assembly and D/E freezes unchanged.
