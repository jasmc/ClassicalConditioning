# Consolidated scientific and scope decisions

## 2026-10-31 — universal last-US assessment and visual exclusion review

Author direction: assess last-US timing and the bout response after the last US
for every fish, including already-flagged, pending and unavailable recordings.
Failures and unevaluable checks require explicit warnings. For a fish flagged
by either last-US check, show one shared tail-angle-over-time plot around that
stimulus, with timing, bout interval and missing/invalid tracking identified.
Every proposed exclusion or failed check must be presented for the author's
visual review, during assessment or at its conclusion; unavailable plotting
inputs remain an explicit unresolved case.

The [editable combined rule worksheet](study-sources.json)
resolves through the study-source registry to JOAQUIM. It collects the legacy
stages, AI recommendations and per-step author input. The inherited 0.4 s and
0–5 s settings, exact trace/window and final exclusion actions remain for
review. This author direction establishes assessment/review requirements; it
does not approve automatic exclusions, freeze membership, claim a completed
paper-scale check or certify that the dedicated warning/plot workflow is
already implemented. Current technical-primary boundaries remain in force.

## 2026-10-31 — robust baseline normalization for all heatmaps

**Author-approved standing direction:** from now on, use median/MAD robust
z-scoring across all heatmaps, including single-fish, population and stratified
heatmaps. This replaces Q10 scaling and baseline-subtraction-only normalization
for new heatmap work. The choice follows the Figure 2 B comparison: the broad
patterns were similar, and the author selected the symmetric, general-purpose
baseline-spread definition rather than a lower-tail-specific reference.

The approved recipe is `baseline-mad-robust-z-v1`. For each displayed time row,
use its finite baseline-bin values over **[−15,0) s relative to CS onset**:

```text
B   = median(baseline bins)
MAD = median(abs(baseline bins - B))
S   = MAD / 0.6744897501960817
Z(t) = (P(t) - B) / S
```

The factor is approximately 1.4826 times MAD. Output is dimensionless, with
baseline median zero and normal-calibrated MAD one; neither baseline Q10 = −1
nor ordinary standard deviation = 1 is required. This is a baseline-relative
descriptive score, not a significance statistic or a normality transformation.

Preserve the upstream metric, bout eligibility, log transform, bin estimator,
timing, cohort and aggregation. For single-fish heatmaps, apply the recipe to
each fish/trial's binned log-vigor row. For population heatmaps, retain individual
fish/trial baseline centring and equal-fish pooling, then apply the recipe
separately to each pooled condition/trial row, as in the reviewed Figure 2 B
alternative. Do not move robust scaling before fish pooling. Stratified
population heatmaps use the same order within each declared group. Other
heatmap structures must explicitly map their rows and baseline reference before
using this rule; do not invent a baseline or reinterpret a non-vigor quantity.

Use finite bins only; preserve missing cells, contributing-fish counts and
coverage. A missing baseline or nonfinite/nonpositive S makes the whole row
undefined. Do not insert an epsilon, fill missing values or numerically clip
scores. Record baseline-bin counts and S for each row, and inspect small scales;
no new minimum-bin or small-scale threshold is approved by this decision.
Verify baseline median zero and normal-calibrated MAD one on defined rows.
Legends/colourbars must identify robust normalization and dimensionless units;
colour limits are a separate scoped choice (the B comparison used ±2).

This approves the method for future heatmap work, not a retroactive rewrite of
accepted freezes or an automatic selection of the exploratory B artwork.
Existing frozen bytes and reproduction recipes remain immutable; current
selected artwork/legends continue to describe their actual saved method until
new versions are computed, verified and published through the scientific route.
Implementation is not claimed paper-wide by this decision record. In particular,
stability under whole-fish resampling has not been established by the visual
comparison, and normalization does not change non-heatmap outcomes or inference.

## 2026-10-12 — frozen versions define current

Author instruction: the current versions are the frozen versions. The latest explicitly accepted freeze governs its own scope; an earlier accepted panel remains current until replaced. Other recipes are legacy or proposed. Scientific definitions and accepted limitations stay exactly as frozen; a panel freeze does not automatically approve the full assembly or first release.


Resolve current accepted scopes through [Current figures](figures/README.md)
and the exact [selection records](../../records/frozen-analyses/selections.md).
The chronology below preserves earlier instructions, including superseded
Figure 1 E/F/G/H V12 and Figure 2 rank-test/ratio choices. It does not override
later scoped selections or the separate working-presentation state. Detector,
cohort, learner and release gates remain separate. No new approval is created here.

**2026-10-09 standing author update:** [all analytical vigor is bout-only](../../records/scientific-evidence/decisions/BOUT_ONLY_VIGOR_POLICY_2026-10-09.md). No-bout periods are NaN and ignored before every vigor calculation. This supersedes earlier all-frame vigor definitions; historical/frozen artifacts remain preserved. The D/E mean-ratio and log-median comparison has been rebuilt under this shared rule; D/E version B was subsequently selected and frozen; that selection does not cover other panels.

This document exists to stop the analysis from feeling open-ended. It
collapses the scattered per-step "gate" checklists into one place, records
the actual decisions made, and sets implementation priorities so the full
roadmap is not silently re-litigated per step.

Earlier step plans and their acceptance criteria remain in Git history. Items
outside the immediate priority lane are deferred, not silently declared done,
and can be resumed when the corrected paper-analysis path is further advanced.

Update this file when a decision changes. Do not duplicate its content back
into the individual step plans beyond a short pointer.

## Standing decisions and their chronology

| Gate | Decision |
| --- | --- |
| G0 (scope) | Same core claim and same fish/experiments as the original paper. No scope expansion. |
| T0 (raw tracking semantics) | Raw `angleN` columns are **radians**. Legacy analysis converts them to degrees with `* (180/pi)` before vigor (`data_io.read_tail_tracking_data`, legacy `my_functions`). Candidate metrics keep radians. `angle1..angle14` behave as local intersegment bends (agree with XY-derived segment orientation changes on the pilot); `angle15` is a terminal placeholder. Measured `xN`/`yN` are present and treated as **tracking-image pixels**; absolute µm calibration is **not required** for relative activity metrics. Confidence and independent body-axis fields are absent. Synchronized video remains optional/deferred for blinded validation only. |
| T1 (activity metric) | **Metric frozen by the author on 2026-10-06: tail bend angular speed** (`legacy_distal_angular_speed`, rad/ms), using the existing corrected measured-time formula. All paper activity analyses, assessments and panels follow this choice. Angular L1 and whole-tail XY remain explicit comparison/sensitivity metrics. Evidence, name, formula, limitations and downstream consequences are in [the selection record](../../records/scientific-evidence/decisions/metric-selection-2026-10-06.md). Shared bout segmentation remains unchanged; detector/smoothing policy approval remains separate. |
| C0 (cohort inclusion) | **Primary paper policy remains open; boundary decided.** One `assess-discarding` command runs technical assessment, then exploratory source-linked preprocessing checks and a merged learner-input prerequisite. Primary membership may use only approved label-independent technical criteria. Response-window movement, CR strength, and learner status cannot determine it. The exploratory combined status is a projection on the selected metric, not a historical or primary cohort. Normalized-vigor trial/block filtering is deliberately outside this command; LME trial eligibility and learner model-derived feature failures remain explicit in their respective later analyses. See the [current assessment behavior](methods/recording-quality-and-cohorts.md) and [active cohort work](../../Plans/02_DISCARDING_AND_COHORT_IMPLEMENTATION.md). |
| S (statistics) | **Open in the [analysis and statistics design](../../Plans/06_ANALYSIS_AND_STATISTICS.md).** The existing fish-grouped LME, fish permutation, and fish bootstrap are exploratory engineering scaffolds, not the approved strategy. The primary estimand, outcome family, condition contrast, random-effects structure, diagnostics, uncertainty, multiplicity, and validation status must be decided together. The LME may be improved, demoted to sensitivity, or replaced. The legacy statistical route remains frozen as `legacy-paper` reproduction only—not extended or “fixed.” |
| L (learner analysis) | **Required paper workstream; method open.** Complete the [learner classification and stratified analysis plan](../../Plans/04_FIGURE_3_LEARNER_ANALYSIS.md). Begin with continuous fish-level effects and heterogeneity and compare continuous, longitudinal, probabilistic, and categorical representations. A binary or multiclass classifier is used only if it adds defensible meaning and avoids circular confirmation. If a hard classifier is rejected, the paper still reports the approved continuous or model-based learner result and the reason classes were not imposed. |
| F (figures) | Only two figure modes are actively maintained going forward: static PNG and publication SVG/PDF. Interactive local HTML figures are frozen as-is (already implemented, not broken, not deleted) but receive no further investment. |

**Gate F scope clarification:** this historical limit concerns interactive
analysis renderers. Later author requests explicitly maintain HTML reading,
annotation, selected-figure display and version-comparison/navigation pages.
Those derivatives wrap authenticated saved artwork; they do not authorize new
analytical dashboards, scientific recomputation or replacement of editable
publication masters. See the [figure workflow](figures/shared/iteration-workflow.md#display-and-working-formats)
and [ordered review library](figures/review-library.md).

### Addendum (2026-08-31) — historical pickle timebase

Stage-1 gzip pickles for `20221115_04` and `20221116_12` advance
`Original frame number` within trials at `expected/predicted` (reciprocal of
current interpolate). Package Parquet matches current
`FrameID *= expected/predicted` physics (`predicted/expected` Original slope).
CS onset still agrees; edge misalignment ~0.33 s. Do **not** invert
`legacy-paper` interpolate to match pickles. The underlying finding is also in
[the analysis audit](../../records/scientific-evidence/audits/01_ANALYSIS_FINDINGS.md#h1-historical-per-fish-pickles-warp-within-trial-time-via-reciprocal-original-frame-rate).

## Legacy D/E ratio instructions — superseded by accepted B within that scope

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
[panel review comments](figures/README.md). The D/E instructions do
not change other panels, the separate onset/trajectory review, or literal
`legacy-paper` reproduction. F remains inconclusive without authenticated data.

## Figure element identities and future freezes (2026-10-09)

The author selected a uniform paper style and a durable scientific element
contract for each explicit panel/assembly freeze. The
[single figure guide](figures/shared/iteration-workflow.md#what-happens-when-the-author-freezes)
owns the procedure; the [unified contract](../../configs/paper-figures/figure-elements.json)
owns current machine-readable roles and presentation values. Freeze records bind
the contract version/hash, registry, resolved styles, scoped exceptions, source/data
hashes and final exports. Explicit prior approvals persist within recorded scope;
unresolved exceptions need actual author decisions, distinct from scientific definitions.
The author selected code-enforced checks **only at explicit freeze time**; ordinary
rendering/export does not invoke the gate. Scientific mappings, structure and
unmeasurable renderer properties still require genuine completed review evidence.
The [SciFigEditor review](../analysis/figures/SCIFIGEDITOR_REVIEW_2026-10-09.md)
retains the historical integration rationale. Existing frozen figures, data and
specifications remain unchanged; this does not reactivate the deferred general
artifact-schema registry below.

## What is deferred or minimized in the current priority lane

- **Schema registry / logical-content hashing / typed artifact metadata**: the optional design remains in Git history, with no active implementation step. Current SHA-256 byte hashing and atomic transactional publication are sufficient for the first scoped analysis release. Reopen only a specific part if a concrete need appears.
- **Historical interactive analysis renderers**: frozen, no further polish;
  later-authorized HTML reading/display/version navigation has the separate
  scope explained after Gate F above.
- **The 10-gate formalism as a recurring per-step ritual**: replaced by this single document. Gates are not re-asked per step; they are only revisited if new information changes a decision (for example, real cohort numbers when deciding Gate C0).
- **Categorical learner implementation**: final implementation waits for stable
  preprocessing, cohort, and outcomes, but the learner-method review and design
  are active and required (see Gate L).
- **Behavior-dependent legacy discard as primary QC**: rejected. Its exact
  checks remain available only as a named sensitivity analysis under the
  [integrated cohort/CR-profile plan](../../Plans/05_FIGURE_4_CR_PROFILES.md).
- **First analysis release**: release main Figures 1–4 when their code, scientific decisions, analysis and evidence are ready. Supplementary composition, tail mechanisms and imaging are separate. Preserve a tagged commit and frozen manifest linking inputs, cohort, configuration, results, and figures; candidate and legacy runs are analysis evidence, not interim releases. See [the final release plan](../../Plans/08_FIRST_RELEASE_FIGURES_1_TO_4.md).
- **Expansion of legacy characterization**: lower priority because the existing characterization (see [the codebase behavior map](../../records/scientific-evidence/legacy/02_CODEBASE_BEHAVIOR_MAP.md) and [analysis findings](../../records/scientific-evidence/audits/01_ANALYSIS_FINDINGS.md)) is sufficient for the immediate path. Additional characterization remains in scope when required by equivalence testing or a concrete migration risk.

## Figure 2 B D/E selection and freeze — 2026-10-09

Joaquim authorized correcting the statistical tests and freezing the selected version B. D/E now use positive valid bout-frame median natural-log response minus baseline, followed by a median of eligible trials per fish/block. No-bout and invalid frames are excluded before every calculation; empty windows remain undefined. Scientific windows, cohorts and eligibility thresholds are unchanged.

Exact two-sided sign tests handle paired fish changes and zero-reference comparisons; Brunner–Munzel t tests handle independent conditioned/control rank/probability comparisons without requiring equal shapes. One Holm36 family covers all D/E comparisons. Fish independence is assumed; day/tank adjustment and exploratory selection remain explicit limitations. Delay effects survive correction; no Trace comparison survives. Earlier Wilcoxon/Mann–Whitney results are superseded, preserved historical evidence.

[D freeze](../../records/repository-archive/ARTIFACT_INDEX.md#artifact-2243b6469ea3) and [E freeze](../../records/repository-archive/ARTIFACT_INDEX.md#artifact-84915add0333) passed the explicit specification 1.1.1 gate. [Scoped selection](../../configs/paper-figures/selections/figure2-DE-B-freeze-20261009.json) records the authoritative panel exports and hashes. This scope does not freeze F or the whole assembly. Existing whole-figure assembly and historical freezes remain preserved. Future assembly must accommodate 54.9 × 53.2 mm per panel to retain effective fonts/strokes.

[Processing handoff](../../records/figure-history/shared/bout-log-processing-and-adaptation-history-2026-10-09.md) supplies the shared scaffold for Figure 1 E–H and whole Figure 2. Other panels require their own estimator, source coverage, statistical scope and freeze; existing Figure 1 quantile-scaled definitions are not replaced by this selection.

### 2026-10-09 — G selected Historical LogMedian freeze
Author explicitly selected native bout-only Historical LogMedian with D/M/R and phase-aware LMM trial statistics. See configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json. No second log or smoothing. Test3 local M/R n/a; exploratory, no onset claim. All alternatives consolidated in reviews/figure2_delay_all_versions_20261009.html; verified stale Delay copies removed. Whole assembly and D/E freezes unchanged.


## 2026-10-10 — release and storage scope

Author-approved cleanup: the first release contains main Figures 1–4 and necessary scientific evidence. Supplementary figure composition, tail mechanistic work and imaging remain later/separate work; necessary main-claim validation is retained. This supersedes earlier wording requiring one final release after every follow-on analysis.

Keep supported pipeline/tests/tools and exact frozen definitions/settings/manifests/code in the repository. Move figures, visual HTML archives, numerical payloads and unconsumed alternative code to JOAQUIM only after verified transfer. The author explicitly deferred transfer until the disconnected SSD is available. Existing freezes are neither restyled nor refrozen by this organizational decision.

## Fish-selection review scope — 2026-10-11

The author requested: “combine all steps of discarding from legacy.... we will review them later”. [Plan 02](../../Plans/02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) is the single owner, before Analysis and Statistics. Its combined legacy review includes preprocessing screening, discard-list propagation, normalized-vigor plotting/model filters, paired-block eligibility and all learner variants. Collection does not authorize applying all rules jointly, change any threshold or freeze paper membership. The label-independent technical primary boundary remains decided; C0 numerical criteria and C1 reviewed paper cohort remain open pending later review.


## Training US numbering and protocol verification — 2026-10-31

The author confirmed that all 18 pre-training USs are retained and the 19th US
starts training. Active one-based training US IDs are 19–64; legacy 18–63 labels
must not select events in new analysis. No first-US discard is introduced.
New temporal/trial outcomes bind the corrected trial map; prior frozen results
and legacy source remain immutable. Protocol verification uses ±0.2 s for CS
and paired-US timing and warns while continuing, without changing cohort
membership. Shared acquisition facts and diagnostic settings live in
`src/classical_conditioning/config/protocols.py`; operational behavior and
rebuild requirements are in [the pipeline guide](../running-analysis/run-pipeline.md#automatic-protocol-verification).

Read-only source check: Delay `20221115_04` (stimulus-log SHA-256
`2a24ffa397fc089196784de42c869983aaf30d4df8405c8938257f006df8d21f`)
and fixed 3sTrace `20230306_01` (`e443d70e499098e450bd5689b42b9cd356143e48cfa0607a6f4cd191a591db54`)
on JOAQUIM each contain 94 CSs and 79 USs, with 46 matched training pairings
beginning at US 19. One control from each assay remained unpaired-compatible;
recorded pulse-duration deviations were retained as warnings. Fixed 10sTrace
coverage for this code change is synthetic; no paper dataset was regenerated.


### Exact stimulus totals — author clarification, 2026-10-31

For the supported Delay, fixed 3sTrace and fixed 10sTrace assays and their
controls, every complete recording must contain exactly 94 CS events and 79 US
events (78 regular plus the final viability pulse). There is no minimum-count
acceptance rule. Both totals must always be confirmed against the configured
expectation, including resumed preprocessing. Missing and additional events are
reported as discrepancies under the existing warn-and-continue policy. Runtime
configuration and exports use `expected_cs_count` and `expected_us_count`;
legacy minimum-count names/values remain only in historical reproduction inputs.
