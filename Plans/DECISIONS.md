# Consolidated scientific and scope decisions

This document exists to stop the analysis from feeling open-ended. It
collapses the scattered per-step "gate" checklists into one place, records
the actual decisions made, and sets implementation priorities so the full
roadmap is not silently re-litigated per step.

Nothing in this document deletes, supersedes, or weakens the original plans.
All original step plans and their acceptance criteria remain preserved. Items
outside the immediate priority lane are deferred, not discarded, and can be
resumed when the corrected paper-analysis path is further advanced.

Update this file when a decision changes. Do not duplicate its content back
into the individual step plans beyond a short pointer.

## Decisions made (2026-08-30)

| Gate | Decision |
| --- | --- |
| G0 (scope) | Same core claim and same fish/experiments as the original paper. No scope expansion. |
| T0 (raw tracking semantics) | Raw `angleN` columns are **radians**. Legacy analysis converts them to degrees with `* (180/pi)` before vigor (`data_io.read_tail_tracking_data`, legacy `my_functions`). Candidate metrics keep radians. `angle1..angle14` behave as local intersegment bends (agree with XY-derived segment orientation changes on the pilot); `angle15` is a terminal placeholder. Measured `xN`/`yN` are present and treated as **tracking-image pixels**; absolute µm calibration is **not required** for relative activity metrics. Confidence and independent body-axis fields are absent. Synchronized video remains optional/deferred for blinded validation only. |
| T1 (activity metric) | Do **not** freeze a single metric yet. All 5 candidate metrics (manuscript segment angular-speed sum, all-segment angular RMS, whole-tail XY RMS speed, whole-tail XY mean speed, curvature-change RMS) run through the full pipeline (detection, outcomes, statistics, figures) generically, producing a side-by-side comparison report. The paper will report whichever metric(s) hold up under that comparison. |
| C0 (cohort inclusion) | **Deferred.** For now, do not discard any fish based on per-block trial-count completeness. Only basic technical QC (raw data present, passes acquisition integrity checks) gates inclusion. The legacy rule used OR logic across blocks (a fish is kept if it clears the minimum trial count in *any one* required block, not *every* required block) — this is a known bug, not a design choice. The fix (require every block) will be evaluated later at Step 08 with real cohort numbers in front of the user, comparing cohort size/composition under both rules before deciding. |
| S (statistics) | **Engineering default (revisable in Step 10.0):** corrected primary analysis uses a simple mixed-effects model with fish as a random effect, plus a holdout/cross-validation sanity check, implemented once and applied identically to all 5 candidate metrics. **Before Gate S freeze**, run the statistics-methodology workshop ([Notes/STATISTICS_METHODOLOGY_WORKSHOP.md](./Notes/STATISTICS_METHODOLOGY_WORKSHOP.md); Step 10.0): criticize legacy Mann-Whitney/per-trial LME/ratio/bootstrap practice **and** the default LME itself, and consider drastically different families (fish permutation, Bayes, GEE, functional/GAM, bout point-process, HMM, design-based, multivariate, predictive). The legacy statistical route remains frozen as `legacy-paper-v1` reproduction only — not extended or “fixed.” |
| L (learner classification) | **Deferred, not required for the metric comparison.** No learner classification work is needed to compare the 5 candidate metrics or report the population-level conditioning effect. When revisited, it folds into the existing post-refactor workshop (see [Plans/steps/11_LEARNER_CLASSIFICATION.md](./steps/11_LEARNER_CLASSIFICATION.md) Work Package 11.0), which now also includes brainstorming alternatives to and criticism of the original single-fish learner classification approach (not just PCA/power as candidate tools, but questioning the approach itself). |
| F (figures) | Only two figure modes are actively maintained going forward: static PNG and publication SVG/PDF. Interactive local HTML figures are frozen as-is (already implemented, not broken, not deleted) but receive no further investment. |

### Addendum (2026-08-31) — historical pickle timebase

Stage-1 gzip pickles for `20221115_04` and `20221116_12` advance
`Original frame number` within trials at `expected/predicted` (reciprocal of
current interpolate). Package Parquet matches current
`FrameID *= expected/predicted` physics (`predicted/expected` Original slope).
CS onset still agrees; edge misalignment ~0.33 s. Do **not** invert
`legacy-paper-v1` interpolate to match pickles. Details:
[Archive/Notes/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md](./Archive/Notes/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md).

## What is deferred or minimized in the current priority lane

- **Schema registry / logical-content hashing / typed `ArtifactRef` metadata** (Step 02): deferred, not removed from the full plan. Current SHA-256 byte hashing and atomic transactional publication are sufficient for the immediate analysis path.
- **Interactive HTML figures**: frozen, no further polish (see Gate F above).
- **The 10-gate formalism as a recurring per-step ritual**: replaced by this single document. Gates are not re-asked per step; they are only revisited if new information changes a decision (e.g. real cohort numbers at Step 08 for Gate C0).
- **Learner classification**: fully deferred (see Gate L above), not scheduled as near-term work.
- **Formal immutable release packaging (Step 13)**: full implementation is deferred until the analysis is ready for release. A tagged commit plus a frozen cohort/config/data-hash manifest is sufficient for interim milestones; the complete Step 13 plan remains preserved.
- **Expansion of legacy characterization**: lower priority because the existing characterization (see [docs/analysis/CODEBASE_BEHAVIOR_MAP.md](../docs/analysis/CODEBASE_BEHAVIOR_MAP.md) and [docs/analysis/ANALYSIS_ISSUES.md](../docs/analysis/ANALYSIS_ISSUES.md)) is sufficient for the immediate path. Additional characterization remains in scope when required by equivalence testing or a concrete migration risk.

## Practical shortened critical path

Given the decisions above, the realistic remaining path to a corrected,
defensible result is:

1. Step 03 (domain/configuration) — small, mechanical.
2. Step 04/05 (ingestion + legacy equivalence) — implement and debug against
   local fixtures; expand coverage when more raw recordings are available.
3. Step 06/07 (corrected preprocessing + the 5 candidate metrics) —
   implement generically so all 5 run the same way; local fish are debug
   fixtures, not selection/confirmation cohorts.
4. Step 08 (cohort assembly) — this is where the Gate C0 per-block inclusion
   question gets revisited with real numbers.
5. Step 09/10 (outcomes + the mixed-effects statistics route) run identically
   for each of the 5 metrics, producing one comparison report.
6. Step 12 (PNG + publication figures only) for the comparison and final
   results.
7. A lightweight Step 13 (tagged commit + frozen manifest), not a full
   release-engineering exercise, unless requested.

Local recordings currently on disk (`20221115_04`, `20221116_12`) are the
**real fixtures for end-to-end pipeline tests**, including cohort freeze and
multi-recording runners. They are not paper-scale N and do not authorize
Gate T1/C0 scientific decisions. Synthetic clone fish were removed once these
two recordings were available.

Step 00 governance evidence remains required before final paper-authoritative
claims and release. It is not required to block every engineering increment.
Step 11 learner classification remains fully preserved but deferred until the
non-learner analysis and five-metric comparison are complete.
