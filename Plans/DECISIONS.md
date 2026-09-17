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
| T1 (activity metric) | Do **not** freeze a single metric yet. **Three metrics run** through the candidate pipeline generically: tail-length-weighted angular L1, tail-length-normalized whole-tail XY mean speed, and the legacy-derived distal cumulative-angle-speed benchmark on measured time. The legacy metric is a historical benchmark, not itself a selection winner. **Bout segmentation is shared across all three metrics** so comparisons use the same behavioral episodes; no metric-specific detector is created. The shared detector source, smoothing, thresholds, and validity rules still require a later T1 decision before paper use. The unweighted manuscript segment-speed sum, all-segment angular RMS, whole-tail XY RMS, and curvature-change RMS are superseded because they are respectively sampling-density-dependent, unnecessarily peak-weighted, or redundant/noise-sensitive alternatives. |
| C0 (cohort inclusion) | **Deferred.** For now, do not discard any fish based on per-block trial-count completeness. Only basic technical QC (raw data present, passes acquisition integrity checks) gates inclusion. The legacy rule used OR logic across blocks (a fish is kept if it clears the minimum trial count in *any one* required block, not *every* required block) — this is a known bug, not a design choice. The fix (require every block) will be evaluated later at Step 08 with real cohort numbers in front of the user, comparing cohort size/composition under both rules before deciding. |
| S (statistics) | **Open in the [analysis and statistics design](./Analysis/1_ANALYSIS_AND_STATISTICS.md).** The existing fish-grouped LME, fish permutation, and fish bootstrap are exploratory engineering scaffolds, not the approved strategy. The primary estimand, outcome family, condition contrast, random-effects structure, diagnostics, uncertainty, multiplicity, and validation status must be decided together. The LME may be improved, demoted to sensitivity, or replaced. The legacy statistical route remains frozen as `legacy-paper-v1` reproduction only—not extended or “fixed.” |
| L (learner analysis) | **Required paper workstream; method open.** Complete the [learner classification and stratified analysis plan](./Analysis/2_LEARNER_CLASSIFICATION.md). Begin with continuous fish-level effects and heterogeneity and compare continuous, longitudinal, probabilistic, and categorical representations. A binary or multiclass classifier is used only if it adds defensible meaning and avoids circular confirmation. If a hard classifier is rejected, the paper still reports the approved continuous or model-based learner result and the reason classes were not imposed. |
| F (figures) | Only two figure modes are actively maintained going forward: static PNG and publication SVG/PDF. Interactive local HTML figures are frozen as-is (already implemented, not broken, not deleted) but receive no further investment. |

### Addendum (2026-08-31) — historical pickle timebase

Stage-1 gzip pickles for `20221115_04` and `20221116_12` advance
`Original frame number` within trials at `expected/predicted` (reciprocal of
current interpolate). Package Parquet matches current
`FrameID *= expected/predicted` physics (`predicted/expected` Original slope).
CS onset still agrees; edge misalignment ~0.33 s. Do **not** invert
`legacy-paper-v1` interpolate to match pickles. Details:
[Archive/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md](./Archive/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md).

## What is deferred or minimized in the current priority lane

- **Schema registry / logical-content hashing / typed artifact metadata** ([semantic provenance plan](./Deferred/SCHEMA_SEMANTIC_PROVENANCE_AND_LEGACY_CONVERSION.md)): deferred, not removed from the full plan. Current SHA-256 byte hashing and atomic transactional publication are sufficient for the immediate analysis path.
- **Interactive HTML figures**: frozen, no further polish (see Gate F above).
- **The 10-gate formalism as a recurring per-step ritual**: replaced by this single document. Gates are not re-asked per step; they are only revisited if new information changes a decision (for example, real cohort numbers when deciding Gate C0).
- **Categorical learner implementation**: final implementation waits for stable
  preprocessing, cohort, and outcomes, but the learner-method review and design
  are active and required (see Gate L).
- **Formal immutable release packaging**: full implementation is deferred until the analysis is ready for release. A tagged commit plus a frozen cohort/config/data-hash manifest is sufficient for interim milestones; the complete release plan remains active.
- **Expansion of legacy characterization**: lower priority because the existing characterization (see [the codebase behavior map](../docs/analysis/legacy/CODEBASE_BEHAVIOR_MAP.md) and [analysis findings](../docs/analysis/audits/ANALYSIS_FINDINGS.md)) is sufficient for the immediate path. Additional characterization remains in scope when required by equivalence testing or a concrete migration risk.

## Practical shortened critical path

Given the decisions above, the remaining path to a corrected, defensible
result is (fixture-scoped Steps 03–05 are already complete; see
[IMPLEMENTATION_STEP_INDEX.md](./IMPLEMENTATION_STEP_INDEX.md)):

1. Complete the paper baseline evidence that still matters for release;
   deferred schema/logical-hash work stays deferred.
2. Approve corrected preprocessing and the three-metric candidate set —
   keep the pipeline generic so all three run the same way; local fish are
   debug fixtures, not selection/confirmation cohorts. Gate P and Gate T1
   remain open.
3. Cohort assembly — revisit Gate C0 per-block inclusion with
   real paper-scale numbers when available.
4. Outcomes and statistics — run identically across the three-metric set;
   freeze Gate O/S after the design record is decided.
5. Complete required learner analysis and its validation mode.
6. Build PNG + publication figures only; interactive HTML remains frozen.
7. Create a lightweight tagged release with a frozen manifest, not a full
   release-engineering exercise, unless requested.

Local recordings currently on disk (`20221115_04`, `20221116_12`) are the
**real fixtures for end-to-end pipeline tests**, including cohort freeze and
multi-recording runners. They are not paper-scale N and do not authorize
Gate T1/C0 scientific decisions. Synthetic clone fish were removed once these
two recordings were available.

Step 00 governance evidence remains required before final paper-authoritative
claims and release. It is not required to block every engineering increment.
Learner source plans remain preserved in the archive. Their operational content
is restored in the active learner plan, and learner analysis is required for
the paper even though the categorical-versus-continuous representation remains
open.
