# Technical Assessment Before Cohort Review

> **Archived incomplete on 2026-09-23 for plan cleanup.** The command behavior
> is documented in the [discarding assessment guide](../../docs/analysis/DISCARDING_ASSESSMENT.md).
> Numerical paper policy approval and cohort freeze remain open in the
> [active cohort plan](../01_COHORT_IMPLEMENTATION.md).

**Status:** Implemented as an auditable draft; numerical paper policy and reviewed cohort remain open.

After per-fish corrected frames, metrics, bouts, temporal profiles, and trial
outcomes, `assess-discarding` begins with the complete raw recording inventory.
It retains incomplete, ambiguous, requested-but-undiscovered, and failed
recordings in its technical table. It records identity/condition, tracking-header
schema, processing status, authenticated artifact lineage, matched and valid
frame counts, and protocol timing evidence.

A policy file may specify `approval_status`, `min_matched_frames`, and
`min_valid_frame_fraction`. The default policy is a draft evidence audit, not
paper approval. An approved policy also needs `approved_by` and `approved_at`.
Only technical evidence may inform the reviewed primary-cohort manifest; no
CR strength, US bout, or learner feature may change that population. The
command records a candidate disposition but does not freeze a cohort.

The output bundle has stable paths under `Processed data/Discarding/<analysis_id>/`
and authenticates its inputs with an assessment hash. Processing failures stay
visible; rerunning after a source, policy, or metric change atomically replaces
derived assessment files at those paths. Reviewed cohort manifests remain
immutable. Raw and processed inputs are never moved or deleted.

The focused command is `classical-conditioning assess-discarding --raw-dir ...
--project-dir ... --analysis-id ... --experiment ... --metric ...`. The routine
pipeline calls the same assessment after per-fish outcomes and before cohort
comparison. See [the exploratory second stage](../../docs/analysis/DISCARDING_ASSESSMENT.md#exploratory-stage)
and [the cohort boundary](./SINGLE_COHORT_AND_EXCLUSION.md).
