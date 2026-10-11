# Paper Input Validation and Reproducibility

## Purpose and boundary

Authenticate the paper-scale inputs and reproducible processing state for the first Figures 1–4 release. Existing pipeline, provenance and fixture infrastructure are implemented; this plan owns the remaining paper-scale evidence, not a fresh implementation from zero. [Plan 02](02_COHORT_IMPLEMENTATION.md) owns fish inclusion/eligibility policy and the reviewed cohort; [Plan 03](03_ANALYSIS_AND_STATISTICS.md) owns outcome/inference choices.

## Remaining evidence

1. Produce the expected paper recording inventory by experiment/condition, with source paths/hashes and missing, incomplete, ambiguous and unavailable cases. Reconcile all raw triplets against that denominator; keep external source data in place.
2. Identify the exact code revision, locked environment and resolved input/processing configuration for the reference run. Authenticate existing outputs before scheduling any regeneration; record source, configuration, stage and output fingerprints.
3. Validate corrected measured-time preprocessing, synchronization/frame gaps and shared detector behavior across the paper inputs. Use the author-selected tail bend angular speed metric and approved bout-only vigor sampling; detector/smoothing validation does not reopen the selected metric or historical panel freezes.
4. Reconcile protocol/timing evidence, including expected 3sTrace US events versus observed events. Record coverage and unresolved mappings explicitly; missing inputs or processing failures are not automatically biological exclusions.
5. Deliver one traceable input-validation report and authenticated stage identities to Plan 02 and the outcome/inference owners. Distinguish full paper validation from fixture tests and provisional scientific runs.

## Acceptance

The expected recording inventory, source/stage identities, code/configuration/environment, timing reconciliation and preprocessing/detector validation evidence are mutually traceable at paper scale. Every unresolved case has a disposition and owner. Existing evidence is reused where authenticated; this plan does not authorize recomputation or refreezing during repository cleanup. Progress is tracked only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md); enduring rules live in [governance](../docs/maintenance/ANALYSIS_GOVERNANCE.md).
