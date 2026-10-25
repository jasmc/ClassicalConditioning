# Paper Input Validation and Reproducibility

## Purpose and boundary

Authenticate the paper-scale inputs and reproducible processing state for the first Figures 1–4 release. Existing pipeline, provenance and fixture infrastructure are implemented; this plan owns the remaining paper-scale evidence, not a fresh implementation from zero. [Plan 02](02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) owns fish inclusion/eligibility policy and the reviewed cohort; [Plan 06](06_ANALYSIS_AND_STATISTICS.md) owns outcome/inference choices.

## Remaining evidence

1. Produce the expected paper recording inventory by experiment/condition, with source paths/hashes and missing, incomplete, ambiguous and unavailable cases. Reconcile all raw triplets against that denominator; keep external source data in place.
2. Identify the exact code revision, locked environment and resolved input/processing configuration for the reference run. Authenticate existing outputs before scheduling any regeneration; record source, configuration, stage and output fingerprints.
3. Validate corrected measured-time preprocessing, synchronization/frame gaps and shared detector behavior across the paper inputs. Use the author-selected tail bend angular speed metric and approved bout-only vigor sampling; detector/smoothing validation does not reopen the selected metric or historical panel freezes.
4. Reconcile protocol/timing evidence, including expected 3sTrace US events versus observed events. Record coverage and unresolved mappings explicitly; missing inputs or processing failures are not automatically biological exclusions.
5. Deliver one traceable input-validation report and authenticated stage identities to Plan 02 and the outcome/inference owners. Distinguish full paper validation from fixture tests and provisional scientific runs.

## Saved-analysis provenance follow-up

Use the [G/H/I audit's exact source manifest and gaps](../records/scientific-evidence/analyses/figure-02-ghi-statistical-audit.json)
to close original-run reproducibility gaps alongside paper input validation:

- Recover the exact H/I driver and phase-helper revisions still missing from
  supported recovery, plus the G imported fitter/runtime closure. Search by
  recorded identity/hash; extend the existing recovery index only after exact
  verification. Reuse the 86 G artifact aliases and recovered H/I fitter.
- Recover the original producer binding for I's saved bootstrap draw file, if
  available. Its audit snapshot and successful index/percentile replay do not
  establish that historical binding.
- Establish the historical environment/library identities and effective fit
  settings not recorded in saved diagnostics, including optimizer iteration
  defaults. Do not substitute current defaults or rewrite old expected hashes.
- Document exhausted searches and unrecoverable evidence honestly. If an
  independent reproduction is required, prepare it separately on authenticated
  inputs under the owning scientific plan; compare it with the preserved result
  and seek the necessary scientific acceptance. A new run cannot repair the
  identity of the original run retrospectively.

Acceptance for this follow-up is an authenticated source/runtime/producer map,
or explicit reviewed limitations and a separately traceable reproduction route.
It is not another general repository/SSD cleanup audit.

## Acceptance

The expected recording inventory, source/stage identities, code/configuration/environment, timing reconciliation and preprocessing/detector validation evidence are mutually traceable at paper scale. Every unresolved case has a disposition and owner. Existing evidence is reused where authenticated; this plan does not authorize recomputation or refreezing during repository cleanup. Progress is tracked only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md); enduring rules live in [governance](../docs/project-maintenance/governance.md).
