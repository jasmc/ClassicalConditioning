# Technical cohort CSV schema

The preserved `configs/allDelay-full-v1-cohort.csv` contains 57 fish (29 Delay, 28 control), marked technically approved by `Codex-technical-all-complete`. It does not establish final author-approved paper inclusion.

The full original table is external. Resolve it with:

```python
from classical_conditioning.external_artifacts import resolve_artifact
csv_path = resolve_artifact("configs/allDelay-full-v1-cohort.csv")
```

Use BOM-aware UTF-8 decoding (`utf-8-sig`). Original columns, in order:

- `experiment_id`
- `recording_id`
- `fish_id`
- `condition_id`
- `technical_valid`
- `technical_exclusion_reason`
- `behavioral_engagement`
- `behavioral_engagement_reason`
- `primary_included`
- `sensitivity_population_ids`
- `review_status`
- `reviewer`
- `reviewed_at`
- `source_qc_artifact_id`

Keep exclusions and their reasons when building a reviewed cohort; follow the supported freeze-cohort schema and validation. This guide introduces no new cohort selection or approval. The [verified transfer inventory](../../maintenance/archive/configuration-cleanup-20261011/transfer-inventory.json) records exact bytes and SHA-256.
