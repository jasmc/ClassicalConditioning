# Content and evidence design

Read when planning a substantial guide. Select what serves the user's decision;
do not reproduce this entire outline mechanically.

## Establish a reliable inventory

For each version or method, record:

| Field | What to establish |
|---|---|
| Identity | Method/version ID, source/config/run identity and selected/historical role |
| Status | Discussed, code present, tested, run, validated, published and/or selected |
| Question | Scientific target and allowed interpretation |
| Inputs | Cohort, condition/experiment, metric, transformation, windows and units |
| Selection | Technical exclusions, trial eligibility, feature/model requirements |
| Estimation | Aggregation, fixed/random structure, reference, smoothing/pooling |
| Uncertainty | Target, covariance terms, resampling unit, dependence, multiplicity |
| Decision | Threshold/direction/conjunction and what a label actually means |
| Results | Verified outputs, counts/denominators, missing cases, warnings/failures |
| Limits | Validation mode, assumptions, sensitivity and unresolved evidence |

A method can have several status attributes: a published exploratory run may
remain scientifically unvalidated. A default in code does not establish the
active setting in a saved run. A historical label is not ground truth.

Bind each reported count to its denominator, input generation and eligibility
policy. Reconcile flagged, eligible and unclassified totals where the saved
decision table permits it. Similarly, distinguish cohort size from trial/phase
contributors, resamples requested from accepted, and independent simulated
datasets from multiple models fitted to each dataset. If a complete table is
missing but a saved sidecar preserves annotations, identify the partial evidence
without inventing the missing non-rejections.

For discussions spanning revisions, make a short chronology: what was requested,
what was built, what the author corrected, what remains selected and what was
started but not completed. Preserve the distinction between the author's choice
and the assistant's recommendation. Background studies consulted during the chat
are not automatically analyses performed in that chat.

## A reading path that supports decisions

Useful content modules include:

- Orientation: current products, completion gaps and how to read the guide.
- Scientific questions: population change, individual expression, prediction and
  evidence for categories are different questions.
- Data lineage/protocol: observation units, inventory, time/trial mappings and
  relevant event definitions.
- Quantities: metric versus outcome; transform and aggregation order; zero,
  immobility, undefined conditional outcomes and unavailable observation.
- Guided output reading: saved figures plus matching legends and case-level
  evidence. Explain graphical encodings before showing classification badges.
- Method cards/comparison: exact recipes and actual settings/results.
- Exclusion ledger: original inventory → technical cohort → trial outcomes →
  feature/model eligibility → classifiable/uncertain cases → illustrated cases.
- Iterations: why a draft or method changed and what the change did not establish.
- Statistical ideas: layered explanations and clearly synthetic demonstrations.
- Validation/sensitivity: assumptions, failures, leakage and independence.
- Decision worksheet: analysis choice, evidence needed, panel roles and unresolved
  scientific decisions before any eventual explicit freeze.
- Evidence appendix: source paths/hashes, code/config/logs, glossary and optional
  primary statistical references.

Give each major explanation a plain-language meaning, its exact calculation,
and its limits. Optional technical boxes can supply equations, a worked example
and the source function/configuration. Essential warnings belong beside the
relevant result, not only in an appendix.

For a history-heavy request, add a compact topic coverage map linking requested
questions to sections and sources or explicit gaps. Show important saved output
variants with their own status and captions. Use aggregate tables and a few
examples when individual-case detail is not wanted; do not turn an exhaustive
methods review into an unrequested atlas of every subject.

## Statistical distinctions worth teaching when relevant

Choose these based on the study, not as a universal checklist:

- Means, medians, differences of logs and logs of arithmetic-mean ratios can have
  different targets. A change attributed to shrinkage may actually change the
  estimand; use a matched unpooled comparator when available.
- A BLUP random deviation alone is not a fish/subject's total fixed-plus-random
  effect. Conditional random-effect variance is not automatically full prediction
  uncertainty. Estimated covariance/reference uncertainty and dependence matter.
- Acquisition and recovery can define different biological patterns. Requiring
  both is a conjunction to explain, not a generic definition of learning.
- Raw/observed contrast, population mean effect and individual causal effect are
  different quantities. Control comparison does not observe each subject's
  counterfactual.
- Availability and precision are different. Retaining every estimable case does
  not prove adequate information. Missingness can depend on the phenotype.
- A control percentile evaluated on its construction sample is not independent
  false-positive calibration. A score/rank is not a probability of a biological
  class without a justified model/calibration interpretation.
- Pointwise, simultaneous and predictive intervals protect different statements.
  A smooth fit, unit tests or numerical agreement do not prove interval coverage.
- Same-data illustrations, prespecified holdouts, retrospectively inspected data
  and genuinely independent validation must be labeled differently.
- "No label assigned" is not a count of zero positives. "Criterion not met" is
  not evidence of equivalence or absence of learning.

A toy explanation must state its simplifying assumptions and avoid looking like
an additional fit to the real data. Keep toy controls isolated from actual
results, files, labels and decision state.

## Offline implementation and notes

Prefer inline CSS/JS, system fonts, embedded resources and no CDN libraries for
an offline guide. Keep optional bibliographic links separate from runtime
resources. Isolate IDs in embedded SVGs (for example, data-URI images) to prevent
resource collisions; retain original matching legends and source mappings.

Long guides benefit from a sidebar and accessible headings/details. Make wide
tables scroll within their container, not across the page. Include useful alt
text. Avoid injecting enormous inventories into the initial visible layout when
a case selector can retain complete coverage more clearly.

Notes should have stable field IDs, a versioned export schema, transparent local
persistence and an explicit export. Catch storage failures and allow export even
when persistence is unavailable. Import uses text/value assignment, not HTML
execution. A changed guide should restore compatible existing fields rather than
silently clearing notes. Notes are advisory user content; they do not modify
scientific selections or trigger actions.

For the fuller chapter-search, notebook, import-backup and print behavior used
in the Figure 2 G guide, read [html-behavior.md](html-behavior.md). Its optional
browser script is a reusable interaction helper, not a scientific renderer or
a mandatory document template.

## Optional portable provenance format

Use existing project schemas when present. For a standalone guide without one,
`scripts/check_guide.py` accepts:

```json
{
  "guide": {"path": "/absolute/guide.html", "sha256": "64 hex characters"},
  "sources": [
    {"path": "/absolute/saved-output.csv", "sha256": "64 hex characters"}
  ],
  "protected": [
    {"path": "/absolute/current-state.json", "sha256": "64 hex characters"}
  ]
}
```

Paths may be relative to the manifest's directory. Use real hashes. The checker
only reads files. Sources establish consumed-byte identity; protected entries
show unchanged bytes at verification time. Neither authorizes writes, certifies
method adequacy, nor chooses current authority. Record source role/status and
limitations in additional fields when helpful.
