# Individual-learning interfaces and validation

The scientific choice belongs to [individual-learning methods](../paper/methods/individual-learning.md) and [Plan 04](../../Plans/04_FIGURE_3_LEARNER_ANALYSIS.md). There is no approved categorical representation merely because a legacy classifier or schema exists.

The implementation contract requires authenticated experiment/condition/fish identity; cohort and policy hashes; metric, detector and recipe identity; source/code identity; scores or trajectories and uncertainty; explicit eligibility and reasons; and disjoint construction/evaluation trials where claims require independence. Categorical fields and label-based strata are conditional on Gate L.

Joins into derived frame and trial tables must be many-to-one, reject duplicate keys, report unmatched fish, preserve controls and insufficient-information fish, and leave the reviewed primary cohort unchanged. Aggregate samples to trials, within fish, then across fish. Carry coverage, contributing fish/trials, event/window identity and fish-level uncertainty with outputs.

Same-data strata are descriptive. Held-out or cross-fitted validation must exclude evaluation data from score construction, calibration, thresholds, eligibility and labels. Test identity, fixed seeds, joins, cohort preservation, selectors, coverage, aggregation and leakage.

The full [implementation design](../../records/scientific-evidence/designs/individual-learning-implementation.md) and [outputs/validation design](../../records/scientific-evidence/designs/individual-learning-outputs-and-validation.md) preserve all original schemas, conditional examples, output families and acceptance requirements. They are design evidence, not current classifier selection or a second progress board. Remaining work is owned by Plans 04 and 05; Figure 3 owns representation/validation presentation, and Figure 4 owns downstream response-profile presentation.
