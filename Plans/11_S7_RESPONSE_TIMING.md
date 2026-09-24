# S7 independent response timing

**Status:** Planned follow-on. Descriptive Figure 4 block and catch curves do not depend on this analysis.

## Inputs and estimands

Use the authenticated Figure 4 trial-bin and fish-bin tables, the frozen learner manifest, recorded paired-training US times, and explicitly identified evaluation trials. Analyze anticipatory responses only before the expected US. No-bout signed-vigor bins remain missing; movement probability provides the complementary occurrence outcome.

Before fitting, freeze numerical definitions for response onset, peak, negative-response center and offset, plus a rule for traces with no measurable response. Report each timing estimate relative to both CS onset and expected US. Do not estimate timing from post-US paired-training activity as if it were a conditioned response.

## Independent evaluation

Require evaluation trials that did not determine the corresponding fish label, or a cross-fitted/independent cohort design. Verify the classifier training and evaluation trial sets are disjoint for each fish. Re-estimate uncertainty by resampling fish, with trials resampled within fish for pooled groups. Compare Delay and 3sTrace only after the recorded 3sTrace US time agrees with the approved experiment definition. Include controls, nonlearners, all-fish and alternate-metric sensitivity results.

## Deliverables and gate

Save fish-level timing estimates, group summaries, uncertainty draws, trial-set provenance, model diagnostics and an S7 figure with its source hashes. Keep Figure 4 free of timing p-values or onset/offset claims until these definitions, independence checks and diagnostics are reviewed. A missing response or unresolved US timestamp yields an explicit missing reason rather than a fabricated time.
