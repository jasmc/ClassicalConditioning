# Delay cohort: learner and Figure 4 readiness (2026-09-24)

## Authenticated inputs and completed analysis

The original project is `F:\Digested Data\allDelay-full-v1`. The copy on `J:` is not an interchangeable analysis root: its saved summaries name absolute `F:` paths. The frozen `allDelay-full-v1` technical cohort contains 57 fish (29 delay, 28 control), with cohort hash `9ef4b9297c939e0d34a99a4d606d0f9c8c2a42a0a7b9aee20d947823ef66f2d5`. It is an all-complete technical cohort, not a paper-approved inclusion review.

The existing learning-onset run `allDelay-full-learning-onset-v1` uses CS-aligned `total-activity` from `tail_length_weighted_angular_l1`, 499 fish bootstrap replicates, 9,999 permutations, and a three-trial persistence rule with `delta_min = 0`. Its simultaneous band **did not localize an onset** (`threshold_never_exceeded`); 34.1% of bootstrap runs localized one. The summary records `paper_approved: false`. Its two PNG figures already exist. Per-fish PNG profile directories are present for 10 of the 57 fish, so they are not complete for the full cohort.

## Metric and rule for the present descriptive review

Use `tail_length_weighted_angular_l1` as the **working primary activity metric** because it is the metric of the completed delay LME. Keep `whole_tail_xy_mean_speed_normalized` as the independent whole-tail kinematic sensitivity metric and `legacy_distal_angular_speed` as the historical benchmark. All three use the same corrected trial-outcome cohort and shared movement detector. This working choice does not freeze Gate T1 for the paper.

For a **provisional delay-only learner-label review**, use the archived `legacy-improved` rule, whose script SHA-256 is `8987f606fb133d90be93aa83472c1644f368f00a9086353caca47235b5f1102c`. On the working primary metric it flags 5/29 delay fish and 1/28 controls. It is the most stable of the four historical rules across these three metrics by the simple criterion of varying from 5 to 7 delay flags, while `legacy-wip` varies from 13 to 19. The control flag is in-sample and is **not** an independently calibrated false-positive rate. The comparison uses corrected activity units, so these are not reproductions of historical published labels. No canonical Gate L classifier version or validation mode has been approved.

| Archived rule | Tail L1 delay / control flags | Whole-tail XY delay / control flags | Legacy distal delay / control flags |
| --- | ---: | ---: | ---: |
| `legacy-nominal` | 5 / 2 | 10 / 3 | 7 / 0 |
| `legacy-new` | 6 / 2 | 9 / 3 | 7 / 2 |
| `legacy-improved` | 5 / 1 | 7 / 1 | 6 / 1 |
| `legacy-wip` | 13 / 2 | 16 / 2 | 19 / 2 |

Under the provisional working combination, the five flagged delay fish are `20221122_03`, `20240613_02`, `20240618_02`, `20240621_03`, and `20240703_08`. All five are also flagged by `legacy-improved` with whole-tail XY; four of five are flagged with legacy distal speed. The three complete comparison bundles are in `outputs/delay-learner-review-tail-l1`, `outputs/delay-learner-review-whole-tail-xy`, and `outputs/delay-learner-review-legacy-distal`. The tail-L1 bundle includes all four historical summary figure sets and a control-flag audit.

## Figure 4 state

The repository's `figure4-analyze` route is implemented and tested, but its paper-data run requires all three reviewed assay cohorts (`allDelay`, `all3sTrace`, `all10sTrace`), their selected-metric assessments, and a frozen Gate L classifier manifest with one label per fish. The fixed-trace project on `J:` has no frozen cohort yet, no 10sTrace project is present in the inspected `J:\Digested Data` directory, and no Gate L manifest was found in the delay project. The historical comparison above is **not** a substitute for that manifest. No Figure 4 panel data or main figures were generated from paper data.

Once Gate L is frozen, the proposed Figure 4 signal is signed, trial-baseline-centered bout-log-vigor from the selected metric in 0.5-s bins over −20…+20 s, with 0.9 expected-frame coverage. Movement probability and contributing-fish counts are separate companion outputs. The pooled catch set is global CS trials 25, 39, 53, 59, and 65. The fixed 3sTrace raw-protocol audit supports a 13-s paired-US latency; Figure 4 still verifies each authenticated fish's events. Learner-stratified curves from classification and plotting on the same trials remain descriptive.

The next Figure 4 run requires the reviewed three-assay cohort/assessment identities and an approved classifier/validation decision. Preserve the current five provisional labels as sensitivity evidence, not as the final Figure 4 identity.
