# Fixed 3sTrace exploratory run

## Scope and inputs

The raw source is `J:\Raw Data\all3sTtrace`. A completed preflight on 2026-09-24 found **40 fixed 3sTrace fish and 19 matched controls** with complete camera, protocol, and tracking triplets. Four incomplete recording groups remain in the inventory and are excluded from processing. Two duplicate source filenames are ignored by the triplet scan. The historical Figure 1 example is fish `20230307_12`, identified in the archived March 2023 example and vigor scripts.

Only `J:\Digested Data\all3sTrace-full-v1` is moved to `F:\Digested Data\all3sTrace-full-v1`. The copy script checks each copied file by SHA-256 before the pipeline starts. Existing completed fish artifacts carry absolute `J:` provenance, so the pipeline must verify or rebuild them under `F:`. The `J:` project is removed only after the `F:` copy, full run, and downstream outputs are verified. Other derived and raw projects are outside the move scope.

## Metric and learner choice

The working primary activity metric is **`tail_length_weighted_angular_l1`**, the metric used in the completed Delay learning-onset run. **`whole_tail_xy_mean_speed_normalized`** is the whole-tail sensitivity metric and **`legacy_distal_angular_speed`** is the historical benchmark. The trial-outcome recipe is `tail-candidate-corrected`.

For this exploratory run, use archived classifier version **`legacy-wip`**. In the authenticated Delay comparison on the primary metric it flagged **13/29 conditioned fish**, the most of the four variants (nominal 5, new 6, improved 5). It also flagged 2/28 Delay controls. The 3sTrace run records its own conditioned and control counts. These labels describe the same fish and trials used in the plots; they do not constitute an independently validated learning classifier. This explicit user choice supersedes the earlier Delay-only provisional recommendation of `legacy-improved` for the present 3sTrace review.

## Output sequence

1. Run `scripts/run-all3sTrace-full-windows.ps1 -AllowIncomplete` against the F: project. This inventories all raw groups, selects the 59 complete fish, performs verified intake, corrected three-metric analysis, selected-metric assessment, and routine figures.
2. Freeze `all3sTrace-full-exploratory` as an all-complete technical cohort, independent of the assessment's exploratory discarding recommendations. Build authenticated cohort trial outcomes.
3. Render Figure 1F for `20230307_12`; Figure 2B as signed CS-aligned vigor and coverage; Figure 2E/H as selected-block and trial-level response/baseline ratios. Run the exploratory learning-onset model and its diagnostics.
4. Compare all four archived learner scripts on the 3sTrace cohort using the working primary metric. Render their descriptive summaries and a provisional Figure 3 review showing WIP scores, flags, paired change, and score-selected example trajectories. Package `legacy-wip` labels in a hash-bound provisional classifier manifest.
5. Run the 3sTrace-only `figure4-analyze` and `figure4-render` routes. Their saved analysis has `partial_assay_review` scope and `descriptive_provisional_legacy_rule` status; Figure 4B and companion panels are review outputs.

The frozen cohort and learner manifest carry separate identities. The assessment is retained as provenance for the selected metric, but its discard calls do not determine this exploratory cohort. A future paper cohort or Gate L classifier can replace these review identities without rewriting the raw or corrected fish data.
