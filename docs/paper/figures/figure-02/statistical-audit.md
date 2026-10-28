# Figure 2 G/H/I statistical audit

The editable [audit manuscript](</Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-02/panels-g-h-i/studies/statistical-audit/report/study.md>)
and [HTML review](</Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-02/panels-g-h-i/studies/statistical-audit/report/study.html>)
live together on JOAQUIM. The source registry authenticates the manuscript;
the [compact audit record](../../../../records/scientific-evidence/analyses/figure-02-ghi-statistical-audit.json)
binds inputs, outputs and remaining coverage gaps.

The [current methods reference](current-methods.md) separates selected working
definitions from preserved historical Methods. Reproduction uses
`scripts/paper/figure_02/panels_g_h_i/audit_saved_statistics.py --manifest
<authenticated audit-inputs.json> --output-dir <new external directory>
--notebook-script <personal helper> --notebook-sha256 <exact hash>`.
The command replays saved arithmetic and bootstrap indices, preserves current
state, and publishes a new audit bundle transactionally. It never refits models,
draws new resamples, overwrites an existing study or selects science. Keep manual
manuscript revisions and notes through the existing study-artifact workflow.
