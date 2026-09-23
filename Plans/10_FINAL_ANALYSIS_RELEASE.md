# Final analysis release

There will be **one release** for this analysis, after the code and the
scientific analysis are ready. Development runs, fixture results, candidate
metrics, and draft figures are working evidence; they are not releases.

## When to make it

The final release is ready when the decisions in [DECISIONS.md](./DECISIONS.md)
are settled for the paper, the selected pipeline has run on the reviewed paper
cohort, required validation and diagnostics pass, Figure 4's pooled-learner CR
analysis and the subsequent tail mechanistic work are recorded, and the results
and figures agree with the manuscript's methods, sample sizes, and claims. The
[implementation index](./IMPLEMENTATION_STEP_INDEX.md) tracks those open items.
Finishing code alone does not approve an unresolved scientific choice.

## What to freeze

Keep one readable record that identifies the code commit and environment,
source data inventory and hashes, resolved configuration and recipe, selected
metric and detector, reviewed cohort and its hash, outcome and model or learner
methods, diagnostics, result tables, figure inputs and final figures, and the
manuscript claims they support. Record the commands or steps used to verify the
analysis. A tag and a frozen manifest are sufficient if they make these links
unambiguous; no separate release framework is required.

Legacy behavior and candidate comparisons remain in the audit and analysis
records where needed to explain decisions. They do not need their own releases.
Do not retire a legacy executable until its remaining consumers and required
reproduction use are accounted for under [governance](./GOVERNANCE.md); that
cleanup is independent of the final analysis release.
