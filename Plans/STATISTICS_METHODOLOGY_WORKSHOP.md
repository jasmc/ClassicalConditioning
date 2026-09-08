# Statistics methodology workshop (Step 10.0)

**Status:** Workshop recorded — Gate S not frozen  
**When:** Before locking confirmatory mixed-effects (or any other family)  
**Analog:** Learner workshop in `Plans/11_LEARNER_CLASSIFICATION.md` §11.0  

This note criticizes the **current legacy statistics** and the **default corrected
mixed-effects plan**, and records **drastically different** inferential families
that may fit the scientific estimand better. Step 10.0 workshop content is
recorded; it does **not** approve a paper model. Gate S still freezes the
primary strategy after decisions are written into `Plans/DECISIONS.md`.

Sources already on record:

- `docs/analysis/ANALYSIS_ISSUES.md` §§8–11 (bootstrap unit, fragile LME, ratios)
- `docs/analysis/CODEBASE_BEHAVIOR_MAP.md` (Mann-Whitney / FDR / per-trial LME)
- `Plans/DECISIONS.md` Gate S (default LME + holdout; legacy frozen as reproduction)
- `Plans/10_STATISTICS_AND_SENSITIVITY.md`

## 1. What the legacy route actually does

Frozen `legacy-paper-v1` statistics mix:

1. **Block/phase nonparametric tests** (Mann-Whitney / Wilcoxon) on ratio or
   summary vigor, with **separate BH-FDR families**.
2. **Many small mixed models**: global interaction terms, **per-block** mean/slope
   models, and **per-trial** condition models on `log(response)` with
   `log(baseline)` covariate.
3. Random-effects formula defaulting toward `~Log_Baseline` rather than a
   justified fish intercept / trial slope.
4. Fit failures converted to **strings**; pipeline continues; weak or missing
   convergence / singularity checks.
5. Display bootstrap often at **trial/row** level with too few resamples relative
   to the manuscript claim.

This route is preserved only as a **faithful reproduction**. It is not a
template to “fix” into the corrected paper analysis.

## 2. Critique of the legacy approach (not just implementation bugs)

| Problem | Why it matters scientifically |
| --- | --- |
| Many weak tests | Per-block and per-trial families inflate researcher degrees of freedom even with FDR. |
| Ratio outcomes | Baseline noise in the denominator dominates; reduced movement is the biology. |
| Independence assumptions | Nonparametric block tests treat fish/trials as if exchangeable without a clear hierarchy. |
| Model failure as string | A non-converged or singular fit can disappear into a plot path. |
| Wrong bootstrap unit | Trial rows ≠ independent fish; intervals understate uncertainty. |
| Estimand unclear | Mixing “is block A different?” with “does learning trajectory differ?” without one primary estimand. |

## 3. Critique of the default corrected plan (Gate S LME)

`DECISIONS.md` currently defaults the corrected primary analysis to:

> mixed-effects with fish as random effect + holdout/CV sanity check,
> applied identically to all six metrics in the candidate comparison.

That is a **sensible engineering default**, not a settled scientific choice.
Criticisms to confront before Gate S freeze:

1. **Gaussian / log-linear LME may be the wrong family** for movement
   probability, bout counts, zero-inflated intensity, or skewed totals.
2. **“Fish random intercept” alone** may be too thin for learning trajectories
   (trial slopes, block structure, day/rig batches).
3. **Identical formulas across five metrics** aid fair comparison but can hide
   that some metrics need binomial/beta/Tweedie links.
4. **Holdout/CV on N≈2 fixtures is meaningless**; validation design must wait for
   paper-scale N and must not reuse metric-selection observations (Step 10
   confirmatory dataset rule).
5. **LME p-values are not the only credible endpoint**; interval estimands,
   predictive checks, and fish-level effect distributions may answer the paper
   question better.
6. Treating LME as inevitable risks **recreating legacy complexity** (many
   contrasts) under a nicer API.

The workshop may keep LME as primary, demote it to sensitivity, or replace it.

## 4. Drastically different inferential families (candidates)

Record these as named alternatives. Each needs an estimand, unit of analysis,
assumptions, and a kill criterion.

### A. Fish-level summary + exact/permutation inference

Collapse each fish to a small vector (e.g. CS response−baseline by phase), then
permutation or nonparametric tests across fish.  
**Strength:** Matches fish as the biological unit; hard to cheat with trial N.  
**Weakness:** Throws away within-fish dynamics; low power at small N.

### B. Hierarchical Bayesian model (fish → trial → frame/window)

Full posterior for condition × phase effects with fish-level partial pooling;
posterior predictive checks for suppression during CS.  
**Strength:** Uncertainty is honest; missingness and zero-inflation are
modelable.  
**Weakness:** Heavier stack; Gate S must freeze priors and diagnostics.

### C. Marginal / GEE / cluster-robust GLM

Population-average effects with fish as cluster; robust SEs.  
**Strength:** Directly targets average conditioning effect; fewer random-effect
pathologies.  
**Weakness:** Weaker for fish-specific learning curves.

### D. Functional / longitudinal curve models (GAM, FDA, growth curves)

Model the CS-aligned temporal profile as a curve; test condition differences in
function space.  
**Strength:** Uses the rich temporal artifacts already built.  
**Weakness:** Needs careful alignment and multiplicity control over the curve.

### E. Point-process / bout models

Model bout onsets (and optionally durations) as a marked point process or
survival/hazard during CS vs baseline.  
**Strength:** Matches “movement events” rather than arbitrary window means.  
**Weakness:** Depends on detector quality (Gate T1).

### F. State-space / HMM movement occupancy

Latent move/quiet states with trial-varying transition probabilities under CS.  
**Strength:** Separates detection from learning of state occupancy.  
**Weakness:** Identifiability and label-switching; not a small patch on LME.

### G. Causal / design-based estimands

Define the paper claim as a contrast under the experimental assignment
(paired CS/US structure, within-fish baselines) and use randomization inference
or design-based SEs.  
**Strength:** Ties inference to the protocol, not to a convenience GLM.  
**Weakness:** Requires an explicit causal graph and exclusion rules (Gate C0).

### H. Multivariate multi-metric model

One joint model (or hierarchical meta-model) over the five candidates instead of
five identical univariate LMEs.  
**Strength:** Metric comparison becomes a formal estimand.  
**Weakness:** Harder to communicate; needs care with scale differences.

### I. Pure predictive evaluation

Train on held-out fish to predict CS suppression; report calibration and
decision curves rather than coefficient tables.  
**Strength:** Aligns with “does the signal carry learning?” without p-hacking
coefficients.  
**Weakness:** Not a drop-in methods paragraph for classical physiology papers.

## 5. What we implement now (engineering, not Gate S)

**Not** “many mixed-effects versions.” Exactly three inference routes for now:

| Route | Recipe | Role |
| --- | --- | --- |
| One simple fish-grouped LME | `candidate-mixed-effects-v1` | Shared plumbing / diagnostic scaffold (Gate S *draft* default) |
| One drastically different alternative | `candidate-fish-permutation-v1` | Fish-level early→late effect + sign-flip permutation (family A) |
| Fish-unit uncertainty | `candidate-fish-bootstrap-v1` | Percentile CI on the same fish-level effect (Step 10.4) |

Everything else in §4 stays on the candidate list until Gate S. Do not implement
Bayes/GEE/HMM/etc. until the workshop picks them.

Until Gate S:

1. Keep legacy statistics frozen and comparable.
2. Run both routes identically across the five metrics.
3. Treat two-fish fixture results as pipeline tests only (LME often singular;
   permutation p-values coarse).

## 6. Decision log (fill during Gate S)

| Question | Decision | Date |
| --- | --- | --- |
| Primary estimand | _TBD_ | |
| Primary model family | _TBD (default draft: LME)_ | |
| Kill LME? | _TBD_ | |
| First alternative family | Fish-level permutation (implemented as exploratory) | 2026-08-31 |
| Validation mode | _TBD_ | |
| Multiplicity policy | _TBD_ | |
