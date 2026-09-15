# Step 10A — Exploratory Inference Scaffolding

**Status:** Complete for engineering scaffolding; archived 2026-09-15
**Scientific status:** Exploratory only; no confirmatory result authorized

## Closed scope

Build enough shared infrastructure to prove that authenticated trial outcomes
can feed several fish-aware statistical routes without embedding statistics in
plotting code.

## Implemented

- `candidate-model-input-v1` builds a shared, authenticated long-form model
  input from trial outcomes.
- `candidate-mixed-effects-v1` fits an exploratory fish-grouped mixed-effects
  model and publishes coefficient and diagnostic artifacts.
- `candidate-fish-permutation-v1` collapses observations to fish-level
  early-to-late effects and runs a sign-flip permutation test.
- `candidate-fish-bootstrap-v1` produces deterministic fish-unit percentile
  intervals for the same exploratory effect.
- All three active activity metrics use the same plumbing.
- Outputs are versioned, hashed, and explicitly marked non-paper-approved.

## Evidence

- `src/classical_conditioning/analysis/inference/model_input.py`
- `src/classical_conditioning/analysis/inference/mixed_effects.py`
- `src/classical_conditioning/analysis/inference/fish_permutation.py`
- `src/classical_conditioning/analysis/inference/fish_bootstrap.py`
- Corresponding `tests/test_model_input.py`, `tests/test_mixed_effects.py`,
  `tests/test_fish_permutation.py`, and `tests/test_fish_bootstrap.py`

## Important boundary

This archive proves engineering feasibility only. The current LME lacks the
approved conditioned-versus-control estimand; permutation/bootstrap summaries
do not yet implement that contrast; the paper cohort/outcome contract is not
frozen; and fit failures are not yet strict publication blockers. The active
analysis and statistics plan owns those decisions and corrections.
