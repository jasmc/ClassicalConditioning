# Tail Mechanistic Analyses

**Status:** Planned active follow-on analysis; step 09 after Figure 4

**Starts after:** Figure 4's conditioned-response (CR) analysis based on pooled
learners is completed and reviewed. Reuse its approved learner definition,
cohort, outcome identities, and timing conventions rather than choosing them
inside this exploratory work.

**Also depends on:** Approved preprocessing, shared bout segmentation, and the
selected activity metric/outcome contract.

**Sequence:** Figure 4 is completed first; these exploratory analyses follow,
then the single final analysis release records the finished work and its status.

## Purpose

Preserve the scientifically interesting mechanistic work from the former tail
dynamics plan without presenting it as unfinished core metric implementation.
Begin this follow-on analysis only after Figure 4 establishes the pooled-learner
CR result. Tail geometry, rhythmic power, and traveling-wave analyses can then
explore how movement mechanics relate to that result.
Current metric and bout behavior is documented in
[Activity Metrics, Smoothing, and Shared Bout Segmentation](../docs/analysis/05_METRICS_AND_BOUTS.md).
Earlier tail and metric-selection plans remain in Git history.

## Questions

- Does conditioning change the geometry of tail movement, not only its total
  amount or occurrence?
- Do tail-shape modes or their temporal transitions differ across conditions or
  learner representations?
- Does rhythmic movement power change in biologically meaningful windows?
- Is traveling-wave direction or energy altered by acquisition, extinction, or
  trace interval?

## Work package A — PCA or eigen-tail dynamics

Construct an approved body-centred, time-aware tail representation and assess:

- variance explained and loading stability;
- sensitivity to segment count, missing tracking points, and scaling;
- fish-level rather than pooled-frame weighting;
- score trajectories aligned to CS and expected/actual US events;
- phase/block and learner-representation summaries; and
- reproducibility across fish and acquisition days.

PCA is a descriptive coordinate system, not evidence that biological learner
classes exist.

## Work package B — Windowed rhythmic movement power

Define the input signal, time-frequency method, frequency range, window length,
normalization, and missing-data behavior before comparing conditions. Store
power summaries at fish/trial/window level and avoid treating overlapping time
windows as independent animals.

## Work package C — Traveling-wave energy

Estimate propagation across tail position and time only after spatial identity
and timebase are validated. Candidate outputs include direction-resolved energy,
phase gradient, propagation speed, and event-aligned summaries. Compare against
synthetic standing waves, traveling waves, noise, missing segments, and reversed
tail order.

## Validation

- known-answer synthetic shapes and waves;
- invariance to translation and approved scaling;
- sensitivity to sampling rate and smoothing;
- fish-level stability and uncertainty;
- null-condition behavior;
- bounded preprocessing sensitivity; and
- explicit distinction between exploratory and confirmatory use.

## Exit gate

A mechanistic method becomes active only when its biological question, input
representation, expected invariants, validation set, aggregation order,
uncertainty, and manuscript role are approved. Until then these analyses must
not influence selection of the primary activity metric or learner threshold.
