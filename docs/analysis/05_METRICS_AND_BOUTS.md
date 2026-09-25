# Activity Metrics, Smoothing, and Shared Bout Segmentation

## Purpose

This document records the current behavior-analysis contract extracted from the
archived tail and metric-selection plans. It describes what exists and which
scientific choices remain open; it is not a substitute for the decision
register or an implementation plan.

## Active candidate metrics

The candidate pipeline carries exactly three activity metrics:

1. **Tail-length-weighted angular L1** — absolute local-angle change weighted
   by represented tail length.
2. **Tail-length-normalized whole-tail XY mean speed** — measured tracking-point
   displacement normalized by represented tail length and measured time.
3. **Legacy-derived distal cumulative-angle speed** — the historical distal
   cumulative-angle formula evaluated on measured time as a benchmark.

The legacy-derived metric is a historical comparator, not an automatic winner.
The manuscript's unweighted segment-speed sum and the retired RMS/curvature
candidates are not active selection candidates.

## Shared bout segmentation

Bout segmentation is shared by all three metrics. Metrics are compared on the
same behavioral episodes; no metric-specific detector may redefine which
samples count as movement.

The detector therefore has its own identity and configuration, separate from
the metric used to quantify an accepted bout. Its provenance must include:

- detector input signal;
- smoothing method and parameters;
- onset and offset thresholds;
- minimum duration and gap-merging rules;
- invalid-sample and recording-gap behavior;
- detector implementation and parameter-set versions.

## Smoothing

Smoothing is not inherently a defect. The problem is that it changes peak
amplitude, bout onset/offset, short-bout detectability, and apparent temporal
precision. If smoothing is selected after inspecting which setting produces the
most favorable biological result, it also becomes an undisclosed analysis
choice.

The current code contains a smoothing path, but the paper-authoritative method
still needs an approved contract. Gate T1 must record:

- which signal is smoothed: detector source, metric values, or both;
- filter family and exact parameters;
- whether smoothing is causal or centered;
- edge and missing-data handling;
- time units rather than frame-count assumptions;
- the expected effect on onset, duration, and peak measurements; and
- a bounded sensitivity comparison against plausible neighboring settings.

Until that decision, smoothing is implemented engineering behavior, not a
frozen scientific method.

## Implemented and documented behavior

- The three-metric fixture pipeline exists for the two local real recordings.
- Both corrected input routes use the shared candidate-metric kernel.
- Shared bout segmentation is implemented and is the decided architecture.
- Temporal and trial outcomes can be generated for all three candidates.
- Synthetic and local-fixture tests establish important plumbing and selected
  invariants.

## Still open before paper use

- preprocessing policy for interpolation, gaps, filtering, and corrected
  scaling (Gate P);
- the approved detector source, smoothing, thresholds, duration, and merging
  rules (Gate T1);
- remaining known-answer and legacy-regression coverage;
- metric selection on a scientifically valid cohort and outcome contract;
- robustness to the bounded preprocessing/detector choices; and
- confirmation or explicitly labeled exploratory status.

Manual/video bout validation remains outside the active requirement by user
decision. It is not listed as a completion blocker.

## Outcome separation

For every approved temporal window, preserve distinct outcomes:

- total or zero-inclusive activity;
- movement probability or fraction of valid time moving;
- bout count/rate with observation time as exposure; and
- activity intensity conditional on movement.

This avoids treating immobility as missing behavior or confusing the
probability of moving with how intensely a fish moves once a bout occurs.

## Authority and history

The scientific decisions are in
[Plans/DECISIONS.md](../../Plans/DECISIONS.md). Earlier metric and tail plans
remain in Git history. Future mechanistic extensions are separated into
[Tail Mechanistic Analyses](../../Plans/09_TAIL_MECHANISTIC_ANALYSES.md).

## Current recipe reference

The active candidate recipe writes three frame-level metrics:

| Metric | Calculation and role |
| --- | --- |
| `tail_length_weighted_angular_l1` | Tail-length-weighted mean absolute segment angular speed; reduces dependence on tracking-point spacing. |
| `whole_tail_xy_mean_speed_normalized` | Tail-length-weighted mean XY point speed divided by the recording-wide median tail arc length; units are tail lengths/ms rather than pixels/ms. |
| `legacy_distal_angular_speed` | Absolute speed of the sum of local tail angles; retained as the historical benchmark. Opposing segment changes can cancel. |

Earlier experimental alternatives are no longer part of the active package or
artifact metadata. Rebuild existing candidate artifacts with `overwrite`
enabled when deliberately applying this narrowed metric set.

For the moving-only vigor outcome, baseline and CS/trace response intensity are
averaged only over frames inside shared detected bouts. A window with no bout is
`NaN`, not zero. Population inference uses the zero-offset contrast
`log(baseline) - log(response)` (the negative of the model's
`log_response - log_baseline` adjustment); non-positive or no-bout pairs cannot
be logged and are excluded with coverage remaining explicit.

## Current shared-detector reference

One detector runs per recording because bout segmentation is a property of the
animal's behaviour, not a property of a metric. It reproduces the historical
four-step rule from `legacy/modules/my_functions.py`:

1. Build an envelope: centred rolling **max minus rolling min** of smoothed
   distal cumulative-angle speed, with 28.6 ms and 571.4 ms windows.
2. Threshold the envelope at **4 deg/ms**.
3. Merge bouts separated by less than **14.3 ms**, then drop bouts shorter than
   **57.1 ms**.
4. Drop bouts whose peak instantaneous angular speed never reaches
   **1 deg/ms**.

The original constants were frame counts at interpolated 700 FPS (20, 400, 10,
40 frames). The active route applies them as milliseconds of *measured* time
and converts them to rad/ms; windows never span a tracking discontinuity.
