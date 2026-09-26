# Pre-CS baseline review for Figure 1D and related figures

**Status:** comparison only. No repository-wide baseline default has been changed. The intended alternatives for per-trial Figure 1D scaling are **[−15, 0) s** and **[−20, 0) s** relative to CS onset. Both use the same trial's log-vigor P10/P90, with 0–1 clipping after scaling.

The two [Figure 1 C/D review plots](../../../outputs/figure1-cd-baseline-review/) use Delay recording `20221115_07`, global CS trials 9, 17, 63, 66 and 93, `tail_length_weighted_angular_l1`, and positive frames in detected moving bouts. The new renderer calculates each bout's mean raw metric, takes its log, assigns it to the bout's positive moving frames, then calculates P10/P90 from the chosen pre-CS interval. It averages the resulting scaled frame values in 0.5 s bins for D. Missing bins stay empty. The paired C angle trace is unchanged. Alternative all-valid-frame review outputs are in the same folder; they use log raw frame values and therefore show a different signal.

For this one example, both moving-bout baselines yielded a valid scale in all five selected trials. [−20, 0) used 1,321, 955, 1,680, 2,144 and 2,484 baseline moving frames across the five trials; [−15, 0) used 823, 731, 1,197, 1,586 and 1,700. Among 16,922 frames with finite scaled values under both, 11,912 changed by more than 1e−9; the mean absolute difference was 0.0209 on the 0–1 scale, maximum 0.3412. These are **one-fish display differences**, not evidence that either interval is more biologically appropriate.

## Baseline definitions currently in use

| Code path | Current reference | Meaning and review consequence |
| --- | --- | --- |
| `figures/example_traces.py` Figure 1C angle centering | All finite pre-CS frames in the displayed [−20, 0) s window | The C trace itself is unchanged in the two D comparison exports. A literal common baseline rule would also change its zero reference, so include C in the final review. |
| `analysis/trial_outcomes.py`, `analysis/metric_comparison.py`, `figures/cohort_response.py`, `analysis/discarding.py` | [−15, 0) s | Outcome ratios, metric comparisons and baseline bout flags already use the proposed shorter interval. Changing these affects numerical outcomes and possibly inference. |
| `figures/signed_bout_heatmap.py` and current Figure 1E/F/G and Figure 2A/B signed heatmaps | [−20, 0) s | Positive moving-bout log vigor is **median-centred**, then binned; it is not P10/P90 scaled. A [−15, 0) change needs a new start-bound parameter and fresh fish and pooled panel data. |
| `analysis/figure4.py` | [−20, 0) s | Signed Figure 4 profiles are tied to a declared −20…0 s baseline. Review before changing their estimates or labels. |
| `analysis/temporal_profiles.py` `TemporalProfileConfig.scaling_baseline_end_s=-15` | All available samples earlier than −15 s | This historical two-layer scaler does **not** mean [−15, 0) s. It can use times earlier than −20 s when the profile window allows them. |
| `figures/per_trial_scaled_vigor.py` default `baseline_end_s=-15`, no start | All available covered bins earlier than −15 s | Another historical P10/P90 recipe; callers must set `baseline_start_s=-15`, `baseline_end_s=0` to test the proposed interval. |
| `scripts/render_log_scaled_vigor_heatmaps.py` frame-first Figure 1 alternative | [−20, 0) s | This earlier 0–1 heatmap scales log bout-mean frames before binning. The new D moving-bout review follows this operation order but offers both baseline intervals. |
| `analysis/movement_state.py` calibration controls | [−5, −1) s | Detector calibration serves a different purpose; changing a display baseline does not imply changing this control interval. |

## Decision sequence

1. Choose whether D and the main heatmaps should show moving-bout vigor or all valid-frame activity; the two differ visibly and in missing-data meaning.
2. Compare [−15, 0) and [−20, 0) on Figure 1 single-fish heatmaps and Figure 2 equal-fish pooled heatmaps, retaining coverage and saturation diagnostics. Compare any Figure 4 signed profiles affected by a shared change.
3. Decide whether the scaled-log D uses the same signal and baseline as the main heatmaps. If the heatmaps remain median-centred signed log vigor, explain that D's P10/P90 scale is a different display measure even when its frame mask and baseline agree.
4. Only then update defaults, regenerate affected panel data and figures, and record source/input hashes and selected interval in the release manifest. Historical outputs and their sidecars remain identifiable.
