# G: late-row yellow and missingness audit

This is a diagnostic of the user's latest screenshot, not another change to F/G/H. The current panel data hash and independent representative-frame artifact hash were verified before numerical reading. The diagnostic is at `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/fgh-late-trace-diagnostic-20261007/`, built by `audit_late_trace_display.py` in this folder. Current F/G/H files remain unchanged.

## NaNs are present and correctly distinct from low movement coverage

G has 698 NaN bins / 7,200 total (9.69%). The Test block has 130 / 2,400 (5.42%). Trial 93 has 10 / 80. Trial 91 is the only Test row with zero NaN bins. A separate black/white missingness image makes these visible independently of the dark-purple midpoint in managua_r.

The established bin definition yields a finite conditional-amplitude value whenever ANY finite eligible bout frames contribute. It does not require a bout to occupy the whole half-second. Example: G trial 93 [6.0,6.5) has only 42 of 352 frames eligible (11.9%), yet its bin is finite. Off-bout frames are omitted from both numerator and denominator. A row with every bin finite need not be continuously moving.

The representative-frame eligibility counts independently reproduce all 80 per-bin coverage counts for trial 93. There is no NaN-to-zero conversion, interpolation or gap filling in the bin stage. Actual detected-bout support must still be distinguished from independently validated biological motion; this audit is not a video-based detector validation.

## Yellow is spread plus saturation, not failed median subtraction

Every trial's finite [−15,0) baseline-bin median is zero. Pooled Test baseline statistics are:

| Statistic | Value |
|---|---:|
| Median | 0.000000 |
| Mean | +0.072092 |
| 10th percentile | −0.325907 |
| 25th percentile | −0.168221 |
| 75th percentile | +0.328380 |
| 90th percentile | +0.602096 |
| Finite baseline bins beyond +0.25 | 31.19% |
| Finite baseline bins below −0.25 | 16.94% |
| Finite baseline bins within ±0.05 | 16.24% |

Median zero does not imply values cluster tightly at zero. The positive tail is larger than the negative tail, so the mean is positive and yellow saturation occurs more frequently. The requested median reference does not equal arithmetic mean centring or trial-by-trial spread normalization. No mathematical centring rule alone guarantees a predominantly midpoint-coloured panel.

The latest correction moved the reference from frame logs to displayed bins. In trial 93 it added +0.271605 to every previously displayed bin. This removes the negative baseline median but also shifts positive bouts further into yellow. That is the documented consequence of changing reference weighting; it does not recover a new underlying raw signal.

At ±0.75, the same stored Test baseline values have only 1.29% yellow-end saturation and 0.12% blue-end saturation. The diagnostic deliberately holds all data, support, references and managua_r fixed and changes only colour limits. This isolates a presentation choice, not a scientific recomputation. It remains a diagnostic; no final colour range was silently changed.

## Why some bright stretches extend across several bins

A single within-bout median is repeated over every eligible frame of the bout. In trial 93, a stronger bout near −6 to −4 s produces several neighboring positive bins near +0.65. Their equal values are expected under the bout-summary definition; they are not proof that instantaneous vigor stayed constant. Raw eligible vigor and repeated bout values are displayed separately in `G_trial93_support.png`.

## Disposition

The audit supports widening the shared display limits as a presentation candidate, while retaining managua_r and explicit missingness. It does not justify another reference shift, dividing each trial by its variance, inventing NaNs, or imposing an arbitrary coverage cutoff to make the plot look sparse. Such changes would alter the measure and need separate justification. The F/G/H numerical pipeline and detector remain provisional within the ongoing full preprocessing review.
