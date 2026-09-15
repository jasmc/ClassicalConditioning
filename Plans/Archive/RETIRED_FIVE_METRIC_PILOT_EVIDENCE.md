# Retired five-metric pilot evidence

This note preserves the local-fixture detector evidence that predates the
active three-metric set. It is historical sensitivity evidence only: it does
not select a metric, detector, smoothing strength, cohort, or paper result.

## First local fish, 10 ms detector

| Metric | Movement fraction | Bouts | Post-US minus pre-US movement |
| --- | ---: | ---: | ---: |
| Segment angular-speed sum | 0.045 | 2,732 | 0.413 |
| Segment angular RMS | 0.047 | 2,709 | 0.392 |
| Whole-tail XY mean speed | 0.147 | 4,700 | 0.627 |
| Whole-tail XY RMS speed | 0.217 | 7,223 | 0.591 |
| Curvature-change RMS | 0.015 | 1,617 | 0.011 |

## First local fish, smoothing sensitivity

| Metric | 0 ms movement / bouts / US delta | 10 ms movement / bouts / US delta | 20 ms movement / bouts / US delta |
| --- | --- | --- | --- |
| Segment angular-speed sum | 0.005 / 500 / 0.203 | 0.045 / 2,732 / 0.413 | 0.057 / 2,886 / 0.363 |
| Segment angular RMS | 0.005 / 521 / 0.183 | 0.047 / 2,709 / 0.392 | 0.059 / 2,932 / 0.411 |
| Whole-tail XY mean speed | 0.192 / 8,654 / 0.541 | 0.147 / 4,700 / 0.627 | 0.129 / 4,106 / 0.656 |
| Whole-tail XY RMS speed | 0.206 / 9,506 / 0.498 | 0.217 / 7,223 / 0.591 | 0.164 / 5,275 / 0.646 |
| Curvature-change RMS | 0.000 / 1 / 0.000 | 0.015 / 1,617 / 0.011 | 0.030 / 2,692 / 0.019 |

## Second local fish, 10 ms detector

| Metric | Movement fraction | Bouts | Post-US minus pre-US movement |
| --- | ---: | ---: | ---: |
| Segment angular-speed sum | 0.030 | 1,990 | 0.524 |
| Segment angular RMS | 0.030 | 1,995 | 0.512 |
| Whole-tail XY mean speed | 0.117 | 4,915 | 0.642 |
| Whole-tail XY RMS speed | 0.128 | 5,866 | 0.629 |
| Curvature-change RMS | 0.012 | 1,131 | 0.010 |

Thresholds were recalibrated for each smoothing variant. The material changes
in movement fraction, bout count, and US contrast demonstrate sensitivity, not
accuracy. The active set is exactly the three metrics in `DECISIONS.md`.
