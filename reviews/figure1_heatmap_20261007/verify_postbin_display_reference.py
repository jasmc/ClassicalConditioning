"""Read-only proof that all F/G/H display references use post-binning medians.

Recompute from uncentred, already aggregated half-second bins, independently
of the stored trial reference and rendered-option helper. Each bin gets one
vote regardless of its eligible frame count. No frames are loaded here.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
review = ROOT/'fgh-all-options-baseline-colour-centred-20261007'
manifest = json.loads((review/'colour_review_manifest.json').read_text())
path = review/'all_options_bins.parquet'
record = next(item for item in manifest['outputs'] if Path(item['path']) == path)
assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
bins = pd.read_parquet(path)
maximum_reference_error = 0.
maximum_signal_error = 0.
count = 0
undefined_ranges = []
for (panel, trial), group in bins.groupby(['panel', 'trial']):
    # These are already binned values, not frame/bout observations.
    raw_bins = group.uncentred_bout_log_bin.to_numpy()
    times = group.bin_center_s.to_numpy()
    baseline_bins = raw_bins[(times >= -15) & (times < 0) & np.isfinite(raw_bins)]
    reference = float(np.median(baseline_bins))
    assert len(baseline_bins) == int(group.display_baseline_bin_count.iloc[0])
    stored_reference = group.display_baseline_log_bin_median.to_numpy()
    np.testing.assert_allclose(stored_reference, reference, atol=1e-12, rtol=0)
    maximum_reference_error = max(maximum_reference_error, float(np.max(np.abs(stored_reference-reference))))
    centred_bins = raw_bins-reference
    np.testing.assert_allclose(group.colour_centred_log_bin, centred_bins,
                               atol=1e-12, rtol=0, equal_nan=True)
    maximum_signal_error = max(maximum_signal_error, float(np.nanmax(np.abs(group.colour_centred_log_bin-centred_bins))))
    # Quantiles, too, are taken from the binned baseline before clipping.
    p10, p50, p90 = np.quantile(baseline_bins, [.1, .5, .9])
    for option, scale in [('C', (p90-p10)/2), ('D', max(p50-p10, p90-p50))]:
        if scale > 0:
            expected = np.clip(centred_bins/scale, -1, 1)
        else:
            expected = np.full(centred_bins.shape, np.nan)
            undefined_ranges.append((panel, int(trial), option))
        np.testing.assert_allclose(group[option], expected, atol=1e-12, rtol=0, equal_nan=True)
    count += 1
print(json.dumps({'trials_verified': count, 'reference_stage': 'after 0.5 s bin aggregation',
                  'baseline_s': [-15, 0], 'weighting': 'one vote per finite bin',
                  'maximum_reference_error': maximum_reference_error,
                  'maximum_display_signal_error': maximum_signal_error,
                  'undefined_quantile_ranges': undefined_ranges}, indent=2))
