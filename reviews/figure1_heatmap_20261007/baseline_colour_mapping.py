"""Median-preserving colour contract for the F/G/H review options."""
import numpy as np
from matplotlib.colors import CenteredNorm


def baseline_centred_options(values, baseline_mask):
    """Return unclipped and clipped options from the same finite baseline bins.

    A/B retain log units. C divides by half the P10--P90 width; D divides
    by the larger P50--P10/P90--P50 distance. Every clipping interval is
    symmetric about zero, so odd and even sample medians remain centred.
    """
    values = np.asarray(values, dtype=float)
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    if values.ndim != 1 or values.shape != baseline_mask.shape:
        raise ValueError('Expected aligned one-dimensional values and baseline mask')
    finite = np.isfinite(values)
    base = values[finite & baseline_mask]
    empty = np.full(values.shape, np.nan)
    result = {'log': empty.copy(), 'C_unclipped': empty.copy(),
              'C': empty.copy(), 'D_unclipped': empty.copy(), 'D': empty.copy()}
    meta = {'baseline_count': int(len(base)), 'reference': np.nan,
            'p10': np.nan, 'p50': np.nan, 'p90': np.nan,
            'C_scale': np.nan, 'D_scale': np.nan, 'status': 'missing baseline'}
    if not len(base):
        return result, meta
    lo, reference, hi = np.quantile(base, [.1, .5, .9])
    meta.update(reference=float(reference), p10=float(lo), p50=float(reference),
                p90=float(hi), C_scale=float((hi-lo)/2),
                D_scale=float(max(reference-lo, hi-reference)), status='ok')
    result['log'][finite] = values[finite] - reference
    for name in ['C', 'D']:
        scale = meta[name + '_scale']
        if scale > 0:
            result[name + '_unclipped'][finite] = result['log'][finite] / scale
            result[name][finite] = np.clip(result[name + '_unclipped'][finite], -1, 1)
    if not meta['C_scale'] > 0 or not meta['D_scale'] > 0:
        meta['status'] = 'undefined quantile range'
    return result, meta


def baseline_colour_norm(half_range):
    """Zero is explicitly fixed at palette coordinate 0.5."""
    if not np.isfinite(half_range) or half_range <= 0:
        raise ValueError('A positive finite colour half-range is required')
    return CenteredNorm(vcenter=0, halfrange=float(half_range), clip=True)
