"""Bin-based trial reference applied to unbinned repeated bout samples."""
import numpy as np
from baseline_colour_mapping import baseline_centred_options


def hybrid_sample_values(seconds, repeated_bout_log):
    seconds=np.asarray(seconds,float)
    values=np.asarray(repeated_bout_log,float)
    if seconds.ndim!=1 or values.shape!=seconds.shape:
        raise ValueError('Expected one value per aligned time sample')
    if not np.isfinite(seconds).all() or np.any((seconds < -20)|(seconds >= 20)):
        raise ValueError('Sample times must be finite and within [-20,20)')
    index=np.floor((seconds+20)/.5).astype(int)
    finite=np.isfinite(values)
    counts=np.bincount(index[finite],minlength=80)
    sums=np.bincount(index[finite],weights=values[finite],minlength=80)
    bins=np.divide(sums,counts,out=np.full(80,np.nan),where=counts>0)
    centres=np.arange(-19.75,20,.5)
    baseline=(centres>=-15)&(centres<0)
    bin_options,meta=baseline_centred_options(bins,baseline)
    reference=meta['reference']
    centred=np.where(finite,values-reference,np.nan)
    samples={'centred_log':centred}
    for option in ['C','D']:
        scale=meta[option+'_scale']
        uncapped=centred/scale if scale>0 else np.full(values.shape,np.nan)
        samples[option+'_unclipped']=uncapped
        samples[option]=np.clip(uncapped,-1,1)
        # Linear transformation commutes with bin means; reference stays post-bin.
        if scale>0:
            rebinned=np.bincount(index[finite],weights=uncapped[finite],minlength=80)
            rebinned=np.divide(rebinned,counts,out=np.full(80,np.nan),where=counts>0)
            np.testing.assert_allclose(rebinned,bin_options[option+'_unclipped'],atol=1e-10,rtol=1e-11,equal_nan=True)
    if np.isfinite(reference):
        assert abs(np.nanmedian((bins-reference)[baseline]))<1e-12
    return samples, bins, counts, meta
