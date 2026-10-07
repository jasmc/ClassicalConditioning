"""Explicit quantile candidates on the SAME displayed bout-log bins."""
import numpy as np

def trial_quantile_variants(values, baseline_mask):
    values=np.asarray(values,float);baseline_mask=np.asarray(baseline_mask,bool)
    if values.shape!=baseline_mask.shape:raise ValueError('Values and baseline must align')
    finite=np.isfinite(values);base=values[finite&baseline_mask]
    linear=np.full(values.shape,np.nan);anchored=linear.copy()
    if len(base)<2:return linear,anchored,{'n':len(base),'p10':np.nan,'p50':np.nan,'p90':np.nan,'status':'insufficient baseline bins'}
    lo,med,hi=np.quantile(base,[.1,.5,.9])
    meta={'n':len(base),'p10':float(lo),'p50':float(med),'p90':float(hi),
          'width':float(hi-lo),'negative_width':float(med-lo),'positive_width':float(hi-med),
          'linear_low_clipped_bins':int((values[finite]<lo).sum()),
          'linear_high_clipped_bins':int((values[finite]>hi).sum()),'status':'ok'}
    if hi>lo:
        linear[finite]=np.clip((values[finite]-lo)/(hi-lo),0,1)
        meta['baseline_median_linear']=float(np.median(linear[finite&baseline_mask]))
    else:meta['status']='degenerate P10-P90'
    radius=max(med-lo,hi-med)
    meta['symmetric_radius']=float(radius)
    if radius>0:
        anchored[finite]=np.clip((values[finite]-med)/radius,-1,1)
        meta['baseline_median_anchored']=float(np.median(anchored[finite&baseline_mask]))
    else:meta['anchored_status']='degenerate symmetric quantile range'
    return linear,anchored,meta
