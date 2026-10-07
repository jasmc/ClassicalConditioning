"""F/G/H-specific display-signal reference; no shared preprocessing mutation."""
import numpy as np
import pandas as pd

BASELINE_S = (-15.0, 0.0)
SIGNAL = 'bout_median_log_finite_frame_bin_mean_trial_baseline_bin_median_centred'

def centre_trial_heatmap_bins(source: pd.DataFrame) -> pd.DataFrame:
    """Centre each displayed row on its finite baseline bins, equally weighted.

    Source signed_bout_log_bin is frame-reference centred. Remove that reference
    to record absolute log-bin values, and replace it with the displayed-bin
    reference. Subtracting the old row's baseline-bin median is algebraically
    identical, avoiding an unnecessary second frame-to-bin calculation.
    No division, value clipping, zero fill or smoothing occurs here.
    """
    required={'trial','bin_center_s','signed_bout_log_bin','baseline_log_median'}
    if required-set(source):raise ValueError(f'Missing fields: {required-set(source)}')
    if source.duplicated(['trial','bin_center_s']).any():raise ValueError('Duplicate trial/time bin')
    if not np.isfinite(source.bin_center_s).all():raise ValueError('Nonfinite bin times')
    result=source.copy()
    result['frame_baseline_centred_bin']=result.signed_bout_log_bin.where(np.isfinite(result.signed_bout_log_bin))
    result['uncentred_bout_log_bin']=result.frame_baseline_centred_bin+result.baseline_log_median
    result['display_baseline_log_bin_median']=np.nan
    result['display_centre_offset_from_previous']=np.nan
    result['display_baseline_bin_count']=0
    result['signed_bout_log_bin']=np.nan
    for trial,group in result.groupby('trial',sort=False):
        references=group.baseline_log_median.dropna().unique()
        if len(references)>1:raise ValueError(f'Trial {trial} has inconsistent source frame reference')
        baseline=group.loc[group.bin_center_s.ge(BASELINE_S[0])&group.bin_center_s.lt(BASELINE_S[1]),'frame_baseline_centred_bin'].dropna()
        result.loc[group.index,'display_baseline_bin_count']=len(baseline)
        if baseline.empty:continue
        if len(references)!=1 or not np.isfinite(references[0]):raise ValueError('Finite bins require a finite source frame reference')
        shift=float(np.median(baseline))
        result.loc[group.index,'display_centre_offset_from_previous']=shift
        result.loc[group.index,'display_baseline_log_bin_median']=float(references[0])+shift
        result.loc[group.index,'signed_bout_log_bin']=group.frame_baseline_centred_bin-shift
    result['Signal semantics']=SIGNAL
    result['Baseline start (s)']=BASELINE_S[0]
    result['Baseline end (s)']=BASELINE_S[1]
    return result
