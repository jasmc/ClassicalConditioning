"""Repeat angles -> raw vigor -> detector -> signed bins with explicit NaNs."""
import json
import numpy as np
import pandas as pd
from audit_figure1_vigor_alignment import ROOT, PROJECT, TIME, COL, digest
from build_figure1_legacy_vigor_heatmaps import read_windows
from classical_conditioning.analysis.movement_state import (
    smooth_contiguous_median, rolling_extreme_envelope, detect_legacy_envelope_bouts)
from classical_conditioning.analysis.temporal_profiles import _signed_bout_log_vigor

OUT = ROOT / 'audit-vigor-explicit-nans-repeat-20261006'


def main():
    emeta = json.loads((ROOT/'traces/Fig1_PanelE_RawVigor_HeatmapOverlay_allTrials_v4.svg.json').read_text())
    fmeta = json.loads((ROOT/'heatmaps/Fig1_PanelF_Delay_legacy-vigor_v2.svg.json').read_text())
    for item in emeta['input_artifacts'] + fmeta['input_artifacts']:
        assert digest(item['path']) == item['sha256']
    assert digest(fmeta['panel_data']) == fmeta['panel_data_sha256']
    proc = PROJECT/'Processed data/20221115_07'
    protocol = pd.read_parquet(proc/'stimulus_events.parquet')
    cycles = protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    heat = pd.read_parquet(fmeta['panel_data'])
    output, checks = [], []
    for trial in emeta['global_cs_trials']:
        onset = int(cycles.iloc[trial-1].Beg)
        interval = [(onset-22000, onset+22001)]
        angles = [f'angle{i}' for i in range(16)]
        c = read_windows(proc/'frame_preprocessed_corrected-v1.parquet',
                         ['FrameID','AbsoluteTime','ElapsedTime','FrameStep','DeltaTimeMs',
                          'frame_valid','timestamp_valid','derivative_valid',*angles], interval)
        m = read_windows(proc/'frame_activity_candidates-corrected-v1.parquet',
                         ['FrameID','AbsoluteTime',COL,'valid_derivative','angular_valid_tail_fraction'],interval)
        d = read_windows(proc/'movement_state_candidates-corrected-v2.parquet',
                         ['FrameID','AbsoluteTime','valid','moving','bout_id'],interval)
        np.testing.assert_array_equal(c[['FrameID','AbsoluteTime']],m[['FrameID','AbsoluteTime']])
        np.testing.assert_array_equal(c[['FrameID','AbsoluteTime']],d[['FrameID','AbsoluteTime']])
        # An angle is a position measurement; an invalid time derivative does
        # not itself invalidate that angle. Invalid position rows are NaN.
        invalid_position = ~c.frame_valid | ~c.timestamp_valid
        c.loc[invalid_position,angles] = np.nan
        cumulative = c[angles].to_numpy().sum(axis=1)
        delta = np.r_[np.nan,np.diff(cumulative)]
        raw = np.abs(np.arctan2(np.sin(delta),np.cos(delta)))/c.DeltaTimeMs.to_numpy()
        raw[~c.derivative_valid.to_numpy()] = np.nan
        # Only the context's first frame lacks its predecessor in this read.
        np.testing.assert_allclose(raw[1:],m[COL].to_numpy()[1:],rtol=0,atol=1e-12,equal_nan=True)
        smooth = smooth_contiguous_median(raw,c.FrameStep.to_numpy(),window_samples=7)
        envelope = rolling_extreme_envelope(smooth,c.FrameStep.to_numpy(),max_window_samples=21,min_window_samples=403)
        valid = c.derivative_valid.to_numpy() & np.isfinite(envelope) & (m.angular_valid_tail_fraction.to_numpy()>=.8)
        moving, bout_ids = detect_legacy_envelope_bouts(envelope,raw,c.DeltaTimeMs.to_numpy(),c.FrameStep.to_numpy(),valid,
            envelope_threshold=np.deg2rad(4),amplitude_threshold=np.deg2rad(1),
            minimum_bout_duration_ms=40/700*1000,maximum_interbout_gap_ms=10/700*1000)
        seconds=(c.AbsoluteTime.to_numpy()-onset)/1000
        selected=(seconds>=-20)&(seconds<20)
        np.testing.assert_array_equal(valid[selected],d.valid.to_numpy()[selected])
        np.testing.assert_array_equal(moving[selected],d.moving.to_numpy()[selected])
        c=c.loc[selected].copy(); raw=raw[selected]; valid=valid[selected]
        moving=moving[selected]; bout_ids=bout_ids[selected]; seconds=seconds[selected]
        support=valid & moving & (bout_ids>0) & np.isfinite(raw) & (raw>0)
        c['Trial number']=trial; c[TIME]=seconds
        c['raw_vigor']=raw
        c['detector_vigor']=np.where(valid,raw,np.nan)
        c['bout_raw_vigor']=np.where(support,raw,np.nan)
        baseline=float(np.median(np.log(raw[support & (seconds>=-15) & (seconds<0)])))
        c['scaled_vigor']=np.where(support,np.log(np.where(support,raw,np.nan))-baseline,np.nan)
        c['valid']=valid; c['moving']=moving; c['bout_id']=bout_ids
        c['Vigor']=raw; c['eligible']=support; c['centred_log']=c.scaled_vigor
        mapping=c.loc[support].groupby('bout_id').scaled_vigor.median()
        c['bout_median']=c.bout_id.map(mapping).where(support)
        c['bin_index']=np.floor((seconds+20)/.5).astype(int)
        np.testing.assert_array_equal(np.isfinite(c.bout_raw_vigor),np.isfinite(c.scaled_vigor))
        result=_signed_bout_log_vigor(c.bout_raw_vigor.to_numpy(),seconds,c.bin_index.to_numpy(),valid,moving,bout_ids,
                                     bin_count=80,baseline_start_s=-15,baseline_end_s=0)
        expected=heat.loc[heat['Trial number'].eq(trial)].sort_values('Time bin center (s)')['Signed log vigor'].to_numpy()
        np.testing.assert_allclose(result,expected,rtol=0,atol=1e-12,equal_nan=True)
        output.append(c)
        checks.append(dict(trial=trial,invalid_angle_rows=int(invalid_position.sum()),
                           invalid_raw_frames=int((~np.isfinite(raw)).sum()),
                           invalid_detector_frames=int((~valid).sum()),
                           shared_bout_frames=int(support.sum()),
                           maximum_bin_error=float(np.nanmax(np.abs(result-expected)))))
    OUT.mkdir(parents=True,exist_ok=True)
    data=OUT/'explicit_nan_frames.parquet'
    pd.concat(output,ignore_index=True).to_parquet(data,index=False)
    summary=dict(checks=checks,data_sha256=digest(data),angle_mask='Invalid position/timestamp rows become NaN',
                 raw_mask='Invalid derivatives become NaN',detector_mask='Invalid detector rows become NaN',
                 comparison_mask='Raw and scaled use identical eligible bout frames',
                 binning='Only finite eligible bout medians contribute; empty bins remain NaN')
    (OUT/'repeat.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
