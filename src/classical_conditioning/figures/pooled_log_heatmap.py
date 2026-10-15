"""Figure 1 F–H frame pipeline adapted to equal-fish population heatmaps.

The recording stage ports the named cadence/metric/detector computation from
render_figure1_best_learners_version08.analyze, used by the selected F–H
1 s review. It supports matched controls without inventing paired-US events.
Source markers authenticate every recording; loss evidence stops calculation.
"""
from pathlib import Path
from dataclasses import asdict
import gc
import numpy as np
import pandas as pd
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import sha
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
from classical_conditioning.analysis.figure4 import verify_expected_us
from classical_conditioning.analysis.movement_state import (
    MovementCalibrationConfig, _odd_window_samples, smooth_contiguous_median,
    rolling_extreme_envelope, detect_legacy_envelope_bouts,
)
import json

class StreamingSnapshot:
    """Recheck authenticated large frame files without reading them into RAM."""
    def __init__(self, hashes):
        self.hashes = hashes
    def check(self):
        for path, expected in self.hashes.items():
            if sha256_file(Path(path)) != expected:
                raise ValueError('Source changed: ' + path)

def read_windows(path, columns, intervals):
    """Read the same row-group-filtered frame windows as the example renderer."""
    import pyarrow.parquet as pq
    parquet=pq.ParquetFile(path);parts=[]
    time_index=parquet.schema.names.index('AbsoluteTime')
    for i in range(parquet.metadata.num_row_groups):
        stats=parquet.metadata.row_group(i).column(time_index).statistics
        if stats is None or stats.min is None or stats.max is None:
            raise ValueError('Missing time statistics: '+str(path))
        overlap=[(a,b) for a,b in intervals if stats.max>=a and stats.min<b]
        if not overlap:continue
        part=parquet.read_row_group(i,columns=columns).to_pandas()
        times=part.AbsoluteTime.to_numpy();keep=np.zeros(len(part),bool)
        for a,b in overlap:keep|=(times>=a)&(times<b)
        if keep.any():parts.append(part.loc[keep])
    if not parts:raise ValueError('No frames: '+str(path))
    return pd.concat(parts,ignore_index=True)

def direct_log_bins(seconds, raw_vigor):
    """Direct original-frame log medians; baseline-bin medians each get one vote."""
    seconds=np.asarray(seconds,float);raw_vigor=np.asarray(raw_vigor,float)
    good=np.isfinite(seconds)&np.isfinite(raw_vigor)&(raw_vigor>0)&(seconds>=-20)&(seconds<20)
    indices=np.searchsorted(np.arange(-20,21),seconds[good],side='right')-1
    logs=np.log(raw_vigor[good]);counts=np.bincount(indices,minlength=40)
    medians=np.array([np.median(logs[indices==i]) if counts[i] else np.nan for i in range(40)])
    finite=medians[5:20];finite=finite[np.isfinite(finite)]
    baseline=float(np.median(finite)) if len(finite) else np.nan
    return medians, baseline, medians-baseline, counts

def pool_equal_fish(fish_bins, condition_by_recording):
    """Average each fish's centred trial/bin value with equal weight per fish."""
    data=fish_bins.copy()
    keys=['fish','acquisition_trial','bin_index']
    if data.duplicated(keys).any():raise ValueError('Duplicate fish/trial/bin')
    data['condition_id']=data.fish.map(condition_by_recording)
    if data.condition_id.isna().any():raise ValueError('Unknown cohort recording')
    pooled=data.groupby(['condition_id','acquisition_trial','bin_index'],observed=True).delta_log_vigor.agg(mean_delta_log_vigor='mean',contributing_fish='count').reset_index()
    totals=pd.Series(condition_by_recording).value_counts()
    pooled['cohort_fish']=pooled.condition_id.map(totals)
    pooled['coverage_fraction']=pooled.contributing_fish/pooled.cohort_fish
    return pooled

def analyze_recording(project, fish, suffix, experiment, condition):
    direct_medians, bin_width_s = True, 1
    proc = project / 'Processed data' / fish
    metadata = project / 'Metadata'
    manifest_path = metadata / f'{fish}_source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['recording_id'] == fish
    assert manifest['condition_id'] == condition
    frame_marker = metadata / f'{fish}_corrected-preprocess{suffix}_complete.json'
    metric_marker = metadata / f'{fish}_candidate-corrected{suffix}_complete.json'
    fm, mm = json.loads(frame_marker.read_text()), json.loads(metric_marker.read_text())
    paths = dict(camera=proc/'camera.parquet', protocol=proc/'stimulus_events.parquet',
                 frames=proc/f'frame_preprocessed_corrected{suffix}.parquet',
                 coverage=proc/f'frame_activity_candidates-corrected{suffix}.parquet')
    expected = dict(camera=manifest['artifacts']['camera']['sha256'], protocol=manifest['artifacts']['protocol']['sha256'],
                    frames=fm['frames_sha256'], coverage=mm['metrics_sha256'])
    assert fm['status'] == mm['status'] == 'complete'
    expected_hashes = {str(path.resolve()): expected[key] for key,path in paths.items()}
    expected_hashes.update({str(path.resolve()): sha256_file(path) for path in (manifest_path, frame_marker, metric_marker)})
    snapshot = StreamingSnapshot(expected_hashes)
    snapshot.check()
    camera = pd.read_parquet(paths['camera'])
    cadence = estimate_camera_cadence(camera)
    assert not cadence.has_frame_loss_evidence
    epoch = float(camera.iloc[cadence.reference_position].AbsoluteTime)
    protocol = pd.read_parquet(paths['protocol'])
    us_time, paired = (None, 0) if condition == 'control' else verify_expected_us(protocol, experiment)
    if condition != 'control':
        assert paired == 46
    cycles = protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    assert len(cycles) == 94
    del camera
    cfg = MovementCalibrationConfig()
    context = 5000
    while True:
        windows = [(int(t)-20000-context, int(t)+20001+context) for t in cycles.iloc[4:].Beg]
        frames = read_windows(paths['frames'], ['FrameID','AbsoluteTime','frame_valid','timestamp_valid',*[f'angle{k}' for k in range(16)]], windows)
        coverage = read_windows(paths['coverage'], ['FrameID','AbsoluteTime','angular_valid_tail_fraction'], windows)
        np.testing.assert_array_equal(frames[['FrameID','AbsoluteTime']], coverage[['FrameID','AbsoluteTime']])
        ids = frames.FrameID.to_numpy(np.int64)
        steps = np.r_[0, np.diff(ids)]
        assert np.all(steps[1:] > 0)
        times = epoch + (ids-cadence.reference_frame_id)*cadence.interval_ms
        dt = steps*cadence.interval_ms
        position = frames.frame_valid.to_numpy(bool) & frames.timestamp_valid.to_numpy(bool)
        angles = frames[[f'angle{k}' for k in range(16)]].to_numpy(float)
        angles[~position] = np.nan
        delta = np.r_[np.nan, np.diff(angles.sum(axis=1))]
        derivative = position & np.r_[False,position[:-1]] & (steps==1) & (dt>0) & (dt<=10)
        raw = np.abs(np.arctan2(np.sin(delta),np.cos(delta)))/np.where(dt>0,dt,np.nan)
        raw[~derivative] = np.nan
        sizes = [_odd_window_samples(w,cadence.interval_ms) for w in (cfg.smoothing_window_ms,cfg.envelope_max_window_ms,cfg.envelope_min_window_ms)]
        smooth = smooth_contiguous_median(raw,steps,window_samples=sizes[0])
        envelope = rolling_extreme_envelope(smooth,steps,max_window_samples=sizes[1],min_window_samples=sizes[2])
        valid = derivative & np.isfinite(envelope) & (coverage.angular_valid_tail_fraction.to_numpy() >= cfg.minimum_valid_tail_fraction)
        moving,bouts = detect_legacy_envelope_bouts(envelope,raw,dt,steps,valid,
            envelope_threshold=cfg.envelope_threshold_rad_per_ms, amplitude_threshold=cfg.bout_amplitude_threshold_rad_per_ms,
            minimum_bout_duration_ms=cfg.minimum_bout_duration_ms,maximum_interbout_gap_ms=cfg.maximum_interbout_gap_ms)
        eligible = valid & moving & (bouts>0) & np.isfinite(raw) & (raw>0)
        slices=[];shown=np.zeros(len(ids),bool)
        for trial in range(5,95):
            onset=int(cycles.iloc[trial-1].Beg);a,b=np.searchsorted(times,[onset-20000,onset+20000])
            slices.append((trial,onset,a,b));shown[a:b]=True
        selected=np.unique(bouts[shown & eligible]);index=np.arange(len(ids));first=np.full(int(bouts.max())+1,len(ids));last=np.full_like(first,-1)
        np.minimum.at(first,bouts,index);np.maximum.at(last,bouts,index)
        starts=np.r_[0,np.flatnonzero(steps[1:]!=1)+1];ends=np.r_[starts[1:]-1,len(ids)-1];segments=np.searchsorted(starts,first[selected],side='right')-1
        margins=np.minimum(times[first[selected]]-times[starts[segments]],times[ends[segments]]-times[last[selected]])
        if len(margins) and np.min(margins) >= 1000:break
        assert context < 120000, 'Cannot certify detector context'
        context*=2
        del frames,coverage,angles,smooth,envelope,raw;gc.collect()
    width=(.5 if direct_medians else 1) if bin_width_s is None else bin_width_s
    edges=np.arange(-20,20+width,width)
    n_bins=len(edges)-1; baseline_start=int(5/width) if direct_medians else 0; baseline_end=int(20/width)
    tables=[];matrix=[];counts=[]
    for trial,onset,a,b in slices:
        t=(times[a:b]-onset)/1000;good=eligible[a:b];values=np.log(raw[a:b][good]);bins=np.searchsorted(edges,t[good],side='right')-1
        assert np.all((bins>=0)&(bins<n_bins))
        means,reference,centred,count = direct_log_bins(t[good], raw[a:b][good])
        if np.isfinite(reference):
            assert abs(np.nanmedian(centred[5:20])) < 1e-12
        assert np.array_equal(np.isnan(means),count==0)
        matrix.append(centred);counts.append(count)
        tables.append(pd.DataFrame(dict(fish=fish,acquisition_trial=trial,display_trial=trial-4,bin_index=np.arange(n_bins),start_s=edges[:-1],end_s=edges[1:],eligible_frames=count,bin_log_vigor=means,baseline_log_vigor=reference,delta_log_vigor=centred)))
    if not direct_medians:
        for table in tables:table.rename(columns={'bin_log_vigor':'mean_log_vigor','baseline_log_vigor':'baseline_mean_log_vigor'},inplace=True)
    result=np.asarray(matrix);snapshot.check()
    info=dict(fish=fish,experiment=experiment,cadence=asdict(cadence),detector=asdict(cfg),context_ms=context,
              minimum_selected_bout_context_margin_ms=float(np.min(margins)),expected_us_s=us_time,paired_training_trials=paired,
              sources=snapshot.hashes,finite_cells=int(np.isfinite(result).sum()),defined_trials=int(np.any(np.isfinite(result),axis=1).sum()),
              saturated_cells=int(np.sum(np.isfinite(result)&(np.abs(result)>.25))),baseline_zero_verified=True,
              bin_estimator='median' if direct_medians else 'mean',
              independent_frame_bins_verified=True,mean_log_values_sha256=sha(np.asarray(result,dtype='<f8').tobytes()))
    return result,pd.concat(tables,ignore_index=True),info
