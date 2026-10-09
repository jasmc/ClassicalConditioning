"""Recover valid non-bout frames separately from invalid/missing tracking."""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_full_bouts_baseline_samples_20261009'
sys.path[:0]=[str(REPO/'reviews/fgh_c_bout_vs_bins_20261008')]
from build_comparison import FISH
from build_review import (read_windows,digest,estimate_camera_cadence,CFG,_odd_window_samples,
    smooth_contiguous_median,rolling_extreme_envelope,detect_legacy_envelope_bouts)

def main():
    manifest=json.loads((SOURCE/'data_manifest.json').read_text())
    specs=list(FISH);g=list(specs[1]);g[2]='20230310_08';specs[1]=tuple(g)
    reports=[]
    for letter,_,fish,project,metric,*_ in specs:
        print('Classifying valid non-bout frames',letter,fish,flush=True)
        meta=next(r for r in manifest['panels'] if r['panel']==letter)
        for record in meta['inputs']:
            assert digest(Path(record['path']))==record['sha256']
        proc=Path(project)/'Processed data'/fish
        camera=pd.read_parquet(proc/'camera.parquet');cadence=estimate_camera_cadence(camera)
        anchor=float(camera.iloc[cadence.reference_position].AbsoluteTime)
        protocol=pd.read_parquet(proc/'stimulus_events.parquet')
        cycles=protocol[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
        context=meta['context_ms']
        intervals=[(int(t)-20000-context,int(t)+20001+context) for t in cycles.iloc[4:].Beg]
        suffix='' if letter=='G' else '-v1'
        frames=read_windows(proc/f'frame_preprocessed_corrected{suffix}.parquet',
            ['FrameID','AbsoluteTime','frame_valid','timestamp_valid',*[f'angle{k}' for k in range(16)]],intervals)
        coverage=read_windows(proc/metric,['FrameID','AbsoluteTime','angular_valid_tail_fraction'],intervals)
        np.testing.assert_array_equal(frames[['FrameID','AbsoluteTime']],coverage[['FrameID','AbsoluteTime']])
        ids=frames.FrameID.to_numpy(np.int64);steps=np.r_[0,np.diff(ids)];dt=steps*cadence.interval_ms
        position=frames.frame_valid.to_numpy(bool)&frames.timestamp_valid.to_numpy(bool)
        angles=frames[[f'angle{k}' for k in range(16)]].to_numpy(float);angles[~position]=np.nan
        delta=np.r_[np.nan,np.diff(angles.sum(axis=1))]
        derivative=position&np.r_[False,position[:-1]]&(steps==1)&(dt>0)&(dt<=10)
        raw=np.abs(np.arctan2(np.sin(delta),np.cos(delta)))/np.where(dt>0,dt,np.nan);raw[~derivative]=np.nan
        windows=[_odd_window_samples(w,cadence.interval_ms) for w in
            [CFG.smoothing_window_ms,CFG.envelope_max_window_ms,CFG.envelope_min_window_ms]]
        smoothed=smooth_contiguous_median(raw,steps,window_samples=windows[0])
        envelope=rolling_extreme_envelope(smoothed,steps,max_window_samples=windows[1],min_window_samples=windows[2])
        valid=derivative&np.isfinite(raw)&np.isfinite(envelope)&(coverage.angular_valid_tail_fraction.to_numpy()>=CFG.minimum_valid_tail_fraction)
        moving,bouts=detect_legacy_envelope_bouts(envelope,raw,dt,steps,valid,
            envelope_threshold=CFG.envelope_threshold_rad_per_ms,amplitude_threshold=CFG.bout_amplitude_threshold_rad_per_ms,
            minimum_bout_duration_ms=CFG.minimum_bout_duration_ms,maximum_interbout_gap_ms=CFG.maximum_interbout_gap_ms)
        eligible=valid&moving&(bouts>0)&(raw>0)
        saved_path=SOURCE/f'Panel{letter}_complete_bout_sample_data.parquet'
        source_hash=next(r['sha256'] for r in manifest['data_files'] if Path(r['path'])==saved_path)
        assert digest(saved_path)==source_hash
        shown=pd.read_parquet(saved_path,columns=['trial','FrameID','time_s','eligible','bout_id','log_vigor'])
        indices=np.searchsorted(ids,shown.FrameID.to_numpy())
        np.testing.assert_array_equal(ids[indices],shown.FrameID)
        np.testing.assert_array_equal(eligible[indices],shown.eligible)
        np.testing.assert_array_equal(bouts[indices],shown.bout_id)
        np.testing.assert_allclose(np.log(raw[indices[shown.eligible]]),shown.loc[shown.eligible,'log_vigor'],atol=1e-12,rtol=0)
        shown['data_valid']=valid[indices]
        shown['valid_nonbout']=valid[indices]&(~moving[indices])&(bouts[indices]==0)
        assert not (shown.eligible&shown.valid_nonbout).any()
        path=HERE/f'Panel{letter}_frame_classification.parquet'
        shown.to_parquet(path,index=False)
        reports.append({'panel':letter,'fish':fish,'frame_count':len(shown),'eligible_bout_frames':int(shown.eligible.sum()),
            'valid_nonbout_frames':int(shown.valid_nonbout.sum()),'remaining_ineligible_frames':int((~shown.eligible&~shown.valid_nonbout).sum()),
            'path':str(path),'sha256':digest(path),'source_table_sha256':source_hash,'detector_and_eligible_values_match':True})
        del camera,protocol,cycles,frames,coverage,angles,raw,smoothed,envelope,shown;gc.collect()
    record={'classification':'existing valid frame/derivative/envelope/coverage mask, then detected bout membership',
            'source_manifest':str(SOURCE/'data_manifest.json'),'source_manifest_sha256':digest(SOURCE/'data_manifest.json'),
            'ineligible_inside_bout_frames':'remain missing; are not automatically called non-bout','panels':reports}
    (HERE/'frame_classification_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
