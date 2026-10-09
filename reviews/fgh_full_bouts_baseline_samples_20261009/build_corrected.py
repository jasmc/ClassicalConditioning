"""Complete-bout medians and a common timepoint baseline for all three rows."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import sys,json,gc
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(REPO/'reviews/fgh_c_bout_vs_bins_20261008'),str(REPO/'reviews/fgh_v1_trace20230310_08_20261009')]
import build_comparison as original
import build_new_versions as plotting
from build_review import (read_windows,digest,estimate_camera_cadence,CFG,_odd_window_samples,
    smooth_contiguous_median,rolling_extreme_envelope,detect_legacy_envelope_bouts)
SOURCE_FREEZE=REPO/'reviews/fgh_v1_trace20230310_08_20261009/frozen-version1/freeze.json'
EDGES=np.arange(-20,20.5,.5)

def reconstruct(spec,metadata,context_ms):
    letter,_,fish,project_name,metric_name,*_=spec
    proc=Path(project_name)/'Processed data'/fish
    camera=pd.read_parquet(proc/'camera.parquet');cadence=estimate_camera_cadence(camera)
    assert not cadence.has_frame_loss_evidence
    anchor=float(camera.iloc[cadence.reference_position].AbsoluteTime)
    protocol=pd.read_parquet(proc/'stimulus_events.parquet')
    cycles=protocol[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    assert len(cycles)==94
    intervals=[(int(t)-20000-context_ms,int(t)+20001+context_ms) for t in cycles.iloc[4:].Beg]
    suffix='' if letter=='G' else '-v1'
    frames=read_windows(proc/f'frame_preprocessed_corrected{suffix}.parquet',
        ['FrameID','AbsoluteTime','frame_valid','timestamp_valid',*[f'angle{k}' for k in range(16)]],intervals)
    coverage=read_windows(proc/metric_name,['FrameID','AbsoluteTime','angular_valid_tail_fraction'],intervals)
    np.testing.assert_array_equal(frames[['FrameID','AbsoluteTime']],coverage[['FrameID','AbsoluteTime']])
    ids=frames.FrameID.to_numpy(np.int64);steps=np.r_[0,np.diff(ids)]
    times=anchor+(ids-cadence.reference_frame_id)*cadence.interval_ms;dt=steps*cadence.interval_ms
    position=frames.frame_valid.to_numpy(bool)&frames.timestamp_valid.to_numpy(bool)
    angles=frames[[f'angle{k}' for k in range(16)]].to_numpy(float);angles[~position]=np.nan
    delta=np.r_[np.nan,np.diff(angles.sum(axis=1))]
    derivative=position&np.r_[False,position[:-1]]&(steps==1)&(dt>0)&(dt<=10)
    raw=np.abs(np.arctan2(np.sin(delta),np.cos(delta)))/np.where(dt>0,dt,np.nan);raw[~derivative]=np.nan
    sizes=[_odd_window_samples(w,cadence.interval_ms) for w in
        [CFG.smoothing_window_ms,CFG.envelope_max_window_ms,CFG.envelope_min_window_ms]]
    smooth=smooth_contiguous_median(raw,steps,window_samples=sizes[0])
    envelope=rolling_extreme_envelope(smooth,steps,max_window_samples=sizes[1],min_window_samples=sizes[2])
    valid=derivative&np.isfinite(envelope)&(coverage.angular_valid_tail_fraction.to_numpy()>=CFG.minimum_valid_tail_fraction)
    moving,bouts=detect_legacy_envelope_bouts(envelope,raw,dt,steps,valid,
        envelope_threshold=CFG.envelope_threshold_rad_per_ms,amplitude_threshold=CFG.bout_amplitude_threshold_rad_per_ms,
        minimum_bout_duration_ms=CFG.minimum_bout_duration_ms,maximum_interbout_gap_ms=CFG.maximum_interbout_gap_ms)
    eligible=valid&moving&(bouts>0)&np.isfinite(raw)&(raw>0)
    trial_slices=[];display=np.zeros(len(ids),bool)
    for trial in range(5,95):
        onset=int(cycles.iloc[trial-1].Beg)
        start,stop=np.searchsorted(times,[onset-20000,onset+20000])
        trial_slices.append((trial,onset,int(start),int(stop)));display[start:stop]=True
    selected=np.unique(bouts[display&eligible]);assert (selected>0).all()
    # Whole selected bouts must end safely inside each loaded frame segment.
    # Expand context automatically if a bout could have been clipped by the read.
    indices=np.arange(len(ids));first=np.full(int(bouts.max())+1,len(ids),np.int64);last=np.full_like(first,-1)
    np.minimum.at(first,bouts,indices);np.maximum.at(last,bouts,indices)
    starts=np.r_[0,np.flatnonzero(steps[1:]!=1)+1];ends=np.r_[starts[1:]-1,len(ids)-1]
    seg=np.searchsorted(starts,first[selected],side='right')-1
    margins=np.minimum(times[first[selected]]-times[starts[seg]],times[ends[seg]]-times[last[selected]])
    if (margins<1000).any():
        print('Expanding read context for',letter,'to',context_ms*2,'ms',flush=True)
        del frames,coverage,camera,angles,smooth,envelope,raw;gc.collect()
        assert context_ms<120000,'A selected bout cannot yet be certified complete'
        return None
    log=np.full(len(raw),np.nan);log[eligible]=np.log(raw[eligible])
    full=pd.DataFrame({'bout_id':bouts[eligible],'log_vigor':log[eligible]})
    summaries=full.groupby('bout_id').log_vigor.agg(['median','count'])
    median=np.full(len(ids),np.nan);median[eligible]=summaries['median'].reindex(bouts[eligible]).to_numpy()
    support=eligible&np.isin(bouts,selected)
    pd.DataFrame({'FrameID':ids[support],'absolute_ms':times[support],'bout_id':bouts[support],
        'log_vigor':log[support]}).to_parquet(HERE/f'Panel{letter}_complete_bout_support.parquet',index=False)
    bout_rows=[]
    for bout in selected:
        bout_rows.append({'bout_id':int(bout),'first_frame_id':int(ids[first[bout]]),'last_frame_id':int(ids[last[bout]]),
            'start_absolute_ms':float(times[first[bout]]),'end_absolute_ms':float(times[last[bout]]+cadence.interval_ms),
            'eligible_sample_count':int(summaries.loc[bout,'count']),'median_log_vigor':float(summaries.loc[bout,'median']),
            'minimum_context_margin_ms':float(margins[np.searchsorted(selected,bout)])})
    shown=[];boundary_counts=[]
    for trial,onset,start,stop in trial_slices:
        p=pd.DataFrame({'trial':trial,'FrameID':ids[start:stop],'time_s':(times[start:stop]-onset)/1000,
            'eligible':eligible[start:stop],'bout_id':bouts[start:stop],'raw_vigor':np.where(eligible[start:stop],raw[start:stop],np.nan),
            'log_vigor':log[start:stop],'bout_median_log':median[start:stop]})
        trial_bouts=np.unique(p.loc[p.eligible,'bout_id'])
        crossing=[int(b) for b in trial_bouts if first[b]<start or last[b]>=stop]
        baseline_bouts=np.unique(p.loc[p.eligible&p.time_s.ge(-15)&p.time_s.lt(0),'bout_id'])
        crossing_cs=[int(b) for b in baseline_bouts if times[last[b]]>=onset]
        cropped=p.loc[p.eligible].groupby('bout_id').log_vigor.transform('median')
        changed=~np.isclose(p.loc[p.eligible,'bout_median_log'],cropped,rtol=0,atol=1e-12)
        boundary_counts.append({'trial':trial,'bouts_crossing_plot_boundary':len(crossing),
            'changed_eligible_sample_medians':int(changed.sum()),'baseline_bouts_crossing_CS':len(crossing_cs)})
        shown.append(p)
    del camera,frames,coverage,angles,smooth,envelope,raw,full;gc.collect()
    return pd.concat(shown,ignore_index=True),cadence,pd.DataFrame(bout_rows),boundary_counts

def build():
    freeze=json.loads(SOURCE_FREEZE.read_text())
    prior=json.loads((REPO/'reviews/fgh_c_bout_vs_bins_20261008/outputs/manifest.json').read_text())
    source_tables={r['panel']:r for r in freeze['original_sample_tables']}
    allstats=[];metadata=[]
    specs=list(original.FISH);g=list(specs[1]);g[2]='20230310_08';specs[1]=tuple(g)
    for spec in specs:
        letter=spec[0];fish=spec[2]
        meta=freeze['replacement_inputs'] if letter=='G' else next(r for r in prior['source_inputs'] if r['panel']==letter)
        for item in meta['inputs']:assert digest(Path(item['path']))==item['sha256']
        context=5000;result=None
        print('Reconstructing complete bouts',letter,fish,flush=True)
        while result is None:
            result=reconstruct(spec,meta,context);context*=2
        frames,cadence,bouts,boundaries=result
        original_frames=pd.read_parquet(source_tables[letter]['path'],columns=['FrameID','trial','eligible','log_vigor'])
        np.testing.assert_array_equal(frames[['trial','FrameID']],original_frames[['trial','FrameID']])
        np.testing.assert_array_equal(frames.eligible,original_frames.eligible)
        np.testing.assert_allclose(frames.log_vigor,original_frames.log_vigor,atol=1e-12,rtol=0,equal_nan=True)
        sample_runs={m:[] for m in 'CD'};allbins=[]
        frames['C_sample']=np.nan;frames['D_sample']=np.nan
        for trial,p in frames.groupby('trial',sort=True):
            base=p.time_s.ge(-15)&p.time_s.lt(0)&p.eligible
            base_values=p.loc[base,'bout_median_log']
            if len(base_values):lo,m,hi=np.quantile(base_values,[.1,.5,.9],method='linear')
            else:lo=m=hi=np.nan
            c_width=(hi-lo)/2;d_width=max(m-lo,hi-m)
            allstats.append({'panel':letter,'fish':fish,'trial':int(trial),'baseline_timepoint_count':len(base_values),
                'baseline_distinct_bouts':int(p.loc[base,'bout_id'].nunique()),'p10':float(lo),'p50':float(m),'p90':float(hi),
                'C_scale':float(c_width),'D_scale':float(d_width)})
            for mode,width in [('C',c_width),('D',d_width)]:
                value=np.clip((p.bout_median_log.to_numpy()-m)/width,-1,1) if width>0 else np.full(len(p),np.nan)
                if width>0:assert abs(np.nanmedian(value[base]))<1e-11
                frames.loc[p.index,mode+'_sample']=value
                q=p.copy();q['C_sample']=value
                sample_runs[mode].append(original.sample_runs(q,cadence).rename(columns={'C':mode}))
            index=np.searchsorted(EDGES,p.time_s.to_numpy(),side='right')-1;good=p.eligible.to_numpy()
            count=np.bincount(index[good],minlength=80)
            sums=np.bincount(index[good],weights=p.log_vigor.to_numpy()[good],minlength=80)
            means=np.divide(sums,count,out=np.full(80,np.nan),where=count>0)
            direct=p.loc[good].assign(bin_index=index[good]).groupby('bin_index').log_vigor.mean().reindex(range(80)).to_numpy()
            np.testing.assert_allclose(means,direct,atol=1e-12,rtol=0,equal_nan=True)
            normalized=np.clip((means-m)/c_width,-1,1) if c_width>0 else np.full(80,np.nan)
            allbins.append(pd.DataFrame({'trial':int(trial),'bin_index':range(80),'start_s':EDGES[:-1],'end_s':EDGES[1:],
                'eligible_frame_count':count,'mean_log_vigor':means,'C':normalized,
                'baseline_reference_median':m,'baseline_reference_P10':lo,'baseline_reference_P90':hi}))
        for mode in 'CD':
            runs=pd.concat([r for r in sample_runs[mode] if len(r)],ignore_index=True)
            runs.to_csv(HERE/f'Panel{letter}_{mode}_sample_runs.csv',index=False)
        pd.concat(allbins,ignore_index=True).to_csv(HERE/f'Panel{letter}_direct_bins.csv',index=False)
        frames.to_parquet(HERE/f'Panel{letter}_complete_bout_sample_data.parquet',index=False)
        bouts.to_csv(HERE/f'Panel{letter}_complete_bouts.csv',index=False)
        pd.DataFrame(boundaries).to_csv(HERE/f'Panel{letter}_boundary_audit.csv',index=False)
        metadata.append({'panel':letter,'fish':fish,'inputs':meta['inputs'],'cadence':asdict(cadence),
            'complete_selected_bouts':len(bouts),'minimum_context_margin_ms':float(bouts.minimum_context_margin_ms.min()),
            'context_ms':context//2,'eligible_support_and_direct_logs_identical_to_previous':True,
            'changed_eligible_sample_medians':sum(r['changed_eligible_sample_medians'] for r in boundaries),
            'baseline_bouts_crossing_CS':sum(r['baseline_bouts_crossing_CS'] for r in boundaries)})
        del frames,original_frames;gc.collect()
        print('Saved corrected',letter,flush=True)
    pd.DataFrame(allstats).to_csv(HERE/'baseline_statistics.csv',index=False)
    manifest={'status':'corrected complete-bout review; supersedes crop-limited bout summaries and binned baselines',
        'source_freeze':str(SOURCE_FREEZE),'source_freeze_sha256':digest(SOURCE_FREEZE),'fish':freeze['fish'],'panels':metadata,
        'C':'clip((x-P50)/((P90-P10)/2),-1,1)','D':'clip((x-P50)/max(P50-P10,P90-P50),-1,1)',
        'baseline':'P10/P50/P90 from all eligible baseline timepoints carrying complete-bout median log values; same reference for every variant',
        'baseline_interval':'[-15,0)','display_interval':'[-20,20)','bout_summary':'median of every eligible log sample of each complete detected bout, beyond plotting crop',
        'Version2':'direct eligible framewise log means in 0.5 s cells; no bout-median substitution in displayed means',
        'scaling':'median-centred percentile C/D; conventional affine min-max rejected because it does not preserve baseline median zero',
        'quantile_method':'NumPy linear','palette':'managua_r','limits':[-1,1],
        'data_files':[{'path':str(p),'sha256':digest(p)} for p in list(HERE.glob('*.csv'))+list(HERE.glob('*.parquet'))]}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    if '--data-only' in sys.argv:build()
    else:
        plotting.OUT=HERE
        plotting.render(sys.argv[1])
