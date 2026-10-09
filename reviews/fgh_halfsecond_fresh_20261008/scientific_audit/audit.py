"""Scientific diagnostics; leave all figures and upstream artifacts unchanged."""
from pathlib import Path
import sys, json, gc
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_review import (FISH, read_windows, digest, estimate_camera_cadence,
    CFG, _odd_window_samples, smooth_contiguous_median, rolling_extreme_envelope,
    detect_legacy_envelope_bouts)

OUT=HERE/'outputs'
BINNED=HERE.parent/'outputs'

def qsummary(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return dict(zip(['min','p10','median','p90','max'],map(float,np.quantile(x,[0,.1,.5,.9,1])))) if len(x) else {}

def bins_mean(values,index):
    good=np.isfinite(values)
    count=np.bincount(index[good],minlength=80)
    total=np.bincount(index[good],weights=values[good],minlength=80)
    return np.divide(total,count,out=np.full(80,np.nan),where=count>0)

def audit(spec):
    panel,_,fish,project_name,metric_name,*_=spec
    print('Auditing source',panel,flush=True)
    proc=Path(project_name)/'Processed data'/fish
    old_manifest=json.loads((BINNED/'manifest.json').read_text())
    old_metadata=next(r for r in old_manifest['panels'] if r['panel']==panel)
    for r in old_metadata['inputs']:assert digest(Path(r['path']))==r['sha256']
    camera=pd.read_parquet(proc/'camera.parquet')
    cadence=estimate_camera_cadence(camera)
    anchor=float(camera.iloc[cadence.reference_position].AbsoluteTime)
    cycles=pd.read_parquet(proc/'stimulus_events.parquet')
    cycles=cycles[cycles.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    intervals=[(int(t)-22000,int(t)+22001) for t in cycles.iloc[4:].Beg]
    suffix='' if panel=='G' else '-v1'
    a=read_windows(proc/f'frame_preprocessed_corrected{suffix}.parquet',
        ['FrameID','AbsoluteTime','frame_valid','timestamp_valid',*[f'angle{k}' for k in range(16)]],intervals)
    cov=read_windows(proc/metric_name,['FrameID','AbsoluteTime','angular_valid_tail_fraction'],intervals)
    np.testing.assert_array_equal(a[['FrameID','AbsoluteTime']],cov[['FrameID','AbsoluteTime']])
    ids=a.FrameID.to_numpy(np.int64);steps=np.r_[0,np.diff(ids)]
    times=anchor+(ids-cadence.reference_frame_id)*cadence.interval_ms
    dt=steps*cadence.interval_ms
    position=a.frame_valid.to_numpy(bool)&a.timestamp_valid.to_numpy(bool)
    angles=a[[f'angle{k}' for k in range(16)]].to_numpy(float);angles[~position]=np.nan
    difference=np.r_[np.nan,np.diff(angles.sum(axis=1))]
    derivative=position&np.r_[False,position[:-1]]&(steps==1)&(dt>0)&(dt<=10)
    raw=np.abs(np.arctan2(np.sin(difference),np.cos(difference)))/np.where(dt>0,dt,np.nan)
    raw[~derivative]=np.nan
    windows=[_odd_window_samples(w,cadence.interval_ms) for w in
        [CFG.smoothing_window_ms,CFG.envelope_max_window_ms,CFG.envelope_min_window_ms]]
    sm=smooth_contiguous_median(raw,steps,window_samples=windows[0])
    en=rolling_extreme_envelope(sm,steps,max_window_samples=windows[1],min_window_samples=windows[2])
    valid=derivative&np.isfinite(en)&(cov.angular_valid_tail_fraction.to_numpy()>=CFG.minimum_valid_tail_fraction)
    moving,bout=detect_legacy_envelope_bouts(en,raw,dt,steps,valid,
        envelope_threshold=CFG.envelope_threshold_rad_per_ms,
        amplitude_threshold=CFG.bout_amplitude_threshold_rad_per_ms,
        minimum_bout_duration_ms=CFG.minimum_bout_duration_ms,
        maximum_interbout_gap_ms=CFG.maximum_interbout_gap_ms)
    eligible=valid&moving&(bout>0)&np.isfinite(raw)&(raw>0)
    log=np.log(np.where(eligible,raw,np.nan))
    context=pd.DataFrame({'bout_id':bout[eligible],'log_raw':log[eligible],'time_ms':times[eligible]})
    context_medians=context.groupby('bout_id').log_raw.median()
    context_extent=context.groupby('bout_id').time_ms.agg(['min','max'])
    delivered=pd.read_parquet(BINNED/f'Panel{panel}_scalar_bins.parquet')
    binparts=[];trials=[];boutparts=[];crossings=[]
    for trial in range(5,95):
        onset=int(cycles.iloc[trial-1].Beg)
        first,last=np.searchsorted(times,[onset-20000,onset+20000]);s=slice(first,last)
        t=(times[s]-onset)/1000;index=np.floor((t+20)/.5).astype(int)
        p=pd.DataFrame({'time_s':t,'bin_index':index,'bout_id':bout[s],
            'eligible':eligible[s],'log_raw':log[s],'detector_valid':valid[s],
            'moving':moving[s],'raw':raw[s]})
        e=p[p.eligible].copy()
        visible_q=e.groupby('bout_id').log_raw.median()
        e['q_visible']=e.bout_id.map(visible_q)
        e['q_context']=e.bout_id.map(context_medians)
        current=bins_mean(e.q_visible.to_numpy(),e.bin_index.to_numpy())
        direct=bins_mean(e.log_raw.to_numpy(),e.bin_index.to_numpy())
        context_bins=bins_mean(e.q_context.to_numpy(),e.bin_index.to_numpy())
        old=delivered[delivered.trial.eq(trial)].sort_values('bin_index')
        np.testing.assert_allclose(current,old.uncentred_log_bin,atol=1e-12,rtol=0,equal_nan=True)
        zcurrent=current-np.nanmedian(current[10:40])
        zdirect=direct-np.nanmedian(direct[10:40])
        base=p[(p.time_s>=-15)&(p.time_s<0)]
        baseline_bouts=e[(e.time_s>=-15)&(e.time_s<0)].bout_id.nunique()
        eg=e.groupby('bout_id').agg(start_s=('time_s','min'),end_s=('time_s','max'),
            eligible_frames=('time_s','size'),bins_spanned=('bin_index','nunique'),
            visible_median=('q_visible','first'),context_median=('q_context','first'))
        eg['context_start_s']=(eg.index.map(context_extent['min'])-onset)/1000
        eg['context_end_s']=(eg.index.map(context_extent['max'])-onset)/1000
        eg['window_truncated']=(eg.context_start_s<-20)|(eg.context_end_s>=20)
        eg['eligible_span_s']=eg.end_s-eg.start_s+cadence.interval_ms/1000
        eg['median_crop_difference']=eg.visible_median-eg.context_median
        eg['trial']=trial;eg['panel']=panel
        boutparts.append(eg.reset_index())
        for boundary in [-15,0,9 if panel=='F' else 13 if panel=='G' else 10,10]:
            for bout_id,g in e.groupby('bout_id'):
                pre=g[g.time_s<boundary];post=g[g.time_s>=boundary]
                if len(pre) and len(post):
                    crossings.append({'panel':panel,'trial':trial,'boundary_s':boundary,'bout_id':int(bout_id),
                        'start_s':float(g.time_s.min()),'end_s':float(g.time_s.max()),
                        'median_pre':float(pre.log_raw.median()),'median_post':float(post.log_raw.median()),
                        'median_assigned':float(g.q_visible.iloc[0]),
                        'pre_samples':len(pre),'post_samples':len(post)})
        counts=np.bincount(index[eligible[s]],minlength=80)
        validcounts=np.bincount(index[valid[s]],minlength=80)
        total=np.bincount(index,minlength=80)
        row=pd.DataFrame({'panel':panel,'trial':trial,'bin_index':range(80),
            'eligible_frames':counts,'detector_valid_frames':validcounts,'total_frames':total,
            'current_log_bin':current,'direct_log_bin':direct,'context_bout_log_bin':context_bins,
            'current_centred':zcurrent,'direct_centred':zdirect,
            'difference_after_own_baseline':zcurrent-zdirect})
        binparts.append(row)
        trials.append({'panel':panel,'trial':trial,'finite_baseline_bins':int(np.isfinite(current[10:40]).sum()),
            'unique_baseline_scalars':len(np.unique(current[10:40][np.isfinite(current[10:40])])),
            'distinct_baseline_bouts':int(baseline_bouts),
            'baseline_eligible_fraction':float(base.eligible.mean()),
            'baseline_detector_valid_fraction':float(base.detector_valid.mean()),
            'C_scale':float(old.C_scale.iloc[0]),'D_scale':float(old.D_scale.iloc[0]),
            'crop_changed_bins':int((np.abs(current-context_bins)>1e-12).sum()),
            'current_vs_direct_mean_absolute_log_difference':float(np.nanmean(np.abs(zcurrent-zdirect)))})
    bins=pd.concat(binparts,ignore_index=True); ts=pd.DataFrame(trials); bs=pd.concat(boutparts,ignore_index=True)
    cross=pd.DataFrame(crossings).drop_duplicates(['panel','trial','boundary_s','bout_id'])
    finite=bins.current_log_bin.notna()
    absent=~finite
    summary={'panel':panel,'fps':cadence.framerate,'detector_samples':windows,
        'detector_maximum_future_support_ms':((windows[0]-1)/2+(windows[2]-1)/2)*cadence.interval_ms,
        'finite_bins':int(finite.sum()),'empty_bins':int(absent.sum()),
        'empty_bins_with_full_detector_support':int((absent&(bins.detector_valid_frames==bins.total_frames)).sum()),
        'empty_bins_with_no_detector_support':int((absent&bins.detector_valid_frames.eq(0)).sum()),
        'eligible_fraction_of_finite_bins':qsummary((bins.eligible_frames/bins.total_frames)[finite]),
        'finite_bins_with_10_or_fewer_eligible_samples':int((finite&bins.eligible_frames.le(10)).sum()),
        'baseline_finite_bins':qsummary(ts.finite_baseline_bins),
        'trials_with_fewer_than_10_finite_baseline_bins':int(ts.finite_baseline_bins.lt(10).sum()),
        'distinct_baseline_bouts':qsummary(ts.distinct_baseline_bouts),
        'C_scale':qsummary(ts.C_scale),'D_scale':qsummary(ts.D_scale),
        'defined_C_scale_max_min_ratio':float(ts.C_scale.max()/ts.loc[ts.C_scale.gt(0),'C_scale'].min()),
        'defined_D_scale_max_min_ratio':float(ts.D_scale.max()/ts.loc[ts.D_scale.gt(0),'D_scale'].min()),
        'bout_count':len(bs),'bout_eligible_span_s':qsummary(bs.eligible_span_s),
        'bouts_spanning_multiple_halfsecond_bins':int(bs.bins_spanned.gt(1).sum()),
        'bouts_crossing_CS_onset':int(cross.boundary_s.eq(0).sum()),
        'bouts_crossing_baseline_start':int(cross.boundary_s.eq(-15).sum()),
        'window_truncated_bouts':int(bs.window_truncated.sum()),
        'window_truncated_bouts_with_changed_medians':int((bs.median_crop_difference.abs()>1e-12).sum()),
        'bins_changed_by_context_vs_visible_bout_median':int((np.abs(bins.current_log_bin-bins.context_bout_log_bin)>1e-12).sum()),
        'direct_vs_bout_centred_absolute_bin_difference':qsummary(bins.difference_after_own_baseline.abs()),
        'direct_vs_bout_opposite_sign_bins':int((bins.current_centred*bins.direct_centred<0).sum()),
        'visible_wrapped_changes_over_pi':int(sum(np.sum(np.abs(difference[np.searchsorted(times,int(onset)-20000):np.searchsorted(times,int(onset)+20000)])>np.pi) for onset in cycles.iloc[4:].Beg)),
        'clipping':{col:{'all_finite_endpoint_fraction':float(delivered[col].dropna().abs().ge(1).mean()),
            'baseline_finite_endpoint_fraction':float(delivered.loc[delivered.bin_index.between(10,39),col].dropna().abs().ge(1).mean())} for col in ['C','D']}}
    print(json.dumps(summary,indent=2),flush=True)
    del camera,a,cov,angles,sm,en,context
    gc.collect()
    return summary,bins,ts,bs,cross

def main():
    OUT.mkdir(exist_ok=False)
    summaries=[];bins=[];trials=[];bouts=[];cross=[]
    for spec in FISH:
        s,b,t,q,c=audit(spec);summaries.append(s);bins.append(b);trials.append(t);bouts.append(q);cross.append(c)
    pd.concat(bins).to_parquet(OUT/'bin_diagnostics.parquet',index=False)
    pd.concat(trials).to_csv(OUT/'trial_diagnostics.csv',index=False)
    pd.concat(bouts).to_parquet(OUT/'bout_diagnostics.parquet',index=False)
    allcross=pd.concat(cross,ignore_index=True);allcross.to_csv(OUT/'boundary_crossing_bouts.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps({'panels':summaries,'code_sha256':digest(Path(__file__)),
        'context_note':'context medians use the loaded 44 s neighbourhood; not certified whole-recording bouts'},indent=2))
    print('Audit complete',flush=True)

if __name__=='__main__':main()
