"""Reproduce Figure 1E/F frame, detector and bin alignment without changing panels."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_figure1_legacy_vigor_heatmaps import read_windows
from classical_conditioning.analysis.movement_state import smooth_contiguous_median, rolling_extreme_envelope, detect_legacy_envelope_bouts
from classical_conditioning.analysis.temporal_profiles import _signed_bout_log_vigor

ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
PROJECT=Path('J:/Digested Data/allDelay-full-v1')
OUT=ROOT/'audit-vigor-alignment-20261006'
COL='legacy_distal_angular_speed_rad_per_ms'
TIME='Time relative to CS onset (s)'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def main():
    e=json.loads((ROOT/'traces/Fig1_PanelE_RawVigor_HeatmapOverlay_allTrials_v4.svg.json').read_text())
    f=json.loads((ROOT/'heatmaps/Fig1_PanelF_Delay_legacy-vigor_v2.svg.json').read_text())
    checked={}
    for a in e['input_artifacts']+f['input_artifacts']+[
        {'path':e[k],'sha256':e[k+'_sha256']} for k in ('frames','events','svg','panel_data')]+[
        {'path':f[k],'sha256':f[k+'_sha256']} for k in ('panel_data','svg')]:
        if a['path'] not in checked:
            checked[a['path']]=digest(a['path']); assert checked[a['path']]==a['sha256'],a['path']
    meta=PROJECT/'Metadata'; fish='20221115_07'; proc=PROJECT/'Processed data'/fish
    summary_path=PROJECT/'Quality checks'/fish/'movement-candidate-corrected-v2_summary.json'
    marker=json.loads((meta/f'{fish}_movement-candidate-corrected-v2_complete.json').read_text())
    assert marker['status']=='complete' and digest(summary_path)==marker['summary_sha256']
    summary=json.loads(summary_path.read_text()); assert summary['inputs']['candidate_metrics']['sha256']==checked[str(proc/'frame_activity_candidates-corrected-v1.parquet')]
    freeze=json.loads((Path(__file__).resolve().parents[1]/'configs/paper-figures/figure1-freeze.json').read_text())
    for panel in freeze['panels']:
        assert digest(ROOT/panel['source'])==panel['sha256']
    protocol=pd.read_parquet(proc/'stimulus_events.parquet')
    cycles=protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    raw=pd.read_parquet(e['frames']); heat=pd.read_parquet(f['panel_data'])
    OUT.mkdir(parents=True,exist_ok=True)
    joined=[]; binrows=[]; results=[]
    for trial in e['global_cs_trials']:
        onset=int(cycles.iloc[trial-1].Beg)
        interval=[(onset-22000,onset+22001)]
        m=read_windows(proc/'frame_activity_candidates-corrected-v1.parquet', ['FrameID','AbsoluteTime','ElapsedTime','FrameStep','DeltaTimeMs','valid_derivative','angular_valid_tail_fraction',COL],interval)
        d=read_windows(proc/'movement_state_candidates-corrected-v2.parquet',['FrameID','AbsoluteTime','valid','moving','bout_id'],interval)
        np.testing.assert_array_equal(m[['FrameID','AbsoluteTime']],d[['FrameID','AbsoluteTime']])
        smooth=smooth_contiguous_median(m[COL].to_numpy(),m.FrameStep.to_numpy(),window_samples=7)
        env=rolling_extreme_envelope(smooth,m.FrameStep.to_numpy(),max_window_samples=21,min_window_samples=403)
        valid=m.valid_derivative.to_numpy() & np.isfinite(env) & (m.angular_valid_tail_fraction.to_numpy()>=.8)
        moving,ids=detect_legacy_envelope_bouts(env,m[COL].to_numpy(),m.DeltaTimeMs.to_numpy(),m.FrameStep.to_numpy(),valid,envelope_threshold=np.deg2rad(4),amplitude_threshold=np.deg2rad(1),minimum_bout_duration_ms=40/700*1000,maximum_interbout_gap_ms=10/700*1000)
        m=m.assign(valid=d.valid.to_numpy(),moving=d.moving.to_numpy(),bout_id=d.bout_id.to_numpy(),smoothed=smooth,envelope=env,recomputed_valid=valid,recomputed_moving=moving)
        p=raw.loc[raw['Trial number'].eq(trial)].merge(m,on='FrameID',validate='one_to_one')
        assert len(p)==len(raw.loc[raw['Trial number'].eq(trial)])
        np.testing.assert_allclose(p.Vigor,p[COL],rtol=0,atol=0,equal_nan=True)
        timing=np.abs(p[TIME]-(p.AbsoluteTime-onset)/1000)
        assert timing.max()==0
        assert p.valid.eq(p.recomputed_valid).all() and p.moving.eq(p.recomputed_moving).all()
        p=p.loc[p[TIME].lt(20)].copy()
        p['cs_onset_ms']=onset; p['bin_index']=np.floor((p[TIME]+20)/.5).astype(int)
        p['eligible']=p.valid & p.moving & p.bout_id.gt(0) & np.isfinite(p.Vigor) & p.Vigor.gt(0)
        baseline=np.log(p.loc[p.eligible & p[TIME].ge(-15) & p[TIME].lt(0),'Vigor']).median()
        p['centred_log']=np.where(p.eligible,np.log(p.Vigor.where(p.Vigor.gt(0)))-baseline,np.nan)
        mapping=p.loc[p.eligible].groupby('bout_id').centred_log.median()
        p['bout_median']=p.bout_id.map(mapping).where(p.eligible)
        stored=heat.loc[heat['Trial number'].eq(trial)].sort_values('Time bin center (s)')['Signed log vigor'].to_numpy()
        rebuilt=_signed_bout_log_vigor(p.Vigor.to_numpy(),p[TIME].to_numpy(),p.bin_index.to_numpy(),p.valid.to_numpy(),p.moving.to_numpy(),p.bout_id.to_numpy(),bin_count=80,baseline_start_s=-15,baseline_end_s=0)
        np.testing.assert_allclose(rebuilt,stored,rtol=0,atol=1e-12,equal_nan=True)
        p['stored_bin_value']=stored[p.bin_index]; joined.append(p)
        for k in range(80):
            q=p.loc[p.bin_index.eq(k)]
            binrows.append(dict(trial=trial,bin_index=k,left_s=-20+k*.5,right_s=-19.5+k*.5,raw_max=q.Vigor.max(),frames=len(q),valid=int(q.valid.sum()),moving=int(q.moving.sum()),eligible=int(q.eligible.sum()),stored=float(stored[k]),rebuilt=float(rebuilt[k])))
        results.append(dict(trial=trial,frames=len(p),timing_error_ms=float(timing.max()*1000),baseline_log=float(baseline),valid_fraction=float(p.valid.mean()),eligible_fraction=float(p.eligible.mean()),missing_bins=int(np.isnan(stored).sum()),bin_max_error=float(np.nanmax(np.abs(stored-rebuilt))),frame_interval_ms_quantiles=p.DeltaTimeMs.quantile([0,.5,.99,1]).to_dict(),frame_gaps=int(p.FrameStep.gt(1).sum()),invalid_high_peaks=int((p.Vigor.gt(.1)&~p.valid).sum())))
    allframes=pd.concat(joined); bins=pd.DataFrame(binrows)
    allframes.to_parquet(OUT/'joined_frames.parquet',index=False); bins.to_csv(OUT/'bins.csv',index=False)
    candidates=bins.loc[bins.stored.isna() & bins.raw_max.gt(.1)].sort_values('raw_max',ascending=False)
    example=candidates.iloc[0]; trial=int(example.trial); left=float(example.left_s)-.5; right=float(example.right_s)+.5
    p=allframes.loc[allframes['Trial number'].eq(trial)&allframes[TIME].between(left,right)]
    p.to_csv(OUT/'example_frames.csv',index=False)
    plt.rcParams['svg.fonttype']='none'
    fig,ax=plt.subplots(4,1,figsize=(10,7),sharex=True,layout='constrained')
    ax[0].plot(p[TIME],p.Vigor,color='black',lw=.7); ax[0].set_ylabel('Raw vigor\nrad/ms')
    ax[1].plot(p[TIME],p.smoothed,label='7-frame median'); ax[1].plot(p[TIME],p.envelope,label='Envelope'); ax[1].axhline(np.deg2rad(4),ls='--',color='grey',label='4 deg/ms gate'); ax[1].legend(fontsize=8); ax[1].set_ylabel('Detector\nrad/ms')
    for col,y in [('valid',2),('moving',1),('eligible',0)]: ax[2].scatter(p[TIME],np.where(p[col],y,np.nan),s=3,label=col)
    ax[2].set_yticks([0,1,2],['Eligible','Moving','Valid']); ax[2].set_ylim(-.5,2.5)
    b=bins.loc[bins.trial.eq(trial)&bins.left_s.lt(right)&bins.right_s.gt(left)]
    for row in b.itertuples():
        if np.isfinite(row.stored): ax[3].bar(row.left_s,row.stored,width=.5,align='edge',color='#c85a17')
        else: ax[3].axvspan(row.left_s,row.right_s,color='black',alpha=.15); ax[3].text(row.left_s+.25,0,'NaN',ha='center',fontsize=8)
    ax[3].axhline(0,color='grey',lw=.5); ax[3].set_ylabel('Stored signed\nlog vigor'); ax[3].set_xlabel('Seconds relative to measured CS onset')
    for a in ax:
        a.set_xlim(left,right)
        for edge in np.arange(-20,20.5,.5): a.axvline(edge,color='grey',alpha=.2,lw=.5)
    fig.suptitle(f'Figure 1E alignment audit: trial {trial}, CS onset {int(p.cs_onset_ms.iloc[0])} ms')
    fig.savefig(OUT/'diagnostic.svg'); fig.savefig(OUT/'diagnostic.png',dpi=160); plt.close(fig)
    example_frames=allframes.loc[allframes['Trial number'].eq(trial)&allframes.bin_index.eq(int(example.bin_index))]
    failures=dict(total_frames=len(allframes),invalid_derivatives=int((~allframes.valid_derivative).sum()),nonfinite_envelopes=int((~np.isfinite(allframes.envelope)).sum()),example_invalid_derivatives=int((~example_frames.valid_derivative).sum()),example_nonfinite_smoothed=int((~np.isfinite(example_frames.smoothed)).sum()),example_nonfinite_envelopes=int((~np.isfinite(example_frames.envelope)).sum()))
    report=dict(verified_hashes=checked,frozen_panels_verified=[p['id'] for p in freeze['panels']],detector_summary=summary,trial_checks=results,example=example.to_dict(),validity_failures=failures,output=str(OUT),conclusion='All five raw traces match metric frames exactly; timing exact; detector masks reproduced; all 400 stored bins reproduced including NaNs. No signal alignment or binning defect established.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'trial_checks':results,'example':example.to_dict(),'output':str(OUT)},indent=2))
if __name__=='__main__': main()
