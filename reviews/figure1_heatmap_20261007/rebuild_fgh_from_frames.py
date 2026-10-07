"""Only F/G/H: independent frame-first reconstruction of corrected E contract.

Does not invoke the v5 builder or signed-log helper. Never creates an assembly.
"""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image,ImageOps,ImageDraw
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT,FISH,read_windows,digest
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
from classical_conditioning.analysis.movement_state import MovementCalibrationConfig,_odd_window_samples,smooth_contiguous_median,rolling_extreme_envelope,detect_legacy_envelope_bouts
OUT=ROOT/'fgh-rebuild-panel-e-contract-20261007'
V5=ROOT/'cadence-review-v5-20261007'
TIME='Time relative to CS onset (s)'
OUT.mkdir(parents=True,exist_ok=True)
reports=[];contributions=[]
cfg=MovementCalibrationConfig()
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':9})
cmap=plt.get_cmap('managua_r').copy();cmap.set_bad('black');norm=Normalize(-.25,.25,clip=True)

def plot_panel(spec,bins):
    panel,name,fish,*_=spec
    fig=plt.figure(figsize=(6.1,5.5))
    grid=fig.add_gridspec(3,1,height_ratios=[10,50,30],left=.17,right=.78,bottom=.16,top=.80,hspace=.10)
    matrix=bins.pivot(index='trial',columns='bin_center_s',values='signed_bout_log_bin')
    for i,(phase,lo,hi) in enumerate([('Pre-Train',5,14),('Train',15,64),('Test',65,94)]):
        a=fig.add_subplot(grid[i]);values=matrix.reindex(range(lo,hi+1)).to_numpy()
        a.imshow(values,aspect='auto',origin='upper',interpolation='nearest',extent=(-20,20,hi+.5,lo-.5),cmap=cmap,norm=norm)
        a.set_ylim(hi+.5,lo-.5);a.set_yticks([lo,hi]);a.tick_params(labelsize=8,length=2)
        a.set_ylabel(phase,fontsize=9)
        for sec in [0,10]:a.axvline(sec,color='#0d7f3c',lw=.8,ls='--' if sec==10 else '-')
        if phase=='Train' and spec[-1] is not None:a.axvline(spec[-1],color='#78358c',ls=':',lw=.9)
        a.set_xticks([-20,-10,0,10,20]);a.set_xlim(-20,20)
        if i<2:a.tick_params(labelbottom=False)
        else:a.set_xlabel('Time from measured CS onset (s)',fontsize=10)
    fig.text(.04,.955,panel,fontsize=19,weight='bold')
    fig.text(.17,.955,f'{name} · {fish}',fontsize=13,weight='bold')
    fig.text(.17,.905,'Tail bend angular speed · trial-centred bout log vigor',fontsize=9)
    fig.text(.17,.865,'Each trial uses its own baseline [−15, 0) s',fontsize=9)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.825,.16,.023,.64]),ticks=[-.25,0,.25],extend='both')
    cb.ax.tick_params(labelsize=8);cb.set_label('Mean centred bout log vigor',fontsize=9)
    fig.text(.17,.055,'0.5 s bins; finite eligible frames only; black = no contribution',fontsize=8)
    fig.text(.17,.026,'managua_r · colour saturation only; stored values are uncapped',fontsize=8)
    stem=OUT/f'Fig1_Panel{panel}_{name.replace(" ","")}_rebuilt'
    for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    return stem

for spec in FISH:
    panel,name,fish,project_name,metric_name,*_=spec
    print('Independent frame rebuild:',panel,name,flush=True)
    oldsvg=V5/f'Fig1_Panel{panel}_{name.replace(" ","")}_presumed-cadence_v5.svg'
    oldmeta=json.loads(oldsvg.with_suffix('.svg.json').read_text())
    for item in oldmeta['input_artifacts']:assert digest(Path(item['path']))==item['sha256'],item['path']
    assert digest(Path(oldmeta['panel_data']))==oldmeta['panel_data_sha256']
    project=Path(project_name);proc=project/'Processed data'/fish
    source_manifest=project/'Metadata'/f'{fish}_source_manifest.json'
    manifest=json.loads(source_manifest.read_text())
    camera_path=proc/'camera.parquet';assert digest(camera_path)==manifest['artifacts']['camera']['sha256']
    camera=pd.read_parquet(camera_path);cadence=estimate_camera_cadence(camera)
    assert not cadence.has_frame_loss_evidence
    anchor=float(camera.iloc[cadence.reference_position].AbsoluteTime)
    suffix='' if panel=='G' else '-v1'
    angles_path=proc/f'frame_preprocessed_corrected{suffix}.parquet'
    marker_path=project/'Metadata'/f'{fish}_corrected-preprocess{suffix}_complete.json'
    marker=json.loads(marker_path.read_text());assert marker['status']=='complete'
    assert digest(angles_path)==marker['frames_sha256']
    protocol_path=proc/'stimulus_events.parquet'
    assert digest(protocol_path)==manifest['artifacts']['protocol']['sha256']
    protocol=pd.read_parquet(protocol_path)
    cycles=protocol[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True);assert len(cycles)==94
    intervals=[(int(t)-22000,int(t)+22001) for t in cycles.iloc[4:].Beg]
    angles=read_windows(angles_path,['FrameID','AbsoluteTime','frame_valid','timestamp_valid',*[f'angle{k}' for k in range(16)]],intervals)
    coverage=read_windows(proc/metric_name,['FrameID','AbsoluteTime','angular_valid_tail_fraction'],intervals)
    np.testing.assert_array_equal(angles[['FrameID','AbsoluteTime']],coverage[['FrameID','AbsoluteTime']])
    ids=angles.FrameID.to_numpy(np.int64);steps=np.r_[0,np.diff(ids)]
    acquired_ms=anchor+(ids-cadence.reference_frame_id)*cadence.interval_ms
    dt=steps*cadence.interval_ms
    valid_position=angles.frame_valid.to_numpy(bool)&angles.timestamp_valid.to_numpy(bool)
    local=angles[[f'angle{k}' for k in range(16)]].to_numpy(float);local[~valid_position]=np.nan
    bend=np.sum(local,axis=1)
    difference=np.r_[np.nan,np.diff(bend)]
    derivative_valid=valid_position&np.r_[False,valid_position[:-1]]&(steps==1)&(dt>0)&(dt<=10)
    raw=np.abs(np.arctan2(np.sin(difference),np.cos(difference)))/np.where(dt>0,dt,np.nan)
    raw[~derivative_valid]=np.nan
    windows=[_odd_window_samples(w,cadence.interval_ms) for w in [cfg.smoothing_window_ms,cfg.envelope_max_window_ms,cfg.envelope_min_window_ms]]
    smoothed=smooth_contiguous_median(raw,steps,window_samples=windows[0])
    envelope=rolling_extreme_envelope(smoothed,steps,max_window_samples=windows[1],min_window_samples=windows[2])
    detector_valid=derivative_valid&np.isfinite(envelope)&(coverage.angular_valid_tail_fraction.to_numpy()>=cfg.minimum_valid_tail_fraction)
    moving,boutids=detect_legacy_envelope_bouts(envelope,raw,dt,steps,detector_valid,envelope_threshold=cfg.envelope_threshold_rad_per_ms,amplitude_threshold=cfg.bout_amplitude_threshold_rad_per_ms,minimum_bout_duration_ms=cfg.minimum_bout_duration_ms,maximum_interbout_gap_ms=cfg.maximum_interbout_gap_ms)
    trialstats=[];binparts=[];support=[];exampleframes=[]
    for trial in range(5,95):
        onset=int(cycles.iloc[trial-1].Beg);start=np.searchsorted(acquired_ms,onset-20000);stop=np.searchsorted(acquired_ms,onset+20000)
        s=slice(start,stop);t=(acquired_ms[s]-onset)/1000
        eligible=detector_valid[s]&moving[s]&(boutids[s]>0)&np.isfinite(raw[s])&(raw[s]>0)
        p=pd.DataFrame({'trial':trial,'FrameID':ids[s],'time_s':t,'raw_on_eligible':np.where(eligible,raw[s],np.nan),
            'detector_valid':detector_valid[s],'moving':moving[s],'eligible':eligible,'bout_id':boutids[s],
            'bin_index':np.floor((t+20)/.5).astype(int)})
        p['frame_log']=np.log(p.raw_on_eligible)
        baseline_values=p.loc[p.time_s.ge(-15)&p.time_s.lt(0),'frame_log'].dropna()
        baseline=float(baseline_values.median()) if len(baseline_values) else np.nan
        p['centred_log']=p.frame_log-baseline
        p['bout_median']=p.loc[p.eligible].groupby('bout_id').centred_log.transform('median').reindex(p.index)
        for c in ['raw_on_eligible','frame_log','centred_log','bout_median']:
            np.testing.assert_array_equal(np.isfinite(p[c]),eligible if len(baseline_values) or c in ['raw_on_eligible','frame_log'] else np.zeros(len(p),bool))
        ag=p.groupby('bin_index').agg(signed_bout_log_bin=('bout_median','mean'),framewise_log_bin=('centred_log','mean'),eligible_frames=('bout_median','count'),total_frames=('FrameID','size'))
        ag=ag.reindex(range(80));ag['trial']=trial;ag['bin_center_s']=np.arange(-19.75,20,.5)
        ag['baseline_log_median']=baseline;ag['baseline_frame_count']=len(baseline_values)
        ag['eligible_frames']=ag.eligible_frames.fillna(0).astype(int)
        ag['total_frames']=ag.total_frames.fillna(0).astype(int)
        ag['eligible_fraction']=ag.eligible_frames/ag.total_frames.replace(0,np.nan)
        binparts.append(ag.reset_index(names='bin_index'))
        trialstats.append({'trial':trial,'baseline_log_median':baseline,'baseline_frames':len(baseline_values),'baseline_bouts':p.loc[p.eligible&p.time_s.ge(-15)&p.time_s.lt(0),'bout_id'].nunique(),'eligible_frames':int(eligible.sum()),'nan_bins':int(ag.signed_bout_log_bin.isna().sum())})
        if trial in [9,17,63,66,93]:exampleframes.append(p)
        # Every row has coverage denominators, keeping missingness separate from intensity.
    bins=pd.concat(binparts,ignore_index=True);frames=pd.concat(exampleframes,ignore_index=True)
    old=pd.read_parquet(oldmeta['panel_data']).sort_values(['Trial number','Time bin center (s)'])
    np.testing.assert_allclose(bins.signed_bout_log_bin,old['Signed log vigor'],atol=1e-12,rtol=0,equal_nan=True)
    report={'panel':panel,'fish':fish,'fps':cadence.framerate,'interval_ms':cadence.interval_ms,'detector_samples':windows,
        'baseline_s':[-15,0],'palette':'managua_r','colour_limits':[-.25,.25],
        'matched_v5_bins':7200,'maximum_v5_error':float(np.nanmax(np.abs(bins.signed_bout_log_bin.to_numpy()-old['Signed log vigor'].to_numpy()))),
        'missing_bins':int(bins.signed_bout_log_bin.isna().sum()),'eligible_frames':int(bins.eligible_frames.sum()),
        'finite_framewise_vs_bout_bin_mean_abs_difference':float((bins.signed_bout_log_bin-bins.framewise_log_bin).abs().mean()),
        'finite_framewise_vs_bout_bin_opposite_sign_count':int(((bins.signed_bout_log_bin*bins.framewise_log_bin)<0).sum()),
        'trial_baselines':trialstats,'verified_input_artifacts':oldmeta['input_artifacts'],
        'scope':'Only F/G/H. Timing-only reconstructed-acquisition recipe; full preprocessing contract unconfirmed.'}
    if panel=='F':
        ep=V5/'Fig1_PanelsD-E_presumed-cadence_frames_v5.parquet'
        em=json.loads((V5/'Fig1_PanelE_RawSignedBoutVigor_presumed-cadence_v5.svg.json').read_text())
        assert digest(ep)==em['panel_data_sha256']
        corrected_e=pd.read_parquet(ep)
        for trial in [9,17,63,66,93]:
            p=frames[frames.trial.eq(trial)];e=corrected_e[corrected_e['Trial number'].eq(trial)]
            np.testing.assert_array_equal(p.FrameID,e.FrameID)
            np.testing.assert_array_equal(p.eligible,e.eligible)
            for our,their in [('raw_on_eligible','Vigor'),('centred_log','centred_log'),('bout_median','bout_median')]:
                np.testing.assert_allclose(p[our],e[their],atol=1e-12,rtol=0,equal_nan=True)
        report['corrected_E_match']='all FrameIDs, eligibility, raw, centred log and bout medians match in trials 9/17/63/66/93; 400 bins match'
        # Concrete two-bout bin: exact frame weighting and contrasting framewise log mean.
        q=frames[frames.trial.eq(93)&frames.eligible]
        counts=q.groupby('bin_index').bout_id.nunique();k=int(counts[counts.ge(2)].index[0])
        for bout,g in q[q.bin_index.eq(k)].groupby('bout_id'):
            contributions.append({'trial':93,'bin_index':k,'bin_start_s':-20+k*.5,'bout_id':int(bout),'frames':len(g),'bout_median':float(g.bout_median.iloc[0])})
        cb=bins[bins.trial.eq(93)&bins.bin_index.eq(k)].iloc[0]
        report['numerical_example']={'trial':93,'bin_start_s':float(cb.bin_center_s-.25),'heatmap_bin':float(cb.signed_bout_log_bin),'framewise_log_mean_same_frames':float(cb.framewise_log_bin),'eligible_frames':int(cb.eligible_frames)}
        # Verification evidence for F only, not an E redesign.
        fig,axs=plt.subplots(5,2,figsize=(13,8),sharex=True,layout='constrained')
        for i,trial in enumerate([9,17,63,66,93]):
            p=frames[frames.trial.eq(trial)];b=bins[bins.trial.eq(trial)]
            axs[i,0].plot(p.time_s,p.bout_median,color='#555555',lw=.8)
            axs[i,0].set_ylabel(f'Trial {trial}\nbout log')
            axs[i,1].bar(b.bin_center_s,b.signed_bout_log_bin,width=.5,color=cmap(norm(b.signed_bout_log_bin)),linewidth=0)
            axs[i,1].set_ylim(-1.6,.9)
            for a in axs[i]:a.set_xlim(-20,20);a.axvline(0,color='green',lw=.6);a.axvline(10,color='green',lw=.6,ls='--')
        for a in axs[:,0]:a.set_ylim(-1.6,.9)
        axs[0,0].set_title('Corrected E bout signal · identical eligible frame support')
        axs[0,1].set_title('F heatmap values · finite-frame-weighted 0.5 s means')
        fig.suptitle('F verification only: unbinned signal → exact heatmap bin values; bar heights uncapped')
        fig.savefig(OUT/'F_signal_to_bins_verification.png',dpi=160);plt.close(fig)
    path=OUT/f'Panel{panel}_rebuilt_bins.parquet';bins.to_parquet(path,index=False)
    pd.DataFrame(trialstats).to_csv(OUT/f'Panel{panel}_baselines.csv',index=False)
    frames.to_parquet(OUT/f'Panel{panel}_representative_frames.parquet',index=False)
    stem=plot_panel(spec,bins);report['panel_data']=str(path);report['panel_data_sha256']=digest(path)
    report['svg_sha256']=digest(stem.with_suffix('.svg'))
    stem.with_suffix('.svg.json').write_text(json.dumps(report,indent=2));reports.append(report)
    del angles,coverage,camera,local,raw,smoothed,envelope,frames;gc.collect()

pd.DataFrame(contributions).to_csv(OUT/'F_example_bin_contributions.csv',index=False)
thumbs=[]
for spec in FISH:
    panel,name,*_=spec
    with Image.open(OUT/f'Fig1_Panel{panel}_{name.replace(" ","")}_rebuilt.png') as im:thumbs.append(ImageOps.contain(im.convert('RGB'),(900,820)))
gallery=Image.new('RGB',(2700,820),'white')
for i,im in enumerate(thumbs):gallery.paste(im,(900*i,0))
gallery.save(OUT/'F-G-H_rebuilt.png')
manifest={'panels':reports,'builder':str(Path(__file__)),'builder_sha256':digest(Path(__file__)),
    'all_21600_bins_independently_rebuilt':True,'selection_status':'review only','full_assembly_created':False,
    'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()]}
(OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k not in ['verified_input_artifacts','trial_baselines']} for r in reports],indent=2));print(OUT)
