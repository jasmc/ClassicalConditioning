"""User-requested C comparison: bout sample display versus direct half-second bins."""
from pathlib import Path
import sys,json,gc,re,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm,to_hex
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from PIL import Image

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'reviews/fgh_halfsecond_fresh_20261008'))
from build_review import (FISH,read_windows,digest,estimate_camera_cadence,CFG,
    _odd_window_samples,smooth_contiguous_median,rolling_extreme_envelope,detect_legacy_envelope_bouts)
OUT=HERE/'outputs'
PRIOR=REPO/'reviews/fgh_halfsecond_fresh_20261008/outputs'
DIAGNOSTIC=REPO/'reviews/fgh_halfsecond_fresh_20261008/scientific_audit/outputs/bin_diagnostics.parquet'
PHASES=[('Pre-Train',5,14),('Train',15,64),('Test',65,94)]
EDGES=np.arange(-20.,20.5,.5)
NORM=CenteredNorm(vcenter=0,halfrange=1,clip=True)
CMAP=plt.get_cmap('managua_r').copy();CMAP.set_bad('black')
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':9})

def load_frames(spec):
    panel,_,fish,project_name,metric_name,*_=spec
    metadata=next(r for r in json.loads((PRIOR/'manifest.json').read_text())['panels'] if r['panel']==panel)
    for r in metadata['inputs']:assert digest(Path(r['path']))==r['sha256']
    proc=Path(project_name)/'Processed data'/fish
    camera=pd.read_parquet(proc/'camera.parquet');cadence=estimate_camera_cadence(camera)
    assert not cadence.has_frame_loss_evidence
    anchor=float(camera.iloc[cadence.reference_position].AbsoluteTime)
    protocol=pd.read_parquet(proc/'stimulus_events.parquet')
    cycles=protocol[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
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
    moving,bouts=detect_legacy_envelope_bouts(en,raw,dt,steps,valid,
        envelope_threshold=CFG.envelope_threshold_rad_per_ms,
        amplitude_threshold=CFG.bout_amplitude_threshold_rad_per_ms,
        minimum_bout_duration_ms=CFG.minimum_bout_duration_ms,
        maximum_interbout_gap_ms=CFG.maximum_interbout_gap_ms)
    eligible=valid&moving&(bouts>0)&np.isfinite(raw)&(raw>0)
    frames=[]
    for trial in range(5,95):
        onset=int(cycles.iloc[trial-1].Beg)
        first,last=np.searchsorted(times,[onset-20000,onset+20000]);s=slice(first,last)
        p=pd.DataFrame({'trial':trial,'FrameID':ids[s],'time_s':(times[s]-onset)/1000,
            'eligible':eligible[s],'bout_id':bouts[s],'raw_vigor':np.where(eligible[s],raw[s],np.nan)})
        p['log_vigor']=np.log(p.raw_vigor)
        p['bout_median_log']=p.loc[p.eligible].groupby('bout_id').log_vigor.transform('median').reindex(p.index)
        np.testing.assert_array_equal(p.bout_median_log.notna(),p.eligible)
        frames.append(p)
    del camera,a,cov,angles,sm,en,raw
    gc.collect()
    return pd.concat(frames,ignore_index=True),cadence,metadata

def c_values(x,baseline):
    x=np.asarray(x,float);baseline=np.asarray(baseline,bool)
    base=x[baseline&np.isfinite(x)]
    if len(base):
        p10,m,p90=np.quantile(base,[.1,.5,.9],method='linear');scale=(p90-p10)/2
    else:p10=m=p90=scale=np.nan
    unclipped=(x-m)/scale if scale>0 else np.full_like(x,np.nan)
    result=np.clip(unclipped,-1,1)
    if scale>0:
        assert np.array_equal(np.isnan(result),np.isnan(x))
        assert abs(np.nanmedian(result[baseline]))<1e-11
    return result,unclipped,{'baseline_count':len(base),'p10':float(p10),'p50':float(m),
        'p90':float(p90),'C_scale':float(scale),'defined':bool(scale>0)}

def sample_runs(p,cadence):
    """Lossless vector encoding of repeated values; never average or bridge gaps."""
    good=np.isfinite(p.C_sample.to_numpy())
    if not good.any():return pd.DataFrame(columns=['trial','first_frame_id','last_frame_id','start_s','end_s','C','sample_count'])
    idx=np.flatnonzero(good);vals=p.C_sample.to_numpy()[idx];ids=p.FrameID.to_numpy()[idx]
    bouts=p.bout_id.to_numpy()[idx]
    breaks=np.r_[True,(np.diff(ids)!=1)|(np.diff(idx)!=1)|(vals[1:]!=vals[:-1])|(bouts[1:]!=bouts[:-1])]
    begins=np.flatnonzero(breaks);ends=np.r_[begins[1:]-1,len(idx)-1]
    seconds=p.time_s.to_numpy()[idx];half=cadence.interval_ms/2000
    runs=pd.DataFrame({'trial':int(p.trial.iloc[0]),'first_frame_id':ids[begins],
        'last_frame_id':ids[ends],'start_s':np.maximum(-20,seconds[begins]-half),
        'end_s':np.minimum(20,seconds[ends]+half),'C':vals[begins],
        'sample_count':ends-begins+1})
    assert runs.sample_count.sum()==good.sum()
    assert np.all(runs.end_s>runs.start_s)
    # Every encoded run contains exactly its original, unchanged sample value.
    for b,e in zip(begins,ends):np.testing.assert_array_equal(vals[b:e+1],np.full(e-b+1,vals[b]))
    return runs

def verify_sample_svg(path,runs):
    ns={'s':'http://www.w3.org/2000/svg'};root=ET.parse(path).getroot()
    groups=[g for g in root.findall('.//s:g',ns) if g.get('id','').startswith('sample_runs_')]
    assert len(groups)==3
    checks=[]
    for group,(phase,first,last) in zip(groups,PHASES):
        sub=runs[runs.trial.between(first,last)];paths=group.findall('s:path',ns)
        assert len(paths)==len(sub)
        widths=[];expected_widths=[]
        for path,row in zip(paths,sub.itertuples()):
            xy=np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',path.get('d'))]).reshape(-1,2)
            widths.append(np.ptp(xy[:,0]))
            # Saved SVG coordinates are in points; account for their rounding.
            scale=(.82-.14)*8.2*72/40
            expected_widths.append((row.end_s-row.start_s)*scale)
            np.testing.assert_allclose(np.min(xy[:,0]),.14*8.2*72+(row.start_s+20)*scale,rtol=0,atol=2e-6)
            fill=re.search(r'fill:\s*(#[0-9a-f]+)',path.get('style',''))
            assert (fill.group(1) if fill else '#000000')==to_hex(CMAP(NORM(row.C)))
        np.testing.assert_allclose(widths,expected_widths,rtol=0,atol=2e-6)
        checks.append({'phase':phase,'sample_runs':len(paths),'fills_and_sample_duration_widths_verified':True})
    return checks

def make_axes(spec,variant):
    panel,name,fish,*_=spec
    fig=plt.figure(figsize=(8.2,7.1));grid=fig.add_gridspec(3,1,height_ratios=[10,50,30],
        left=.14,right=.82,bottom=.15,top=.83,hspace=.11)
    axes=[]
    for k,(phase,first,last) in enumerate(PHASES):
        ax=fig.add_subplot(grid[k]);ax.set_facecolor('black');ax.set_ylim(last+.5,first-.5)
        ax.set_yticks([first,last]);ax.set_ylabel(phase);ax.set_xlim(-20,20)
        ax.set_xticks([-20,-10,0,10,20]);ax.tick_params(labelbottom=k==2,length=2)
        if k==2:ax.set_xlabel('Time from measured CS onset (s)')
        axes.append(ax)
    title='Bout medians | no binning' if variant=='BoutSamples' else 'Direct log vigor | 0.5 s means'
    fig.text(.035,.95,panel,fontsize=20,weight='bold')
    fig.text(.14,.95,f'{name} | {fish} | C',fontsize=13,weight='bold')
    fig.text(.14,.90,title,fontsize=12,weight='bold')
    baseline='Baseline median/P10/P90 from eligible displayed samples' if variant=='BoutSamples' else 'Baseline median/P10/P90 from finite half-second bin scalars'
    fig.text(.14,.86,baseline,fontsize=9)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),
        cax=fig.add_axes([.855,.15,.025,.68]),ticks=[-1,0,1]);cb.set_label('C-scaled log vigor')
    fig.text(.14,.073,'C = clip((value - P50) / ((P90 - P10) / 2), -1, +1)',fontsize=9)
    fig.text(.14,.046,'Each trial baseline [-15, 0) s | managua_r | zero = midpoint',fontsize=9)
    fig.text(.14,.022,'Black = no eligible contribution, or undefined C scale',fontsize=8)
    return fig,axes

def add_guides(spec,axes):
    for ax,(phase,_,_) in zip(axes,PHASES):
        for t in [0,10]:ax.axvline(t,color='#168047',lw=.8,ls='--' if t==10 else '-')
        if phase=='Train' and spec[-1] is not None:ax.axvline(spec[-1],color='#964bad',lw=.9,ls=':')

def render(spec,runs,bins):
    from build_review import verify_svg
    checks=[]
    fig,axes=make_axes(spec,'BoutSamples')
    for ax,(phase,first,last) in zip(axes,PHASES):
        sub=runs[runs.trial.between(first,last)]
        patches=[Rectangle((r.start_s,r.trial-.5),r.end_s-r.start_s,1) for r in sub.itertuples()]
        coll=PatchCollection(patches,facecolors=CMAP(NORM(sub.C.to_numpy())),
            edgecolors='none',antialiaseds=False)
        coll.set_gid('sample_runs_'+phase);ax.add_collection(coll)
    add_guides(spec,axes)
    stem=OUT/f'C_BoutSamples_Panel{spec[0]}'
    for ext in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    checks.append({'panel':spec[0],'version':'BoutSamples','baseline_unit':'eligible sample',
        'geometry':verify_sample_svg(stem.with_suffix('.svg'),runs)})
    fig,axes=make_axes(spec,'DirectBins')
    matrix=bins.pivot(index='trial',columns='bin_index',values='C_bin').to_numpy()
    assert matrix.shape==(90,80)
    for ax,(phase,first,last) in zip(axes,PHASES):
        mesh=ax.pcolormesh(EDGES,np.arange(first-.5,last+1.5),matrix[first-5:last-4],
            cmap=CMAP,norm=NORM,shading='flat',edgecolors='none',antialiased=False)
        mesh.set_gid('trial_cells_'+phase)
        np.testing.assert_array_equal(mesh.get_coordinates()[0,:,0],EDGES)
    add_guides(spec,axes)
    stem=OUT/f'C_DirectBins_Panel{spec[0]}'
    for ext in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    checks.append({'panel':spec[0],'version':'DirectBins','baseline_unit':'finite half-second scalar',
        'geometry':verify_svg(stem.with_suffix('.svg'),matrix,NORM)})
    return checks

def main():
    resume='--resume' in sys.argv
    OUT.mkdir(exist_ok=resume)
    audits=[];checks=[];metadata=[]
    diagnostic=pd.read_parquet(DIAGNOSTIC)
    for spec in FISH:
        print('Reconstructing comparison',spec[0],flush=True)
        cache=OUT/f'Panel{spec[0]}_sample_data.parquet'
        if resume and cache.exists():
            from types import SimpleNamespace
            frames=pd.read_parquet(cache)
            meta=next(r for r in json.loads((PRIOR/'manifest.json').read_text())['panels'] if r['panel']==spec[0]).copy()
            cadence=SimpleNamespace(interval_ms=meta['interval_ms'])
            meta['local_sample_cache']={'path':str(cache),'sha256_before_resume':digest(cache)}
        else:frames,cadence,meta=load_frames(spec)
        meta['interval_ms']=cadence.interval_ms
        metadata.append(meta);allruns=[];allbins=[]
        frames['C_sample']=np.nan;frames['C_sample_unclipped']=np.nan
        for trial,p in frames.groupby('trial',sort=True):
            # Version 1 has NO half-second aggregation at any processing stage.
            baseline=p.time_s.ge(-15)&p.time_s.lt(0)
            values,uncapped,stats=c_values(p.bout_median_log,baseline)
            frames.loc[p.index,'C_sample']=values;frames.loc[p.index,'C_sample_unclipped']=uncapped
            p=p.copy();p['C_sample']=values
            allruns.append(sample_runs(p,cadence))
            audits.append({'panel':spec[0],'trial':int(trial),'version':'BoutSamples',**stats})
            # Version 2 uses direct eligible log samples, never bout medians.
            index=np.searchsorted(EDGES,p.time_s.to_numpy(),side='right')-1
            finite=p.log_vigor.notna().to_numpy();count=np.bincount(index[finite],minlength=80)
            sums=np.bincount(index[finite],weights=p.log_vigor.to_numpy()[finite],minlength=80)
            b=np.divide(sums,count,out=np.full(80,np.nan),where=count>0)
            prior=diagnostic[diagnostic.panel.eq(spec[0])&diagnostic.trial.eq(trial)].sort_values('bin_index')
            np.testing.assert_allclose(b,prior.direct_log_bin,atol=1e-12,rtol=0,equal_nan=True)
            base=np.zeros(80,bool);base[10:40]=True
            cv,cu,stats=c_values(b,base)
            allbins.append(pd.DataFrame({'trial':trial,'bin_index':range(80),'start_s':EDGES[:-1],
                'end_s':EDGES[1:],'eligible_frames':count,'direct_log_bin':b,'C_bin':cv,'C_bin_unclipped':cu}))
            audits.append({'panel':spec[0],'trial':int(trial),'version':'DirectBins',**stats})
        runs=pd.concat(allruns,ignore_index=True);bins=pd.concat(allbins,ignore_index=True)
        frames.to_parquet(OUT/f'Panel{spec[0]}_sample_data.parquet',index=False)
        runs.to_csv(OUT/f'Panel{spec[0]}_display_sample_runs.csv',index=False)
        bins.to_parquet(OUT/f'Panel{spec[0]}_direct_halfsecond_bins.parquet',index=False)
        bins.to_csv(OUT/f'Panel{spec[0]}_direct_halfsecond_bins.csv',index=False)
        checks.extend(render(spec,runs,bins))
        del frames,runs,bins;gc.collect()
    pd.DataFrame(audits).to_csv(OUT/'C_trial_baseline_statistics.csv',index=False)
    for variant in ['BoutSamples','DirectBins']:
        ims=[Image.open(OUT/f'C_{variant}_Panel{panel}.png').convert('RGB') for panel in 'FGH']
        gallery=Image.new('RGB',(sum(im.width for im in ims),max(im.height for im in ims)),'white')
        x=0
        for im in ims:gallery.paste(im,(x,0));x+=im.width
        gallery.save(OUT/f'C_{variant}_FGH.png')
    report={'status':'user-requested review comparison','scope':'F/G/H only',
        'C_formula':'clip((value - P50)/((P90 - P10)/2), -1, +1)',
        'versions':{'BoutSamples':'eligible log -> per-bout visible median -> sample baseline quantiles -> C on samples -> sample interval display; no half-second binning anywhere',
            'DirectBins':'eligible log -> finite-sample half-second mean -> baseline quantiles on bins -> C on bins -> 80-cell display; no bout-median substitution'},
        'sample_rendering':'Lossless vector run encoding of equal-valued, consecutive eligible samples; no averaging, interpolation, or bridging gaps. Sample intervals centred on timestamps with cadence/2 edges, clipped at display boundaries.',
        'baseline_s':[-15,0],'palette':'managua_r','zero_palette_coordinate':float(NORM(0)),
        'zero_colour':to_hex(CMAP(NORM(0))),'source_inputs':metadata,'render_validation':checks,
        'undefined_trials':[r for r in audits if not r['defined']],
        'code_sha256':digest(Path(__file__)),
        'diagnostic_reference':{'path':str(DIAGNOSTIC),'sha256':digest(DIAGNOSTIC)},
        'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()]}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'undefined_trials':report['undefined_trials'],'render_checks':checks},indent=2),flush=True)

if __name__=='__main__':main()
