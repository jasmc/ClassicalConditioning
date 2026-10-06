"""Rebuild Figure 1 D-H on presumed acquisition cadence; preserve all snapshots."""
import json
import math
import gc
import argparse
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from build_figure1_legacy_vigor_heatmaps import ROOT, FISH, read_windows, digest, draw as draw_heatmap
from build_figure1_trace_panels import draw as draw_trace, TRIALS, STAGES
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
from classical_conditioning.analysis.movement_state import (
    MovementCalibrationConfig, _odd_window_samples, smooth_contiguous_median,
    rolling_extreme_envelope, detect_legacy_envelope_bouts)
from classical_conditioning.analysis.temporal_profiles import _signed_bout_log_vigor
from classical_conditioning.analysis.figure4 import verify_expected_us
from assemble_svg_figure import render, export_with_inkscape

REPO=Path(__file__).resolve().parents[1]
OUT=ROOT/'cadence-review-v5-20261007'
TIME='Time relative to CS onset (s)'
METRIC='legacy_distal_angular_speed'
COL=METRIC+'_rad_per_ms'


def fish_data(spec):
    panel,name,fish,project_name,metric_name,_,role,us_s=spec
    project=Path(project_name); proc=project/'Processed data'/fish
    old=json.loads((ROOT/f'heatmaps/Fig1_Panel{panel}_{name.replace(" ","")}_legacy-vigor_v2.svg.json').read_text())
    sources=[]
    for item in old['input_artifacts']:
        assert digest(Path(item['path']))==item['sha256'],item['path']
        sources.append(item)
    manifest_path=project/'Metadata'/f'{fish}_source_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    camera_path=proc/'camera.parquet'
    assert digest(camera_path)==manifest['artifacts']['camera']['sha256']
    sources.append({'path':str(camera_path),'sha256':manifest['artifacts']['camera']['sha256']})
    suffix='' if panel=='G' else '-v1'
    corrected_path=proc/f'frame_preprocessed_corrected{suffix}.parquet'
    marker_path=project/'Metadata'/f'{fish}_corrected-preprocess{suffix}_complete.json'
    marker=json.loads(marker_path.read_text())
    assert marker['status']=='complete' and digest(corrected_path)==marker['frames_sha256']
    sources.append({'path':str(corrected_path),'sha256':marker['frames_sha256']})
    camera=pd.read_parquet(camera_path)
    cadence=estimate_camera_cadence(camera)
    assert not cadence.has_frame_loss_evidence
    ref=camera.iloc[cadence.reference_position]
    protocol=pd.read_parquet(proc/'stimulus_events.parquet')
    cycles=protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    assert len(cycles)==94
    if panel in ('F','G'):
        measured_us,count=verify_expected_us(protocol,'allDelay' if panel=='F' else 'all3sTrace')
        assert count==46 and abs(measured_us-us_s)<.1
    else: measured_us=None
    intervals=[(int(b)-22000,int(b)+22001) for b in cycles.iloc[4:].Beg]
    angles=[f'angle{i}' for i in range(16)]
    c=read_windows(corrected_path,['FrameID','AbsoluteTime','FrameStep','frame_valid','timestamp_valid',*angles],intervals)
    m=read_windows(proc/metric_name,['FrameID','AbsoluteTime','angular_valid_tail_fraction'],intervals)
    np.testing.assert_array_equal(c[['FrameID','AbsoluteTime']],m[['FrameID','AbsoluteTime']])
    ids=c.FrameID.to_numpy(dtype=np.int64)
    absolute=float(ref.AbsoluteTime)+(ids-cadence.reference_frame_id)*cadence.interval_ms
    steps=np.r_[0,np.diff(ids)]
    dt=steps*cadence.interval_ms
    position_valid=c.frame_valid.to_numpy()&c.timestamp_valid.to_numpy()
    derivative_valid=position_valid&np.r_[False,position_valid[:-1]]&(steps==1)&(dt>0)&(dt<=10)
    angle_matrix=c[angles].to_numpy()
    angle_matrix[~position_valid]=np.nan
    angle=angle_matrix.sum(axis=1)
    delta=np.r_[np.nan,np.diff(angle)]
    raw=np.abs(np.arctan2(np.sin(delta),np.cos(delta)))/np.where(dt>0,dt,np.nan)
    raw[~derivative_valid]=np.nan
    cfg=MovementCalibrationConfig()
    windows=[_odd_window_samples(w,cadence.interval_ms) for w in
             (cfg.smoothing_window_ms,cfg.envelope_max_window_ms,cfg.envelope_min_window_ms)]
    smoothed=smooth_contiguous_median(raw,steps,window_samples=windows[0])
    envelope=rolling_extreme_envelope(smoothed,steps,max_window_samples=windows[1],min_window_samples=windows[2])
    valid=derivative_valid&np.isfinite(envelope)&(m.angular_valid_tail_fraction.to_numpy()>=cfg.minimum_valid_tail_fraction)
    moving,boutids=detect_legacy_envelope_bouts(envelope,raw,dt,steps,valid,
        envelope_threshold=cfg.envelope_threshold_rad_per_ms,amplitude_threshold=cfg.bout_amplitude_threshold_rad_per_ms,
        minimum_bout_duration_ms=cfg.minimum_bout_duration_ms,maximum_interbout_gap_ms=cfg.maximum_interbout_gap_ms)
    rows=[]; traces=[]; events=[]; trial_checks=[]
    for trial in range(5,95):
        onset=int(cycles.iloc[trial-1].Beg)
        start=np.searchsorted(absolute,onset-20000); stop=np.searchsorted(absolute,onset+20000)
        s=slice(start,stop); seconds=(absolute[s]-onset)/1000
        indices=np.floor((seconds+20)/.5).astype(np.int32)
        values=_signed_bout_log_vigor(raw[s],seconds,indices,valid[s],moving[s],boutids[s],
                                      bin_count=80,baseline_start_s=-15,baseline_end_s=0)
        rows.extend({'Recording ID':fish,'Metric ID':METRIC,'Trial number':trial,
                     'Time bin center (s)':float(t),'Signed log vigor':float(v),
                     'Baseline start (s)':-15.,'Baseline end (s)':0.}
                    for t,v in zip(np.arange(-19.75,20,.5),values))
        eligible=valid[s]&moving[s]&(boutids[s]>0)&np.isfinite(raw[s])&(raw[s]>0)
        trial_checks.append({'trial':trial,'frames':len(seconds),'detector_valid':int(valid[s].sum()),
                             'eligible':int(eligible.sum()),'missing_bins':int(np.isnan(values).sum())})
        if panel=='F' and trial in TRIALS:
            baseline=float(np.median(np.log(raw[s][eligible&(seconds>=-15)&(seconds<0)])))
            centred=np.where(eligible,np.log(np.where(eligible,raw[s],np.nan))-baseline,np.nan)
            p=pd.DataFrame({'Trial number':trial,'FrameID':ids[s],'AbsoluteTime':absolute[s],TIME:seconds,
                            'Vigor':np.where(eligible,raw[s],np.nan),'Raw frame vigor':raw[s],
                            'Tail angle (rad)':angle[s],'valid':valid[s],'moving':moving[s],
                            'bout_id':boutids[s],'eligible':eligible,'centred_log':centred,'bin_index':indices})
            p['Tail angle (rad)']-=np.nanmedian(p.loc[p[TIME].ge(-15)&p[TIME].lt(0),'Tail angle (rad)'])
            mapping=p.loc[p.eligible].groupby('bout_id').centred_log.median()
            p['bout_median']=p.bout_id.map(mapping).where(p.eligible)
            np.testing.assert_array_equal(np.isfinite(p.Vigor),np.isfinite(p.bout_median))
            rebuilt=p.groupby('bin_index').bout_median.mean().reindex(range(80)).to_numpy()
            np.testing.assert_allclose(rebuilt,values,atol=1e-12,rtol=0,equal_nan=True)
            traces.append(p)
            events.extend([{'Trial number':trial,'Event':'CS onset','Time (s)':0.},
                           {'Trial number':trial,'Event':'CS offset','Time (s)':10.}])
            for beg in protocol.loc[protocol.Type.eq('Reinforcer'),'Beg']:
                relative=(int(beg)-onset)/1000
                if -20<=relative<20: events.append({'Trial number':trial,'Event':'actual US onset','Time (s)':relative})
    provenance={'recording_id':fish,'condition':name,'metric_id':METRIC,'baseline_s':[-15,0],
        'time_source':'presumed constant acquisition cadence; stable legacy reference',
        'cadence':asdict(cadence),'fps':cadence.framerate,'detector_windows_samples':windows,
        'input_artifacts':sources,'verified_paired_us_onset_s':measured_us,'trial_checks':trial_checks,
        'selection_status':'provisional timing review; full preprocessing confirmation pending',
        'processing_scope':'Acquisition clock repair only; no additional angle filtering or 700 FPS resampling'}
    return pd.DataFrame(rows),traces,events,provenance


def raw_scaled_panel(frames,output):
    plt.rcParams.update({'svg.fonttype':'none','path.simplify':False,'savefig.bbox':None})
    fig,axes=plt.subplots(5,2,figsize=(8.15,4.8),sharex=True,sharey='col',layout='none')
    fig.subplots_adjust(left=.20,right=.975,top=.80,bottom=.15,wspace=.22,hspace=.18)
    fig.text(.09,.96,'Bout vigor · Delay fish 20221115_07',fontsize=13,weight='bold',va='top')
    fig.text(.09,.91,'Raw and signed values use exactly the same detected bout frames',fontsize=8.5,va='top')
    raw_high=math.ceil(float(frames.Vigor.max())*2)/2
    signed_low=min(-.25,math.floor(float(frames.bout_median.min())*4)/4)
    signed_high=max(.25,math.ceil(float(frames.bout_median.max())*4)/4)
    for i,(trial,stage) in enumerate(zip(TRIALS,STAGES)):
        p=frames.loc[frames['Trial number'].eq(trial)]
        axes[i,0].plot(p[TIME],p.Vigor,color='#111111',lw=.45)
        axes[i,1].plot(p[TIME],p.bout_median,color='#c85a17',lw=.65)
        axes[i,0].set_ylim(0,raw_high); axes[i,0].set_yticks([0,raw_high])
        axes[i,1].set_ylim(signed_low-.05,signed_high+.05); axes[i,1].set_yticks([signed_low,0,signed_high])
        axes[i,0].text(-.12,.5,stage,transform=axes[i,0].transAxes,ha='right',va='center',fontsize=8,weight='bold')
        for a in axes[i]:
            a.set_xlim(-20,20); a.axvline(0,color='#0d7f3c',lw=.6); a.axvline(10,color='#0d7f3c',ls='--',lw=.6)
            a.spines[['top','right']].set_visible(False); a.tick_params(labelsize=7,length=2)
        axes[i,1].axhline(0,color='#c85a17',lw=.35,alpha=.5)
    axes[0,0].set_title('Raw vigor (rad/ms)',fontsize=9)
    axes[0,1].set_title('Signed bout log vigor',fontsize=9)
    for a in axes[-1]: a.set_xticks([-20,0,20]);a.set_xlabel('Time relative to CS onset (s)',fontsize=8)
    fig.savefig(output);plt.close(fig)


def save_meta(svg,meta,data):
    svg.with_suffix('.svg.json').write_text(json.dumps({**meta,'svg':str(svg),'svg_sha256':digest(svg),
        'panel_data':str(data),'panel_data_sha256':digest(data)},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true',help='Reuse verified saved data for layout repairs')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((REPO/'configs/paper-figures/figure1-freeze.json').read_text())
    for p in freeze['panels']: assert digest(ROOT/p['source'])==p['sha256']
    outputs={}; reports={}
    for spec in FISH:
        panel,name,*_=spec
        svg=OUT/f'Fig1_Panel{panel}_{name.replace(" ","")}_presumed-cadence_v5.svg'
        data=svg.with_suffix('.parquet')
        if args.render_only:
            meta=json.loads(svg.with_suffix('.svg.json').read_text())
            assert digest(data)==meta['panel_data_sha256']
            bins=pd.read_parquet(data)
        else:
            print(f'Calculating {panel}: {name}',flush=True)
            bins,traces,events,meta=fish_data(spec)
            bins.to_parquet(data,index=False)
        assert len(bins)==7200
        tree=draw_heatmap(spec,bins)
        tree.set('data-time-source','presumed-acquisition-cadence')
        ET.ElementTree(tree).write(svg,encoding='utf-8',xml_declaration=True)
        save_meta(svg,meta,data);outputs[panel]=svg;reports[panel]=meta
        if panel=='F':
            frame_path=OUT/'Fig1_PanelsD-E_presumed-cadence_frames_v5.parquet'
            event_path=OUT/'Fig1_PanelsD-E_events_v5.parquet'
            if args.render_only:
                dmeta=json.loads((OUT/'Fig1_PanelD_TailAngle_presumed-cadence_v5.svg.json').read_text())
                assert digest(frame_path)==dmeta['panel_data_sha256']
                frames=pd.read_parquet(frame_path);eventdata=pd.read_parquet(event_path)
            else:
                frames=pd.concat(traces,ignore_index=True);eventdata=pd.DataFrame(events)
                frames.to_parquet(frame_path,index=False);eventdata.to_parquet(event_path,index=False)
            dsvg=OUT/'Fig1_PanelD_TailAngle_presumed-cadence_v5.svg'
            draw_trace(frames,eventdata,panel='D',output=dsvg,baseline_s=(-15,0));save_meta(dsvg,meta,frame_path);outputs['D']=dsvg
            esvg=OUT/'Fig1_PanelE_RawSignedBoutVigor_presumed-cadence_v5.svg'
            raw_scaled_panel(frames,esvg);save_meta(esvg,meta,frame_path);outputs['E']=esvg
        gc.collect()
    layout=json.loads((REPO/'configs/paper-figures/figure1-assembly.json').read_text())
    layout['title']='Figure 1 · presumed acquisition timing review; preprocessing confirmation pending'
    layout['output']=str(OUT/'figure1-presumed-acquisition-review-v5.svg')
    for p in layout['panels']:
        if p['id'] in outputs:
            p['source']=str(outputs[p['id']]);p.pop('frozen_sha256',None)
            p['selection_status']='updated timing review; preprocessing confirmation pending'
            if p['id']=='D': p['role']='selected-fish tail-angle traces on presumed acquisition clock'
            if p['id']=='E': p['role']='paired raw and signed vigor on identical detected bout frames'
    configpath=REPO/'configs/paper-figures/figure1-cadence-review-v5.json'
    configpath.write_text(json.dumps(layout,indent=2)+'\n')
    assembly=Path(layout['output'])
    render(configpath,assembly,strict=True)
    export_with_inkscape(assembly,['png','pdf'],font_directory=ROOT/'fonts')
    for p in freeze['panels']: assert digest(ROOT/p['source'])==p['sha256']
    (OUT/'build_manifest.json').write_text(json.dumps({'panels':reports,'assembly':str(assembly),
        'assembly_sha256':digest(assembly),'assembly_config':str(configpath),'frozen_snapshots_verified':['A','B','C','D'],
        'replaced_panels':['D','E','F','G','H'],'prior_variants_preserved':True},indent=2)+'\n')
    print(assembly,flush=True)


if __name__=='__main__': main()
