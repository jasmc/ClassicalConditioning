#!/usr/bin/env python3
"""Render two candidate learners with numbered #08's one-second mean analysis."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import gc
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from build_figure1_legacy_vigor_heatmaps import read_windows
from build_figure_legend_review import svg_editing_page, figure_font_paths
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot, sha, state_lock, tag
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
from classical_conditioning.analysis.figure4 import verify_expected_us
from classical_conditioning.analysis.movement_state import (
    MovementCalibrationConfig, _odd_window_samples, smooth_contiguous_median,
    rolling_extreme_envelope, detect_legacy_envelope_bouts,
)


def analyze(project, fish, suffix, experiment, *, direct_medians=False, bin_width_s=None):
    proc = project / 'Processed data' / fish
    metadata = project / 'Metadata'
    manifest_path = metadata / f'{fish}_source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['recording_id'] == fish
    assert manifest['condition_id'] == ('delay' if experiment == 'allDelay' else 'trace')
    frame_marker = metadata / f'{fish}_corrected-preprocess{suffix}_complete.json'
    metric_marker = metadata / f'{fish}_candidate-corrected{suffix}_complete.json'
    fm, mm = json.loads(frame_marker.read_text()), json.loads(metric_marker.read_text())
    paths = dict(camera=proc/'camera.parquet', protocol=proc/'stimulus_events.parquet',
                 frames=proc/f'frame_preprocessed_corrected{suffix}.parquet',
                 coverage=proc/f'frame_activity_candidates-corrected{suffix}.parquet')
    expected = dict(camera=manifest['artifacts']['camera']['sha256'], protocol=manifest['artifacts']['protocol']['sha256'],
                    frames=fm['frames_sha256'], coverage=mm['metrics_sha256'])
    assert fm['status'] == mm['status'] == 'complete'
    for key, path in paths.items():
        assert sha256_file(path) == expected[key], str(path)
    snapshot = Snapshot([*paths.values(), manifest_path, frame_marker, metric_marker])
    camera = pd.read_parquet(paths['camera'])
    cadence = estimate_camera_cadence(camera)
    assert not cadence.has_frame_loss_evidence
    epoch = float(camera.iloc[cadence.reference_position].AbsoluteTime)
    protocol = pd.read_parquet(paths['protocol'])
    us_time, paired = verify_expected_us(protocol, experiment)
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
        count=np.bincount(bins,minlength=n_bins);sums=np.bincount(bins,weights=values,minlength=n_bins)
        if direct_medians:
            means=np.array([np.median(values[bins==i]) if count[i] else np.nan for i in range(n_bins)])
            independent=pd.DataFrame({'bin':bins,'log':values}).groupby('bin').log.median().reindex(range(n_bins)).to_numpy()
        else:
            means=np.divide(sums,count,out=np.full(n_bins,np.nan),where=count>0)
            independent=pd.DataFrame({'bin':bins,'log':values}).groupby('bin').log.mean().reindex(range(n_bins)).to_numpy()
        np.testing.assert_allclose(means,independent,atol=1e-12,rtol=0,equal_nan=True)
        base=means[baseline_start:baseline_end];base=base[np.isfinite(base)]
        reference=float(np.median(base) if direct_medians else base.mean()) if len(base) else np.nan;centred=means-reference
        if len(base):assert abs((np.nanmedian if direct_medians else np.nanmean)(centred[baseline_start:baseline_end])) < 1e-12
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


def main(output):
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json'
    registry=ROOT/'records/figure-reviews/figure-01/fgh-version-numbering-2026-10-13.json'
    recipe=next(e for e in json.loads(registry.read_text())['numbered_versions'] if e['number']==8)
    assert sha256_file(Path(recipe['source']['reference']['path'])) == recipe['source']['reference']['sha256']
    snap=Snapshot([state_path,registry,Path(__file__),recipe['source']['reference']['path']])
    specs=[('F','Delay','20221122_03',Path('/Volumes/JOAQUIM/Digested Data/allDelay-full-v1'),'-v1','allDelay'),
           ('G','3sTrace','20230519_11',Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-best-learners-version08-2026-10-13-inputs'),'','all3sTrace')]
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':7,'axes.labelweight':'bold'})
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black')
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.2),sharey=True)
    fig.subplots_adjust(left=.12,right=.87,bottom=.16,top=.85,wspace=.2)
    records=[];tables=[]
    for ax,(letter,name,fish,project,suffix,experiment) in zip(axes,specs):
        print('Analyzing',name,fish,flush=True)
        prepared=specs[1][3]/'delay-version08-prepared.json'
        if letter=='F' and prepared.exists():
            info=json.loads(prepared.read_text());cache_values=prepared.with_suffix('.npz');cache_table=prepared.with_suffix('.parquet')
            assert info['fish']==fish and info['experiment']==experiment
            for path,expected in info['sources'].items():assert sha256_file(Path(path))==expected
            values=np.load(cache_values)['values'];table=pd.read_parquet(cache_table)
            assert sha(np.asarray(values,dtype='<f8').tobytes())==info['mean_log_values_sha256']
            np.testing.assert_allclose(table.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').to_numpy(),values,rtol=0,atol=0,equal_nan=True)
            snap.hashes.update(Snapshot([prepared,cache_values,cache_table]).hashes)
        else:
            values,table,info=analyze(project,fish,suffix,experiment)
        records.append(info);tables.append(table);snap.hashes.update(info['sources'])
        artist=ax.imshow(values,extent=(-20,20,90.5,.5),aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25);artist.set_gid(f'{fish}__direct-log-frame-means__1s')
        ax.axvline(0,color='#0d8136',lw=1.5,alpha=.8);ax.axvline(10,color='#0d8136',lw=1.5,alpha=.8,ls='--')
        ax.plot([9 if experiment=='allDelay' else 13]*2,[10.5,60.5],color='#702e78',lw=.7,ls=':')
        for y in (10.5,60.5):ax.axhline(y,color='white',lw=.8)
        ax.set_xticks([-20,0,20]);ax.set_yticks([1,10,20,30,40,50,60,70,80,90]);ax.set_xlabel('Time from CS onset (s)',fontsize=8,weight='bold');ax.tick_params(labelsize=7,direction='out',length=2,width=.5)
        ax.set_title(name+'\n'+fish,fontsize=8,pad=8);ax.text(-.12,1.08,letter,transform=ax.transAxes,fontsize=10,weight='bold')
        for spine in ax.spines.values():spine.set_linewidth(.5)
    for label,y in [('Pre',5.5),('Train',35.5),('Test',75.5)]:axes[0].text(-.28,y,label,transform=axes[0].get_yaxis_transform(),rotation=90,va='center',ha='center',fontsize=8)
    cax=fig.add_axes([.9,.23,.018,.55]);bar=fig.colorbar(artist,cax=cax,ticks=[-.25,0,.25]);bar.set_label('Log vigor relative to baseline',fontsize=8,weight='bold');bar.ax.tick_params(labelsize=7,length=2,width=.5);bar.outline.set_linewidth(.5)
    buffer=io.BytesIO();fig.savefig(buffer,format='svg');plt.close(fig);root=ET.fromstring(buffer.getvalue())
    defs=ET.SubElement(root,tag('defs'));css=[]
    for weight,path in zip((400,700),figure_font_paths()):css.append('@font-face{font-family:"DejaVu Sans";font-weight:'+str(weight)+';src:url(data:font/ttf;base64,'+base64.b64encode(path.read_bytes()).decode()+') format("truetype")}')
    ET.SubElement(defs,tag('style')).text=''.join(css)
    report=dict(status='new-fish scientific draft; not frozen',comparison_number=8,original_recipe='V5 one-second direct means',panels=records,
                bin_width_s=1,baseline_s=[-20,0],bin_estimator='arithmetic mean of eligible positive frame natural-log vigor',baseline_estimator='arithmetic mean of finite baseline-bin means; one vote per bin',percentile_scaling=False,numerical_clipping=False,colour_limits=[-.25,.25],palette='managua_r',trial_mapping='acquisition minus 4; acquisition 5–94 maps to displayed 1–90',timing='accepted legacy cadence reconstruction, anchored to camera arrival reference',source_snapshot=snap.hashes)
    ET.SubElement(root,tag('metadata')).text=json.dumps(report)
    legend='(F–G) Provisional example learners: Delay 20221122_03 and 3sTrace 20230519_11. Numbered version #08 analysis: positive finite valid adjacent detected-bout frames are natural-log transformed, then averaged within left-inclusive/right-exclusive 1 s bins. Subtract the arithmetic mean of finite baseline-bin means over [−20,0), with one vote per finite bin. One finite baseline bin is sufficient; no baseline leaves the trial undefined. No percentile scaling or numerical clipping is applied. Colours saturate at ±0.25 log units; black indicates missing/undefined bins. Timing retains the established cadence reconstruction. Displayed trials 1–90 map to acquisition trials 5–94. Green guides mark CS onset/nominal 10 s offset; dotted purple guides show expected US at 9 s (Delay) or 13 s (3sTrace) within training. Horizontal boundaries separate Pre, Train and Test. Fish selection comes from provisional Figure 3 learner results; these are descriptive examples.'
    page=svg_editing_page(ET.tostring(root),scope_label='Figure 1 · best learner examples · version #08',legend=legend).encode();data=pd.concat(tables,ignore_index=True).to_csv(index=False).encode()
    report['html']=dict(path=str(output),sha256=sha(page));report['data']=dict(path=str(output.with_suffix('.csv')),sha256=sha(data))
    evidence=ROOT/'records/figure-reviews/figure-01/best-learners-version08-2026-10-13.json'
    with state_lock(state_path):
        snap.check()
        for path,raw in [(output,page),(output.with_suffix('.csv'),data),(evidence,(json.dumps(report,indent=2)+'\n').encode())]:
            with path.open('xb') as stream:stream.write(raw)
            assert sha256_file(path)==sha(raw)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.output)
