"""Fresh frame reconstruction and explicit vector scalar cells, F/G/H only."""
from pathlib import Path
import sys, json, gc, re, xml.etree.ElementTree as ET
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, to_hex
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import FISH, read_windows, digest
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
from classical_conditioning.analysis.movement_state import (
    MovementCalibrationConfig, _odd_window_samples, smooth_contiguous_median,
    rolling_extreme_envelope, detect_legacy_envelope_bouts)

OUT = HERE/'outputs'
EDGES = np.arange(-20., 20.5, .5)
CENTRES = (EDGES[:-1]+EDGES[1:])/2
PHASES = [('Pre-Train',5,14),('Train',15,64),('Test',65,94)]
CFG = MovementCalibrationConfig()
CMAP = plt.get_cmap('managua_r').copy()
CMAP.set_bad('black')
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':9})

def load_scalars(spec):
    panel, name, fish, project_name, metric_name, *_ = spec
    project = Path(project_name)
    proc = project/'Processed data'/fish
    suffix = '' if panel == 'G' else '-v1'
    manifest_path = project/'Metadata'/f'{fish}_source_manifest.json'
    marker_path = project/'Metadata'/f'{fish}_corrected-preprocess{suffix}_complete.json'
    metric_marker_path = project/'Metadata'/f'{fish}_candidate-corrected{suffix}_complete.json'
    manifest = json.loads(manifest_path.read_text())
    marker = json.loads(marker_path.read_text())
    metric_marker = json.loads(metric_marker_path.read_text())
    paths = {'camera':proc/'camera.parquet', 'protocol':proc/'stimulus_events.parquet',
             'angles':proc/f'frame_preprocessed_corrected{suffix}.parquet',
             'coverage':proc/metric_name}
    expected = {'camera':manifest['artifacts']['camera']['sha256'],
                'protocol':manifest['artifacts']['protocol']['sha256'],
                'angles':marker['frames_sha256'], 'coverage':metric_marker['metrics_sha256']}
    assert marker['status'] == metric_marker['status'] == 'complete'
    hashes = {k:digest(p) for k,p in paths.items()}
    assert hashes == expected
    camera = pd.read_parquet(paths['camera'])
    cadence = estimate_camera_cadence(camera)
    assert not cadence.has_frame_loss_evidence
    anchor = float(camera.iloc[cadence.reference_position].AbsoluteTime)
    protocol = pd.read_parquet(paths['protocol'])
    cycles = protocol[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    assert len(cycles) == 94
    intervals = [(int(t)-22000,int(t)+22001) for t in cycles.iloc[4:].Beg]
    angles = read_windows(paths['angles'], ['FrameID','AbsoluteTime','frame_valid',
        'timestamp_valid',*[f'angle{k}' for k in range(16)]], intervals)
    coverage = read_windows(paths['coverage'],
        ['FrameID','AbsoluteTime','angular_valid_tail_fraction'], intervals)
    np.testing.assert_array_equal(angles[['FrameID','AbsoluteTime']],coverage[['FrameID','AbsoluteTime']])
    ids = angles.FrameID.to_numpy(np.int64)
    assert np.all(np.diff(ids)>0)
    steps = np.r_[0,np.diff(ids)]
    acquired = anchor+(ids-cadence.reference_frame_id)*cadence.interval_ms
    dt = steps*cadence.interval_ms
    valid = angles.frame_valid.to_numpy(bool)&angles.timestamp_valid.to_numpy(bool)
    local = angles[[f'angle{k}' for k in range(16)]].to_numpy(float)
    local[~valid] = np.nan
    bend = local.sum(axis=1)
    diff = np.r_[np.nan,np.diff(bend)]
    derivative_valid = valid&np.r_[False,valid[:-1]]&(steps==1)&(dt>0)&(dt<=10)
    raw = np.abs(np.arctan2(np.sin(diff),np.cos(diff)))/np.where(dt>0,dt,np.nan)
    raw[~derivative_valid] = np.nan
    windows = [_odd_window_samples(w,cadence.interval_ms) for w in
        [CFG.smoothing_window_ms, CFG.envelope_max_window_ms, CFG.envelope_min_window_ms]]
    smoothed = smooth_contiguous_median(raw,steps,window_samples=windows[0])
    envelope = rolling_extreme_envelope(smoothed,steps,
        max_window_samples=windows[1],min_window_samples=windows[2])
    detector_valid = derivative_valid&np.isfinite(envelope)&(
        coverage.angular_valid_tail_fraction.to_numpy()>=CFG.minimum_valid_tail_fraction)
    moving,bouts = detect_legacy_envelope_bouts(envelope,raw,dt,steps,detector_valid,
        envelope_threshold=CFG.envelope_threshold_rad_per_ms,
        amplitude_threshold=CFG.bout_amplitude_threshold_rad_per_ms,
        minimum_bout_duration_ms=CFG.minimum_bout_duration_ms,
        maximum_interbout_gap_ms=CFG.maximum_interbout_gap_ms)
    rows, audits, examples = [], [], []
    for trial in range(5,95):
        onset = int(cycles.iloc[trial-1].Beg)
        first,last = np.searchsorted(acquired,[onset-20000,onset+20000])
        s = slice(first,last)
        t = (acquired[s]-onset)/1000
        eligible = detector_valid[s]&moving[s]&(bouts[s]>0)&np.isfinite(raw[s])&(raw[s]>0)
        index = np.searchsorted(EDGES,t,side='right')-1
        assert np.all((index>=0)&(index<80))
        p = pd.DataFrame({'time_s':t,'FrameID':ids[s],'bin_index':index,
            'bout_id':bouts[s],'eligible':eligible,
            'log_raw':np.log(np.where(eligible,raw[s],np.nan))})
        p['bout_log'] = p.loc[p.eligible].groupby('bout_id').log_raw.transform('median').reindex(p.index)
        assert np.array_equal(p.bout_log.notna(),eligible)
        # Independent sum/count and pandas groupby constructions must agree.
        counts = np.bincount(index[eligible],minlength=80)
        sums = np.bincount(index[eligible],weights=p.bout_log.to_numpy()[eligible],minlength=80)
        b = np.divide(sums,counts,out=np.full(80,np.nan),where=counts>0)
        check = p.groupby('bin_index').bout_log.mean().reindex(range(80)).to_numpy()
        np.testing.assert_allclose(b,check,atol=1e-12,rtol=0,equal_nan=True)
        base = b[10:40]; base = base[np.isfinite(base)]
        assert len(base)>0
        p10,m,p90 = np.quantile(base,[.1,.5,.9],method='linear')
        z = b-m
        cscale = (p90-p10)/2
        dscale = max(m-p10,p90-m)
        c = z/cscale if cscale>0 else np.full(80,np.nan)
        d = z/dscale if dscale>0 else np.full(80,np.nan)
        data = pd.DataFrame({'panel':panel,'trial':trial,'bin_index':np.arange(80),
            'bin_start_s':EDGES[:-1],'bin_end_s':EDGES[1:],'bin_center_s':CENTRES,
            'eligible_frames':counts,'total_frames':np.bincount(index,minlength=80),
            'uncentred_log_bin':b,'baseline_median':m,'p10':p10,'p90':p90,
            'centred_log_bin':z,'C_scale':cscale,'D_scale':dscale,
            'C_unclipped':c,'D_unclipped':d,'C':np.clip(c,-1,1),'D':np.clip(d,-1,1)})
        for col in ['centred_log_bin','C','D']:
            values = data[col].to_numpy()
            if np.isfinite(values).any():
                assert np.array_equal(np.isnan(values),counts==0)
                assert abs(np.nanmedian(values[10:40]))<1e-11
        rows.append(data)
        audits.append({'panel':panel,'trial':trial,'finite_baseline_bins':len(base),
            'p10':p10,'median':m,'p90':p90,'C_scale':cscale,'D_scale':dscale,
            'missing_bins':int((counts==0).sum()),'centred_baseline_median':float(np.nanmedian(z[10:40])),
            'C_baseline_median':float(np.nanmedian(data.C.iloc[10:40])) if cscale>0 else None,
            'D_baseline_median':float(np.nanmedian(data.D.iloc[10:40])) if dscale>0 else None})
        if trial in [9,16,17,63,66,93]:
            q=p.loc[p.eligible].groupby(['bin_index','bout_id']).agg(
                eligible_frames=('FrameID','size'),bout_log_median=('bout_log','first')).reset_index()
            q['panel']=panel;q['trial']=trial;examples.append(q)
    b = pd.concat(rows,ignore_index=True)
    metadata = {'panel':panel,'fish':fish,'fps':cadence.framerate,
        'interval_ms':cadence.interval_ms,'reference_frame_id':int(cadence.reference_frame_id),
        'reference_camera_position':int(cadence.reference_position),'anchor_ms':anchor,
        'detector_samples':windows,'inputs':[{'kind':k,'path':str(p),'sha256':hashes[k]} for k,p in paths.items()],
        'metadata_inputs':[{'path':str(p),'sha256':digest(p)} for p in
            [manifest_path,marker_path,metric_marker_path]],'missing_scalar_bins':int(b.uncentred_log_bin.isna().sum())}
    del camera,angles,coverage,local,raw,smoothed,envelope
    gc.collect()
    return b,audits,pd.concat(examples,ignore_index=True),metadata

def verify_svg(path,matrix,norm):
    root = ET.parse(path).getroot(); ns={'s':'http://www.w3.org/2000/svg'}
    groups=[g for g in root.findall('.//s:g',ns) if g.get('id','').startswith('trial_cells_')]
    assert len(groups)==3
    report=[]
    for group,(_,first,last) in zip(groups,PHASES):
        paths=group.findall('s:path',ns)
        values=matrix[first-5:last-4].ravel()
        assert len(paths)==len(values)==(last-first+1)*80
        widths=[];heights=[]
        for path,value in zip(paths,values):
            xy=np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',path.get('d'))]).reshape(-1,2)
            assert len(xy)==5
            widths.append(float(np.ptp(xy[:,0])));heights.append(float(np.ptp(xy[:,1])))
            expected=to_hex(CMAP(norm(value))) if np.isfinite(value) else '#000000'
            match=re.search(r'fill:\s*(#[0-9a-f]+)',path.get('style',''))
            # SVG's default fill is black, which Matplotlib omits from style.
            actual=match.group(1) if match else '#000000'
            assert actual==expected,(value,path.get('style'),expected)
        np.testing.assert_allclose(widths,widths[0],atol=2e-6,rtol=0)
        np.testing.assert_allclose(heights,heights[0],atol=2e-6,rtol=0)
        xs=np.unique(np.round(np.concatenate([np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',p.get('d'))]).reshape(-1,2)[:,0] for p in paths]),6))
        assert len(xs)==81
        report.append({'phase':group.get('id'),'cells':len(paths),'x_edges':len(xs),
            'cell_width_points':widths[0],'cell_height_points':heights[0],
            'all_cell_fills_match_scalar_palette':True,'black_cells':int(np.isnan(values).sum())})
    return report

def render(spec,b,col,label,half):
    panel,name,fish,*_=spec
    matrix=b.pivot(index='trial',columns='bin_index',values=col).reindex(
        index=range(5,95),columns=range(80)).to_numpy()
    assert matrix.shape==(90,80)
    norm=CenteredNorm(vcenter=0,halfrange=half,clip=True)
    assert norm(0)==.5
    fig=plt.figure(figsize=(8.2,7.1))
    grid=fig.add_gridspec(3,1,height_ratios=[10,50,30],left=.14,right=.82,bottom=.13,top=.84,hspace=.11)
    for k,(phase,first,last) in enumerate(PHASES):
        ax=fig.add_subplot(grid[k]); values=matrix[first-5:last-4]
        mesh=ax.pcolormesh(EDGES,np.arange(first-.5,last+1.5),values,
            cmap=CMAP,norm=norm,shading='flat',edgecolors='none',antialiased=False,rasterized=False)
        mesh.set_gid('trial_cells_'+phase)
        coords=mesh.get_coordinates()
        assert coords.shape==(last-first+2,81,2)
        np.testing.assert_array_equal(coords[0,:,0],EDGES)
        np.testing.assert_allclose(np.diff(coords[:,:,0],axis=1),.5,atol=0,rtol=0)
        np.testing.assert_allclose(np.ma.filled(mesh.get_array(),np.nan),values,equal_nan=True)
        ax.set_ylim(last+.5,first-.5); ax.set_yticks([first,last]);ax.set_ylabel(phase)
        ax.set_xlim(-20,20);ax.set_xticks([-20,-10,0,10,20])
        ax.tick_params(labelbottom=k==2,length=2)
        if k==2:ax.set_xlabel('Time from measured CS onset (s)')
        for t in [0,10]:ax.axvline(t,color='#168047',lw=.8,ls='--' if t==10 else '-')
        if phase=='Train' and spec[-1] is not None:ax.axvline(spec[-1],color='#964bad',lw=.9,ls=':')
    fig.text(.035,.95,panel,fontsize=20,weight='bold')
    fig.text(.14,.95,f'{name} | {fish} | {label}',fontsize=13,weight='bold')
    fig.text(.14,.90,'80 half-second cells per trial | baseline bins [-15, 0)',fontsize=10)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),
        cax=fig.add_axes([.855,.13,.025,.71]),ticks=[-half,0,half])
    cb.set_label('Centred log units' if col=='centred_log_bin' else 'Trial-scaled bin value')
    fig.text(.14,.055,'Bin scalar: finite-sample mean of eligible bout-median log vigor',fontsize=9)
    fig.text(.14,.027,'managua_r | zero = midpoint | black = no scalar or undefined quantile scale',fontsize=8)
    stem=OUT/f'{label}_Panel{panel}'
    for ext in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=200)
    plt.close(fig)
    geometry=verify_svg(stem.with_suffix('.svg'),matrix,norm)
    # Enlarged first six test trials: visibly separated half-second cells.
    fig,ax=plt.subplots(figsize=(16,2.7)); fig.subplots_adjust(left=.05,right=.93,bottom=.25,top=.78)
    mesh=ax.pcolormesh(EDGES,np.arange(64.5,71.5),matrix[60:66],
        cmap=CMAP,norm=norm,shading='flat',edgecolors='#ffffff',linewidth=.15,antialiased=False)
    ax.set_ylim(70.5,64.5);ax.set_yticks(range(65,71));ax.set_xticks(np.arange(-20,21,5))
    ax.set_xlabel('Seconds from CS onset; every rectangle is one 0.5 s scalar')
    ax.set_title(f'{panel} {name} | {label} | geometry detail, trials 65-70')
    fig.colorbar(mesh,cax=fig.add_axes([.95,.25,.01,.53]),ticks=[-half,0,half])
    fig.savefig(OUT/f'{label}_Panel{panel}_cells_detail.png',dpi=150);plt.close(fig)
    return {'panel':panel,'variant':label,'matrix_shape':[90,80],'normalization_half_range':half,
        'zero_palette_coordinate':float(norm(0)),'zero_colour':to_hex(CMAP(norm(0))),
        'svg_geometry':geometry,'missing_display_bins':int(np.isnan(matrix).sum())}

def main():
    OUT.mkdir(exist_ok=False)
    panels=[];audits=[];examples=[];geometry=[]
    for spec in FISH:
        print('Reconstructing',spec[0],flush=True)
        b,a,e,metadata=load_scalars(spec)
        b.to_parquet(OUT/f'Panel{spec[0]}_scalar_bins.parquet',index=False)
        b.to_csv(OUT/f'Panel{spec[0]}_scalar_bins.csv',index=False)
        audits.extend(a);examples.append(e);panels.append(metadata)
        for col,label,half in [('centred_log_bin','LogReference',.25),('C','C',1.),('D','D',1.)]:
            geometry.append(render(spec,b,col,label,half))
    pd.DataFrame(audits).to_csv(OUT/'trial_baseline_audit.csv',index=False)
    pd.concat(examples).to_csv(OUT/'representative_bout_contributions.csv',index=False)
    for label in ['LogReference','C','D']:
        images=[Image.open(OUT/f'{label}_Panel{p}.png').convert('RGB') for p in 'FGH']
        overview=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)),'white')
        x=0
        for im in images:overview.paste(im,(x,0));x+=im.width
        overview.save(OUT/f'{label}_FGH.png')
    report={'scope':'F/G/H only; unapproved review; new direct reconstruction',
        'scalar_assumption':'finite-sample mean of per-bout median natural-log raw vigor, on eligible samples only',
        'baseline':'median of finite bin scalars at bin indices 10 through 39, one vote per bin',
        'quantile_method':'numpy linear, finite baseline bin scalars only',
        'C':'clip((b-P50)/((P90-P10)/2), -1, 1)',
        'D':'clip((b-P50)/max(P50-P10,P90-P50), -1, 1)',
        'undefined_range':'all transformed bins NaN; no scale borrowed or support invented',
        'matplotlib_version':matplotlib.__version__,'detector_config':asdict(CFG),
        'panels':panels,'render_validation':geometry,
        'code':[{'path':str(p),'sha256':digest(p)} for p in [Path(__file__),
            REPO/'src/classical_conditioning/analysis/movement_state.py',
            REPO/'src/classical_conditioning/preprocessing/acquisition_timing.py',
            REPO/'scripts/build_figure1_legacy_vigor_heatmaps.py']],
        'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()]}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'panels':[{k:r[k] for k in ['panel','fps','missing_scalar_bins']} for r in panels],
        'svg_cell_counts':[sum(g['cells'] for g in r['svg_geometry']) for r in geometry],
        'midpoint':to_hex(CMAP(.5))},indent=2),flush=True)

if __name__=='__main__':main()
