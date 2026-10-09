"""Temporary builder; exact code is embedded in the sole consolidated HTML."""
from pathlib import Path
import ast, base64, functools, gc, hashlib, html, io, json, re, sys, warnings
from datetime import datetime, timezone
from dataclasses import asdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgb

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.cohort import load_cohort_manifest, logical_cohort_hash
from classical_conditioning.analysis.figure4 import verify_expected_us
from classical_conditioning.figures.theme import apply_theme, condition_color
from classical_conditioning.config.experiments import get_experiment_spec
sys.path.insert(0,str(ROOT/'scripts'))
from render_legacy_ssd_figure2_delay import _verify_cohort
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
from classical_conditioning.analysis.movement_state import MovementCalibrationConfig, _odd_window_samples, smooth_contiguous_median, rolling_extreme_envelope, detect_legacy_envelope_bouts

HTML=ROOT/'reviews/fgh_candidate_palette_versions_20261009/index.html'
SRC=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/sources/20261008T200908002798Z')
KEY='figure2-row1-v12-pooled-review'
COL='legacy_distal_angular_speed_rad_per_ms'
EDGES=np.arange(-20,20.5,.5); BASE=(EDGES[:-1]>=-15)&(EDGES[:-1]<0)
TRIALS=np.arange(5,95)
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
@functools.lru_cache(None)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def art(p): return {'path':str(p),'sha256':sha(str(p))}
def get_script(text,key):
    return re.search(r'<script type="application/json" id="'+key+r'">(.*?)</script>',text,re.S).group(1)
def safe_json(v):
    return json.dumps(v,allow_nan=False,ensure_ascii=True).replace('<','\\u003c')
def tag(k,obj): return '<script type="application/json" id="'+k+'">'+safe_json(obj)+'</script>'

def normalize(m):
    """V12, per fish/trial. Physical d survives a failed scaling screen."""
    q=m[BASE&np.isfinite(m)]
    b=float(np.median(q)) if len(q) else np.nan
    d=m-b; z=np.full_like(m,np.nan)
    q10=q90=s=np.nan; reason=''
    if len(q)<10: reason='fewer_than_10_finite_baseline_bins'
    else:
        q10,q90=np.quantile(d[BASE&np.isfinite(d)],[.1,.9],method='linear')
        s=max(-q10,q90)
        if s<=1e-12: reason='collapsed_baseline_spread'
        else: z=.7*d/s
    return d,z,b,float(q10),float(q90),float(s),len(q),reason

def bin_values(t,L,good):
    bins=np.floor((t+20)/.5).astype(int)
    m=np.full(80,np.nan); n=np.zeros(80,np.int32)
    ids=np.unique(bins[good])
    for j in ids:
        x=L[good&(bins==j)]; m[j]=np.median(x); n[j]=len(x)
    return m,n

def invariant_checks():
    tested=[]
    for n in [10,11,30]:
        m=np.full(80,np.nan);m[np.flatnonzero(BASE)[:n]]=np.linspace(-2,3,n);m[40:]=np.linspace(-4,4,40)
        d,z,b,lo,hi,s,nb,why=normalize(m)
        assert not why and nb==n
        assert abs(np.nanmedian(z[BASE]))<1e-12
        assert abs(max(abs(np.nanquantile(z[BASE],[.1,.9])))-.7)<1e-12
        assert np.array_equal(np.isnan(z),np.isnan(m))
        dm,zm,*_=normalize(-m)
        np.testing.assert_allclose(zm,-z,atol=1e-12,equal_nan=True)
        _,zu,*_=normalize(m+np.log(1000))
        np.testing.assert_allclose(zu,z,atol=1e-12,equal_nan=True)
        x=m.copy();x[~BASE]=100
        assert normalize(x)[2:7]==normalize(m)[2:7]
    for n in [0,1,9]:
        m=np.full(80,np.nan);m[np.flatnonzero(BASE)[:n]]=np.arange(n)
        assert np.isnan(normalize(m)[1]).all()
    m=np.ones(80);assert normalize(m)[-1]=='collapsed_baseline_spread'
    t=np.array([-20.,-19.5,-15.,0.,19.999]);L=np.array([1.,2.,3.,4.,5.]);g=np.ones(5,bool)
    m,n=bin_values(t,L,g)
    assert m[0]==1 and m[1]==2 and m[10]==3 and m[40]==4 and m[79]==5
    assert np.isnan(m[2]) and n[0]==1
    # No-bout outlier cannot affect bins.
    m2,n2=bin_values(np.r_[t,0.1],np.r_[L,1e100],np.r_[g,False])
    np.testing.assert_allclose(m,m2,equal_nan=True);assert np.array_equal(n,n2)
    # Binwise population medians need not leave a zero baseline median.
    f=np.array([[1,1,0,-1,-1],[0,1,1,-1,-1],[1,0,1,-1,-1]])
    assert np.median(f,axis=1).sum()==0 and np.median(np.median(f,axis=0))==1
    return ['odd/even and sparse baselines','empty and constant baseline exclusion','half-open bin boundaries','single-sample bins','missing masks','baseline median zero','wider percentile side at +/-0.7','reciprocal symmetry','unit invariance','scale independent of response','no-bout outlier invariance','population-centre counterexample']

def read_windows(path,columns,intervals):
    f=pq.ParquetFile(path);idx=f.schema.names.index('AbsoluteTime');parts=[]
    for k in range(f.num_row_groups):
        s=f.metadata.row_group(k).column(idx).statistics
        assert s and s.min is not None
        overlaps=[(a,b) for a,b in intervals if s.max>=a and s.min<b]
        if not overlaps:continue
        d=f.read_row_group(k,columns=columns).to_pandas();t=d.AbsoluteTime.to_numpy(float)
        good=np.zeros(len(d),bool)
        for a,b in overlaps:good|=(t>=a)&(t<b)
        parts.append(d.loc[good])
    assert parts
    return pd.concat(parts,ignore_index=True)

def build_fish(exp,project,cohort,side):
    evidence=[];all_m=[];all_d=[];all_z=[];all_n=[];params=[];joins=[]
    recorded={r['path']:r['sha256'] for r in side['inputs']}
    for i,r in enumerate(cohort.itertuples(index=False),1):
        rid=str(r.recording_id);proc=project/'Processed data'/rid;meta=project/'Metadata'
        delay=exp=='allDelay'
        metric=proc/('frame_activity_candidates-corrected-v1.parquet' if delay else 'frame_activity_candidates-corrected.parquet')
        movement=proc/('movement_state_candidates-corrected-v2.parquet' if delay else 'movement_state_candidates-corrected.parquet')
        protocol=proc/'stimulus_events.parquet'
        marker_m=meta/(rid+('_candidate-corrected-v1_complete.json' if delay else '_candidate-corrected_complete.json'))
        marker_v=meta/(rid+('_movement-candidate-corrected-v2_complete.json' if delay else '_movement-candidate-corrected_complete.json'))
        source=meta/(rid+'_source_manifest.json')
        print(f'{exp} {i}/{len(cohort)} {rid}: authenticate/read',flush=True)
        for p in [metric,protocol]:
            assert str(p) in recorded,str(p)
            assert sha(str(p))==recorded[str(p)],str(p)
        mm=json.loads(marker_m.read_text());mv=json.loads(marker_v.read_text());sm=json.loads(source.read_text())
        for x in [mm,mv]:assert x['status']=='complete' and x['recording_id']==rid
        assert mm['metrics_sha256']==sha(str(metric))
        assert sm['recording_id']==rid and sm['artifacts']['protocol']['sha256']==sha(str(protocol))
        for p in [marker_m,marker_v,source]:
            if str(p) in recorded:assert sha(str(p))==recorded[str(p)]
        camera_path=proc/'camera.parquet'
        corrected=proc/('frame_preprocessed_corrected-v1.parquet' if delay else 'frame_preprocessed_corrected.parquet')
        corrected_marker=meta/(rid+('_corrected-preprocess-v1_complete.json' if delay else '_corrected-preprocess_complete.json'))
        cm=json.loads(corrected_marker.read_text())
        assert cm['status']=='complete' and cm['recording_id']==rid
        assert sha(str(corrected))==cm['frames_sha256']
        assert sha(str(camera_path))==sm['artifacts']['camera']['sha256']
        evidence.extend(art(p) for p in [metric,protocol,marker_m,source,camera_path,corrected,corrected_marker])
        camera=pd.read_parquet(camera_path);cadence=estimate_camera_cadence(camera)
        assert not cadence.has_frame_loss_evidence,(rid,asdict(cadence))
        anchor=float(camera.iloc[cadence.reference_position].AbsoluteTime)
        del camera;gc.collect()
        events=pq.read_table(protocol).to_pandas();cycles=events.loc[events.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
        assert len(cycles)>=94 and np.all(np.diff(cycles.Beg)>0)
        if r.condition_id!='control':verify_expected_us(events,exp)
        starts=cycles.iloc[TRIALS-1].Beg.to_numpy(np.int64)
        intervals=[(int(s)-22000,int(s)+22001) for s in starts]
        m=read_windows(corrected,['FrameID','AbsoluteTime','frame_valid','timestamp_valid']+[f'angle{k}' for k in range(16)],intervals)
        v=read_windows(metric,['FrameID','AbsoluteTime','angular_valid_tail_fraction'],intervals)
        assert m.FrameID.is_unique and v.FrameID.is_unique
        assert np.array_equal(m[['FrameID','AbsoluteTime']].to_numpy(),v[['FrameID','AbsoluteTime']].to_numpy())
        arrivals=m.AbsoluteTime.to_numpy(float);ids=m.FrameID.to_numpy(np.int64);steps=np.r_[0,np.diff(ids)]
        assert np.all(np.isfinite(arrivals)) and np.all(np.diff(arrivals)>=0) and np.all(steps[1:]>0)
        t=anchor+(ids-cadence.reference_frame_id)*cadence.interval_ms
        dt=steps*cadence.interval_ms
        valid=m.frame_valid.to_numpy(bool)&m.timestamp_valid.to_numpy(bool)
        local=m[[f'angle{k}' for k in range(16)]].to_numpy(float);local[~valid]=np.nan
        bend=np.sum(local,axis=1);change=np.r_[np.nan,np.diff(bend)]
        adjacent=(steps==1)&(dt>0)&(dt<=10)
        derivative=valid&np.r_[False,valid[:-1]]&adjacent
        raw=np.abs(np.arctan2(np.sin(change),np.cos(change)))/np.where(dt>0,dt,np.nan);raw[~derivative]=np.nan
        cfg=MovementCalibrationConfig()
        windows=[_odd_window_samples(w,cadence.interval_ms) for w in [cfg.smoothing_window_ms,cfg.envelope_max_window_ms,cfg.envelope_min_window_ms]]
        smooth=smooth_contiguous_median(raw,steps,window_samples=windows[0])
        envelope=rolling_extreme_envelope(smooth,steps,max_window_samples=windows[1],min_window_samples=windows[2])
        detector_valid=derivative&np.isfinite(envelope)&(v.angular_valid_tail_fraction.to_numpy()>=cfg.minimum_valid_tail_fraction)
        moving,boutids=detect_legacy_envelope_bouts(envelope,raw,dt,steps,detector_valid,envelope_threshold=cfg.envelope_threshold_rad_per_ms,amplitude_threshold=cfg.bout_amplitude_threshold_rad_per_ms,minimum_bout_duration_ms=cfg.minimum_bout_duration_ms,maximum_interbout_gap_ms=cfg.maximum_interbout_gap_ms)
        eligible=detector_valid&moving&(boutids>0)&np.isfinite(raw)&(raw>0)
        L=np.full(len(m),np.nan);L[eligible]=np.log(raw[eligible])
        fm=[];fd=[];fz=[];fn=[]
        for trial,onset in zip(TRIALS,starts):
            a=np.searchsorted(t,onset-20000,'left');b=np.searchsorted(t,onset+20000,'left')
            seconds=(t[a:b]-onset)/1000
            cell,n=bin_values(seconds,L[a:b],eligible[a:b]);d,z,bl,q10,q90,s,nb,why=normalize(cell)
            if not why:
                np.testing.assert_allclose(np.nanmedian(z[BASE]),0,atol=1e-12)
                assert abs(max(abs(np.nanquantile(z[BASE],[.1,.9])))-.7)<1e-12
                assert np.array_equal(np.isnan(cell),np.isnan(z))
            params.append({'experiment':exp,'recording_id':rid,'condition':r.condition_id,'trial':int(trial),'baseline_log':bl,'q10_delta':q10,'q90_delta':q90,'spread':s,'finite_baseline_bins':nb,'exclusion_reason':why,'defined':not bool(why)})
            fm.append(cell);fd.append(d);fz.append(z);fn.append(n)
        all_m.append(fm);all_d.append(fd);all_z.append(fz);all_n.append(fn)
        joins.append({'recording_id':rid,'window_frame_rows':len(m),'eligible_frame_rows':int(eligible.sum()),'adjacency_excluded_rows':int((~adjacent).sum()),'camera_cadence':asdict(cadence),'detector_configuration':asdict(cfg),'duplicate_frames':0,'join':'one-to-one corrected frames/coverage by FrameID/AbsoluteTime equality checked'})
        del m,v,L,raw,local,bend,change,smooth,envelope,moving,boutids;gc.collect()
    return {'m':np.array(all_m),'d':np.array(all_d),'z':np.array(all_z),'counts':np.array(all_n),'parameters':params,'input_artifacts':evidence,'join_checks':joins,'recording_ids':cohort.recording_id.tolist(),'conditions':cohort.condition_id.tolist()}

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    return v

def summaries(data):
    pooled={};rows=[]
    for exp,d in data.items():
        for c in sorted(set(d['conditions'])):
            x=d['z'][np.array(d['conditions'])==c];n=np.isfinite(x).sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                med=np.nanmedian(x,axis=0)
            mean=np.divide(np.nansum(x,axis=0),n,out=np.full(n.shape,np.nan),where=n>0)
            count=len(x);pooled[(exp,c)]={'median':med,'mean':mean,'contributors':n,'coverage':n/count,'total':count}
            for estimator,Z in [('median',med),('mean',mean)]:
                for phase,sel in [('Pre',TRIALS<=14),('Train',(TRIALS>=15)&(TRIALS<=64)),('Test',TRIALS>=65),('All',np.ones(90,bool))]:
                    q=Z[sel];base=q[:,BASE];finite=np.isfinite(q)
                    rows.append({'assay':exp,'condition':c,'estimator':estimator,'phase':phase,'cohort_n':count,'finite_cells':int(finite.sum()),'total_cells':q.size,'pooled_baseline_median':float(np.nanmedian(base)),'baseline_above_zero':int((base>1e-12).sum()),'baseline_below_zero':int((base<-1e-12).sum()),'baseline_zero':int((np.abs(base)<=1e-12).sum()),'coverage_min':float((n[sel]/count).min()),'coverage_median':float(np.median(n[sel]/count)),'saturated_cells':int((np.abs(q)>1).sum()),'saturation_fraction':float((np.abs(q)>1).sum()/max(1,finite.sum()))})
            print(exp,c,'defined fish/trials',int(np.isfinite(x).any(axis=2).sum()),'/',count*90,'median baseline',np.nanmedian(med[:,BASE]),'mean baseline',np.nanmedian(mean[:,BASE]),flush=True)
    return pooled,rows

def draw_row(pooled,which,cmap):
    apply_theme();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.titlesize':7,'svg.fonttype':'none','svg.hashsalt':'fig2-row1-v12-'+which,'savefig.bbox':None})
    fig=plt.figure(figsize=(183/25.4,94/25.4));gs=fig.add_gridspec(1,3,left=.065,right=.98,bottom=.23,top=.82,wspace=.23,width_ratios=[1,1,1])
    maps=[]
    for col,(exp,paired,panel) in enumerate([('allDelay','delay','A'),('all3sTrace','trace','B')]):
        sub=gs[col].subgridspec(1,2,wspace=.12)
        for k,c in enumerate(['control',paired]):
            ax=fig.add_subplot(sub[k]);p=pooled[(exp,c)]
            arr=p['coverage'] if which=='coverage' else p[which]
            ax.pcolormesh(EDGES,np.arange(4.5,95.5),arr,cmap=cmap,vmin=0 if which=='coverage' else -1,vmax=1,rasterized=False,edgecolors='none',linewidth=0).set_gid(f'fig2__{panel.lower()}__{c}__{which}-cells')
            ax.set_xlim(-20,20);ax.set_ylim(94.5,4.5);ax.set_xticks([-20,0,20]);ax.set_yticks([10,30,50,70,90]);ax.tick_params(length=2,width=.5,pad=2)
            if k or col:ax.set_yticklabels([])
            if not col and not k:ax.set_ylabel('Global CS trial')
            spec=next(s for s in get_experiment_spec(exp).conditions if s.condition_id==c)
            ax.set_title(('Control' if c=='control' else 'Delay' if c=='delay' else '3sTrace')+'\n(n='+str(p['total'])+')',color=condition_color(spec),fontsize=6.5,pad=4)
            for t,ls in [(0,'-'),(10,'--')]:ax.axvline(t,color='#168241',ls=ls,lw=.8,alpha=.9,zorder=3)
            if c!='control':ax.axvline(9 if c=='delay' else 13,color='#9257a5',ls=':',lw=.7,zorder=3)
            for t in [14.5,64.5]:ax.axhline(t,color='white',lw=.65,zorder=4)
            for spine in ax.spines.values():spine.set_linewidth(.5)
        bb=gs[col].get_position(fig);fig.text((bb.x0+bb.x1)/2,.945,f'{panel}  '+('Delay' if col==0 else '3sTrace'),ha='center',weight='bold',fontsize=8)
    ax=fig.add_subplot(gs[2]);ax.set_facecolor('#f1f1ef');ax.set_xticks([]);ax.set_yticks([])
    ax.text(.5,.62,'C  10sTrace',ha='center',va='center',transform=ax.transAxes,weight='bold',fontsize=8)
    ax.text(.5,.42,'Unavailable\nNo authenticated cohort / frames\nInterpretation: inconclusive',ha='center',va='center',transform=ax.transAxes,fontsize=7)
    fig.text(.365,.165,'Time from CS onset (s)',ha='center',fontsize=7)
    fig.text(.065,.985,'Pooled V12 Â· '+which+' Â· review candidate',fontsize=9,weight='bold',va='top')
    cax=fig.add_axes([.13,.065,.50,.027]);norm=Normalize(0 if which=='coverage' else -1,1)
    bar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,.5,1] if which=='coverage' else [-1,-.7,0,.7,1]);bar.ax.tick_params(labelsize=6,length=2,pad=1)
    bar.set_label('Contributing fish / cohort fish' if which=='coverage' else 'Pooled normalized fish deviation (baseline-spread units)',fontsize=6.5,labelpad=1)
    fig.canvas.draw();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=190,facecolor='white');png=buf.getvalue()
    plt.close(fig)
    return png

def main():
    original=HTML.read_bytes();text=original.decode('utf-8');assert f'id="{KEY}"' not in text
    archive_text=get_script(text,'version12-frozen-archive');archive=json.loads(archive_text)
    for name,item in archive['files'].items():assert sha_bytes(base64.b64decode(item['base64']))==item['sha256'],name
    palette_code=base64.b64decode(archive['files']['palette-functions-snapshot.py']['base64']).decode('utf-8')
    ns={'np':np};tree=ast.parse(palette_code);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'frozen-palette','exec'),ns)
    anchors=['#00bfff','#383842','#ff5252'];lab=ns['rgb_lab']([to_rgb(c) for c in anchors]);u=np.linspace(0,1,513)
    rgb=np.clip(ns['lab_rgb'](np.stack([np.interp(u,[0,.5,1],lab[:,i]) for i in range(3)],axis=-1)),0,1)
    np.testing.assert_allclose(rgb[[0,256,512]],[to_rgb(c) for c in anchors],atol=1e-6)
    cmap=ListedColormap(rgb);cmap.set_bad('black');coverage=plt.get_cmap('cividis').copy();coverage.set_bad('black')
    checks=invariant_checks();print('Synthetic invariant checks passed',flush=True)
    layout=json.loads((ROOT/'configs/paper-figures/figure2-assembly.json').read_text())
    protect_paths=[ROOT/'configs/paper-figures/figure2-assembly.json',ROOT/'configs/paper-figures/figure2-DE-B-freeze-20261009.json',ROOT/'configs/paper-figures/figure2-G-logmedian-freeze-20261009.json']
    protected=[art(p) for p in protect_paths if p.exists()]
    data={};bindings={};cohort_records=[]
    for label,exp,project,cid in [('A','allDelay',Path('J:/Digested Data/allDelay-full-v1'),'allDelay-full-v1'),('B','all3sTrace',Path('F:/Digested Data/all3sTrace-full-v1'),'all3sTrace-full-exploratory')]:
        p=next(x for x in layout['panels'] if x['id']==label);side=Path(p['source_provenance']['sidecar'])
        assert sha(str(side))==p['source_provenance']['sidecar_sha256']
        identity=json.loads(side.read_text());assert identity['analysis_identity']['baseline_s']==[-15,0]
        if exp=='allDelay':
            cohort,logical,cohort_inputs=_verify_cohort(project)
            cohort_records.extend(cohort_inputs)
        else:
            cohort=load_cohort_manifest(project,cid);logical=logical_cohort_hash(cohort)
        assert logical==identity['analysis_identity']['cohort_hash'],(logical,identity['analysis_identity']['cohort_hash'])
        cohort=cohort.loc[cohort.primary_included.eq(True)].sort_values('recording_id').reset_index(drop=True)
        assert not cohort.recording_id.duplicated().any();assert cohort.groupby('condition_id').size().to_dict()==identity['analysis_identity']['cohort_counts']
        for path in [project/'Processed data/Cohorts'/cid/('cohort-manifest-v1.parquet' if exp=='allDelay' else 'cohort-manifest.parquet')]:cohort_records.append(art(path))
        bindings[label]={'cohort_hash':logical,'counts':identity['analysis_identity']['cohort_counts'],'old_source':art(Path(p['source'])),'old_sidecar':art(side)}
        data[exp]=build_fish(exp,project,cohort,identity)
    example_audit=json.loads(get_script(text,'version11-audit'))
    example_checks=[]
    for panel,rid in example_audit['fish'].items():
        exp='all3sTrace' if panel=='G' else 'allDelay';d=data[exp]
        idx=d['recording_ids'].index(rid)
        for cell in example_audit['cells']:
            if cell['panel']==panel:
                np.testing.assert_allclose(d['m'][idx,int(cell['trial'])-5],np.array(cell['bin_median_log'],float),atol=1e-12,rtol=0,equal_nan=True)
        example_checks.append({'panel':panel,'fish':rid,'result':'all 90 direct-bin median arrays independently reproduced from corrected frames'})
    print('Frozen F/G/H example arrays reproduced exactly',flush=True)
    pooled,stats=summaries(data)
    arrays={}
    for exp,d in data.items():
        for k in ['m','d','z','counts']:arrays[exp+'_'+k]=d[k]
    for (exp,c),p in pooled.items():
        for k in ['median','contributors','coverage']:arrays[exp+'_'+c+'_'+k]=p[k]
    buf=io.BytesIO();np.savez_compressed(buf,**arrays);blob=buf.getvalue()
    provenance={'status':'new review candidate; not selected or scientifically frozen','created_utc':datetime.now(timezone.utc).isoformat(),'authorization':'ok. adapt handout to pooled data a make a new version of pooled heatmpas','scope':'Figure 2 A/B population review; C unavailable; all other rows preserved','recipe':'V12 direct original-log-frame bin medians; equal-bin pre15 median; 0.7 symmetric P10/P90 spread; >=10 finite baseline bins, spread >1e-12; no numeric clipping; normalize per fish/trial then pool one finite value per fish; no second centring/scaling','timing':'Same camera-cadence-reconstructed acquisition time and wrapped summed-angle vigor calculation as frozen Fig1 V12. Cadence estimated from authenticated camera records; frame-loss evidence rejected. Legacy smoothing/envelope used only for the inherited movement detector; original unsmoothed eligible raw vigor enters log/bin calculations. This is not newly authenticated hardware exposure timing.','eligibility':'Same reconstructed derivative/coverage and legacy envelope-bout detector as frozen example frames; adjacent valid corrected frames; finite strictly positive unsmoothed vigor on valid detected bouts','cohort_bindings':bindings,'cohort_artifacts':cohort_records,'bin_edges_s':EDGES.tolist(),'trials':TRIALS.tolist(),'baseline_bin_mask':BASE.tolist(),'palette_rgb_513':rgb.tolist(),'palette_source_archive_hash':sha_bytes(archive_text.encode()),'palette_source_file_hash':archive['files']['palette-functions-snapshot.py']['sha256'],'previous_html_sha256':sha_bytes(original),'immutable_archive_entries_verified':{k:v['sha256'] for k,v in archive['files'].items()},'protected_artifacts':protected,'validation':checks,'frozen_example_reproduction':example_checks,'population_statistics':stats,'fish_parameters':{exp:clean(d['parameters']) for exp,d in data.items()},'fish_order_and_conditions':{exp:{'recording_ids':d['recording_ids'],'conditions':d['conditions']} for exp,d in data.items()},'source_artifacts':{exp:d['input_artifacts'] for exp,d in data.items()},'join_checks':{exp:d['join_checks'] for exp,d in data.items()},'array_archive':{'format':'NumPy compressed NPZ; fish x trial x bin; NaNs retained','sha256':sha_bytes(blob),'base64':base64.b64encode(blob).decode()},'builder_code':{'sha256':sha_bytes(Path(__file__).read_bytes()),'base64':base64.b64encode(Path(__file__).read_bytes()).decode()},'presentation':{'row_width_mm':183,'row_height_mm':94,'font':'DejaVu Sans','panel_box_width_mm':54.9,'nominal_event_guides':'green solid CS onset 0; green dashed CS offset 10; purple dotted paired US 9/13; omission/catch rows retain nominal guides; control no guessed US','phase_boundaries':[14.5,64.5],'colorbar_ticks':[-1,-.7,0,.7,1],'coverage_palette':'cividis; 0..1; separate from vigor','publication_status':'review PNG embedded in-memory; no new freeze command requested'}}
    plots=''
    for which,pal in [('median',cmap)]:
        img=draw_row(pooled,which,pal)
        plots+=f'<h3>Equal-fish {which if which!="coverage" else "coverage (shared by both estimators)"}</h3><img style="width:100%;max-width:1200px;height:auto" alt="Figure 2 row 1 {which}" src="data:image/png;base64,{base64.b64encode(img).decode()}">'
    table=pd.DataFrame(stats);shown=table.loc[table.phase.eq('All')].drop(columns=['phase','baseline_zero'])
    exc=[]
    for exp,d in data.items():
        p=pd.DataFrame(d['parameters'])
        for (c,why),q in p.groupby(['condition','exclusion_reason']):exc.append({'assay':exp,'condition':c,'status':why or 'defined','fish_trials':len(q)})
    # Include unchanged old current previews for a concrete comparison, no copies.
    historical=''
    for label,exp in [('A','allDelay'),('B','all3sTrace')]:
        p=SRC/f'Fig2_Panel{label}_{exp}_signed-pre15.png';b=p.read_bytes()
        historical+=f'<h4>Preserved older {label}: signed bout-summary bins, physical +/-0.25 managua</h4><img style="width:48%;min-width:300px" src="data:image/png;base64,{base64.b64encode(b).decode()}" alt="Older {label}">'
    section='<section id="figure2-row1-v12-population" style="background:white;padding:24px;margin:24px 0;border:2px solid #383842"><h2>Figure 2 row 1: pooled Figure 1 V12</h2><p>A: Delay 29 / control 28. B: full exploratory 3sTrace 40 / control 19. C remains unavailable and inconclusive.</p><p>Same single-fish V12 calculation: eligible original log-frame medians in 0.5-s bins; baseline-bin median over [-15,0); symmetric baseline P10/P90 spread; factor 0.7; at least 10 finite baseline bins. Then take the equal-fish median of finite normalized cells. No second centring, scaling or numeric clipping. Exact frozen V12 palette, limits [-1,+1], black missing cells.</p>'+plots+'<p>Colors express fractions of individual fish/trial baseline spread. The pooled baseline median need not be zero. Nominal event guides are not proof of delivery on every trial. This is a new review version; the frozen Figure 1 archive and other Figure 2 rows remain unchanged.</p><details><summary>Embedded scientific provenance</summary><p>Source hashes, cohorts, joins, camera cadence, detector settings, bin counts, physical values, normalized values and exclusions are embedded below. Source and numerical checks passed. The camera-cadence timing assumption is inherited from Figure 1 V12; upstream audits and population selection remain open.</p></details>'+tag(KEY,clean(provenance))+'</section>'
    # Detect concurrent edits and preserve all existing bytes verbatim.
    latest=HTML.read_bytes();assert latest==original,'Review HTML changed during calculation; merge required'
    for r in protected:assert sha_bytes(Path(r['path']).read_bytes())==r['sha256']
    insert=text.rfind('</body>');assert insert>=0
    revised=text[:insert]+section+text[insert:]
    assert get_script(revised,'version12-frozen-archive')==archive_text
    old_imgs=re.findall(r'<img\b[^>]*src="([^"]+)"',text)
    assert re.findall(r'<img\b[^>]*src="([^"]+)"',revised)[:len(old_imgs)]==old_imgs
    j=json.loads(get_script(revised,KEY));assert sha_bytes(base64.b64decode(j['array_archive']['base64']))==j['array_archive']['sha256']
    loaded=np.load(io.BytesIO(base64.b64decode(j['array_archive']['base64'])))
    for k,x in arrays.items():np.testing.assert_allclose(loaded[k],x,equal_nan=True)
    tmp=HTML.with_name('.index-row1-v12-writing.tmp');tmp.write_bytes(revised.encode('utf-8'));tmp.replace(HTML)
    assert get_script(HTML.read_bytes().decode('utf-8'),'version12-frozen-archive')==archive_text
    print('COMPLETED',str(HTML),'bytes',HTML.stat().st_size,flush=True)

if __name__=='__main__':main()
