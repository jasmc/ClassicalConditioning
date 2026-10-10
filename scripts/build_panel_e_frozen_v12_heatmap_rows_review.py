"""Single HTML Panel E review using the current frozen V12 half-second bins."""
from pathlib import Path
import sys,json,base64,hashlib,ast,io,html,argparse,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,Normalize,to_rgb,to_hex
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from classical_conditioning.figures.theme import apply_theme
ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
OUT=ROOT/'panel-e-frozen-v12-heatmap-rows-review-20261009'
TRIALS=[9,17,63,66,93];STAGES=['Habituation','Early Train','Late Train','Early Test','Late Test']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(text,key):return json.loads(re.search('<script type="application/json" id="'+key+'">(.*?)</script>',text,re.S).group(1))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-preview');args=parser.parse_args()
    pointer=REPO/'configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json'
    scoped=REPO/'configs/paper-figures/selections/figure1-fgh-full-bout-correction-20261009.json'
    current=json.loads(scoped.read_text());assert current['version12_frozen_selection']['sha256']==sha(pointer)
    rec=json.loads(pointer.read_text());container=Path(rec['container']['path'])
    assert sha(container)==rec['container']['sha256']
    text=container.read_text(encoding='utf-8');archive=audit(text,'version12-frozen-archive')
    blobs={}
    for name,item in archive['files'].items():
        raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];blobs[name]=raw
    assert hashlib.sha256(blobs['freeze.json']).hexdigest()==rec['freeze_manifest']['sha256']
    frozen=json.loads(blobs['freeze.json']);selection=json.loads(blobs['selection.json']);source_audit=audit(text,selection['source_audit_id'])
    cell_digest=hashlib.sha256(json.dumps(source_audit['cells'],ensure_ascii=True,allow_nan=False,sort_keys=True).encode()).hexdigest()
    assert cell_digest==selection['source_cells_sha256']
    source=next(x for x in source_audit['source_frames'] if x['panel']=='F');assert sha(source['path'])==source['sha256']
    f=pd.read_parquet(source['path']);f=f[f.trial.isin(TRIALS)].copy()
    edges=np.array(selection['bin_edges_s']);assert len(edges)==81;np.testing.assert_allclose(np.diff(edges),.5)
    bm=(edges[:-1]>=-15)&(edges[:-1]<0);rows=[];checks=[]
    for t,p in f.groupby('trial'):
        good=p.eligible.to_numpy(bool);L=p.log_vigor.to_numpy();time=p.time_s.to_numpy()
        np.testing.assert_allclose(L[good],np.log(p.raw_vigor.to_numpy()[good]),atol=1e-12,rtol=0)
        np.testing.assert_array_equal(np.isfinite(p.raw_vigor),np.isfinite(L))
        counts=np.array([np.sum(good&(time>=lo)&(time<hi)) for lo,hi in zip(edges[:-1],edges[1:])])
        m=np.array([np.median(L[good&(time>=lo)&(time<hi)]) if n else np.nan for lo,hi,n in zip(edges[:-1],edges[1:],counts)])
        cell=next(c for c in source_audit['cells'] if c['panel']=='F' and c['trial']==t)
        np.testing.assert_allclose(m,np.array(cell['bin_median_log'],float),atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(counts,cell['eligible_frame_count'])
        baseline=m[bm&np.isfinite(m)];z=np.full(80,np.nan);params=next(c for c in selection['trial_parameters'] if c['panel']=='F' and c['trial']==t)
        assert len(baseline)==params['finite_baseline_bins']
        if len(baseline)>=10:
            centre=float(np.median(baseline));d=m-centre;lo,mid,hi=np.quantile(d[bm&np.isfinite(d)],[.1,.5,.9],method='linear');den=max(-lo,hi)
            if den>1e-12:
                z=.7*d/den;assert params['defined']
                np.testing.assert_allclose([centre,lo,hi,den],[params['baseline_log'],params['p10_log'],params['p90_log'],params['denominator_log']],atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.nanmedian(z[bm]),0,atol=1e-12)
                np.testing.assert_allclose(max(abs(np.nanquantile(z[bm],[.1,.9]))),.7,atol=1e-12)
                np.testing.assert_array_equal(np.isnan(z),counts==0)
        checks.append({'trial':int(t),'baseline_bins':len(baseline),'defined':params['defined'],'finite_bins':int(np.isfinite(z).sum()),'values_beyond_colour_limits':int((np.abs(z)>1).sum())})
        rows.extend({'trial':int(t),'bin_index':i,'start_s':float(edges[i]),'stop_s':float(edges[i+1]),'eligible_frames':int(counts[i]),'median_log_vigor':float(m[i]),'frozen_scaled_vigor':float(z[i])} for i in range(80))
    # Reconstruct the exact frozen 513-colour CIELAB ramp from its archived functions.
    ns={'np':np};tree=ast.parse(blobs['palette-functions-snapshot.py'].decode())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'frozen_palette_functions','exec'),ns)
    anchors=selection['palette'];lab=ns['rgb_lab']([to_rgb(c) for c in anchors]);u=np.linspace(0,1,513)
    rgb=ns['lab_rgb'](np.stack([np.interp(u,[0,.5,1],lab[:,i]) for i in range(3)],axis=-1))
    cmap=ListedColormap(np.clip(rgb,0,1));norm=Normalize(-1,1,clip=True)
    apply_theme();plt.rcParams.update({'svg.fonttype':'none','path.simplify':False,'figure.autolayout':False,'figure.constrained_layout.use':False})
    fig=plt.figure(figsize=(9,6),layout='none');fig.set_layout_engine(None)
    grid=fig.add_gridspec(5,1,left=.18,right=.79,top=.82,bottom=.16,hspace=.55)
    axes=[];strips=[]
    for i in range(5):
        sub=grid[i].subgridspec(2,1,height_ratios=[4,1],hspace=.10)
        a=fig.add_subplot(sub[0]);b=fig.add_subplot(sub[1],sharex=a)
        axes.append(a);strips.append(b)
    fig.text(.08,.965,'Raw vigor + frozen heatmap rows',fontsize=13,weight='bold')
    fig.text(.08,.915,'Delay 20221115_07 · 0.5 s bins · baseline [−15,0) s',fontsize=8)
    fig.text(.08,.87,'Each strip is the matching frozen V12 heatmap row · all 80 bins shown',fontsize=8,color='#555555')
    ep=ROOT/'cadence-review-v5-20261007/Fig1_PanelsD-E_events_v5.parquet';e=pd.read_parquet(ep)
    raw_clipped=scaled_saturated=0;patches=[]
    for a,b,t,stage in zip(axes,strips,TRIALS,STAGES):
        p=f[f.trial.eq(t)]
        for r in rows:
            if r['trial']!=t:continue
            value=r['frozen_scaled_vigor'];colour=cmap(norm(value)) if np.isfinite(value) else '#000000'
            patch=b.bar(r['start_s'],1,width=.5,align='edge',color=colour,edgecolor='none',linewidth=0)[0]
            patch.set_gid(f"v12_frozen_bin_trial_{t}_bin_{r['bin_index']}")
            np.testing.assert_allclose([patch.get_x(),patch.get_width(),patch.get_height()],[r['start_s'],.5,1],atol=1e-12,rtol=0)
            assert to_hex(patch.get_facecolor())==to_hex(colour);patches.append(patch)
            scaled_saturated+=bool(np.isfinite(value) and abs(value)>1)
        b.set_ylim(0,1);b.set_yticks([]);b.set_yticks([],minor=True)
        for spine in b.spines.values():spine.set_linewidth(.4);spine.set_color('#999999')
        b.tick_params(axis='x',bottom=b is strips[-1],labelbottom=b is strips[-1],labelsize=7,length=3,width=.55,colors='#555555')
        a.plot(p.time_s,p.raw_vigor,color='black',lw=.65,alpha=.75);a.set_ylim(0,1);a.set_yticks([0,.5,1],labels=['0','0.5','1']);a.set_xlim(-20,20)
        a.text(-.12,.5,f'{stage}\nTrial {t}',transform=a.transAxes,ha='right',va='center',fontsize=7,color='#333333')
        a.spines[['top','right','bottom']].set_visible(False);a.spines['left'].set_bounds(0,1);a.spines['left'].set_color('#999999');a.spines['left'].set_linewidth(.5)
        a.tick_params(axis='y',labelsize=7,length=3,width=.55,pad=4,colors='#555555');a.tick_params(axis='x',bottom=False,labelbottom=False)
        for event in e[e['Trial number'].eq(t)].itertuples(index=False):a.axvline(float(event[2]),color='#702e78' if event.Event=='actual US onset' else '#0d8136',lw=.55,alpha=.7,ls='--' if event.Event=='CS offset' else '-')
        raw_clipped+=int(p.raw_vigor.gt(1).sum())
    fig.text(.006,.49,'Raw vigor (rad/ms)',rotation=90,fontsize=8,va='center');strips[-1].set_xlabel('Time from measured CS onset (s)',fontsize=8)
    for a in axes:a.set_xticks([-20,-10,0,10,20])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.85,.39,.011,.20]),orientation='vertical',ticks=[-1,0,1]);cb.ax.set_yticklabels(['−1','0','+1']);cb.ax.tick_params(labelsize=7,length=3,width=.5);cb.set_label('Scaled deviation from baseline',fontsize=8);cb.outline.set_edgecolor('#999999');cb.outline.set_linewidth(.5)
    fig.text(.18,.05,f'Raw display capped at 1 rad/ms: {raw_clipped} frame values exceed the range.',fontsize=7,color='#555555')
    fig.text(.18,.027,f'Frozen colours saturate at ±1 ({scaled_saturated} bins); missing bins are black. Signed data are not clipped.',fontsize=7,color='#555555')
    buf=io.StringIO();fig.savefig(buf,format='svg');svg=buf.getvalue()
    assert svg.count('id="v12_frozen_bin_')==len(patches)
    if args.qa_preview:fig.savefig(args.qa_preview,dpi=160)
    plt.close(fig)
    provenance={'current_selection':{'path':str(pointer),'sha256':sha(pointer)},'scoped_selection':{'path':str(scoped),'sha256':sha(scoped)},'container':rec['container'],'embedded_freeze':rec['freeze_manifest'],'scientific_definition':selection['scientific_definition'],'source_frames':source,'events':{'path':str(ep),'sha256':sha(ep)},'palette':anchors,'palette_interpolation':'exact archived 513-step CIELAB ramp','bin_parameters_verified_against_frozen_selection':True,'all_400_log_medians_and_counts_verified_against_frozen_cells':True,'verification':checks,'selected_bins':rows,'display':{'raw_cap':1,'raw_alpha':.75,'raw_linewidth':.65,'secondary_ticks':[],'heatmap':'one exact 80-bin row below each raw trace; equal 0.5 s widths and heights; NaN black','strip_x_limits':[-20,20],'colour_limits':[-1,1],'heatmap_opacity':1,'all_bins_rendered':400},'status':'review only; frozen FGH unchanged'}
    assert sha(container)==rec['container']['sha256'];assert sha(source['path'])==source['sha256']
    OUT.mkdir(exist_ok=True);output=OUT/'PanelE_frozen-v12-heatmap-rows-review.html'
    page='<!doctype html><html><meta charset="utf-8"><title>Panel E · frozen heatmap rows under raw traces</title><style>body{font:15px system-ui;max-width:1200px;margin:24px auto;padding:0 20px;color:#222}svg{width:100%;height:auto}pre{white-space:pre-wrap;font-size:12px}</style><body>'+svg
    page+='<p>Each raw trace is paired with its exact matching V12 heatmap row below it. Every row contains 80 equal-sized 0.5 s cells spanning [−20,20). Finite values use the exact frozen palette and limits; missing values are black. All 400 bin medians, counts and trial scaling parameters are verified against the frozen archive. Heatmap cell opacity is 100%; raw styling remains 75% opacity and 0.65 pt.</p><p>The frozen recipe is median eligible original log vigor in each bin; subtract the median of finite baseline-bin medians in [−15,0); scale by 0.7/max(−P10,P90) of centred baseline bins, with ≥10 finite baseline bins and noncollapsed spread. Values remain numerically unclipped. Source data, frozen heatmaps and prior reviews are preserved.</p><details><summary>Embedded provenance and numerical verification</summary><pre>'+html.escape(json.dumps(provenance,indent=2))+'</pre></details></body></html>'
    output.write_text(page,encoding='utf-8');print(json.dumps({'output':str(output),'selected_bins':len(rows),'rendered_cells':len(patches),'verification':checks,'protected_freeze_unchanged':True}))
if __name__=='__main__':main()
