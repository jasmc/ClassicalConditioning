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
OUT=ROOT/'panel-e-frozen-v12-halfsecond-review-20261009'
TRIALS=[9,17,63,66,93];STAGES=['Habituation','Early Train','Late Train','Early Test','Late Test']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(text,key):return json.loads(re.search('<script type="application/json" id="'+key+'">(.*?)</script>',text,re.S).group(1))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-preview');args=parser.parse_args()
    pointer=REPO/'configs/paper-figures/figure1-fgh-version12-freeze-20261009.json'
    scoped=REPO/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json'
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
    fig,axes=plt.subplots(5,1,figsize=(9,6),sharex=True,layout='none');fig.set_layout_engine(None)
    fig.subplots_adjust(left=.18,right=.79,top=.82,bottom=.16,hspace=.32)
    fig.text(.08,.965,'Raw vigor + frozen V12 signed bins',fontsize=13,weight='bold')
    fig.text(.08,.915,'Delay 20221115_07 · 0.5 s bins · baseline [−15,0) s',fontsize=8)
    fig.text(.08,.87,'Frozen bin medians, baseline-bin reference and symmetric 0.7 scale',fontsize=8,color='#555555')
    ep=ROOT/'cadence-review-v5-20261007/Fig1_PanelsD-E_events_v5.parquet';e=pd.read_parquet(ep)
    raw_clipped=scaled_clipped=0;patches=[]
    for a,t,stage in zip(axes,TRIALS,STAGES):
        p=f[f.trial.eq(t)];b=a.twinx();b.set_zorder(1);a.set_zorder(2);a.patch.set_visible(False);b.patch.set_visible(False)
        for r in rows:
            if r['trial']!=t or not np.isfinite(r['frozen_scaled_vigor']):continue
            value=r['frozen_scaled_vigor'];patch=b.bar(r['start_s'],value,width=.5,align='edge',color=cmap(norm(value)),edgecolor='none')[0]
            patch.set_gid(f"v12_frozen_bin_trial_{t}_bin_{r['bin_index']}")
            np.testing.assert_allclose([patch.get_x(),patch.get_width(),patch.get_height()],[r['start_s'],.5,value],atol=1e-12,rtol=0)
            assert to_hex(patch.get_facecolor())==to_hex(cmap(norm(value)));patches.append(patch)
            scaled_clipped+=abs(value)>1
        b.set_ylim(-1,1);b.set_yticks([]);b.set_yticks([],minor=True);b.spines[['top','bottom','left','right']].set_visible(False);b.axhline(0,color='#777777',lw=.45)
        a.plot(p.time_s,p.raw_vigor,color='black',lw=.65,alpha=.75);a.set_ylim(-1,1);a.set_yticks([0,.5,1],labels=['0','0.5','1']);a.set_xlim(-20,20)
        a.text(-.12,.5,f'{stage}\nTrial {t}',transform=a.transAxes,ha='right',va='center',fontsize=7,color='#333333')
        a.spines[['top','right','bottom']].set_visible(False);a.spines['left'].set_bounds(0,1);a.spines['left'].set_color('#999999');a.spines['left'].set_linewidth(.5)
        a.tick_params(axis='y',labelsize=7,length=3,width=.55,pad=4,colors='#555555');a.tick_params(axis='x',bottom=a is axes[-1],labelsize=7,length=3,width=.55,colors='#555555')
        for event in e[e['Trial number'].eq(t)].itertuples(index=False):a.axvline(float(event[2]),color='#702e78' if event.Event=='actual US onset' else '#0d8136',lw=.55,alpha=.7,ls='--' if event.Event=='CS offset' else '-')
        raw_clipped+=int(p.raw_vigor.gt(1).sum())
    fig.text(.006,.49,'Raw vigor (rad/ms)',rotation=90,fontsize=8,va='center');axes[-1].set_xlabel('Time from measured CS onset (s)',fontsize=8);axes[-1].set_xticks([-20,-10,0,10,20])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.85,.39,.011,.20]),orientation='vertical',ticks=[-1,0,1]);cb.ax.set_yticklabels(['−1','0','+1']);cb.ax.tick_params(labelsize=7,length=3,width=.5);cb.set_label('Scaled deviation from baseline',fontsize=8);cb.outline.set_edgecolor('#999999');cb.outline.set_linewidth(.5)
    fig.text(.18,.05,f'Display clipping: {raw_clipped} raw frame values above 1 rad/ms; {scaled_clipped} signed bins outside ±1.',fontsize=7,color='#555555')
    fig.text(.18,.027,'Signed data are not clipped; colours saturate at ±1. Empty bins remain gaps.',fontsize=7,color='#555555')
    buf=io.StringIO();fig.savefig(buf,format='svg');svg=buf.getvalue()
    assert svg.count('id="v12_frozen_bin_')==len(patches)
    if args.qa_preview:fig.savefig(args.qa_preview,dpi=160)
    plt.close(fig)
    provenance={'current_selection':{'path':str(pointer),'sha256':sha(pointer)},'scoped_selection':{'path':str(scoped),'sha256':sha(scoped)},'container':rec['container'],'embedded_freeze':rec['freeze_manifest'],'scientific_definition':selection['scientific_definition'],'source_frames':source,'events':{'path':str(ep),'sha256':sha(ep)},'palette':anchors,'palette_interpolation':'exact archived 513-step CIELAB ramp','bin_parameters_verified_against_frozen_selection':True,'all_400_log_medians_and_counts_verified_against_frozen_cells':True,'verification':checks,'selected_bins':rows,'display':{'raw_cap':1,'signed_axis':[-1,1],'raw_alpha':.75,'raw_linewidth':.65,'secondary_ticks':[],'zero_alignment':'centred','bars':'behind raw traces'},'status':'review only; frozen FGH unchanged'}
    assert sha(container)==rec['container']['sha256'];assert sha(source['path'])==source['sha256']
    OUT.mkdir(exist_ok=True);output=OUT/'PanelE_frozen-v12-halfsecond-review.html'
    page='<!doctype html><html><meta charset="utf-8"><title>Panel E · frozen V12 half-second bins</title><style>body{font:15px system-ui;max-width:1200px;margin:24px auto;padding:0 20px;color:#222}svg{width:100%;height:auto}pre{white-space:pre-wrap;font-size:12px}</style><body>'+svg
    page+='<p>This review follows the current author-frozen V12 heatmaps: median eligible natural-log vigor in each left-inclusive 0.5 s bin; subtract the median of finite baseline-bin medians in [−15,0); scale by 0.7/max(−P10,P90) of those centred baseline bins. At least 10 finite baseline bins and nonzero spread are required. Original log-bin medians, eligible counts and trial parameters match the frozen archive. Numeric values remain unclipped; display/colour saturation is labelled.</p><p>The latest frozen selection supersedes the historical Version 1 used by the preceding Panel E reviews. Its frozen bright blue–charcoal–red palette also replaces the earlier managua_r palette. Empty bins remain gaps. Existing source data, heatmaps and prior reviews are preserved.</p><details><summary>Embedded provenance and numerical verification</summary><pre>'+html.escape(json.dumps(provenance,indent=2))+'</pre></details></body></html>'
    output.write_text(page,encoding='utf-8');print(json.dumps({'output':str(output),'selected_bins':len(rows),'finite_bars':len(patches),'verification':checks,'protected_freeze_unchanged':True}))
if __name__=='__main__':main()
