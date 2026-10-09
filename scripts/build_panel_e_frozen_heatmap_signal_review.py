"""One embedded HTML review of Panel E using exact frozen C_BoutSamples values."""
from pathlib import Path
import hashlib,json,io,sys,argparse,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,to_hex
import xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from classical_conditioning.figures.theme import apply_theme
ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
OUT=ROOT/'panel-e-exact-frozen-heatmap-runs-review-20261009'
TRIALS=[9,17,63,66,93]
STAGES=['Habituation','Early Train','Late Train','Early Test','Late Test']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-preview');args=parser.parse_args()
    record=REPO/'configs/paper-figures/figure1-fgh-version1-freeze-20261009.json'
    sel=json.loads(record.read_text());freeze_path=Path(sel['freeze_manifest'])
    assert sha(freeze_path)==sel['freeze_manifest_sha256']
    freeze=json.loads(freeze_path.read_text());assert freeze['selected_version']=='C_BoutSamples'
    protected={p['path']:p['sha256'] for p in freeze['snapshot_files']}
    for p,digest in protected.items():assert sha(p)==digest,p
    source=next(p for p in freeze['original_sample_tables'] if p['panel']=='F')
    assert sha(source['path'])==source['sha256']
    data=pd.read_parquet(source['path']);f=data[data.trial.isin(TRIALS)].copy()
    stat_path=freeze_path.parent/'sample_baseline_statistics.csv'
    stats=pd.read_csv(stat_path);stats=stats[stats.panel.eq('F')&stats.trial.isin(TRIALS)]
    run_path=freeze_path.parent/'PanelF_display_sample_runs.csv'
    runs=pd.read_csv(run_path);runs=runs[runs.trial.isin(TRIALS)]
    verification=[];run_rows=[]
    for t,p in f.groupby('trial'):
        base=p.eligible&p.time_s.ge(-15)&p.time_s.lt(0)
        q10,q50,q90=np.quantile(p.loc[base,'bout_median_log'],[.1,.5,.9],method='linear')
        width=(q90-q10)/2;assert width>0
        expected=np.clip((p.bout_median_log-q50)/width,-1,1)
        np.testing.assert_allclose(expected,p.C_sample,atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(np.isfinite(p.raw_vigor),np.isfinite(p.C_sample))
        np.testing.assert_array_equal(np.isfinite(p.C_sample),p.eligible)
        st=stats[stats.trial.eq(t)].iloc[0]
        np.testing.assert_allclose([q10,q50,q90,width],[st.p10,st.p50,st.p90,st.C_scale],atol=1e-12,rtol=0)
        # Independently compare every frozen display interval and value to its source frames.
        covered=np.zeros(len(p),bool)
        for r in runs[runs.trial.eq(t)].itertuples(index=False):
            interval=p[p.FrameID.between(r.first_frame_id,r.last_frame_id)]
            assert len(interval)==r.sample_count
            np.testing.assert_allclose(interval.C_sample,r.C,atol=1e-12,rtol=0)
            mask=p.FrameID.between(r.first_frame_id,r.last_frame_id).to_numpy()
            assert not covered[mask].any();covered[mask]=True
            assert interval.eligible.all()
            assert ((interval.time_s>=r.start_s)&(interval.time_s<r.end_s)).all()
            run_rows.append({'trial':int(t),'first_frame_id':int(r.first_frame_id),'last_frame_id':int(r.last_frame_id),'start_s':float(r.start_s),'stop_s':float(r.end_s),'frozen_C':float(r.C),'sample_count':int(r.sample_count)})
        np.testing.assert_array_equal(covered,p.eligible)
        verification.append({'trial':int(t),'eligible_frames':int(p.eligible.sum()),'baseline_samples':int(base.sum()),'p10':q10,'p50':q50,'p90':q90,'scale':width,'baseline_scaled_median':float(p.loc[base,'C_sample'].median())})
    apply_theme();plt.rcParams.update({'svg.fonttype':'none','path.simplify':False,'figure.autolayout':False,'figure.constrained_layout.use':False})
    cmap=plt.get_cmap('managua_r');norm=Normalize(-1,1,clip=True)
    fig,axes=plt.subplots(5,1,figsize=(9,6),sharex=True,layout='none');fig.set_layout_engine(None)
    fig.subplots_adjust(left=.18,right=.79,top=.82,bottom=.16,hspace=.32)
    fig.text(.08,.965,'Raw vigor + frozen heatmap bout signal',fontsize=13,weight='bold')
    fig.text(.08,.915,'Delay 20221115_07 · baseline [−15,0) s · C_BoutSamples',fontsize=8)
    fig.text(.08,.87,'Exact frozen sample-run values and intervals · gaps preserved · no time bins',fontsize=8,color='#555555')
    events=ROOT/'cadence-review-v5-20261007/Fig1_PanelsD-E_events_v5.parquet'
    e=pd.read_parquet(events)
    raw_clipped=0
    for a,t,stage in zip(axes,TRIALS,STAGES):
        p=f[f.trial.eq(t)];b=a.twinx();b.set_zorder(1);a.set_zorder(2)
        a.patch.set_visible(False);b.patch.set_visible(False)
        for k,r in enumerate(run_rows):
            if r['trial']!=t:continue
            patches=b.bar(r['start_s'],r['frozen_C'],width=r['stop_s']-r['start_s'],align='edge',color=cmap(norm(r['frozen_C'])),edgecolor='none')
            patch=patches[0];patch.set_gid(f'frozen_scaled_run_trial_{t}_run_{k}')
            np.testing.assert_allclose([patch.get_x(),patch.get_width(),patch.get_height()],[r['start_s'],r['stop_s']-r['start_s'],r['frozen_C']],atol=1e-12,rtol=0)
            assert to_hex(patch.get_facecolor())==to_hex(cmap(norm(r['frozen_C'])))
        b.set_ylim(-1,1);b.set_yticks([]);b.set_yticks([],minor=True)
        b.spines[['top','bottom','left','right']].set_visible(False);b.axhline(0,color='#777777',lw=.45)
        a.plot(p.time_s,p.raw_vigor,color='black',lw=.65,alpha=.75)
        a.set_ylim(-1,1);a.set_yticks([0,.5,1],labels=['0','0.5','1']);a.set_xlim(-20,20)
        a.text(-.12,.5,f'{stage}\nTrial {t}',transform=a.transAxes,ha='right',va='center',fontsize=7,color='#333333')
        a.spines[['top','right','bottom']].set_visible(False);a.spines['left'].set_bounds(0,1)
        a.spines['left'].set_color('#999999');a.spines['left'].set_linewidth(.5)
        a.tick_params(axis='y',labelsize=7,length=3,width=.55,pad=4,colors='#555555')
        a.tick_params(axis='x',bottom=a is axes[-1],labelsize=7,length=3,width=.55,colors='#555555')
        for event in e[e['Trial number'].eq(t)].itertuples(index=False):
            a.axvline(float(event[2]),color='#702e78' if event.Event=='actual US onset' else '#0d8136',lw=.55,alpha=.7,ls='--' if event.Event=='CS offset' else '-')
        raw_clipped+=int(p.raw_vigor.gt(1).sum())
    fig.text(.006,.49,'Raw vigor (rad/ms)',rotation=90,fontsize=8,va='center')
    axes[-1].set_xlabel('Time from measured CS onset (s)',fontsize=8);axes[-1].set_xticks([-20,-10,0,10,20])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.85,.39,.011,.20]),orientation='vertical',ticks=[-1,0,1])
    cb.ax.set_yticklabels(['−1','0','+1']);cb.ax.tick_params(labelsize=7,length=3,width=.5)
    cb.set_label('Vigor relative to baseline',fontsize=8);cb.outline.set_edgecolor('#999999');cb.outline.set_linewidth(.5)
    fig.text(.18,.05,f'Raw display capped at 1 rad/ms: {raw_clipped} frame values exceed the range.',fontsize=7,color='#555555')
    fig.text(.18,.027,'Scaled signal uses the frozen −1…+1 transformation; both zero lines remain centred.',fontsize=7,color='#555555')
    svg_buffer=io.StringIO();fig.savefig(svg_buffer,format='svg');svg=svg_buffer.getvalue()
    tree=ET.fromstring(svg)
    exported={node.get('id'):node for node in tree.iter() if node.get('id','').startswith('frozen_scaled_run_')}
    assert len(exported)==len(run_rows)
    for k,r in enumerate(run_rows):
        node=exported[f"frozen_scaled_run_trial_{r['trial']}_run_{k}"]
        styles=' '.join(x.get('style','') for x in node.iter())
        assert 'fill: '+to_hex(cmap(norm(r['frozen_C']))) in styles
    if args.qa_preview:fig.savefig(args.qa_preview,dpi=160)
    plt.close(fig)
    provenance={'selected_frozen_record':{'path':str(record),'sha256':sha(record)},'freeze_manifest':{'path':str(freeze_path),'sha256':sha(freeze_path)},'source_sample_table':source,'frozen_display_runs':{'path':str(run_path),'sha256':sha(run_path)},'event_source':{'path':str(events),'sha256':sha(events)},'scientific_recipe':freeze['processing'],'formula':freeze['C_formula'],'baseline_interval':[-15,0],'baseline_voting_unit':'eligible displayed sample of repeated bout median log vigor','bar_aggregation':'none; exact frozen heatmap runs with unchanged C, start_s and end_s; no bridging missing frames','raw_scaled_support_identical':True,'frozen_run_support_identical':True,'svg_run_count_and_fill_verified':True,'no_time_binning':True,'scaled_limits':[-1,1],'palette_limits':[-1,1],'raw_display_cap':1,'raw_trace_alpha':.75,'raw_trace_width_pt':.65,'primary_ticks':[0,.5,1],'secondary_ticks':[],'verification':verification,'frozen_sample_runs':run_rows,'not_frozen':'Panel E review only; existing FGH freezes preserved','scope_resolution':'Uses authenticated frozen Version 1 values. Later complete-bout correction reviews are not substituted for the frozen snapshot.'}
    for p,digest in protected.items():assert sha(p)==digest,p
    assert sha(freeze_path)==sel['freeze_manifest_sha256'];assert sha(source['path'])==source['sha256']
    OUT.mkdir(exist_ok=True);output=OUT/'PanelE_exact-frozen-heatmap-runs-review.html'
    page='<!doctype html><html><meta charset="utf-8"><title>Panel E · frozen heatmap signal</title><style>body{font:15px system-ui;max-width:1200px;margin:24px auto;padding:0 20px;color:#222}svg{width:100%;height:auto}summary{cursor:pointer}pre{white-space:pre-wrap;font-size:12px}table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid #ddd}</style><body>'+svg
    page+='<p>The signed values and time intervals are taken directly from the frozen single-fish C_BoutSamples heatmap display runs. Bout log medians are centred on baseline P50 and scaled by half the P90–P10 range, then clipped to [−1,+1]. The baseline is [−15,0), with eligible samples voting. There is no extra averaging, no time binning and no filling across excluded frames. Every finite raw frame is covered by its exact frozen scaled run; missing-frame gaps remain gaps.</p>'
    page+='<p>Only the geometry differs from the heatmap: the frozen coloured intervals are drawn as signed vertical bars behind the raw trace. Values, interval edges, sample support, transformation and managua_r colours are unchanged. Existing raw styling and centred zero lines are retained. This review does not change or freeze any heatmap.</p>'
    page+='<details><summary>Embedded provenance, scientific definitions and verification</summary><pre>'+html.escape(json.dumps(provenance,indent=2))+'</pre></details></body></html>'
    output.write_text(page,encoding='utf-8')
    print(json.dumps({'output':str(output),'eligible_frames':int(f.eligible.sum()),'exact_frozen_runs':len(run_rows),'formula_verified':True,'frozen_values_intervals_support_and_colours_verified':True,'protected_hashes_unchanged':True}))
if __name__=='__main__':main()
