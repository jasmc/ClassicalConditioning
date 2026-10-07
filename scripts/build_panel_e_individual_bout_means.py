"""F layout with full signed bin bars in the heatmap palette; preserve originals."""
import json, sys, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from classical_conditioning.figures.theme import apply_theme
from assemble_svg_figure import render, export_with_inkscape

ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
SRC=ROOT/'cadence-review-v5-20261007'
OUT=ROOT/'panel-e-scaled-bout-means-full-duration-20261007'
TIME='Time relative to CS onset (s)'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mean-mode',choices=['raw-first','log-first'],default='log-first')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    fp=SRC/'Fig1_PanelsD-E_presumed-cadence_frames_v5.parquet'
    hp=SRC/'Fig1_PanelF_Delay_presumed-cadence_v5.parquet'
    ep=SRC/'Fig1_PanelsD-E_events_v5.parquet'
    for path,side in [(fp,SRC/'Fig1_PanelD_TailAngle_presumed-cadence_v5.svg.json'),(hp,hp.with_suffix('.svg.json'))]:
        meta=json.loads(side.read_text());assert sha(path)==meta['panel_data_sha256']
    f,h,e=[pd.read_parquet(p) for p in [fp,hp,ep]]
    trials=[9,17,63,66,93]; stages=['Habituation','Early Train','Late Train','Early Test','Late Test']
    selected=h[h['Trial number'].isin(trials)]
    for t in trials:
        p=f[f['Trial number']==t];q=selected[selected['Trial number']==t].sort_values('Time bin center (s)')
        for c in ['Vigor','centred_log','bout_median']:np.testing.assert_array_equal(np.isfinite(p[c]),p.eligible)
        np.testing.assert_allclose(p.groupby('bin_index').bout_median.mean().reindex(range(80)),q['Signed log vigor'],atol=1e-12,rtol=0,equal_nan=True)
    theme=apply_theme(); cmap=plt.get_cmap('managua_r');norm=Normalize(-.25,.25,clip=True)
    plt.rcParams.update({'svg.fonttype':'none','path.simplify':False,'figure.autolayout':False,'figure.constrained_layout.use':False})
    fig,axes=plt.subplots(5,1,figsize=(9,6),sharex=True,layout='none')
    fig.set_layout_engine(None)
    fig.subplots_adjust(left=.18,right=.79,top=.82,bottom=.16,hspace=.32)
    fig.text(.08,.965,'Raw vigor + individual bout means',fontsize=13,weight='bold')
    fig.text(.08,.915,'Delay fish 20221115_07 · reconstructed clock · baseline [−15,0) s',fontsize=8)
    semantics='ln(mean raw vigor per bout) − baseline median ln(raw)' if args.mean_mode=='raw-first' else 'Mean centred log vigor per bout'
    fig.text(.08,.87,semantics+' · bars span full detected bout durations',fontsize=8,color='#555555')
    rawmax=1.
    clipped={};raw_clipped={};bout_rows=[];segment_rows=[];frame_rows=[]
    for a,t,stage in zip(axes,trials,stages):
        p=f[f['Trial number']==t];q=selected[selected['Trial number']==t].sort_values('Time bin center (s)')
        b=a.twinx();b.set_zorder(1);a.set_zorder(2);a.patch.set_visible(False);b.patch.set_visible(False)
        eligible=p.eligible.to_numpy(bool);time=p[TIME].to_numpy();ids=p.bout_id.to_numpy()
        baseline=float(np.median(np.log(p.loc[eligible & p[TIME].ge(-15) & p[TIME].lt(0),'Vigor'])))
        frame_scaled=np.full(len(p),np.nan)
        frame_scaled[eligible]=np.log(p.Vigor.to_numpy()[eligible])-baseline
        np.testing.assert_array_equal(np.isfinite(frame_scaled),np.isfinite(p.Vigor))
        np.testing.assert_allclose(frame_scaled,p.centred_log,atol=1e-12,rtol=0,equal_nan=True)
        p=p.assign(framewise_scaled_vigor=frame_scaled)
        values={}
        for bid,g in p.loc[eligible].groupby('bout_id'):
            raw_mean=float(g.Vigor.mean())
            value=float(np.log(raw_mean)-baseline) if args.mean_mode=='raw-first' else float(g.framewise_scaled_vigor.mean())
            values[bid]=value
            bout_rows.append({'Trial number':t,'bout_id':int(bid),'eligible_frame_count':len(g),'mean_raw_vigor_rad_per_ms':raw_mean,'baseline_median_ln_raw':baseline,'scaled_bout_mean':value,'mean_mode':args.mean_mode})
        scaled=p.bout_id.map(values).where(p.eligible).to_numpy()
        np.testing.assert_array_equal(np.isfinite(scaled),np.isfinite(p.Vigor))
        frame_rows.append(p.assign(scaled_bout_mean=scaled))
        # A bar is a bout summary spanning its full detected duration, not a frame signal.
        # Framewise scaled and repeated means still retain NaNs at every excluded frame.
        dt=float(np.median(np.diff(time)))
        for bid,g in p.loc[p.bout_id.isin(values)].groupby('bout_id'):
            left=float(g[TIME].iloc[0]);right=min(20.,float(g[TIME].iloc[-1]+dt));v=values[bid]
            b.bar(left,v,width=right-left,align='edge',bottom=0,color=cmap(norm(v)),edgecolor='none',linewidth=0)
            segment_rows.append({'Trial number':t,'bout_id':int(bid),'start_s':left,'stop_s':right,'scaled_bout_mean':v,'detected_frame_count':len(g),'eligible_frame_count':int(g.eligible.sum())})
        clipped[str(t)]=sum(abs(v)>.5 for v in values.values())
        b.set_ylim(-.5,.5);b.set_yticks([]);b.set_yticks([],minor=True)
        b.spines[['top','bottom','left','right']].set_visible(False)
        b.axhline(0,color='#777777',lw=.45)
        a.plot(p[TIME],p.Vigor,color='black',lw=.65,alpha=.75)
        raw_clipped[str(t)]=int(p.Vigor.gt(rawmax).sum())
        a.set_ylim(-rawmax,rawmax);a.set_yticks([0,rawmax/2,rawmax],labels=[f'{x:g}' for x in [0,rawmax/2,rawmax]]);a.set_xlim(-20,20)
        # Symmetric display limits align raw zero with signed zero without transforming data.
        assert np.isclose((0-a.get_ylim()[0])/(a.get_ylim()[1]-a.get_ylim()[0]),.5)
        assert np.isclose((0-b.get_ylim()[0])/(b.get_ylim()[1]-b.get_ylim()[0]),.5)
        a.text(-.12,.5,f'{stage}\nTrial {t}',transform=a.transAxes,ha='right',va='center',fontsize=7,color='#333333')
        a.spines[['top','right','bottom']].set_visible(False)
        a.spines['left'].set_bounds(0,rawmax);a.spines['left'].set_color('#999999');a.spines['left'].set_linewidth(.5)
        a.tick_params(axis='y',labelsize=7,length=3,width=.55,pad=4,colors='#555555')
        a.tick_params(axis='x',bottom=a is axes[-1],labelsize=7,length=3,width=.55,colors='#555555')
        for ev in e[e['Trial number']==t].itertuples(index=False):
            a.axvline(float(ev[2]),color=theme.us_color if ev.Event=='actual US onset' else theme.cs_color,lw=.55,alpha=.7,ls='--' if ev.Event=='CS offset' else '-')
    fig.text(.006,.49,'Raw vigor (rad/ms)',rotation=90,fontsize=8,va='center')
    axes[-1].set_xlabel('Time from measured CS onset (s)',fontsize=8)
    axes[-1].set_xticks([-20,-10,0,10,20])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.85,.39,.011,.20]),orientation='vertical',ticks=[-.25,0,.25])
    cb.ax.set_yticklabels(['−0.25','0','+0.25'])
    cb.ax.tick_params(labelsize=7,length=3,width=.5);cb.set_label('Signed bout log vigor',fontsize=8)
    cb.outline.set_edgecolor('#999999');cb.outline.set_linewidth(.5)
    fig.text(.18,.05,f'Raw display capped at 1 rad/ms: {sum(raw_clipped.values())} frame values exceed the range.',fontsize=7,color='#555555')
    fig.text(.18,.027,f'Scaled display ±0.5: {sum(clipped.values())} bout means exceed the range. Colours saturate at ±0.25.',fontsize=7,color='#555555')
    svg=OUT/'Fig1_PanelE_individual-bout-means.svg'
    for ext in ['svg','png','pdf']:fig.savefig(svg.with_suffix('.'+ext),dpi=180)
    plt.close(fig)
    pd.DataFrame(bout_rows).to_parquet(OUT/'PanelE_bout_means.parquet',index=False)
    pd.DataFrame(segment_rows).to_parquet(OUT/'PanelE_bout_bar_segments.parquet',index=False)
    pd.concat(frame_rows).to_parquet(OUT/'PanelE_same-support_frames.parquet',index=False)
    svg.with_suffix('.svg.json').write_text(json.dumps({'selection_status':'user-directed F zero-aligned trial; review','sources':[{'path':str(p),'sha256':sha(p)} for p in [fp,hp,ep]],'svg_sha256':sha(svg),'bar_semantics':'NaN-ignoring 0.5 s means of repeated eligible-frame bout medians; empty bins omitted','baseline_s':[-15,0],'palette':'managua_r','color_limits':[-.25,.25],'secondary_y_limits':[-.5,.5],'raw_display_limits':[-rawmax,rawmax],'zero_alignment':'both zero lines at row centre; raw values unchanged and nonnegative','render_order':'bars behind black raw trace; transparent raw axis patch','raw_amplitude_clipping':False,'bars_outside_axis_per_trial':clipped,'axis_semantics':'independent raw and signed-log axes; heights not commensurate'},indent=2))
    side=svg.with_suffix('.svg.json');meta=json.loads(side.read_text())
    meta.update({'selection_status':'user-directed mean-scaled full-bout-duration review','bar_semantics':semantics+'; one rectangle across full detected bout duration in displayed trial window','framewise_scaled_semantics':'ln(raw) minus trial baseline median ln(raw); identical finite mask to raw; invalid/excluded frames NaN','bar_vs_frame_support':'bars denote full detected bout intervals, including internal excluded frames; these frames remain NaN in raw/scaled frame signals and are omitted from means','mean_mode':args.mean_mode,'aggregation':'individual bouts; no time bins','scaled_mean_source':'arithmetic mean of eligible framewise scaled values in each displayed trial window','heatmap_comparability':'same palette and baseline; mean-based bout signal differs from stored median-based heatmap bins','raw_trace_alpha':.75,'raw_trace_linewidth_pt':.65,'raw_amplitude_clipping':True,'raw_values_outside_axis_per_trial':raw_clipped,'primary_y_ticks':[0,rawmax/2,rawmax],'secondary_y_ticks':[],'secondary_y_minor_ticks':[],'bar_outlines':False,'colorbar_height_fraction':.20})
    meta['bouts_outside_axis_per_trial']=meta.pop('bars_outside_axis_per_trial')
    side.write_text(json.dumps(meta,indent=2))
    repo=Path(__file__).resolve().parents[1]
    layout=json.loads((repo/'configs/paper-figures/figure1-cadence-review-v5.json').read_text())
    layout['output']=str(OUT/'figure1-individual-bout-means-review.svg')
    for panel in layout['panels']:
        if panel['id']=='E':panel['source']=str(svg);panel.pop('content_box',None);panel['selection_status']='user-directed F revision; review'
    config=repo/'configs/paper-figures/figure1-panel-e-scaled-bout-means-full-duration-review.json'
    config.write_text(json.dumps(layout,indent=2))
    render(config,Path(layout['output']),strict=True)
    export_with_inkscape(Path(layout['output']),['png','pdf'],font_directory=ROOT/'fonts')
    print(svg)
if __name__=='__main__':main()
