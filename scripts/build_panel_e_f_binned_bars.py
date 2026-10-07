"""F layout with full signed bin bars in the heatmap palette; preserve originals."""
import json, sys, hashlib
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
OUT=ROOT/'panel-e-f-binned-bars-rawzoom-20261007'
TIME='Time relative to CS onset (s)'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
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
    fig.text(.08,.965,'Raw vigor + binned signed log vigor',fontsize=13,weight='bold')
    fig.text(.08,.915,'Delay fish 20221115_07 · reconstructed clock · baseline [−15,0) s',fontsize=8)
    fig.text(.08,.87,'Raw vigor · 75% opacity     |     Coloured bars · 0.5 s mean of bout medians',fontsize=8,color='#555555')
    rawmax=1.
    clipped={};raw_clipped={}
    for a,t,stage in zip(axes,trials,stages):
        p=f[f['Trial number']==t];q=selected[selected['Trial number']==t].sort_values('Time bin center (s)')
        b=a.twinx();b.set_zorder(1);a.set_zorder(2);a.patch.set_visible(False);b.patch.set_visible(False)
        good=q['Signed log vigor'].notna();values=q.loc[good,'Signed log vigor'].to_numpy()
        b.bar(q.loc[good,'Time bin center (s)'],values,width=.5,bottom=0,color=cmap(norm(values)),edgecolor='none',linewidth=0)
        clipped[str(t)]=int((np.abs(values)>.5).sum())
        b.set_ylim(-.5,.5);b.set_yticks([-.5,0,.5],labels=['−0.5','0','+0.5'])
        b.yaxis.set_minor_locator(FixedLocator([-.25,.25]))
        b.tick_params(axis='y',labelsize=7,colors='#666666',length=3,width=.55,pad=4)
        b.tick_params(axis='y',which='minor',length=1.5,width=.45,color='#888888')
        b.spines[['top','bottom','left']].set_visible(False)
        b.spines['right'].set_color('#bbbbbb');b.spines['right'].set_linewidth(.5)
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
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.875,.16,.014,.66]),orientation='vertical',ticks=[-.25,0,.25])
    cb.ax.set_yticklabels(['−0.25','0','+0.25'])
    cb.ax.tick_params(labelsize=7,length=3,width=.5);cb.set_label('Binned signed log vigor',fontsize=8)
    cb.outline.set_edgecolor('#999999');cb.outline.set_linewidth(.5)
    fig.text(.18,.05,f'Raw display capped at 1 rad/ms: {sum(raw_clipped.values())} frame values exceed the range.',fontsize=7,color='#555555')
    fig.text(.18,.027,f'Secondary axis ±0.5: {sum(clipped.values())} bin bars exceed the range. Colours saturate at ±0.25.',fontsize=7,color='#555555')
    svg=OUT/'Fig1_PanelE_F_binned-managua-bars.svg'
    for ext in ['svg','png','pdf']:fig.savefig(svg.with_suffix('.'+ext),dpi=180)
    plt.close(fig)
    selected.to_parquet(OUT/'PanelE_stored_bins.parquet',index=False)
    svg.with_suffix('.svg.json').write_text(json.dumps({'selection_status':'user-directed F zero-aligned trial; review','sources':[{'path':str(p),'sha256':sha(p)} for p in [fp,hp,ep]],'svg_sha256':sha(svg),'bar_semantics':'NaN-ignoring 0.5 s means of repeated eligible-frame bout medians; empty bins omitted','baseline_s':[-15,0],'palette':'managua_r','color_limits':[-.25,.25],'secondary_y_limits':[-.5,.5],'raw_display_limits':[-rawmax,rawmax],'zero_alignment':'both zero lines at row centre; raw values unchanged and nonnegative','render_order':'bars behind black raw trace; transparent raw axis patch','raw_amplitude_clipping':False,'bars_outside_axis_per_trial':clipped,'axis_semantics':'independent raw and signed-log axes; heights not commensurate'},indent=2))
    side=svg.with_suffix('.svg.json');meta=json.loads(side.read_text())
    meta.update({'raw_trace_alpha':.75,'raw_trace_linewidth_pt':.65,'raw_amplitude_clipping':True,'raw_values_outside_axis_per_trial':raw_clipped,'primary_y_ticks':[0,rawmax/2,rawmax],'secondary_y_ticks':[-.5,0,.5],'secondary_y_minor_ticks':[-.25,.25],'bar_outlines':False})
    side.write_text(json.dumps(meta,indent=2))
    repo=Path(__file__).resolve().parents[1]
    layout=json.loads((repo/'configs/paper-figures/figure1-cadence-review-v5.json').read_text())
    layout['output']=str(OUT/'figure1-F-binned-bars-review.svg')
    for panel in layout['panels']:
        if panel['id']=='E':panel['source']=str(svg);panel.pop('content_box',None);panel['selection_status']='user-directed F revision; review'
    config=repo/'configs/paper-figures/figure1-panel-e-f-binned-bars-rawzoom-review.json'
    config.write_text(json.dumps(layout,indent=2))
    render(config,Path(layout['output']),strict=True)
    export_with_inkscape(Path(layout['output']),['png','pdf'],font_directory=ROOT/'fonts')
    print(svg)
if __name__=='__main__':main()
