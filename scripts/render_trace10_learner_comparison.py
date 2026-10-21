"""Matched, editable shortlist comparison; does not replace Figure 1."""
from pathlib import Path
import base64,io,sys,xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont,FontProperties
import numpy as np
import pandas as pd
from classical_conditioning.figure_workspace import Snapshot,tag
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme,style_axes
from classical_conditioning.figures.export import FigureProvenance,FigureMode,export_matplotlib_figure,_slug
from analyze_trace10_learners import OUT,DATA,STATE,METHOD,CURRENT,load,ref,save,check

def main():
    calculation=load(OUT/'calculation.json');check(calculation['outputs'])
    shortlist=load(OUT/'shortlist.json')['fish'];scores=pd.read_csv(OUT/'full-score-coverage-exclusion-table.csv').set_index('fish_id')
    snapshot=Snapshot([Path(__file__),METHOD,OUT/'calculation.json',OUT/'full-score-coverage-exclusion-table.csv',REPO/'docs/paper/figures/figure-01/current-state.json',REPO/'docs/paper/figures/figure-02/current-state.json',STATE])
    save(OUT/'comparison-render-ticket-v3.json',snapshot.hashes)
    profile=load_presentation_profile('nature-main');width,height=183.,204.;cw=43.75;ch=99.;left=2.;top=2.;gap=1.
    registry={};allocations=[];clips=[];panel_ids=[]
    with mpl.rc_context():
        theme=apply_theme(profile=profile);fig=plt.figure(figsize=(width/25.4,height/25.4));fig.set_layout_engine('none');fig.patch.set_gid('comparison-canvas')
        def axes(x,y,w,h):return fig.add_axes([x/width,1-(y+h)/height,w/width,h/height])
        def text(x,y,value,key,**kwargs):
            artist=fig.text(x/width,1-y/height,value,fontsize=theme.font_size,va='top',color='black',**kwargs);artist.set_gid(key);return artist
        for k,rid in enumerate(shortlist):
            column,row=k%4,k//4;x=left+column*(cw+gap);y=top+row*(ch+gap)
            score=scores.loc[rid];bp=DATA/'fish'/(rid+'-bins.parquet');wp=DATA/'fish'/(rid+'-windows.parquet')
            bins=pd.read_parquet(bp);values=pd.read_parquet(wp)
            matrix=bins.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
            ax=axes(x+9,y+9,29,47);ax.set_gid(rid+'-heatmap-axes')
            cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black')
            image=ax.imshow(np.ma.masked_invalid(matrix),extent=[-20,20,90.5,.5],aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25);image.set_gid(rid+'-heatmap-values')
            ax.set(xlim=(-20,20),ylim=(90.5,.5),xticks=[-20,0,20],yticks=[1,30,60,90])
            style_axes(ax,theme=theme,xlabel='Time from CS (s)',ylabel='Trial' if column==0 else '',show_yticklabels=column==0)
            ax.spines[['top','right']].set_visible(True)
            for t,ls in [(0,'-'),(10,'--')]:ax.axvline(t,color=theme.cs_color,lw=theme.cs_linewidth,ls=ls)
            ax.plot([20,20],[10.5,60.5],color=theme.us_color,lw=theme.us_linewidth,ls=':',clip_on=False)
            for phase in [10.5,60.5]:ax.axhline(phase,color='white',lw=theme.axes_linewidth)
            ax2=axes(x+9,y+70,29,18);ax2.set_gid(rid+'-outcome-axes')
            line,=ax2.plot(values.trial_number,values.legacy_log_median_difference,color='#f15a29',lw=.5,alpha=.8);line.set_gid(rid+'-trial-outcomes')
            ax2.scatter(values.trial_number,values.legacy_log_median_difference,s=2,color='#f15a29',linewidths=0)
            ax2.axhline(0,color='black',lw=theme.axes_linewidth,ls=':')
            for phase in [14.5,64.5]:ax2.axvline(phase,color='black',lw=theme.axes_linewidth,ls=':')
            for a,b,value in [(5,14,score.median_5_14),(40,64,score.median_40_64)]:ax2.plot([a,b],[value,value],color='black',lw=1.4)
            ax2.set(xlim=(4,95),ylim=(-.7,.7),xticks=[5,30,60,94],yticks=[-.5,0,.5])
            style_axes(ax2,theme=theme,xlabel='Acquisition CS trial',ylabel='Δ ln vigor' if column==0 else '',show_yticklabels=column==0)
            rank=int(score.rank_min3) if np.isfinite(score.rank_min3) else '?';heading=rid+(' · comparator' if rid==CURRENT else ' · rank '+str(rank))
            heading_artist=text(x+cw/2,y+1.5,heading,rid+'-heading',ha='center')
            summary=text(x+cw/2,y+5.2,f'Pre − late Train: {score.primary_score:+.3f}',rid+'-score',ha='center')
            panel_ids.extend([rid+'-heatmap',rid+'-outcome'])
            bx=column*(cw+gap);by=row*100.;bw=(cw+gap) if column<3 else width-bx;bh=100. if row==0 else height-by
            allocations.append({'id':rid,'kind':'panel','box_mm':[bx,by,bw,bh],'group_ids':['axes__'+_slug(rid+'-heatmap')+'__main','axes__'+_slug(rid+'-outcome')+'__main',heading_artist.get_gid(),summary.get_gid()]})
            registry[rid+'-heatmap-values']={'scientific_role':'heatmap.vigor','fish':rid,'data':ref(bp),'shape':[90,40],'colormap':'managua_r','limits':[-.25,.25],'baseline_bin_median_centering_s':[-15,0],'acquisition_trials':[5,94],'display_trials':[1,90]}
            registry[rid+'-trial-outcomes']={'scientific_role':'trajectory.individual','fish':rid,'data':ref(wp),'response_s':[0,20],'baseline_s':[-15,0],'limits':[-.7,.7],'score':ref(OUT/'full-score-coverage-exclusion-table.csv')}
        # A separate owned legend card uses the otherwise empty eighth position.
        x=left+3*(cw+gap);y=top+ch+gap
        ca=axes(x+8,y+13,2,31);ca.set_gid('comparison-colourbar');bar=fig.colorbar(image,cax=ca,ticks=[-.25,0,.25]);bar.set_label('Δ ln vigor',fontsize=theme.font_size);bar.ax.tick_params(labelsize=theme.tick_labelsize)
        note=text(x+4,y+52,'Shared heatmap scale\nBlack: missing\nGreen: CS onset/offset\nViolet: expected US, 20 s\n\nBlack bars: scored epochs\nOrange: observed outcomes\nShared outcome view: ±0.7\n\nProvisional descriptive\ncomparison; no freeze.','comparison-note')
        panel_ids.append('comparison-colourbar')
        allocations.append({'id':'comparison-legend','kind':'shared','box_mm':[3*(cw+gap),100.,width-3*(cw+gap),height-100.],'group_ids':['axes__comparison-colourbar__main','comparison-note']})
        spec={'scope':'figure','final_size_mm':[width,height],'inset_mm':1.,'allocations':allocations,'background_ids':['comparison-canvas'],'alignments':[],'data_axes':[]}
        provenance=FigureProvenance(figure_id='Fixed10sTrace illustrative candidate comparison',analysis_recipe='prespecified-acquisition-ranking-and-matched-caches',source_file=str(Path(__file__)),source_symbol='main',source_hash=ref(Path(__file__))['sha256'],reproduction_snippet='.venv-trace/Scripts/python.exe scripts/render_trace10_learner_comparison.py',input_artifacts=(ref(OUT/'plan.json'),ref(OUT/'calculation.json'),ref(OUT/'shortlist.json')),artist_mappings=registry,analysis_identity={'cohort':{'control':32,'trace':29},'ranking_eligible':{'control':26,'trace':24},'shared_comparator':CURRENT,'profile_inputs':profile.inputs,'canvas_status':'review comparison, not a main-paper submission layout','status':'standalone scientific draft; no Figure1 replacement'})
        export_matplotlib_figure(fig,OUT/'candidate-comparison',provenance,mode=FigureMode.PUBLICATION,panel_ids=panel_ids,allow_dirty_publication=True,formats=('svg',),layout_frames=spec);plt.close(fig)
    # Native text stays editable; embed authentic Arial resources for stable review.
    svg=(OUT/'candidate-comparison.svg').read_bytes();root=ET.fromstring(svg)
    # Exporter binds the SVG, so do not mutate it here; its embedded resources are verified separately.
    legend=('Fixed10sTrace candidate comparison. Each fish uses the same authenticated direct 1-s original-frame positive detected-bout log-median heatmap, centred within trial by finite baseline-bin medians [−15,0) s; managua_r ±0.25, missing black. Acquisition5–94 are displayed1–90. Solid/dashed green: CS onset/10-s offset; violet dotted guide: expected training US20 s (actual timing audited separately). Below: measured-time median log positive valid moving-frame response [0,20) minus baseline [−15,0), with shared ±0.7 display limits; black segments are Pre5–14 and lateTrain40–64 medians used in the score. Rank is among24 feature-eligible conditioned fish; all29 conditioned fish and32 controls remain in the common roster, with full63-inventory coverage/exclusion records. Comparison is descriptive; no validated learner labels, replacement or freeze is implied.')
    with (OUT/'comparison-legend.md').open('x',encoding='utf-8') as f:f.write(legend+'\n')
    page=('<!doctype html><html><meta charset="utf-8"><style>body{margin:20px;font:14px Arial;color:black;background:white}main{max-width:1200px;margin:auto}svg{width:100%;height:auto;display:block}p{line-height:1.45}</style><main>'+svg.decode()+'<p>'+legend+'</p></main></html>').encode()
    with (OUT/'candidate-comparison.html').open('xb') as f:f.write(page)
    save(OUT/'comparison.json',{'status':'rendered; exact saved-render review and preferred-candidate recommendation pending','svg':ref(OUT/'candidate-comparison.svg'),'html':ref(OUT/'candidate-comparison.html'),'legend':ref(OUT/'comparison-legend.md'),'registry':ref(OUT/'candidate-comparison.figure.json'),'generator':ref(Path(__file__)),'source_checks':ref(OUT/'calculation.json'),'shortlist':shortlist,'parent_figures_changed':False})
    snapshot.check();print(OUT/'candidate-comparison.html')

if __name__=='__main__':main()
