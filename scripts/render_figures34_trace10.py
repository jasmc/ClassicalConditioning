"""Nature-profile review drafts for the isolated 10sTrace analysis."""
from pathlib import Path
import io, sys, html, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_figures34_trace10 import OUT,DATA,load,ref,save
from classical_conditioning.figure_workspace import Snapshot
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme,style_axes,condition_color
from classical_conditioning.config import get_experiment_spec
from classical_conditioning.figures.export import FigureProvenance,FigureMode,export_matplotlib_figure
from classical_conditioning.figures.layout_frames import load_policy

COLORS={s.condition_id:condition_color(s) for s in get_experiment_spec('all10sTrace').conditions}
PROFILE=load_presentation_profile('nature-main')
INSET=load_policy()[0]['selected_inset_mm']
def page(svg,legend):
    return '<!doctype html><html><meta charset="utf-8"><style>body{margin:20px;font:14px Arial;background:white;color:black}main{max-width:1100px;margin:auto}svg{display:block;width:100%;height:auto}p{line-height:1.5}</style><main>'+svg+'<p>'+html.escape(legend)+'</p></main></html>'
def write_text(p,s):
    with p.open('x',encoding='utf-8') as f:f.write(s)
def export(fig,name,w,h,ids,allocations,mapping,legend,inputs):
    fig.patch.set_gid(name+'-canvas')
    spec={'scope':'figure','final_size_mm':[w,h],'inset_mm':INSET,'allocations':allocations,'background_ids':[name+'-canvas'],'data_axes':[],'alignments':[]}
    provenance=FigureProvenance(figure_id=name,analysis_recipe='trace10-condition-only-descriptive-v1',source_file=str(Path(__file__)),source_symbol=name,source_hash=ref(Path(__file__))['sha256'],reproduction_snippet=f'.venv-trace/Scripts/python.exe scripts/render_figures34_trace10.py {"fig3" if name=="figure3" else "fig4"}',input_artifacts=tuple(ref(p) for p in inputs),artist_mappings=mapping,analysis_identity={'status':'unfrozen scientific draft; Gate L unresolved','profile_inputs':PROFILE.inputs,'metric':'legacy_distal_angular_speed','roster':{'control':32,'trace':29}})
    export_matplotlib_figure(fig,OUT/name,provenance,mode=FigureMode.PUBLICATION,allow_dirty_publication=True,formats=('svg',),panel_ids=ids,layout_frames=spec)
    plt.close(fig)
    write_text(OUT/(name+'-legend.md'),legend+'\n')
    write_text(OUT/(name+'.html'),page((OUT/(name+'.svg')).read_text(encoding='utf-8'),legend))

def fig3():
    scores=pd.read_csv(OUT/'acquisition-scores-all63.csv'); trials=pd.read_parquet(OUT/'observed-trial-outcomes-all63.parquet')
    active=scores.loc[scores.selected]; trace=active.loc[active.condition_id.eq('trace')&active.estimable].sort_values('acquisition_score')
    # Deterministic illustrations of observed magnitudes, not learner classes.
    examples=[trace.iloc[-1],trace.iloc[(len(trace)-1)//2],trace.iloc[np.argmin(np.abs(trace.acquisition_score.to_numpy()))]]
    if len({x.fish_id for x in examples})<3: raise ValueError('Illustrations need distinct fish')
    inputs=[OUT/'acquisition-scores-all63.csv',OUT/'observed-trial-outcomes-all63.parquet',OUT/'analysis-spec.json']
    save(OUT/('figure3-render-ticket-'+ref(Path(__file__))['sha256'][:10]+'.json'),Snapshot([Path(__file__),*inputs]).hashes)
    w,h=183.,132.; ids=['scores']; allocations=[];mapping={}
    with mpl.rc_context():
        theme=apply_theme(profile=PROFILE); fig=plt.figure(figsize=(w/25.4,h/25.4));fig.set_layout_engine('none')
        def axes(x,y,ww,hh):return fig.add_axes([x/w,1-(y+hh)/h,ww/w,hh/h])
        ax=axes(22,9,153,27)
        for i,c in enumerate(['control','trace']):
            s=active.loc[active.condition_id.eq(c)].sort_values('fish_id')
            ax.scatter(s.acquisition_score,np.full(len(s),i)+np.linspace(-.17,.17,len(s)),s=9,color=COLORS[c],linewidths=0)
        ax.axvline(0,color='black',lw=theme.axes_linewidth,ls=':'); ax.set(yticks=[0,1],yticklabels=['Control','10sTrace'],ylim=(-.5,1.5))
        style_axes(ax,theme=theme,xlabel='Acquisition contrast (Pre − late Train)',ylabel='')
        allocations.append({'id':'score-panel','kind':'panel','box_mm':[0,0,w,45],'group_ids':['axes__scores__main']})
        mapping['axes__scores__main']={'scientific_role':'trajectory.individual','data':ref(inputs[0]),'quantity':'Pre 5–14 minus late Train 40–64 medians of observed trial log-median response-minus-baseline','categories':'none','reference':'zero is a direction guide only'}
        ymin=min(-.25,float(trials.loc[trials.fish_id.isin([x.fish_id for x in examples]),'legacy_log_median_difference'].min())-.03)
        ymax=max(.25,float(trials.loc[trials.fish_id.isin([x.fish_id for x in examples]),'legacy_log_median_difference'].max())+.03)
        for i,row in enumerate(examples):
            x=i*61.; rid=row.fish_id; vals=trials.loc[trials.fish_id.eq(rid)]
            hid=rid+'-heatmap';tid=rid+'-trial';ids.extend([hid,tid])
            bp=DATA/'fish'/f'{rid}-bins.parquet'; bins=pd.read_parquet(bp)
            matrix=bins.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
            a=axes(x+13,54,43,36);cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black')
            a.imshow(np.ma.masked_invalid(matrix),extent=[-20,20,94.5,4.5],aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25)
            a.set(xticks=[-20,0,20],yticks=[5,14,64,94]);style_axes(a,theme=theme,xlabel='Time from CS (s)',ylabel='CS trial')
            a.axvline(0,color=theme.cs_color,lw=theme.cs_linewidth);a.axvline(10,color=theme.cs_color,lw=theme.cs_linewidth,ls='--')
            a.set_title(rid+' · '+format(row.acquisition_score,'+.3f'),fontsize=theme.font_size,pad=3)
            a=axes(x+13,106,43,16)
            a.scatter(vals.trial_number,vals.legacy_log_median_difference,s=3,color=COLORS['trace'],linewidths=0)
            for lo,hi,value in [(5,14,row.pre_median),(40,64,row.late_median)]:a.plot([lo,hi],[value,value],color='black',lw=1.)
            a.set(xlim=(4,95),ylim=(ymin,ymax),xticks=[5,40,64,94]);style_axes(a,theme=theme,xlabel='CS trial',ylabel='Δ ln vigor')
            allocations.append({'id':rid,'kind':'panel','box_mm':[x,45,61,h-45],'group_ids':['axes__'+hid.lower()+'__main','axes__'+tid.lower()+'__main']})
            mapping['axes__'+tid.lower()+'__main']={'scientific_role':'trajectory.individual','data':ref(inputs[1]),'heatmap':ref(bp),'score':float(row.acquisition_score),'example_selection':'largest, middle rank, closest to zero among 28 score-estimable selected conditioned fish','trial_outcome_windows_s':{'baseline':[-15,0],'response':[0,20]},'heatmap':'1 s eligible moving-frame log medians centred by median finite baseline bins; not trial outcome estimator','shared_trial_y_limits':[ymin,ymax]}
        legend=('10sTrace Figure 3 scientific draft. Top: continuous acquisition contrasts are finite in 31/32 Control and 28/29 10sTrace fish. Control 20250715_10 has no finite late-training outcome and 10sTrace 20250715_06 has no finite Pre outcome; both remain in the cohort and atlas. The contrast is: median observed trial outcome in Pre CS 5–14 minus late Train 40–64. Trial outcome is median ln positive valid moving-frame vigor [0,20) s minus [−15,0) s; positive contrast means increased suppression. Zero is a direction guide, not a cutoff. Below: deterministically selected largest, middle-rank and closest-to-zero conditioned contrasts; deliberate descriptive examples, not validation. Heatmaps show direct 1 s moving-frame log medians centred by median finite baseline-bin medians, native managua_r −0.25 to +0.25 (black missing; endpoint colors saturated). Green: CS onset and dashed 10 s offset. Dots show observed trial outcomes, black bars the two score-epoch medians; shared outcome limits retain all plotted values. All 63 inventory identities, both exclusions, epoch counts and missing outcomes remain in the numerical audit and all-fish atlas. Mathematical score estimability does not approve coverage or individual precision. Gate L and learner categories remain unresolved. The 32 controls share all3sTrace acquisitions and must not be counted independently across assays. No Figure 3 assembly selection or freeze is changed.')
        export(fig,'figure3',w,h,ids,allocations,mapping,legend,inputs)
    save(OUT/'figure3-examples.json',{'fish':[x.fish_id for x in examples],'rule':'largest, middle rank, nearest zero observed conditioned score','not_validated_classes':True})
    atlas(scores,trials,theme)

def atlas(scores,trials,theme):
    cards=[]
    with mpl.rc_context():
        apply_theme(theme=theme)
        for row in scores.sort_values(['selected','condition_id','fish_id'],ascending=[False,True,True]).itertuples():
            fig,ax=plt.subplots(figsize=(3.5,1.8));fig.set_layout_engine('none');fig.subplots_adjust(left=.19,bottom=.27,right=.97,top=.88)
            g=trials.loc[trials.fish_id.eq(row.fish_id)]
            ax.scatter(g.trial_number,g.legacy_log_median_difference,s=5,color=COLORS[row.condition_id],linewidths=0)
            for a,b,v in [(5,14,row.pre_median),(40,64,row.late_median)]:ax.plot([a,b],[v,v],color='black',lw=1)
            ax.set(xlim=(4,95),xticks=[5,14,40,64,94]);style_axes(ax,theme=theme,xlabel='CS trial',ylabel='Δ ln vigor')
            buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
            cards.append('<article><h3>'+row.fish_id+' · '+row.condition_id+'</h3>'+buf.getvalue()+'<p>'+html.escape(f'Score {row.acquisition_score:+.4f}; Pre {row.pre_n}/10; late Train {row.late_n}/25; finite outcomes {row.finite_outcomes}/{row.observed_trials}; '+('selected roster' if row.selected else 'EXCLUDED: '+row.exclusion_reason))+'</p></article>')
    write_text(OUT/'all-fish-atlas.html','<!doctype html><meta charset="utf-8"><style>body{font:14px Arial}main{display:grid;grid-template-columns:repeat(3,1fr)}article{border:1px solid #ddd;padding:10px}svg{width:100%;height:auto}</style><p>10sTrace descriptive continuous-score atlas: all 63 identities; no learner labels or independent validation. Each card uses its own y scale to retain all observed outcomes. Black bars: score epochs; missing dots are undefined outcomes. Excluded identities remain audit only.</p><main>'+''.join(cards)+'</main>')

def fig4():
    table=pd.read_parquet(OUT/'profile-condition-bins.parquet')
    groups=table.loc[~table.group_type.eq('individual_catch'),['group_order','group_name']].drop_duplicates().sort_values('group_order')
    inputs=[OUT/'profile-condition-bins.parquet',OUT/'profile-fish-bins.parquet',OUT/'profile-source-manifest.json']
    save(OUT/('figure4-render-ticket-'+ref(Path(__file__))['sha256'][:10]+'.json'),Snapshot([Path(__file__),*inputs]).hashes)
    # Three parallel outcome columns, with an owned row per configured group.
    w,h=183.,170.;ids=[];allocations=[];mapping={}
    signals=[('signed_log_vigor','Signed Δ ln vigor'),('movement_probability','Movement probability'),('scaled_activity','Scaled total activity')]
    with mpl.rc_context():
        theme=apply_theme(profile=PROFILE);fig=plt.figure(figsize=(w/25.4,h/25.4));fig.set_layout_engine('none')
        for row,g in enumerate(groups.itertuples(index=False)):
            for col,(signal,label) in enumerate(signals):
                ident=f'row-{row}-signal-{col}';ids.append(ident);x=61.*col;y=16.5*row
                ax=fig.add_axes([(x+15)/w,1-(y+13)/h,43/w,8/h])
                for condition in ['control','trace']:
                    p=table.loc[table.condition_id.eq(condition)&table.group_order.eq(g.group_order)].sort_values('time_s')
                    t=p.time_s.to_numpy();ax.plot(t,p[signal+'_median'],color=COLORS[condition],lw=theme.axes_linewidth)
                    mask=p[signal+'_fish'].ge(2).to_numpy();ax.fill_between(t,p[signal+'_q25'].to_numpy(),p[signal+'_q75'].to_numpy(),where=mask,color=COLORS[condition],alpha=.2,linewidth=0)
                ax.axvline(0,color=theme.cs_color,lw=theme.cs_linewidth);ax.axvline(10,color=theme.cs_color,lw=theme.cs_linewidth,ls='--')
                ax.axvline(20,color=theme.us_color,lw=theme.us_linewidth,ls=':',clip_on=False)
                ax.set(xlim=(-20,20),xticks=[-20,0,20])
                if signal!='signed_log_vigor':ax.set(ylim=(0,1),yticks=[0,.5,1])
                else:
                    bound=max(.4,float(np.nanmax(np.abs(table[[signal+'_q25',signal+'_q75']].to_numpy())))*1.05)
                    ax.set(ylim=(-bound,bound),yticks=[-round(bound,2),0,round(bound,2)])
                style_axes(ax,theme=theme,xlabel='Time from CS (s)' if row==9 else '',ylabel=g.group_name if col==0 else '',show_xticklabels=row==9)
                if row==0:ax.set_title(label,fontsize=theme.font_size,pad=3)
                allocations.append({'id':ident,'kind':'panel','box_mm':[x,y,61,16.5 if row<9 else h-y],'group_ids':['axes__'+ident+'__main']})
                mapping['axes__'+ident+'__main']={'scientific_role':'trajectory.condition','data':ref(inputs[0]),'group':g.group_name,'outcome':signal,'center':'equal-fish median','band':'fish IQR where >=2 finite fish','coverage':'>=0.9 valid expected fraction; no-bout signed values missing','cs_s':[0,10],'expected_training_us_s':20,'strata':'condition only; Gate L not supplied'}
        legend=('10sTrace Figure 4 condition-only scientific draft. Rows: nine configured CS ten-trial blocks and all configured catches pooled (25,39,53,59,65); trial 65 also belongs to the first Test block. Columns: signed bout-log vigor relative to each trial’s −20 to 0 s baseline, movement probability, and the separately defined supported scaled total activity. Each 0.5 s bin is masked below 0.9 valid expected-frame coverage. Signed vigor uses the supported Figure 4 recalculation; no-bout signed bins stay missing. Within each group and bin, trials are pooled by median within fish, then equal-fish condition medians and IQR bands are shown (bands only with ≥2 finite fish). Blue: 32 Control; brown: 29 10sTrace. Per-bin fish/trial counts and all missingness remain in companion tables; curves do not imply every fish contributes everywhere. Green solid/dashed: CS onset/10 s offset. Violet dotted: verified paired-training expected US at 20 s, including catch rows where no US occurred. No categories, thresholds, p-values or held-out validation are implied. Gate L remains unresolved, so the categorical integrated Figure 4 command has not run. Controls share 3sTrace acquisition identities. Two excluded fish remain in the all-inventory score/outcome audit. This assay-specific draft does not select or freeze shared Figure 4 panels.')
        export(fig,'figure4',w,h,ids,allocations,mapping,legend,inputs)

if __name__=='__main__': {'fig3':fig3,'fig4':fig4}[sys.argv[1]]()
