"""Figure 4 draft tied to Figure 3 scores, with separate data and render stages.

Score thirds are descriptive display strata, never learner labels. All 179
source-roster fish and all configured catch/block groups remain represented.
"""
import argparse,html,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from classical_conditioning.config import get_experiment_spec,Alignment
from classical_conditioning.figures.theme import apply_theme,style_axes,COLORBLIND_QUALITATIVE
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.individual_atlas import svg_bytes
from classical_conditioning.figures.layout_frames import embed_spec
from analyze_figures34_3strace import OUT,ROOT,read,write,bound,verify

DEST=OUT/'figure4-figure3-score-draft';NS='{http://www.w3.org/2000/svg}'
GROUPS=['Control reference','Lower score','Middle score','Higher score']
SIGNALS=['scaled_activity','movement_probability','signed_log_vigor']
LABELS={'scaled_activity':'Scaled total activity','movement_probability':'Movement probability','signed_log_vigor':'Signed bout ln vigor'}
COLORS={'Control reference':'#666666','Lower score':COLORBLIND_QUALITATIVE[3],'Middle score':COLORBLIND_QUALITATIVE[0],'Higher score':COLORBLIND_QUALITATIVE[2]}
ET.register_namespace('',NS[1:-1]);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def guards():
    t=read(DEST/'start-ticket.json')
    for item in [t['figure3_state'],t['score_source'],t['preserve_legacy_review'],*t['inputs']]:verify(item)
    return t

def analyze():
    ticket=guards();scores=pd.read_csv(ticket['score_source']['path'])
    assert len(scores)==179 and not scores.duplicated(['fish_id','condition_id']).any()
    assert scores.acquisition_contrast.notna().all()
    q=scores.loc[scores.condition_id.eq('trace'),'acquisition_contrast'].quantile([1/3,2/3]).to_numpy(float)
    assert q[0]<q[1]
    membership=scores[['fish_id','condition_id','acquisition_contrast','pre_n','late_n']].copy()
    membership['display_group']='Control reference'
    trace=membership.condition_id.eq('trace')
    membership.loc[trace,'display_group']=pd.cut(membership.loc[trace,'acquisition_contrast'],[-np.inf,*q,np.inf],labels=GROUPS[1:],right=True).astype(str)
    membership['group_status']='retrospective score-display stratum; no learner classification'
    membership.to_csv(DEST/'figure3-score-membership.csv',index=False)
    fish=pd.read_parquet(OUT/'fish-bins.parquet')
    joined=fish.merge(membership,on=['fish_id','condition_id'],how='left',validate='many_to_one')
    assert joined.display_group.notna().all() and set(joined.fish_id)==set(scores.fish_id)
    assert not joined.duplicated(['fish_id','group_type','group_name','time_s']).any()
    joined.to_parquet(DEST/'fish-bins-with-score.parquet',index=False)
    keys=['display_group','group_type','group_name','group_order','time_s'];rows=[]
    counts=membership.display_group.value_counts().to_dict()
    for key,part in joined.groupby(keys,sort=False):
        row=dict(zip(keys,key));row['roster_fish']=int(counts[row['display_group']])
        for signal in SIGNALS:
            values=part[signal+'_median'].dropna()
            for quantile,name in [(.25,'q25'),(.5,'median'),(.75,'q75')]:row[signal+'_'+name]=float(values.quantile(quantile)) if len(values) else np.nan
            row[signal+'_fish']=int(len(values));row[signal+'_trials']=int(part[signal+'_count'].sum())
        rows.append(row)
    grouped=pd.DataFrame(rows);grouped.to_parquet(DEST/'score-group-bins.parquet',index=False)
    # Condition overview reuses the authenticated equal-fish source summaries.
    overview=pd.read_parquet(OUT/'group-bins.parquet');overview.to_parquet(DEST/'condition-overview.parquet',index=False)
    coverage=grouped[[*keys,'roster_fish',*[s+'_'+f for s in SIGNALS for f in ['fish','trials']]]]
    coverage.to_csv(DEST/'per-bin-contributions.csv',index=False)
    spec=get_experiment_spec('all3sTrace');catches=list(spec.catch_trial_numbers(Alignment.CS))
    assert sorted(joined.loc[joined.group_type.eq('individual_catch'),'group_name'].unique())==sorted(f'Catch {x}' for x in catches)
    assert len(joined.loc[joined.group_type.eq('block'),'group_name'].unique())==9
    direct=joined[joined.group_type.eq('pooled_catch')].groupby(['display_group','time_s']).scaled_activity_median.median()
    saved=grouped[grouped.group_type.eq('pooled_catch')].set_index(['display_group','time_s']).scaled_activity_median
    np.testing.assert_allclose(direct,saved.loc[direct.index],equal_nan=True)
    # Exact source values are passed through, never recomputed or thresholded as labels.
    check=membership.set_index(['fish_id','condition_id']).acquisition_contrast
    np.testing.assert_array_equal(check,scores.set_index(['fish_id','condition_id']).acquisition_contrast)
    write(DEST/'analysis-spec.json',{'status':'unfrozen retrospective scientific draft, no Gate L decision',
        'figure3_score_source':ticket['score_source'],'figure3_state':ticket['figure3_state'],
        'metric':'legacy_distal_angular_speed','score':'median G outcome on Pre 5–14 minus median G outcome on late Train 40–64',
        'roster_fish':179,'counts':counts,'display_group_rule':{'conditioned':'thirds of observed 3sTrace acquisition contrast, pooled within the assay; equal-score ties retained together','cutpoints':q.tolist(),'interval_closure':'lower side open, upper side closed','controls':'all 90, unsplit reference'},
        'interpretation':'display grouping of continuous Figure 3 scores; no learner/nonlearner labels, thresholds of learning or individual confidence statements',
        'catches':catches,'blocks':[{'name':n,'trials':list(v)} for n,v in spec.trial_blocks(Alignment.CS)],
        'aggregation':'saved trial-within-fish medians -> equal-fish display-group median and fish IQR; no new population selection',
        'minimum_profile_coverage':.9,'quantities':{'scaled_activity':'Plan 05 named saved zero-inclusive Scaled total activity; two-layer per-trial scaling; retain historical metadata tags as recorded','movement_probability':'saved detector-based movement occurrence, not tracking completeness','signed_log_vigor':'supported Figure 4 conditional signed bout-ln-vigor; per-trial [-20,0) baseline, no-bout missing'},
        'main_panel_letters':'local letters for this 3sTrace-only composition candidate; manuscript assay-panel numbering is unchanged',
        'main_panels':{'A':'all-fish condition overview across all nine blocks plus pooled catches, scaled activity','B':'pooled-catch scaled activity by Figure 3 score-display groups and controls','C':'pooled-catch movement probability, same memberships','D':'pooled-catch signed conditional vigor, same memberships'},
        'supplements':['all ten profile groups by score band for all three quantities, two pages','each of five catches for all three quantities','per-bin contributing fish fractions for all three quantities'],
        'legacy_WIP':'earlier 57-classified-fish view preserved separately; not used for membership here',
        'validation_mode':'retrospective descriptive; score construction overlaps catches 53 and 59, so pooled-catch group differences are not independent validation',
        'limitations':['Gate L and primary technical cohort approval remain open','display cutpoints are sample-dependent tertiles, not biological classes','scaled activity and signed bout vigor are distinct quantities, retained because Plan 05 and the current Figure 4 manual emphasize different estimands','no onset/peak/offset estimation or timing p-values','no full three-assay Figure 4 assembly is claimed'],
        'checks':['exact Figure 3 score identity for all 179 fish','unique membership and many-to-one join','all named blocks/catches retained','equal-fish pooled-catch group summary independently checked','per-bin missingness/counts retained']})
    guards();write(DEST/'analysis-complete.json',{'artifacts':[bound(DEST/x) for x in ['figure3-score-membership.csv','fish-bins-with-score.parquet','score-group-bins.parquet','condition-overview.parquet','per-bin-contributions.csv','analysis-spec.json']],'code':bound(Path(__file__))})
    print(json.dumps({'counts':counts,'cutpoints':q.tolist()}))

def compose(panels,boxes,size):
    root=ET.Element(NS+'svg',{'width':f'{size[0]}mm','height':f'{size[1]}mm','viewBox':f'0 0 {size[0]} {size[1]}'})
    allocations=[];background=[];clips=[]
    for name,raw in panels.items():
        sub=ET.fromstring(raw);node=sub.find(NS+'metadata[@id="layout-frames"]');spec=json.loads(node.text);sub.remove(node)
        x,y,w,h=boxes[name];sub.set('x',str(x));sub.set('y',str(y));sub.set('width',str(w));sub.set('height',str(h));root.append(sub)
        allocations.append({'id':name,'kind':'shared' if name in ['heading','key'] else 'panel','box_mm':boxes[name],'group_ids':spec['allocations'][0]['group_ids']})
        background+=spec['background_ids'];clips+=spec['scientific_clips']
    return embed_spec(ET.tostring(root),{'scope':'figure','final_size_mm':size,'inset_mm':1.,'allocations':allocations,'background_ids':background,'scientific_clips':clips})

def render():
    guards();spec=read(DEST/'analysis-spec.json')
    for x in read(DEST/'analysis-complete.json')['artifacts']:verify(x)
    grouped=pd.read_parquet(DEST/'score-group-bins.parquet');overview=pd.read_parquet(DEST/'condition-overview.parquet')
    profile=load_presentation_profile('nature-main')
    render_inputs=[bound(DEST/x) for x in ['analysis-spec.json','score-group-bins.parquet','condition-overview.parquet','figure3-score-membership.csv']]+[bound(ROOT/x) for x in profile.inputs]+[bound(Path(__file__))]
    write(DEST/'render-start.json',{'inputs':render_inputs,'profile_inputs':profile.inputs})
    counts=spec['counts'];main=grouped[grouped.group_type.isin(['block','pooled_catch'])]
    ordered=main[['group_name','group_order','group_type']].drop_duplicates().sort_values('group_order')
    outputs=[]
    with mpl.rc_context():
        theme=apply_theme(profile=profile);mpl.rcParams.update({'figure.autolayout':False,'figure.constrained_layout.use':False,'savefig.bbox':None})
        anno=profile.contract['styles']['annotation']['font_size_pt'];letter=profile.contract['styles']['panel_letter']['font_size_pt']
        def panel(label,title,w,h):
            fig=plt.figure(figsize=(w/25.4,h/25.4))
            if label:fig.text(2/w,1-4.5/h,label.lower(),fontsize=letter,fontweight='bold')
            fig.text(8/w,1-4.5/h,title,fontsize=theme.axes_titlesize)
            return fig
        def curves(ax,data,signal,ylim=None,xticklabels=True,xlabel=None,ylabel=None):
            for group in GROUPS:
                g=data[data.display_group.eq(group)].sort_values('time_s')
                ax.plot(g.time_s,g[signal+'_median'],color=COLORS[group],ls='--' if group=='Control reference' else '-',lw=theme.lines_linewidth,label=f'{group} (n = {counts[group]})')
                ax.fill_between(g.time_s,g[signal+'_q25'],g[signal+'_q75'],where=g[signal+'_fish'].ge(2),color=COLORS[group],alpha=.10,linewidth=0)
            ax.axvspan(0,10,color=theme.cs_color,alpha=.07,linewidth=0)
            ax.axvline(13,color=theme.us_color,ls=':',lw=theme.us_linewidth)
            if signal=='signed_log_vigor':ax.axhline(0,color='#999999',lw=.4,zorder=0)
            ax.set_xlim(-20,20);ax.set_xticks([-20,0,10,20])
            if signal!='signed_log_vigor':ax.set_ylim(0,1);ax.set_yticks([0,.5,1])
            else:
                values=np.abs(data[[signal+'_q25',signal+'_median',signal+'_q75']].to_numpy(float));values=values[np.isfinite(values)]
                limit=ylim or max(.25,np.ceil(values.max()*10)/10)
                ax.set_ylim(-limit,limit);ax.set_yticks([-limit,0,limit])
            style_axes(ax,theme=theme,show_xticklabels=xticklabels,xlabel=xlabel,ylabel=ylabel)
        def output(name,svg,legend):
            (DEST/(name+'.svg')).write_bytes(svg);(DEST/(name+'-legend.md')).write_text(legend,encoding='utf-8')
            (DEST/(name+'.html')).write_text('<!doctype html><meta charset="utf-8"><style>body{margin:12px;font:14px Arial;color:black;background:white}svg{max-width:100%;height:auto}p{max-width:1100px}</style>'+svg.decode()+'<p>'+html.escape(legend)+'</p>',encoding='utf-8')
            outputs.extend([bound(DEST/(name+x)) for x in ['.svg','.html','-legend.md']])
        panels={};boxes={'A':[0,0,183,74],'B':[0,74,91.5,48],'C':[91.5,74,91.5,48],'D':[0,122,183,48]}
        fig=panel('A','3sTrace: block progression in all fish',183,74)
        cmap=mpl.colormaps[theme.probability_cmap].copy();cmap.set_bad(theme.heatmap_bad_color)
        for index,(condition,title,x) in enumerate([('control','Control',22),('trace','3sTrace',100)]):
            ax=fig.add_axes([x/183,1-56/74,61/183,42/74]);d=overview[overview.condition_id.eq(condition)&overview.group_type.isin(['block','pooled_catch'])]
            matrix=d.pivot(index='group_order',columns='time_s',values='scaled_activity_median').reindex(ordered.group_order).to_numpy()
            image=ax.imshow(matrix,origin='upper',extent=[-20,20,9.5,-.5],aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1)
            for t in [0,10]:ax.axvline(t,color=theme.cs_color,lw=theme.cs_linewidth)
            ax.axvline(13,color=theme.us_color,lw=theme.us_linewidth,ls=':');ax.set_yticks(range(10),ordered.group_name)
            ax.set_xticks([-20,0,10,20]);ax.set_title(title,pad=3)
            style_axes(ax,theme=theme,xlabel='Time from CS (s)',show_yticklabels=index==0)
        cbax=fig.add_axes([172/183,1-53/74,2/183,30/74]);bar=fig.colorbar(image,cax=cbax,ticks=[0,.5,1]);bar.ax.tick_params(labelsize=anno)
        fig.text(172/183,1-18/74,'Scaled\nactivity',ha='center',fontsize=anno)
        panels['A']=svg_bytes(fig,'score_fig4_A',183,74)
        catches=grouped[grouped.group_type.eq('pooled_catch')]
        for letter_name,signal,title in [('B','scaled_activity','Pooled catches: total activity'),('C','movement_probability','Pooled catches: occurrence')]:
            fig=panel(letter_name,title,91.5,48);ax=fig.add_axes([16/91.5,1-35/48,71/91.5,24/48])
            curves(ax,catches,signal,xlabel='Time from CS (s)',ylabel=LABELS[signal])
            panels[letter_name]=svg_bytes(fig,'score_fig4_'+letter_name,91.5,48)
        fig=panel('D','Pooled catches: conditional vigor',183,48);ax=fig.add_axes([16/183,1-35/48,104/183,24/48])
        curves(ax,catches,'signed_log_vigor',xlabel='Time from CS (s)',ylabel='Signed bout ln vigor')
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,frameon=False,loc='center right',bbox_to_anchor=(.985,.55),fontsize=theme.legend_fontsize)
        fig.text(126/183,.10,'Figure 3 score thirds; descriptive',fontsize=anno)
        panels['D']=svg_bytes(fig,'score_fig4_D',183,48)
        legend=('Figure 4, 3sTrace — proposed composition aligned with the current Figure 3 continuous analysis. (a) Condition-wide median scaled total activity for all 90 Control and 89 3sTrace fish across all nine configured ten-trial blocks and pooled catches. All fish remain represented; this overview is independent of score grouping. (b–d) Pooled-catch scaled total activity, movement probability and signed conditional bout-ln-vigor, respectively. The 89 3sTrace fish are displayed in lower, middle and higher thirds of the exact Figure 3 acquisition contrast; cutpoints are '+f'{spec["display_group_rule"]["cutpoints"][0]:.6f} and {spec["display_group_rule"]["cutpoints"][1]:.6f}'+'. These are descriptive display bins, not learner/nonlearner labels or biological thresholds. All 90 controls are an unsplit reference. Lines are equal-fish medians after pooling trial medians within each fish; bands are fish IQRs where at least two fish contribute. Pooled catches are '+', '.join(map(str,spec['catches']))+'; trial 65 also belongs to Test 1. Green marks CS 0–10 s; violet marks the protocol-verified paired-training expected US at 13 s, not a US on conditioned catch trials. Controls have unpaired reinforcers. Bins below 0.9 valid expected-frame coverage are missing; no-bout signed vigor stays undefined and movement remains separate. Scaled activity includes valid zeros and retains its saved two-layer per-trial scaling; signed vigor uses a distinct [−20,0) s baseline. All-fish membership, score source hashes and per-bin fish/trial contributions accompany the draft. The score uses Pre 5–14 and late Train 40–64, overlapping catches 53 and 59: these same-data views do not independently validate learning or response timing. Full score-group block profiles, individual catches and coverage are supplementary reviews. The earlier legacy-WIP subset is preserved separately and is not used here. No categorical classifier, freeze, Gate L or primary-cohort approval is claimed.')
        output('figure4',compose(panels,boxes,[183,170]),legend)
        # Retain every block-by-group curve in two readable supplementary pages.
        supplement_sets=[('block-profiles-1',ordered.iloc[:5]),('block-profiles-2',ordered.iloc[5:]),('individual-catches',grouped[grouped.group_type.eq('individual_catch')][['group_name','group_order','group_type']].drop_duplicates().sort_values('group_order'))]
        for name,rows in supplement_sets:
            panels={};boxes={'heading':[0,0,183,14]};fig=panel('',name.replace('-',' ').capitalize(),183,14)
            handles=[mpl.lines.Line2D([0],[0],color=COLORS[g],ls='--' if g==GROUPS[0] else '-',lw=1) for g in GROUPS]
            fig.legend(handles,[f'{g} (n = {counts[g]})' for g in GROUPS],ncol=2,frameon=False,loc='upper right',bbox_to_anchor=(.99,1),fontsize=theme.legend_fontsize)
            panels['heading']=svg_bytes(fig,'score_'+name+'_heading',183,14)
            for row,(_,item) in enumerate(rows.iterrows()):
                for col,signal in enumerate(SIGNALS):
                    identity=f'r{row}c{col}';boxes[identity]=[61*col,14+25*row,61,25]
                    end=16.5 if row==len(rows)-1 else 18
                    fig=plt.figure(figsize=(61/25.4,25/25.4));ax=fig.add_axes([12/61,1-end/25,46/61,11/25])
                    d=grouped[grouped.group_order.eq(item.group_order)]
                    curves(ax,d,signal,xticklabels=row==len(rows)-1,xlabel='Time from CS (s)' if row==len(rows)-1 else None)
                    ax.set_title((item.group_name+' · ' if col==0 else '')+(LABELS[signal] if row==0 else ''),fontsize=anno,pad=2,loc='left')
                    panels[identity]=svg_bytes(fig,'score_'+name+'_'+identity,61,25)
            output(name,compose(panels,boxes,[183,14+25*len(rows)]),'Supplementary '+name.replace('-',' ')+'. Same Figure 3 score memberships, quantities, coverage mask, equal-fish aggregation and retrospective limitations as the main Figure 4 draft. All three outcomes are shown. The first two pages retain every declared block plus pooled catches; the individual-catch page shows each configured catch separately. '+legend)
        # Coverage is explicitly separate from movement and conditional intensity.
        panels={};boxes={'heading':[0,0,183,14]};fig=panel('','Contributing-fish coverage',183,14)
        fig.text(.985,.22,'White = no contributing fish; black = full group',ha='right',fontsize=anno);panels['heading']=svg_bytes(fig,'score_coverage_heading',183,14)
        for row,group in enumerate(GROUPS):
            for col,signal in enumerate(SIGNALS):
                identity=f'r{row}c{col}';boxes[identity]=[61*col,14+34*row,61,34]
                fig=plt.figure(figsize=(61/25.4,34/25.4));ax=fig.add_axes([12/61,1-25/34,46/61,19/34])
                d=main[main.display_group.eq(group)]
                matrix=d.pivot(index='group_order',columns='time_s',values=signal+'_fish').reindex(ordered.group_order).to_numpy()/counts[group]
                ax.imshow(matrix,origin='upper',extent=[-20,20,9.5,-.5],cmap='Greys',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
                ax.set_yticks([0,5,9],['Pre','Train 5','Catch']);ax.set_xticks([-20,0,20]);ax.set_title(group+' · '+LABELS[signal],fontsize=anno,pad=2)
                style_axes(ax,theme=theme,show_xticklabels=row==3,xlabel='Time from CS (s)' if row==3 else None)
                panels[identity]=svg_bytes(fig,'score_coverage_'+identity,61,34)
        output('coverage',compose(panels,boxes,[183,150]),'Contributing-fish fraction (0–1) for every declared block plus pooled catches, shown separately for each Figure 3 score-display group and each outcome. Denominator is the full display-group membership, not the number of observed fish in a bin. Missingness is not treated as immobility; detailed fish/trial counts are in per-bin-contributions.csv. Same source identities and retrospective limitations as the main draft.')
    for item in render_inputs:verify(item)
    write(DEST/'render-complete.json',{'artifacts':outputs,'source_inputs':render_inputs,'status':'draft renders; exact-render/visual verification pending'})
    print('Rendered Figure 4 main draft and four supplementary reviews.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['analyze','render']);args=p.parse_args();analyze() if args.stage=='analyze' else render()
