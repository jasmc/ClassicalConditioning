"""Figure 4B using authenticated saved legacy-WIP groups, with full flow.

No classifier is invented or refitted. Unmatched current fish are Unclassified.
The frozen-manifest integrated route is not impersonated by this draft route.
"""
import json,html,os
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classical_conditioning.analysis.figure4 import load_classification_manifest,assign_plot_strata,summarize_trial_bins,STRATA
from classical_conditioning.figures.figure4 import COLORS,STYLES
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme,style_axes
from classical_conditioning.figures.individual_atlas import svg_bytes
from classical_conditioning.figures.layout_frames import embed_spec
from analyze_figures34_3strace import OUT,ROOT,PROJECT,read,bound,write,verify

DEST=OUT/'figure4-classified-preliminary';NS='{http://www.w3.org/2000/svg}'

def main():
    lock=DEST/'build.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        ticket=read(DEST/'start-ticket.json')
        for x in [ticket['parent_state'],ticket['protected_figure3'],*ticket['inputs']]:verify(x)
        source=PROJECT/'Processed data/Analyses/figure4-3strace-window13-legacy-59fish'
        labels,meta=load_classification_manifest(source/'learner-labels.csv','legacy_distal_angular_speed',('all3sTrace',))
        verify({'path':meta['source_comparison'],'sha256':meta['source_comparison_sha256']})
        comparison=read(meta['source_comparison'])
        for key,sha_key in [('cohort_path','cohort_sha256'),('outcomes_path','outcomes_sha256')]:verify({'path':comparison[key],'sha256':comparison[sha_key]})
        assert comparison['metric_id']=='legacy_distal_angular_speed' and comparison['cohort_hash']==meta['cohort_hashes']['all3sTrace']
        old=read(source/'figure4/analysis.json')
        current_sources={x['path']:x['sha256'] for r in read(OUT/'source-authentication.json') for x in r['sources']}
        for item in old['inputs']:
            if item['path'] in current_sources:assert item['sha256']==current_sources[item['path']],item['path']
        roster=pd.read_csv(OUT/'acquisition-scores.csv')[['fish_id','condition_id']]
        assert not roster.duplicated('fish_id').any() and len(roster)==179
        assert not labels.duplicated(['fish_id','condition_id']).any()
        excluded=labels[~labels.fish_id.isin(roster.fish_id)]
        excluded.to_csv(DEST/'historical-labels-outside-current-roster.csv',index=False)
        flow=roster.merge(labels[['fish_id','condition_id','classifier_label','classification_eligible','ineligible_reason']],on=['fish_id','condition_id'],how='left',validate='one_to_one',indicator=True)
        flow['label_source_status']=np.where(flow['_merge'].eq('both'),'saved legacy-WIP result','no saved classification for this current-roster fish')
        missing=flow['_merge'].eq('left_only')
        flow.loc[missing,'classifier_label']='Unclassified';flow.loc[missing,'classification_eligible']=False
        flow.loc[missing,'ineligible_reason']='No saved legacy-WIP classification on this fish; not assigned a nonlearner label'
        flow['experiment_id']='all3sTrace';flow['cohort_role']=np.where(flow.condition_id.eq('control'),'reference','conditioned')
        flow=assign_plot_strata(flow.drop(columns='_merge'))
        flow.to_csv(DEST/'sample-flow.csv',index=False)
        counts=flow.plot_stratum.value_counts().to_dict()
        trial=pd.read_parquet(OUT/'trial-bins.parquet')
        assert set(trial.fish_id)==set(flow.fish_id) and len(trial)==179*90*80
        trial=trial.merge(flow,on=['fish_id','condition_id'],validate='many_to_one')
        assert trial.classifier_label.notna().all()
        fish,groups=summarize_trial_bins(trial,'all3sTrace')
        fish.to_parquet(DEST/'fish-bins.parquet',index=False);groups.to_parquet(DEST/'group-bins.parquet',index=False)
        # Independently check a full pooled-catch stratum/time grid against trials.
        catches=trial[trial.trial_number.isin([25,39,53,59,65]) & trial.plot_stratum.isin(STRATA)]
        check=catches.groupby(['fish_id','plot_stratum','time_s']).signed_log_vigor.median().reset_index()
        direct=check.groupby(['plot_stratum','time_s']).signed_log_vigor.median()
        saved=groups[groups.group_type.eq('pooled_catch')].set_index(['plot_stratum','time_s']).signed_median
        np.testing.assert_allclose(direct,saved.loc[direct.index],equal_nan=True)
        profile=load_presentation_profile('nature-main');inputs=[*ticket['inputs'],bound(Path(__file__)),*[bound(ROOT/x) for x in profile.inputs],bound(Path(meta['source_comparison']))]
        write(DEST/'analysis.json',{'scope':'3sTrace Figure 4B only','scientific_status':'preliminary retrospective legacy-rule characterization; not Gate L approval',
            'metric':'legacy_distal_angular_speed','classifier_execution_id':meta['classifier_execution_id'],'classifier_table':bound(source/'learner-labels.csv'),'classifier_manifest':bound(source/'learner-labels.manifest.json'),
            'roster_fish':179,'historical_label_overlap':int((~missing).sum()),'newly_unclassified_without_saved_labels':int(missing.sum()),'strata':counts,
            'historical_outside_current_roster':excluded[['fish_id','condition_id','classifier_label']].to_dict('records'),
            'timing':'current profile inputs already passed paired-training 13 s protocol checks; control unpaired US overlap evidence retained in parent assay output',
            'source_consistency':'consumed saved-classification recording inputs match the current profile source hashes; saved cohort/outcome and assessment hashes authenticated',
            'outcome':'signed bout ln vigor; baseline [-20,0); 0.5 s bins; coverage >=0.9; no-bout values missing',
            'aggregation':'trial median within fish; equal-fish stratum median and IQR; pooled catch set from ExperimentSpec',
            'validation_mode':'descriptive_same_data','checks':['179 fish preserved in sample flow','many-to-one join; no duplicate labels','no assignment of absent labels to nonlearner','direct independent pooled-catch aggregation equality'],
            'inputs':inputs,'limitations':['122 current-roster fish remain unclassified or without a saved classification','historical labels do not constitute independently validated learner categories','single control learner-flagged fish has no fish-IQR band','saved WIP labels require acquisition and recovery; not equivalent to Figure 3 continuous acquisition contrast']})
        main_groups=groups[groups.group_type.isin(['block','pooled_catch'])]
        rows=main_groups[['group_order','group_name']].drop_duplicates().sort_values('group_order');assert len(rows)==10
        vals=np.abs(main_groups[['signed_median','signed_q25','signed_q75']].to_numpy(float));vals=vals[np.isfinite(vals)]
        limit=max(.5,np.ceil(vals.max()*4)/4)
        pretty={STRATA[0]:'3sTrace learner',STRATA[1]:'3sTrace nonlearner',STRATA[2]:'Control learner-flagged',STRATA[3]:'Control nonlearner'}
        panels={};boxes={'heading':[0,0,183,18]}
        with mpl.rc_context():
            theme=apply_theme(profile=profile);mpl.rcParams.update({'figure.autolayout':False,'figure.constrained_layout.use':False,'savefig.bbox':None})
            style=profile.contract['styles'];annotation=style['annotation']['font_size_pt']
            fig=plt.figure(figsize=(183/25.4,18/25.4))
            fig.text(2/183,1-5/18,'b',fontsize=style['panel_letter']['font_size_pt'],fontweight='bold')
            fig.text(8/183,1-5/18,'3sTrace · preliminary',fontsize=theme.axes_titlesize)
            handles=[mpl.lines.Line2D([0],[0],color=COLORS[s],ls=STYLES[s],lw=1) for s in STRATA]
            fig.legend(handles,[pretty[s]+f' (n = {counts.get(s,0)})' for s in STRATA],ncol=2,frameon=False,loc='upper right',bbox_to_anchor=(.985,.98),fontsize=theme.legend_fontsize)
            fig.text(8/183,1-14/18,'Saved legacy-WIP groups; 57 classified / 179 current-roster fish',fontsize=annotation)
            panels['heading']=svg_bytes(fig,'trace3_fig4_heading',183,18)
            for i,(_,group) in enumerate(rows.iterrows()):
                name=f'row{i}';boxes[name]=[0,18+22*i,183,22]
                fig=plt.figure(figsize=(183/25.4,22/25.4));ax=fig.add_axes([14/183,5/22,164/183,12/22])
                for stratum in STRATA:
                    part=main_groups[main_groups.group_order.eq(group.group_order)&main_groups.plot_stratum.eq(stratum)].sort_values('time_s')
                    ax.plot(part.time_s,part.signed_median,color=COLORS[stratum],ls=STYLES[stratum],lw=theme.lines_linewidth)
                    ax.fill_between(part.time_s,part.signed_q25,part.signed_q75,where=part.signed_fish.ge(2),color=COLORS[stratum],alpha=.12,linewidth=0)
                ax.axhline(0,color='#888888',lw=.4,zorder=0);ax.axvspan(0,10,color=theme.cs_color,alpha=.08,linewidth=0)
                for t in [0,10]:ax.axvline(t,color=theme.cs_color,lw=theme.cs_linewidth)
                ax.axvline(13,color=theme.us_color,lw=theme.us_linewidth,ls=':')
                ax.set_xlim(-20,20);ax.set_ylim(-limit,limit);ax.set_yticks([-limit,0,limit]);ax.set_xticks([-20,0,10,20])
                style_axes(ax,theme=theme,show_xticklabels=i==9)
                ax.set_title(group.group_name,loc='left',pad=2,fontsize=theme.axes_titlesize)
                panels[name]=svg_bytes(fig,'trace3_fig4_'+name,183,22)
            boxes['xlabel']=[0,238,183,8];fig=plt.figure(figsize=(183/25.4,8/25.4));fig.text(.525,.50,'Time from CS onset (s)',ha='center',fontsize=theme.axes_labelsize)
            fig.text(.025,.50,'Signed ln vigor',fontsize=annotation);panels['xlabel']=svg_bytes(fig,'trace3_fig4_xlabel',183,8)
        import xml.etree.ElementTree as ET
        ET.register_namespace('',NS[1:-1]);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
        root=ET.Element(NS+'svg',{'width':'183mm','height':'246mm','viewBox':'0 0 183 246'});alloc=[];backgrounds=[];clips=[]
        for name,raw in panels.items():
            sub=ET.fromstring(raw);meta_node=sub.find(NS+'metadata[@id="layout-frames"]');spec=json.loads(meta_node.text);sub.remove(meta_node)
            x,y,w,h=boxes[name];sub.set('x',str(x));sub.set('y',str(y));sub.set('width',str(w));sub.set('height',str(h));root.append(sub)
            alloc.append({'id':name,'kind':'panel' if name.startswith('row') else 'shared','box_mm':boxes[name],'group_ids':spec['allocations'][0]['group_ids']})
            backgrounds+=spec['background_ids'];clips+=spec['scientific_clips']
        svg=embed_spec(ET.tostring(root),{'scope':'figure','final_size_mm':[183,246],'inset_mm':profile.layout['selected_inset_mm'],'allocations':alloc,'background_ids':backgrounds,'scientific_clips':clips})
        legend=('Figure 4B. Preliminary 3sTrace profiles grouped by saved legacy-WIP classification (legacy-wip-3strace-distal-window13-59fish), using legacy_distal_angular_speed. Current-roster overlap yields 16 3sTrace learners, 23 3sTrace nonlearners, 1 control learner-flagged fish and 17 control nonlearners. The timing-flagged historical nonlearner 20230803_12 remains excluded. Of the full 179-fish current roster, 121 lack a saved classification and one saved control result is Unclassified; these 122 are retained in sample-flow.csv and are not relabelled as nonlearners or drawn in the four classified curves. Nine configured CS ten-trial blocks and pooled catches (25,39,53,59,65) use trial medians within fish followed by equal-fish stratum medians; bands are fish IQRs where at least two fish contribute. The n = 1 control learner-flagged group has no IQR band. Signed bout ln vigor uses the supported −20 to 0 s baseline and 0.5 s bins; coverage below 0.9 and no-bout values remain missing. Per-bin fish/trial contributions are saved. Green marks CS 0–10 s; violet is the protocol-verified expected paired-training US at 13 s, not an actual catch US; controls have unpaired reinforcers. Trial 65 contributes to both Test 1 and pooled catches. Labels use the same historical acquisition/recovery data as the profiles: this is retrospective rule characterization, not independent validation, a biological learning diagnosis, or a freeze. Gate L, full-roster classification and primary-cohort approval remain open. Figure 3 and its continuous contrasts are unchanged.')
        (DEST/'figure4.svg').write_bytes(svg);(DEST/'legend.md').write_text(legend,encoding='utf-8')
        (DEST/'review.html').write_text('<!doctype html><meta charset="utf-8"><style>body{margin:12px;background:white;font:14px Arial;color:black}svg{max-width:100%;height:auto}p{max-width:1100px}</style>'+svg.decode()+'<p>'+html.escape(legend)+'</p>',encoding='utf-8')
        for item in [ticket['parent_state'],ticket['protected_figure3'],*inputs]:verify(item)
        write(DEST/'build-complete.json',{'artifacts':[bound(DEST/x) for x in ['figure4.svg','legend.md','review.html','sample-flow.csv','fish-bins.parquet','group-bins.parquet','analysis.json']], 'status':'preliminary grouped figure built; visual verification pending','counts':counts})
        print(json.dumps(counts))
    finally:lock.unlink()

if __name__=='__main__':main()
