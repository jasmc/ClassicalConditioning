"""Nature-profile provisional 3sTrace reviews, without shared publication."""
from __future__ import annotations
import html
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classical_conditioning.config import get_experiment_spec
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme, style_axes, condition_color
from classical_conditioning.figures.layout_frames import embed_spec, load_policy
from analyze_figures34_3strace import OUT, BASE, ROOT, PROJECT, bound, read, write, verify

NS='{http://www.w3.org/2000/svg}'
ET.register_namespace('',NS[1:-1])
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def export(fig,name,w,h,allocations,folder=OUT):
    stream=io.BytesIO(); fig.savefig(stream,format='svg',bbox_inches=None)
    root=ET.fromstring(stream.getvalue())
    bg=root.find('.//'+NS+'g[@id="patch_1"]')
    bg.set('id',name+'_background')
    for i,ax in enumerate(fig.axes):
        # Matplotlib axes contain their ticks, labels, paths and titles.
        ax.set_gid(name+f'_axes_{i}')
    stream=io.BytesIO();fig.savefig(stream,format='svg',bbox_inches=None)
    root=ET.fromstring(stream.getvalue());root.find('.//'+NS+'g[@id="patch_1"]').set('id',name+'_background')
    policy,_=load_policy()
    spec={'scope':'figure','final_size_mm':[w,h],'inset_mm':policy['selected_inset_mm'],
        'allocations':allocations,'background_ids':[name+'_background']}
    path=folder/(name+'.svg')
    path.write_bytes(embed_spec(ET.tostring(root),spec));plt.close(fig)
    return path

def review(name,svg,legend):
    (OUT/(name+'-legend.md')).write_text(legend,encoding='utf-8')
    (OUT/(name+'.html')).write_text('<!doctype html><meta charset="utf-8"><style>body{margin:16px;background:white;color:black;font:14px Arial}svg{max-width:100%;height:auto}p{max-width:1000px}</style>'+svg.read_text(encoding='utf-8')+'<p>'+html.escape(legend)+'</p>',encoding='utf-8')

def main():
    analysis=read(OUT/'analysis.json');verify(read(OUT/'complete.json')['analysis'])
    if (OUT/'render-complete.json').exists():raise RuntimeError('Preserve existing render; no overwrite')
    profile=load_presentation_profile('nature-main')
    write(OUT/'render-start.json',{'inputs':[bound(OUT/x) for x in ['analysis.json','acquisition-scores.csv','group-bins.parquet','observed-g-trial-outcomes.parquet']]
        +[bound(ROOT/x) for x in profile.inputs]
        +[bound(ROOT/x) for x in ['src/classical_conditioning/figures/theme.py','src/classical_conditioning/figures/presentation_profiles.py','src/classical_conditioning/figures/layout_frames.py']]
        +[bound(Path(__file__))],'profile_inputs':profile.inputs})
    scores=pd.read_csv(OUT/'acquisition-scores.csv');g=pd.read_parquet(OUT/'group-bins.parquet')
    windows=pd.read_parquet(OUT/'observed-g-trial-outcomes.parquet')
    signed_values=np.abs(g[['signed_log_vigor_median','signed_log_vigor_q25','signed_log_vigor_q75']].to_numpy(float))
    signed_values=signed_values[np.isfinite(signed_values)]
    signed_limit=max(.5,np.ceil(signed_values.max()*4)/4) if len(signed_values) else .5
    spec=get_experiment_spec('all3sTrace')
    colors={c.condition_id:condition_color(c) for c in spec.conditions}
    artifacts=[]
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        mpl.rcParams.update({'figure.autolayout':False,'figure.constrained_layout.use':False,'savefig.bbox':None})
        fig=plt.figure(figsize=(89/25.4,75/25.4));ax=fig.add_axes([.19,.21,.77,.72])
        for j,condition in enumerate(['control','trace']):
            x=scores[scores.condition_id.eq(condition)].sort_values('fish_id')
            # Deterministic offset is visual packing only; no additional estimator.
            jitter=np.linspace(-.17,.17,len(x))
            ax.scatter(j+jitter,x.acquisition_contrast,s=5,color=colors[condition],alpha=.65,linewidths=0)
            q=x.acquisition_contrast.quantile([.25,.5,.75])
            ax.plot([j,j],[q.loc[.25],q.loc[.75]],color='black')
            ax.plot([j-.1,j+.1],[q.loc[.5]]*2,color='black')
        ax.axhline(0,color='#808080',lw=.4,zorder=0)
        ax.set_xticks([0,1],['Control','3sTrace']);ax.set_xlim(-.4,1.4)
        style_axes(ax,theme=theme,ylabel='Acquisition contrast (ln vigor difference)')
        p=export(fig,'figure3-score',89,75,[{'id':'score','kind':'panel','box_mm':[0,0,89,75],'group_ids':['figure3-score_axes_0']}])
        legend=('Provisional Figure 3, 3sTrace. Each point is one source-roster fish: 90 Control and 89 3sTrace. '
            'Acquisition contrast = median observed response-minus-baseline log-median trial outcome on Pre trials 5–14 minus its median on late-training trials 40–64. '
            'Positive values describe greater suppression. Response window [0,13) s; baseline [−15,0) s; positive valid moving-frame legacy distal angular speed only. '
            'Black marks show condition median and fish IQR, not confidence intervals. Missing vigor remains undefined, distinct from immobility. '
            'All fish remain in the numerical inventory; one finite outcome per epoch establishes mathematical estimability only. '
            'This is retrospective description, with no approved learner categories, individual uncertainty, independent validation or frozen primary technical cohort.')
        review('figure3-score',p,legend);artifacts.append(bound(p))
        for name,mask,height in [('figure4-condition-profiles',~g.group_type.eq('individual_catch'),250),('figure4-individual-catches',g.group_type.eq('individual_catch'),135)]:
            data=g[mask]; groups=data[['group_order','group_name']].drop_duplicates().sort_values('group_order')
            n=len(groups);fig=plt.figure(figsize=(183/25.4,height/25.4));alloc=[]
            signals=[('signed_log_vigor','Signed bout ln vigor'),('movement_probability','Movement probability'),('scaled_activity','Scaled total activity')]
            for row,(_,group) in enumerate(groups.iterrows()):
                top=height*(row/n); rh=height/n
                for col,(signal,label) in enumerate(signals):
                    left=col*61
                    ax=fig.add_axes([(left+12)/183,1-(top+rh-7)/height,46/183,(rh-14)/height])
                    for condition in ['control','trace']:
                        sub=data[data.group_order.eq(group.group_order)&data.condition_id.eq(condition)].sort_values('time_s')
                        ax.plot(sub.time_s,sub[signal+'_median'],color=colors[condition],label='Control' if condition=='control' else '3sTrace')
                        ax.fill_between(sub.time_s,sub[signal+'_q25'],sub[signal+'_q75'],where=sub[signal+'_fish'].ge(2),color=colors[condition],alpha=.18,linewidth=0)
                    ax.axvspan(0,10,color='#009d54',alpha=.08)
                    ax.axvline(13,color='#8055b0',lw=.5,linestyle='--')
                    if signal=='signed_log_vigor':
                        ax.axhline(0,color='#808080',lw=.4,zorder=0)
                        ax.set_ylim(-signed_limit,signed_limit);ax.set_yticks([-signed_limit,0,signed_limit])
                    else:ax.set_ylim(0,1);ax.set_yticks([0,.5,1])
                    ax.set_xlim(-20,20);ax.set_xticks([-20,0,20])
                    style_axes(ax,theme=theme,xlabel='Time from CS (s)' if row==n-1 else None,
                        show_xticklabels=row==n-1)
                    if row==0:ax.set_title((group.group_name+' · ' if col==0 else '')+label,loc='left',pad=2)
                    elif col==0:ax.set_title(group.group_name,loc='left',pad=2)
                    if row==0 and col==2:ax.legend(frameon=False,loc='upper right')
                    idx=len(fig.axes)-1
                    alloc.append({'id':f'row{row}_col{col}','kind':'panel','box_mm':[left,top,61,rh],
                        'group_ids':[name+f'_axes_{idx}']})
            p=export(fig,name,183,height,alloc)
            legend=('Provisional Figure 4, 3sTrace condition profiles. Control (90) and 3sTrace (89) use the same authenticated source roster as Figure 3; no learner labels are supplied. '
                'Half-second bins from −20 to 20 s; bins with valid expected-frame coverage below 0.9 are missing. '
                'Left: supported Figure 4 signed bout log vigor, recomputed from processed metrics with baseline [−20,0) s; no-bout values remain undefined. '
                'Middle: movement probability. Right: separate saved scaled total activity, including valid zeros, with its source per-trial scaling baseline earlier than −15 s. '
                'Trials are pooled by median within fish, then equal-fish condition medians and fish IQRs (bands where at least two fish contribute). '
                'All configured ten-trial blocks and pooled catches are shown in the main profile review; individual catches are shown in the supplementary review. '
                'Catches are 25,39,53,59,65; trial 65 contributes to Test 1 and pooled catches. Green spans CS 0–10 s; dashed violet line marks the event-verified conditioned paired-training expected US at 13 s, not a US observed on catches. Controls have unpaired reinforcers, including overlaps with CS-aligned windows; exact latencies are recorded per fish. '
                'Per-bin fish/trial counts, missingness and source hashes accompany the tables. These condition-only descriptions do not resolve Gate L or support learner-group inference; primary technical cohort approval remains open.')
            review(name,p,legend);artifacts.append(bound(p))
        # Complete individual trial-outcome atlas precedes any new illustrative selection.
        atlas=OUT/'individual-atlas';atlas.mkdir(exist_ok=True); entries=[];atlas_sources=[]
        ylim=windows.legacy_log_median_difference.agg(['min','max']).to_numpy(float)
        ylim=ylim+np.array([-1,1])*.05*np.ptp(ylim)
        for fish in scores.sort_values(['condition_id','fish_id']).itertuples():
            d=windows[windows.fish_id.eq(fish.fish_id)]
            rec_path=BASE/'fish'/f'{fish.fish_id}.json';rec=read(rec_path)
            for item in rec['outputs']:verify(item)
            bins_path=BASE/'fish'/f'{fish.fish_id}-bins.parquet';bins=pd.read_parquet(bins_path)
            assert bins.fish.eq(fish.fish_id).all()
            mat=bins.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').sort_index()
            assert list(mat.index)==list(range(5,95)) and mat.shape==(90,40)
            fig=plt.figure(figsize=(183/25.4,65/25.4))
            axh=fig.add_axes([10/183,12/65,60/183,45/65])
            cmap=mpl.colormaps[theme.intensity_cmap].copy();cmap.set_bad(theme.heatmap_bad_color)
            image=axh.imshow(mat.to_numpy(),origin='upper',extent=(-20,20,94.5,4.5),aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25)
            axh.axvline(0,color=theme.cs_color,lw=theme.cs_linewidth)
            axh.axvline(10,color=theme.cs_color,lw=theme.cs_linewidth,ls='--')
            protocol_path=PROJECT/'Processed data'/fish.fish_id/'stimulus_events.parquet'
            verify(next(x for x in rec['window_sources'] if Path(x['path'])==protocol_path))
            protocol=pd.read_parquet(protocol_path)
            cycles=protocol[protocol.Type.astype(str).eq('Cycle')].sort_values('Beg')
            us=protocol.loc[protocol.Type.astype(str).eq('Reinforcer'),'Beg'].to_numpy(float)
            for number,onset in enumerate(cycles.iloc[4:94].Beg,start=5):
                latency=(us-float(onset))/1000;latency=latency[(latency>=-20)&(latency<20)]
                axh.scatter(latency,np.full(len(latency),number),s=2,color='#c18f28',linewidths=0)
            style_axes(axh,theme=theme,xlabel='Time from CS (s)',ylabel='CS trial')
            axh.set_title(f'{fish.fish_id} · {fish.condition_id}');axh.set_yticks([5,14,40,64,94])
            ax=fig.add_axes([107/183,12/65,70/183,45/65])
            ax.scatter(d.trial_number,d.legacy_log_median_difference,s=4,color=colors[fish.condition_id],linewidths=0)
            ax.plot([5,14],[fish.pre_median]*2,color='black');ax.plot([40,64],[fish.late_median]*2,color='black')
            ax.axhline(0,color='#808080',lw=.4,zorder=0);ax.set_xlim(4,95);ax.set_ylim(ylim)
            style_axes(ax,theme=theme,xlabel='CS trial',ylabel='Observed ln vigor difference')
            ax.set_title('Observed outcome and score epochs')
            cbax=fig.add_axes([74/183,18/65,2/183,32/65]);cb=fig.colorbar(image,cax=cbax,ticks=[-.25,0,.25]);cb.set_label('Heatmap Δ ln vigor')
            name='atlas_'+fish.fish_id
            file=export(fig,name,183,65,[{'id':'heatmap','kind':'panel','box_mm':[0,0,91.5,65],
                'group_ids':[name+'_axes_0',name+'_axes_2']},
                {'id':'outcome','kind':'panel','box_mm':[91.5,0,91.5,65],'group_ids':[name+'_axes_1']}],folder=atlas)
            atlas_sources.extend([bound(rec_path),bound(bins_path),bound(protocol_path)])
            text=(f'{fish.fish_id}: contrast {fish.acquisition_contrast:.4f}; finite Pre {fish.pre_n}/10, late Train {fish.late_n}/25; '
                f'{fish.observed_trial_n}/90 finite outcomes. No class assigned.')
            entries.append('<section>'+file.read_text(encoding='utf-8')+'<p>'+html.escape(text)+'</p></section>')
        atlas_legend=('Complete 3sTrace source-roster atlas, ordered by condition and identity. Left heatmaps reuse authenticated direct 1 s median positive valid moving-frame ln vigor, centred by the median finite baseline-bin medians; this differs from the G trial estimator and Figure 4 signed bout signal. Native managua_r saturates at ±0.25; undefined intensity is black. Green lines bound CS 0–10 s; ochre ticks show actual recorded reinforcers, including unpaired control events. Right dots show observed G trial outcomes; black bars are the exact Pre and late-training epoch medians used for the acquisition contrast. All cards use one outcome scale encompassing every observation. Undefined outcomes remain absent and contributing counts are explicit. No individual intervals or classes are inferred; these same-data displays do not validate learning. Timing-flagged identities remain excluded and listed separately. All 179 source-roster fish are shown, without filtering on illustrative-fish selection.')
        (OUT/'individual-atlas-legend.md').write_text(atlas_legend,encoding='utf-8')
        (OUT/'individual-atlas.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:14px Arial}section{display:block;width:100%;max-width:1100px}svg{width:100%;height:auto}</style>'+''.join(entries)+'<p>'+html.escape(atlas_legend)+'</p>',encoding='utf-8')
    for item in read(OUT/'render-start.json')['inputs']:verify(item)
    write(OUT/'render-complete.json',{'artifacts':artifacts,'atlas_sources':atlas_sources,'atlas_fish':len(scores),'presentation_profile':'nature-main',
        'render_verification':'pending actual saved SVG browser inspection; no Nature compliance claim',
        'scientific_status':'unfrozen assay-specific draft; shared assembly pointers untouched'})

if __name__=='__main__':main()
