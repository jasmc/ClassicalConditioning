"""Assay-scoped A–E adaptation of the descriptive Figure 3 structure."""
import io,json,html,os
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme,style_axes
from classical_conditioning.figures.individual_atlas import svg_bytes
from classical_conditioning.figures.layout_frames import embed_spec
from classical_conditioning.figures.pooled_log_heatmap import read_windows
from analyze_figures34_3strace import OUT,BASE,ROOT,PROJECT,read,bound,verify,write

DEST=OUT/'figure3-a-e';NS='{http://www.w3.org/2000/svg}'
EXAMPLES=[('C','20230419_10','Larger contrast'),('D','20230309_11','Intermediate contrast'),('E','20230802_09','Near-zero contrast')]
BOXES={'A':[0,0,104,56],'B':[104,0,79,56],'C':[0,56,61,144],'D':[61,56,61,144],'E':[122,56,61,144],'key':[0,200,183,12]}
ET.register_namespace('',NS[1:-1]);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def build():
    ticket=read(DEST/'start-ticket.json')
    for x in [ticket['parent_state'],ticket['shared_figure3_state'],*ticket['inputs']]:verify(x)
    state=read(OUT/'assay-draft-state.json');manifest=read(OUT/'artifact-manifest.json')
    known={x['path']:x for x in manifest['files']}
    verify(known[str(OUT/'render-complete.json')])
    known.update({x['path']:x for x in read(OUT/'render-complete.json')['atlas_sources']})
    scores=pd.read_csv(OUT/'acquisition-scores.csv');trials=pd.read_parquet(OUT/'observed-g-trial-outcomes.parquet')
    assert len(scores)==179 and scores.fish_id.nunique()==179
    inputs=[bound(Path(__file__))];details=[];heat={};events={}
    for _,fish,_ in EXAMPLES:
        p=BASE/'fish'/f'{fish}.json';rec=read(p);verify(known[str(p)]);inputs.append(bound(p))
        for item in rec['outputs']:verify(item);inputs.append(item)
        for item in rec['window_sources']:verify(item);inputs.append(item)
        bins=pd.read_parquet(BASE/'fish'/f'{fish}-bins.parquet')
        matrix=bins.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').sort_index()
        assert matrix.shape==(90,40) and list(matrix.index)==list(range(5,95))
        heat[fish]=matrix.to_numpy()
        protocol=pd.read_parquet(PROJECT/'Processed data'/fish/'stimulus_events.parquet')
        cycles=protocol[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
        reinforcers=protocol.loc[protocol.Type.eq('Reinforcer'),'Beg'].to_numpy(np.int64)
        intervals=[(int(t)-15001,int(t)+13001) for t in cycles.iloc[4:94].Beg]
        refs=rec['window_sources'];fp=Path(next(x['path'] for x in refs if 'frame_activity' in x['path']));mp=Path(next(x['path'] for x in refs if 'movement_state' in x['path']))
        frames=read_windows(fp,['FrameID','AbsoluteTime','FrameStep','DeltaTimeMs','legacy_distal_angular_speed_rad_per_ms'],intervals)
        moves=read_windows(mp,['FrameID','AbsoluteTime','valid','moving'],intervals)
        np.testing.assert_array_equal(frames[['FrameID','AbsoluteTime']],moves[['FrameID','AbsoluteTime']])
        time=frames.AbsoluteTime.to_numpy(np.int64);duration=frames.DeltaTimeMs.to_numpy(float)
        valid=moves.valid.to_numpy(bool)&frames.FrameStep.eq(1).to_numpy()&np.isfinite(frames.legacy_distal_angular_speed_rad_per_ms.to_numpy())
        moving=moves.moving.to_numpy(bool)
        events[fish]=[]
        for number,onset in enumerate(cycles.iloc[4:94].Beg,start=5):
            row={'fish_id':fish,'trial_number':number}
            for prefix,lo,hi in [('baseline',-15000,0),('response',0,13000)]:
                a,b=np.searchsorted(time,[int(onset)+lo,int(onset)+hi]);v=valid[a:b];m=moving[a:b];w=duration[a:b]
                denominator=w[v].sum();row[prefix+'_fraction_moving']=float(w[v&m].sum()/denominator) if denominator>0 else np.nan
                row[prefix+'_coverage']=min(1.,float(denominator/(hi-lo)))
            lat=(reinforcers-int(onset))/1000
            actual=[float(x) for x in lat[(lat>=-20)&(lat<20)]];row['actual_us_s']=json.dumps(actual);events[fish].append((number,actual));details.append(row)
    data=pd.DataFrame(details);data.to_csv(DEST/'example-movement-and-events.csv',index=False)
    profile=load_presentation_profile('nature-main');inputs.extend(bound(ROOT/x) for x in profile.inputs)
    write(DEST/'build-inputs.json',{'inputs':inputs,'example_selection':{'C':'existing author-selected Figure 1 illustration, larger observed contrast','D':'nearest 3sTrace median acquisition contrast','E':'smallest absolute 3sTrace acquisition contrast'},'status':'same-data descriptive illustrations; no categorical validation'})
    selected=trials[trials.fish_id.isin([x[1] for x in EXAMPLES])]
    limit=max(.6,np.ceil(selected.legacy_log_median_difference.abs().max()*10)/10)
    panels={}
    with mpl.rc_context():
        theme=apply_theme(profile=profile);mpl.rcParams.update({'figure.autolayout':False,'figure.constrained_layout.use':False,'savefig.bbox':None})
        styles=profile.contract['styles'];anno=styles['annotation']['font_size_pt'];letter_size=styles['panel_letter']['font_size_pt']
        def panel(letter,title,w,h):
            fig=plt.figure(figsize=(w/25.4,h/25.4))
            fig.text(2/w,1-4.5/h,letter.lower(),fontsize=letter_size,fontweight='bold')
            fig.text(8/w,1-4.5/h,title,fontsize=theme.axes_titlesize)
            return fig
        fig=panel('A','Outcome and acquisition contrast',104,56)
        ax=fig.add_axes([.09,.70,.85,.15]);ax.set_xlim(-15,22);ax.set_ylim(-.65,1.4);ax.axis('off')
        ax.broken_barh([(-15,15)],(.1,.35),facecolors='#d9e3e7');ax.broken_barh([(0,13)],(.1,.35),facecolors='#f15a29',alpha=.22)
        ax.plot([0,10],[1,1],color=theme.cs_color,lw=theme.cs_linewidth);ax.plot(13,1,marker='v',color=theme.us_color,ms=3)
        for x,y,text in [(-7.5,-.45,'Baseline −15 to 0 s'),(6.5,-.45,'Response 0 to 13 s'),(4,1.15,'CS'),(17,1.0,'US 13 s')]:ax.text(x,y,text,ha='center',fontsize=anno)
        fig.text(.06,.57,'y = median ln(response vigor) − median ln(baseline vigor)',fontsize=anno)
        ax=fig.add_axes([.09,.30,.85,.14]);ax.set_xlim(4.5,94.5);ax.set_ylim(-.65,1);ax.axis('off')
        for lo,hi,col in [(5,14,'#d9e3e7'),(15,64,'#ededed'),(65,94,'#f5f5f5')]:ax.broken_barh([(lo-.5,hi-lo+1)],(.2,.5),facecolors=col)
        ax.broken_barh([(39.5,25)],(.2,.5),facecolors='#f15a29',alpha=.3)
        for x,text in [(9.5,'Pre 5–14'),(52,'Late Train 40–64'),(81,'Test')]:ax.text(x,-.3,text,ha='center',fontsize=anno)
        fig.text(.06,.17,'Score = median(y Pre) − median(y late Train)',fontsize=anno)
        fig.text(.06,.07,'Positive = greater suppression; no learner cutoff.',fontsize=anno)
        panels['A']=svg_bytes(fig,'trace3_A',104,56)
        fig=panel('B','Individual acquisition scores',79,56);ax=fig.add_axes([.20,.25,.75,.54])
        for condition,y,color in [('control',0,'#00aeef'),('trace',1,'#f15a29')]:
            part=scores[scores.condition_id.eq(condition)].sort_values('fish_id');jitter=np.linspace(-.18,.18,len(part))
            ax.scatter(part.acquisition_contrast,y+jitter,s=7,color=color,alpha=.7,linewidths=0)
            for letter,fish,_ in EXAMPLES:
                if fish in list(part.fish_id):
                    row=part[part.fish_id.eq(fish)].iloc[0];position=y+jitter[list(part.fish_id).index(fish)]
                    ax.scatter(row.acquisition_contrast,position,s=21,facecolors='none',edgecolors='black',linewidth=.6)
                    ax.annotate(letter,(row.acquisition_contrast,position),xytext=(0,10 if letter!='D' else -14),textcoords='offset points',ha='center',fontsize=anno)
        ax.axvline(0,color='#888888',lw=.5,ls=':');ax.set_yticks([0,1],['Control','3sTrace']);ax.set_ylim(-.65,1.65)
        ax.set_xlim(scores.acquisition_contrast.min()-.04,scores.acquisition_contrast.max()+.04)
        style_axes(ax,theme=theme,xlabel='Pre − late Train (Δ median ln vigor)')
        fig.text(.09,.055,'Control n = 90; 3sTrace n = 89',fontsize=anno)
        panels['B']=svg_bytes(fig,'trace3_B',79,56)
        for letter,fish,title in EXAMPLES:
            fig=panel(letter,title,61,144);row=scores[scores.fish_id.eq(fish)].iloc[0];d=trials[trials.fish_id.eq(fish)].sort_values('trial_number')
            fig.text(14/61,1-11/144,fish+' · 3sTrace',fontsize=anno)
            cmap=mpl.colormaps[theme.intensity_cmap].copy();cmap.set_bad(theme.heatmap_bad_color)
            ax=fig.add_axes([14/61,1-67/144,43/61,50/144]);ax.imshow(heat[fish],origin='upper',extent=[-20,20,94.5,4.5],aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25)
            for t in [0,10]:ax.axvline(t,color=theme.cs_color,lw=theme.cs_linewidth)
            ax.plot([13,13],[14.5,64.5],color=theme.us_color,lw=theme.us_linewidth,ls=':')
            for b in [14.5,64.5]:ax.axhline(b,color='white',lw=.5)
            for number,actual in events[fish]:ax.plot(actual,np.full(len(actual),number),'|',color='#a87820',ms=2,mew=.5)
            ax.set_yticks([5,15,40,65,94]);ax.set_xticks([-20,0,13,20]);style_axes(ax,theme=theme,xlabel='Time from CS (s)',ylabel='CS trial')
            strip=fig.add_axes([14/61,1-87/144,43/61,5/144]);sub=data[data.fish_id.eq(fish)].sort_values('trial_number')
            strip.imshow(sub[['baseline_fraction_moving','response_fraction_moving']].to_numpy().T,cmap='Greys',vmin=0,vmax=1,extent=[4.5,94.5,1.5,-.5],aspect='auto',interpolation='nearest')
            strip.set_yticks([0,1],['Move B','Move CR'],fontsize=anno);strip.set_xticks([]);strip.tick_params(length=0,pad=1);strip.spines[:].set_visible(False)
            fig.text(2/61,1-94/144,f'Score {row.acquisition_contrast:+.3f} · Pre {row.pre_n}/10 · late {row.late_n}/25',fontsize=anno)
            ax=fig.add_axes([14/61,1-130/144,43/61,25/144]);y=d.legacy_log_median_difference
            ax.scatter(d.trial_number,y,s=5,color='#f15a29',linewidths=0)
            for lo in range(5,95,5):
                yy=y[d.trial_number.between(lo,lo+4)].median()
                if np.isfinite(yy):ax.plot([lo,lo+4],[yy]*2,color='#f15a29',lw=1)
            ax.axhline(0,color='#888888',lw=.5,ls=':')
            for lo,hi,value in [(5,14,row.pre_median),(40,64,row.late_median)]:ax.plot([lo,hi],[value]*2,color='black',lw=1)
            ax.set_xlim(4.5,94.5);ax.set_ylim(-limit,limit);ax.set_yticks([-limit,0,limit]);ax.set_xticks([5,40,65,94])
            style_axes(ax,theme=theme,xlabel='Scheduled CS trial',ylabel='CR − baseline\nmedian ln vigor');ax.set_title('Observed trial outcome',fontsize=anno,pad=3)
            panels[letter]=svg_bytes(fig,'trace3_'+letter,61,144)
        fig=plt.figure(figsize=(183/25.4,12/25.4));ax=fig.add_axes([.35,.55,.30,.18]);bar=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-.25,.25),cmap=cmap),cax=ax,orientation='horizontal',ticks=[-.25,0,.25]);bar.ax.tick_params(labelsize=anno)
        fig.text(.02,.58,'Heatmaps: Δ ln vigor',fontsize=anno);fig.text(.74,.58,'Black: undefined intensity',fontsize=anno)
        panels['key']=svg_bytes(fig,'trace3_key',183,12)
    root=ET.Element(NS+'svg',{'width':'183mm','height':'212mm','viewBox':'0 0 183 212'});alloc=[];bg=[];clips=[]
    for name,raw in panels.items():
        sub=ET.fromstring(raw);meta=sub.find(NS+'metadata[@id="layout-frames"]');spec=json.loads(meta.text);sub.remove(meta)
        x,y,w,h=BOXES[name];sub.set('x',str(x));sub.set('y',str(y));sub.set('width',str(w));sub.set('height',str(h));root.append(sub)
        alloc.append({'id':name,'kind':'shared' if name=='key' else 'panel','box_mm':BOXES[name],'group_ids':spec['allocations'][0]['group_ids']});bg+=spec['background_ids'];clips+=spec['scientific_clips']
    svg=embed_spec(ET.tostring(root),{'scope':'figure','final_size_mm':[183,212],'inset_mm':profile.layout['selected_inset_mm'],'allocations':alloc,'background_ids':bg,'scientific_clips':clips})
    legend=('Figure 3, 3sTrace — unfrozen descriptive draft. (a) Observed trial outcome y: median ln positive valid moving-frame legacy distal angular speed during [0,13) s minus its median during [−15,0) s. Score: median y on Pre trials 5–14 minus median y on late Train trials 40–64; positive indicates greater suppression. (b) All 179 source-roster fish: 90 Control, 89 3sTrace; jitter is visual only, and zero is not a learner cutoff. C–E identify the illustrated fish. (c–e) Larger (existing Figure 1 illustration 20230419_10), intermediate (nearest 3sTrace median contrast, 20230309_11), and near-zero (smallest absolute contrast, 20230802_09) examples, selected retrospectively without filtering the population. Heatmaps retain authenticated direct 1 s frame-log medians, baseline-bin centred, at managua_r ±0.25; black denotes undefined intensity. Green lines bound CS 0–10 s; violet marks the nominal paired-training US at 13 s; ochre ticks mark actual reinforcers. Move B/CR strips show duration-weighted fraction moving among eligible measured time during baseline/response windows (white 0, black 1), not tracking completeness. Outcome dots and orange five-trial medians use the G outcome; black bars are the exact score-epoch medians. Shared outcome axes retain all illustrated observations. All-fish data and the complete atlas remain preserved. Coverage counts describe mathematical estimability, not an approved scientific eligibility rule. No learner classes, individual uncertainty, independent validation or scientific freeze is claimed; Gate L and primary-cohort approval remain unresolved.')
    (DEST/'figure3.svg').write_bytes(svg);(DEST/'legend.md').write_text(legend,encoding='utf-8')
    (DEST/'review.html').write_text('<!doctype html><meta charset="utf-8"><style>body{margin:12px;background:white;font:14px Arial;color:black}svg{max-width:100%;height:auto}p{max-width:1100px}</style>'+svg.decode()+'<p>'+html.escape(legend)+'</p>',encoding='utf-8')
    for item in inputs:verify(item)
    for x in [ticket['parent_state'],ticket['shared_figure3_state'],*ticket['inputs']]:verify(x)
    write(DEST/'build-complete.json',{'artifacts':[bound(DEST/x) for x in ['figure3.svg','review.html','legend.md','example-movement-and-events.csv']], 'status':'built; visual verification and scoped publication pending','examples':EXAMPLES,'scientific_status':'unfrozen; no shared pointer change'})

if __name__=='__main__':build()
