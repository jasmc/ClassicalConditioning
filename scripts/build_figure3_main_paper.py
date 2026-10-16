"""Build and guard the Figure 3 main-paper draft; no classifier fit or freeze."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from classical_conditioning.analysis.review.individual_atlas import sha,save_json
from classical_conditioning.figures.individual_atlas import svg_bytes
from classical_conditioning.figures.layout_frames import embed_spec,assess
from classical_conditioning.figure_workspace import Snapshot,publish_state,state_lock

STATE=ROOT/'docs/paper/figures/figure-03/current-state.json'
ATLAS=ROOT/'docs/paper/figures/figure-03/exploratory-state.json'
NS='{http://www.w3.org/2000/svg}'
EXAMPLES=[('C','20240618_08','Larger acquisition contrast'),
          ('D','20240618_02','Intermediate contrast'),
          ('E','20240612_08','Near-zero contrast')]
COLOURS={'control':'#00aeef','delay':'#ec008c'}
BOXES={'heading':[0,0,183,9],'A':[0,9,104,44],'B':[104,9,79,44],
       'C':[0,53,61,96],'D':[61,53,61,96],'E':[122,53,61,96],
       'colourbar':[0,149,183,9]}


def inputs(out):
    ticket=json.loads((out/'start-ticket.json').read_text())
    for record in [ticket['parent_state'],*ticket['source_inputs'],*ticket.get('previous_draft_inputs',[])]:
        if sha(record['path'])!=record['sha256']:raise ValueError('Changed bound source: '+record['path'])
    return ticket,Path(ticket['source_inputs'][0]['path']).parent


def acquisition_scores(trials,inventory):
    """Descriptive training contrast; retain every inventory fish and all missingness."""
    keys=['experiment_id','condition_id','fish_id','trial_number']
    if trials.duplicated(keys).any():raise ValueError('Duplicate scheduled trial')
    if inventory.duplicated(['condition_id','fish_id']).any():raise ValueError('Duplicate inventory fish')
    rows=[]
    for fish in inventory.itertuples(index=False):
        data=trials.loc[trials.fish_id.eq(fish.fish_id)&trials.condition_id.eq(fish.condition_id)]
        if len(data)!=90 or set(data.trial_number)!=set(range(5,95)):raise ValueError('Incomplete scheduled inventory')
        y=data.response_logmedian-data.baseline_logmedian
        pre=y.loc[data.trial_number.between(5,14)].dropna()
        late=y.loc[data.trial_number.between(40,64)].dropna()
        a=float(pre.median()-late.median()) if len(pre) and len(late) else np.nan
        rows.append({'experiment_id':fish.experiment_id,'condition_id':fish.condition_id,'fish_id':fish.fish_id,
            'pre_median':float(pre.median()) if len(pre) else np.nan,
            'late_training_median':float(late.median()) if len(late) else np.nan,
            'acquisition_score':a,'pre_n':len(pre),'pre_scheduled':10,'late_training_n':len(late),
            'late_training_scheduled':25,'G_trials':int(y.notna().sum()),'estimable':bool(np.isfinite(a)),
            'reason':'observed descriptive contrast' if np.isfinite(a) else 'no defined G outcome in Pre or late Train',
            'status':'descriptive; no binary G classification or individual uncertainty claim'})
    return pd.DataFrame(rows)


def panel(letter,title,width,height):
    fig=plt.figure(figsize=(width/25.4,height/25.4))
    fig.text(1.7/width,1-4.5/height,letter.lower(),fontsize=8,weight='bold')
    fig.text(8/width,1-4.4/height,title,fontsize=7,weight='normal')
    return fig


def finish(fig,identity,w,h):
    return svg_bytes(fig,'fig3_main_'+identity,w,h)


def definition_panel():
    w,h=104,44;fig=panel('A','Outcome and acquisition contrast',w,h)
    ax=fig.add_axes([.08,.69,.84,.13]);ax.set_xlim(-15,25);ax.set_ylim(0,1);ax.axis('off')
    ax.broken_barh([(-15,15)],(.15,.38),facecolors='#d9e3e7')
    ax.broken_barh([(0,9)],(.15,.38),facecolors=COLOURS['delay'],alpha=.28)
    ax.plot([0,10],[.76,.76],color='#009d54',lw=2)
    ax.plot(9,.77,marker='v',color='#8055b0',ms=4)
    ax.text(-7.5,-.15,'Baseline −15 to 0 s',ha='center',fontsize=6)
    ax.text(4.5,-.15,'CR 0 to 9 s',ha='center',fontsize=6)
    ax.text(5,1.04,'CS',ha='center',fontsize=6)
    ax.text(12,.72,'US 9 s',fontsize=6)
    fig.text(.045,.51,'y = median ln(CR bout vigor) − median ln(baseline bout vigor)',fontsize=6)
    ax2=fig.add_axes([.07,.27,.86,.11]);ax2.set_xlim(4.5,94.5);ax2.set_ylim(0,1);ax2.axis('off')
    for lo,hi,colour in [(5,14,'#d9e3e7'),(15,64,'#edf0f1'),(65,94,'#f7f7f7')]:
        ax2.broken_barh([(lo-.5,hi-lo+1)],(.3,.45),facecolors=colour,edgecolors='#6f797f',lw=.4)
    ax2.broken_barh([(39.5,25)],(.3,.45),facecolors=COLOURS['delay'],alpha=.3)
    for x,label in [(9.5,'Pre 5–14'),(52,'Late Train 40–64'),(79.5,'Test')]:
        ax2.text(x,-.27,label,ha='center',fontsize=6)
    fig.text(.045,.12,'Acquisition score = median(y Pre) − median(y late Train)',fontsize=6)
    fig.text(.045,.047,'Positive = greater suppression; recovery is a separate property.',fontsize=6)
    return finish(fig,'A',w,h)


def score_panel(scores):
    w,h=79,44;fig=panel('B','Individual acquisition scores',w,h)
    ax=fig.add_axes([.23,.30,.72,.42]);ax.set_gid('G_acquisition_scores')
    finite=scores.loc[scores.estimable]
    for condition,y0 in [('control',0),('delay',1)]:
        sub=finite.loc[finite.condition_id.eq(condition)]
        jitter=np.array([(int(hashlib.sha256(x.encode()).hexdigest()[:8],16)/0xffffffff-.5)*.35 for x in sub.fish_id])
        ax.scatter(sub.acquisition_score,np.full(len(sub),y0)+jitter,s=13,color=COLOURS[condition],alpha=.8,linewidths=0)
        for letter,fish,_ in EXAMPLES:
            row=sub.loc[sub.fish_id.eq(fish)]
            if len(row):
                x=float(row.acquisition_score.iloc[0]);dy={'C':10,'D':-12,'E':10}[letter]
                y=float(y0+jitter[list(sub.fish_id).index(fish)])
                ax.scatter([x],[y],s=31,facecolors='none',edgecolors='#222222',lw=.6)
                ax.annotate(letter,(x,y),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=6)
    ax.axvline(0,color='#7b858a',ls=':',lw=.6)
    ax.set_yticks([0,1],['Control','Delay']);ax.set_ylim(-.65,1.65)
    lo=min(-.25,float(finite.acquisition_score.min())-.04);hi=max(.25,float(finite.acquisition_score.max())+.04)
    ax.set_xlim(lo,hi);ax.set_xlabel('Pre − late Train (Δ median ln vigor)',fontsize=6,labelpad=3)
    ax.tick_params(labelsize=6,length=2,width=.5,pad=2)
    ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_linewidth(.5)
    fig.text(.23,.80,'G outcome · one point per fish',fontsize=6)
    counts=[]
    for condition in ['control','delay']:
        sub=scores.loc[scores.condition_id.eq(condition)];counts.append(f'{condition.capitalize()} {int(sub.estimable.sum())}/{len(sub)}')
    fig.text(.05,.065,'Estimable: '+ '; '.join(counts),fontsize=6)
    return finish(fig,'B',w,h)


def example_panel(letter,fish,title,trials,arrays,scores):
    w,h=61,96;fig=panel(letter,title,w,h)
    data=trials.loc[trials.fish_id.eq(fish)].sort_values('trial_number');row=scores.loc[scores.fish_id.eq(fish)].iloc[0]
    fig.text(.22,.91,fish+' · Delay',fontsize=6,color=COLOURS['delay'],weight='bold')
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black')
    ax=fig.add_axes([.22,.53,.74,.32]);ax.set_gid('heatmap')
    ax.imshow(arrays[fish+'__heat'],origin='upper',aspect='auto',extent=[-15,25,94.5,4.5],
        cmap=cmap,vmin=-.25,vmax=.25,interpolation='nearest',rasterized=True)
    for boundary in [14.5,64.5]:ax.axhline(boundary,color='white',lw=.5)
    for t in [0,10]:ax.axvline(t,color='#00a05b',lw=.65)
    ax.plot([9,9],[14.5,64.5],color='#8055b0',lw=.65,ls=':')
    for catch in [25,39,53,59,65]:ax.plot(-14,catch,marker='<',color='white',ms=2.5)
    for trial in data.itertuples(index=False):
        for event in json.loads(trial.actual_us_s):ax.plot(event,trial.trial_number,'|',color='#a87820',ms=1.8,mew=.5)
    ax.set(xticks=[-15,0,9,25],yticks=[5,15,40,65,85,94]);ax.tick_params(labelsize=6,length=2,width=.5,pad=1)
    ax.set_xlabel('Time from CS onset (s)',fontsize=6,labelpad=1);ax.set_ylabel('CS trial',fontsize=6,labelpad=1)
    ax.spines[:].set_linewidth(.5)
    strip=fig.add_axes([.22,.41,.74,.055]);strip.set_gid('movement_fraction_strip')
    strip.imshow(np.vstack([data.baseline_fraction_moving,data.response_fraction_moving]),cmap='Greys',vmin=0,vmax=1,
        extent=[4.5,94.5,1.5,-.5],aspect='auto',interpolation='nearest',rasterized=True)
    strip.set_yticks([0,1],['Move B','Move CR'],fontsize=6);strip.set_xticks([]);strip.tick_params(length=0,pad=1)
    strip.spines[:].set_visible(False)
    scalar=fig.add_axes([.27,.16,.69,.20]);scalar.set_gid('G_observed_trials')
    y=(data.response_logmedian-data.baseline_logmedian).to_numpy(float)
    scalar.scatter(data.trial_number,y,s=5,color=COLOURS['delay'],alpha=.65,linewidths=0)
    for start in range(5,95,5):
        mask=data.trial_number.between(start,start+4)&np.isfinite(y)
        if mask.any():scalar.plot([start,start+4],[np.median(y[mask])]*2,color=COLOURS['delay'],lw=1)
    scalar.axhline(0,color='#6e797f',lw=.5,ls=':')
    scalar.axvspan(4.5,14.5,color='#d9e3e7',alpha=.5,zorder=-1)
    scalar.axvspan(39.5,64.5,color=COLOURS['delay'],alpha=.07,zorder=-1)
    for boundary in [14.5,64.5]:scalar.axvline(boundary,color='#89959b',lw=.5,ls=':')
    finite=y[np.isfinite(y)]
    if len(finite) and (finite.min()<-.6 or finite.max()>.6):raise ValueError('Expand shared G axes to retain observations')
    scalar.set_ylim(-.6,.6);scalar.set_yticks([-.5,0,.5])
    for lo,hi in [(5,14),(40,64)]:
        epoch=y[data.trial_number.between(lo,hi).to_numpy()];epoch=epoch[np.isfinite(epoch)]
        if len(epoch):scalar.plot([lo,hi],[np.median(epoch)]*2,color='black',lw=1.,zorder=4)
    scalar.set_xlim(4.5,94.5);scalar.set_xticks([5,40,65,94]);scalar.tick_params(labelsize=6,length=2,width=.5,pad=1)
    scalar.set_ylabel('Median ln vigor\n(CR − baseline)',fontsize=6,labelpad=1)
    scalar.set_xlabel('Scheduled CS trial',fontsize=6,labelpad=1)
    scalar.set_title('G log-median outcome',fontsize=6,pad=3)
    scalar.spines[['top','right']].set_visible(False);scalar.spines[['left','bottom']].set_linewidth(.5)
    fig.text(.10,.045,f'Score {row.acquisition_score:+.2f} · G trials {row.G_trials}/90',fontsize=6)
    fig.text(.10,.018,f'Pre {row.pre_n}/10 · late Train {row.late_training_n}/25',fontsize=6)
    return finish(fig,letter,w,h)


def colourbar_panel():
    fig=plt.figure(figsize=(183/25.4,9/25.4));ax=fig.add_axes([.33,.57,.31,.17])
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black')
    bar=fig.colorbar(ScalarMappable(norm=Normalize(-.25,.25),cmap=cmap),cax=ax,orientation='horizontal',
        ticks=[-.25,0,.25],extend='both');bar.ax.tick_params(labelsize=6,length=2,width=.5,pad=1)
    fig.text(.015,.51,'Heatmaps: Δ ln vigor',fontsize=6)
    fig.text(.72,.51,'■',fontsize=8,color='black');fig.text(.738,.51,'Undefined intensity',fontsize=6)
    return finish(fig,'colourbar',183,9)


def assemble(panels):
    root=ET.Element(NS+'svg',{'width':'183mm','height':'158mm','viewBox':'0 0 183 158'})
    heading=ET.SubElement(root,NS+'g',{'id':'fig3_main_heading'})
    for x,y,text,size,weight in [(1.7,4.8,'Figure 3',3.5,'bold'),(23,4.7,'Individual evidence of conditioned suppression',3,'normal'),(171,4.7,'Draft',2.5,'normal')]:
        n=ET.SubElement(heading,NS+'text',{'x':str(x),'y':str(y),'font-family':'Arial','font-size':str(size),'font-weight':weight});n.text=text
    allocations=[{'id':'heading','kind':'shared','box_mm':BOXES['heading'],'group_ids':['fig3_main_heading']}]
    backgrounds=[];clips=[];roles={}
    for name,data in panels.items():
        sub=ET.fromstring(data);metadata=sub.find(NS+'metadata[@id="layout-frames"]');spec=json.loads(metadata.text);sub.remove(metadata)
        x,y,w,h=BOXES[name];sub.set('x',str(x));sub.set('y',str(y));sub.set('width',str(w));sub.set('height',str(h));root.append(sub)
        groups=spec['allocations'][0]['group_ids'];allocations.append({'id':name,'kind':'shared' if name=='colourbar' else 'panel','box_mm':BOXES[name],'group_ids':groups})
        backgrounds.extend(spec['background_ids']);clips.extend(spec['scientific_clips']);roles[name]={'owner_groups':groups,'box_mm':BOXES[name]}
    spec={'scope':'figure','final_size_mm':[183,158],'inset_mm':1.,'allocations':allocations,
        'background_ids':backgrounds,'scientific_clips':clips}
    return embed_spec(ET.tostring(root),spec),roles


def legend(scores):
    return '\n\n'.join([
        'Figure 3. Individual acquisition of conditioned response suppression. A single descriptive analysis using the legacy angular-speed metric and the Figure 2 G outcome.',
        '(a) Trial outcome y = median ln positive CR-bout vigor [0,9) s minus median ln positive baseline-bout vigor [−15,0) s. Acquisition score = median(y, Pre trials 5–14) − median(y, late Train 40–64); positive indicates greater suppression. Test trials do not enter the score; recovery is a separate property.',
        '(b) One point per fish, including all 28 control and 29 Delay fish; jitter separates observations. Zero is a direction reference, not a learner cutoff. All scores are mathematically estimable, with 4–10 defined Pre and 5–25 late-training trials. This does not establish adequate individual precision.',
        '(c–e) Deliberately selected Delay illustrations of larger, intermediate and near-zero contrasts, identified in b. Descriptive magnitudes are not learner categories. All 57 fish remain in the supplementary atlas. Heatmaps show 1 s median bout-frame ln vigor minus median finite baseline-bin medians: managua_r saturates at ±0.25; black indicates undefined intensity. Green marks CS; violet, nominal training US; ochre, actual US; white, phases/catches. Move B/CR strips show fraction moving (white 0, black 1). Trial dots show y; pink bars show five-trial medians; black bars show the two score-epoch medians. Shared axes retain all observations. No-bout vigor is undefined; movement remains separate. These same-data illustrations do not provide independent validation or binary learner assignments. Group behavioral comparisons remain in Figure 4.'
    ])+'\n'


def build(out):
    ticket,source=inputs(out)
    if STATE.is_file() and Path(json.loads(STATE.read_text())['draft']['working_svg']['path']).parent==out:
        raise ValueError('Preserve published artwork; build a new guarded candidate directory')
    paths=[Path(r['path']) for r in ticket['source_inputs']]
    paths += [Path(__file__),ROOT/'src/classical_conditioning/figures/individual_atlas.py',
        ROOT/'src/classical_conditioning/figures/layout_frames.py',ROOT/'src/classical_conditioning/figures/layout_frames_browser.js',
        ROOT/'src/classical_conditioning/figure_workspace.py',ROOT/'configs/paper-figures/layout-frames.json']
    snapshot=Snapshot(paths)
    trials=pd.read_parquet(source/'trial-data.parquet');inventory=pd.read_csv(source/'fish-inventory.csv')
    if set(trials.metric_id)!={'legacy_distal_angular_speed'}:raise ValueError('Unexpected metric')
    exported=pd.read_csv(source/'g-logmedian-outcomes.csv')
    np.testing.assert_allclose(exported.response_minus_baseline_logmedian,
        trials.response_logmedian-trials.baseline_logmedian,rtol=1e-12,atol=1e-12,equal_nan=True)
    style=json.loads((source/'reference-style.json').read_text());assert style['condition_colors']==COLOURS and style['heatmap_limits']==[-.25,.25]
    scores=acquisition_scores(trials,inventory)
    scores.to_csv(out/'acquisition-scores.csv',index=False)
    save_json(out/'analysis-spec.json',ticket['analysis_choice'])
    with plt.rc_context({'font.family':'Arial','svg.fonttype':'none','svg.hashsalt':'fig3-main-paper-20261016',
        'font.size':7,'axes.unicode_minus':True,'axes.spines.top':False,'axes.spines.right':False}):
        panels={'A':definition_panel(),'B':score_panel(scores)}
        with np.load(source/'heatmaps.npz') as arrays:
            for letter,fish,title in EXAMPLES:panels[letter]=example_panel(letter,fish,title,trials,arrays,scores)
        panels['colourbar']=colourbar_panel();svg,roles=assemble(panels)
    caption=legend(scores);page='<!doctype html><meta charset="utf-8"><title>Figure 3 · main-paper draft</title><style>body{margin:0;background:#eef0f1;color:#152028;font:9pt/1.5 Georgia,serif}main{width:min(100%,1000px);margin:auto;background:white;padding:22px;box-sizing:border-box}svg{display:block;width:183mm;max-width:100%;height:auto;margin:auto}section{max-width:850px;margin:26px auto 0}p{margin:0 0 12px}@media print{body,main{background:white;padding:0}section{break-before:page}}</style><main>'+svg.decode().split('?>')[-1]+'<section>'+''.join('<p>'+html.escape(p)+'</p>' for p in caption.strip().split('\n\n'))+'</section></main>'
    (out/'figure3.svg').write_bytes(svg);(out/'review.html').write_text(page);(out/'legend.md').write_text(caption)
    for record in ticket['source_inputs']:snapshot.hashes[str(Path(record['path']).resolve())]=record['sha256']
    snapshot.check();save_json(out/'build-inputs.json',snapshot.hashes)
    save_json(out/'render-outputs.json',{name:sha(out/name) for name in ['figure3.svg','review.html','legend.md','acquisition-scores.csv','analysis-spec.json']})
    save_json(out/'draft-registry.json',{'scope':'Figure 3 main-paper draft','roles':roles,
        'sources':ticket['source_inputs'],'observed_metric':'legacy_distal_angular_speed',
        'A':ticket['analysis_choice'],'B':{'data':str(out/'acquisition-scores.csv'),'quantity':'Pre minus late-training medians of G trial outcome','inference':'none'},
        'C_E':{'examples':[{'panel':p,'fish_id':f,'selection_reason':r,'heatmap_array':f+'__heat'} for p,f,r in EXAMPLES],
            'trial_data':str(source/'trial-data.parquet'),'heatmaps':str(source/'heatmaps.npz'),
            'trial_quantity':'response_logmedian minus baseline_logmedian','moving_strips':'baseline_fraction_moving and response_fraction_moving'},
        'relationships':[{'type':'presentation_style','scope':['C','D','E'],'palette':'managua_r','limits':[-.25,.25],'undefined':'black'},
            {'type':'scientific','scope':['A','B','C','D','E'],'property':'One G outcome and one temporal contrast throughout; no binary labels'},
            {'type':'layout','scope':['C','D','E'],'property':'native 61 x 96 mm panels; shared G y limits [-0.6, 0.6] retain all three example observations; aligned axis positions; no scaling of legacy artwork'}],
        'limitations':['Retrospective descriptive G score; no calibrated individual class/uncertainty','Delay/control only; Trace inputs pending','Same-data illustrations; no independent validation']})
    dest=out/'source-snapshots';dest.mkdir(exist_ok=True)
    for i,p in enumerate(paths):shutil.copyfile(p,dest/f'{i:02d}-{p.name}')
    print(json.dumps({'figure':str(out/'figure3.svg'),'fish_scores':len(scores),'estimable':int(scores.estimable.sum()),'panels':'A-E'}))


def publish(out):
    ticket,source=inputs(out);bound=json.loads((out/'build-inputs.json').read_text())
    for p,digest in bound.items():
        if sha(p)!=digest:raise ValueError('Changed build input '+p)
    for name,digest in json.loads((out/'render-outputs.json').read_text()).items():
        if sha(out/name)!=digest:raise ValueError('Manual output edit: '+name)
    layout=json.loads((out/'layout-evidence.json').read_text())
    if assess((out/'figure3.svg').read_bytes(),layout['browser'])['status']!='passed':raise ValueError('Resolve exact-render layout')
    visual=json.loads((out/'visual-review.json').read_text())
    if visual.get('svg_sha256')!=sha(out/'figure3.svg') or visual.get('status')!='reviewed':raise ValueError('Visual review missing/stale')
    artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json' and not p.name.endswith('.lock')}
    save_json(out/'artifact-manifest.json',{'scope':'Figure 3 main-paper draft; not frozen','artifacts':artifacts})
    def record(name):return {'path':str(out/name),'sha256':sha(out/name)}
    snapshot=Snapshot([STATE,ATLAS,*[r['path'] for r in ticket.get('previous_draft_inputs',[])],*bound,*[out/n for n in artifacts],out/'artifact-manifest.json'])
    if snapshot.hashes[str(STATE.resolve())]!=ticket['state_before']:raise ValueError('Main Figure 3 state changed')
    for r in ticket.get('previous_draft_inputs',[]):
        if snapshot.hashes[str(Path(r['path']).resolve())]!=r['sha256']:raise ValueError('Previous working copy changed')
    for p,digest in bound.items():
        if snapshot.hashes[str(Path(p).resolve())]!=digest:raise ValueError('Build source changed before publication')
    protected={p:{'before':h,'after':sha(p),'changed_outside_this_task':sha(p)!=h} for p,h in ticket['protected'].items()}
    state={'schema_version':1,'figure_id':'fig3','default_display':'draft','default_display_format':'html-svg-and-legend',
        'whole_figure_frozen':False,'scientific_selection':{'status':'unfrozen; single descriptive G acquisition path','records':[]},
        'draft':{'status':'Main-paper draft A-E; one G acquisition path; descriptive, validation unresolved','working_svg':record('figure3.svg'),
            'review':record('review.html'),'legend':record('legend.md'),'semantic_registry':record('draft-registry.json'),
            'layout_frames':record('layout-evidence.json'),'data':{'acquisition_scores':record('acquisition-scores.csv')},
            'assembly_scale':{'final_width_mm':183,'final_height_mm':158},'source_atlas':ticket['parent_state'],'previous_draft_state':record('previous-current-state.json'),
            'analysis_spec':record('analysis-spec.json'),'verification':{'visual_review':record('visual-review.json'),'artifact_manifest':record('artifact-manifest.json')},
            'limitations':['No final learner method selected; G acquisition score is descriptive only','Retrospective observations and examples do not validate learner categories','Delay/control pilot; full Trace inputs pending']},
        'other_figure_records':protected}
    publish_state(STATE,state,snapshot);print(STATE)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['build','publish']);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if not os.path.ismount('/Volumes/JOAQUIM') or not out.is_relative_to('/Volumes/JOAQUIM') or not out.is_dir():raise ValueError('Use explicit existing mounted JOAQUIM directory')
    if args.stage=='build':
        with state_lock(out/'build'):build(out)
    else:publish(out)
