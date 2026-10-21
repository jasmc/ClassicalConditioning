"""Match expanded Trace panels to the selected Figure 2 first column.

New vectors use one Nature profile. A/C/F/G/I remain authenticated source
subtrees; D data remain the exact selected frozen values with updated joint
D/E correction. Publication is a separate guarded operation after review.
"""
from pathlib import Path
import argparse, base64, copy, hashlib, io, json, re, sys
import xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot, publish_state, tag, sha
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme, style_axes
from classical_conditioning.figures.export import FigureProvenance, FigureMode, export_matplotlib_figure
from classical_conditioning.figures.layout_frames import load_policy
from main_paper_figures import render as render_page
from analyze_figure2_trace3_matched import OUT, D_SOURCE, EXCLUDED

STATE=REPO/'docs/paper/figures/figure-02/current-state.json'
LEGEND=STATE.parent/'legend.md'
SOURCE=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/reviews/figure2-current-minimal.svg')
REGISTRY=Path('J:/ClassicalConditioning Outputs/current/reviews/nature-main-20261016/figure-02-semantic-registry.json')
XLINK='{http://www.w3.org/1999/xlink}href'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def encode(o):return (json.dumps(o,indent=2,allow_nan=False)+'\n').encode('utf-8')
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
def write_new(p,data):
    with Path(p).open('xb') as f:f.write(data)
def glyph(p,maximum=4):return '*'*min(maximum,4 if p<.0001 else 3 if p<.001 else 2 if p<.01 else 1 if p<.05 else 0)


def block_panel(panel,width,profile,theme,tests,*,y_limits=(-.75,.75),y_ticks=None):
    height=43.26647836837704
    data=pd.read_parquet(D_SOURCE if panel=='D' else OUT/'E-fish-blocks.parquet')
    conditioned='delay' if panel=='D' else 'trace'
    colors={'control':'#00aeef','delay':'#ec008c','trace':'#f15a29'}
    fig=plt.figure(figsize=(width/25.4,height/25.4),layout=None)
    fig.patch.set_gid('canvas-'+panel)
    lefts=[12.137128323139343,27.533926805143423] if panel=='D' else [13.84712433660092,29.2439147441586]
    axes=[fig.add_axes([x/width,1-(19.6149649590255+13.782038393992613)/height,13.04813/width,13.782038393992613/height]) for x in lefts]
    mappings={};extra=[]
    def identify(artist,name,role,source=None):
        artist.set_gid(name);mappings[name]={'scientific_role':role,'panel':panel,'data_source':source or ref(OUT/'DE-statistics-Holm36.csv')}
        return artist
    for ax,c in zip(axes,['control',conditioned]):
        identify(ax.patch,panel+'-'+c+'-data-rectangle','background.data_axes',ref(D_SOURCE if panel=='D' else OUT/'E-fish-blocks.parquet'))
        sub=data.loc[data.Eligible&data.condition_id.eq(c)]
        wide=sub.pivot(index='fish_id',columns='Selected block order',values='Fish median response / baseline').reindex(columns=[0,1,2])
        for rid,row in wide.iterrows():
            line,=ax.plot(range(3),row,color=colors[c],lw=.5,alpha=.3,zorder=1)
            identify(line,f'{panel}-{c}-fish-{rid}','trajectory.fish',{'artifact':ref(D_SOURCE if panel=='D' else OUT/'E-fish-blocks.parquet'),'fish_id':rid,'condition_id':c,'field':'Fish median response / baseline'})
            points=ax.scatter(range(3),row,s=4,color=colors[c],alpha=.3,linewidths=.25,zorder=2)
            identify(points,f'{panel}-{c}-points-{rid}','observation.fish',{'fish_id':rid,'condition_id':c,'artifact':ref(D_SOURCE if panel=='D' else OUT/'E-fish-blocks.parquet')})
        q=np.array([sub.loc[sub['Selected block order'].eq(i),'Fish median response / baseline'].quantile([.25,.5,.75]).to_numpy() for i in range(3)])
        parts=ax.errorbar(range(3),q[:,1],yerr=[q[:,1]-q[:,0],q[:,2]-q[:,1]],fmt='o-',color='black',alpha=.72,lw=profile.contract['styles']['median_iqr']['linewidth_pt'],capsize=1.5,markersize=2,zorder=5).lines
        identify(parts[0],f'{panel}-{c}-median','summary.median',ref(D_SOURCE if panel=='D' else OUT/'E-fish-blocks.parquet'))
        for i,artist in enumerate([*parts[1],*parts[2]]):identify(artist,f'{panel}-{c}-iqr-{i}','uncertainty.iqr',ref(D_SOURCE if panel=='D' else OUT/'E-fish-blocks.parquet'))
        identify(ax.axhline(0,color='black',lw=.5,zorder=0),f'{panel}-{c}-zero','reference.signal.zero',{'value':0})
        ax.set(xlim=(-.15,2.15),ylim=y_limits,xticks=[0,1,2],xticklabels=['PT','ET','LT'],yticks=np.arange(-.6,.61,.2) if y_ticks is None else y_ticks)
        style_axes(ax,theme=theme,xlabel='',ylabel='Δ median ln vigor' if panel=='D' and c=='control' else '',show_yticklabels=panel=='D' and c=='control')
        ax.spines['bottom'].set_visible(False);ax.tick_params(axis='x',bottom=False,length=0)
        heading=fig.text((lefts[axes.index(ax)]+13.04813/2)/width,1-4.6/height,'Control' if c=='control' else 'Delay' if c=='delay' else '3sTrace',ha='center',va='top',fontsize=theme.axes_titlesize,color='black')
        identify(heading,f'{panel}-{c}-heading','title.condition',{'condition_id':c});extra.append(heading.get_gid())
        selected=tests.loc[tests.panel.eq(panel)&tests.condition.eq(c)&tests.p_holm36.lt(.05)]
        for level,row in enumerate(selected.loc[selected.family.eq('within')].itertuples()):
            y=1.04+.22*level
            ln,=ax.plot([row.left,row.right],[y,y],transform=ax.get_xaxis_transform(),color='black',lw=.5,clip_on=False)
            identify(ln,f'{panel}-{c}-within-{row.left}-{row.right}','annotation.statistical_comparison')
            text=ax.text((row.left+row.right)/2,y+.025,row.stars,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=theme.tick_labelsize,clip_on=False)
            identify(text,f'{panel}-{c}-within-label-{row.left}-{row.right}','annotation.statistical_label')
        for row in selected.loc[selected.family.eq('zero')].itertuples():
            text=ax.text(row.left,.83,'†'+row.stars,transform=ax.get_xaxis_transform(),ha='center',fontsize=theme.tick_labelsize)
            identify(text,f'{panel}-{c}-zero-label-{row.left}','annotation.statistical_label')
    for row in tests.loc[tests.panel.eq(panel)&tests.family.eq('between')&tests.p_holm36.lt(.05)].itertuples():
        line=ConnectionPatch((row.left,1.52),(row.right,1.52),coordsA=axes[0].get_xaxis_transform(),coordsB=axes[1].get_xaxis_transform(),color='black',lw=.5,clip_on=False)
        fig.add_artist(line);identify(line,f'{panel}-between-{row.left}','annotation.statistical_comparison');extra.append(line.get_gid())
        a=fig.transFigure.inverted().transform(axes[0].get_xaxis_transform().transform((row.left,1.52)));b=fig.transFigure.inverted().transform(axes[1].get_xaxis_transform().transform((row.right,1.52)))
        text=fig.text((a[0]+b[0])/2,a[1]+.009,row.stars,ha='center',va='bottom',fontsize=theme.tick_labelsize)
        identify(text,f'{panel}-between-label-{row.left}','annotation.statistical_label');extra.append(text.get_gid())
    letter=fig.text(1.5/width,1-1.1/height,panel.lower(),va='top',weight='bold',fontsize=profile.contract['styles']['panel_letter']['font_size_pt'])
    identify(letter,panel+'-letter','panel.letter',{'panel':panel});extra.append(letter.get_gid())
    return fig,[panel.lower()+'-control',panel.lower()+'-'+conditioned],mappings,extra,[width,height]


def trajectory_panel(profile,theme):
    width,height=48.71645694404175,60.49550789093819
    fig=plt.figure(figsize=(width/25.4,height/25.4),layout=None);fig.patch.set_gid('canvas-H')
    # Same data-rectangle width/height and y-position as current G, with shared
    # statistical row labels supplied by G rather than duplicated in H.
    left=10.5;top=130.8200488504007-95.97294097147432
    ax=fig.add_axes([left/width,1-(top+17.22759038333434)/height,35.55614768162458/width,17.22759038333434/height])
    ax.patch.set_gid('H-observed-data-rectangle')
    # Browser-measured G row centres: 105.99978 + 3.00010*n mm.
    # Keep phase headers at 102.48612 mm and account for Arial glyph centring.
    stat=fig.add_axes([left/width,1-(8.64135+15.000505)/height,35.55614768162458/width,15.000505/height])
    mappings={};extra=[]
    def identify(artist,key,role,source):artist.set_gid(key);mappings[key]={'scientific_role':role,'panel':'H','source':source};return artist
    trajectory=pd.read_parquet(OUT/'H-trajectory.parquet')
    for c,color in [('control','#00b5eb'),('trace','#f15a29')]:
        d=trajectory.loc[trajectory.condition_id.eq(c)].sort_values('trial_number')
        line,=ax.plot(d.trial_number,d['median'],color=color,lw=profile.contract['styles']['summary_curve']['linewidth_pt'],label='Control' if c=='control' else '3sTrace')
        identify(line,'H-'+c+'-median','summary.curve',{'artifact':ref(OUT/'H-trajectory.parquet'),'condition_id':c,'fields':['trial_number','median']})
        band=ax.fill_between(d.trial_number,d.ci_lower,d.ci_upper,color=color,alpha=.2,lw=0)
        identify(band,'H-'+c+'-CI','uncertainty.confidence_interval',{'artifact':ref(OUT/'H-trajectory.parquet'),'condition_id':c,'fields':['ci_lower','ci_upper']})
    ax.set(xlim=(4,95),ylim=(-.21,.16),xticks=[5,25,45,65,85,94],yticks=[-.2,-.1,0,.1])
    style_axes(ax,theme=theme,xlabel='Global CS trial',ylabel='',show_yticklabels=False)
    identify(ax.axhline(0,color='black',lw=.5),'H-zero','reference.signal.zero',{'value':0})
    for x in [14.5,64.5]:identify(ax.axvline(x,color='black',lw=.5,ls=':'),'H-phase-'+str(x),'boundary.phase',{'trial':x})
    ax.legend(loc='lower right',bbox_to_anchor=(1,1.02),ncol=2,frameon=False,fontsize=theme.legend_fontsize,handlelength=1,columnspacing=.6,handletextpad=.3,borderaxespad=0)
    stat.set(xlim=(4,95),ylim=(-.5,4.5),xticks=[],yticks=[])
    stat.patch.set_visible(False);stat.spines[:].set_visible(False)
    dtests=pd.read_csv(OUT/'D-interaction-tests.csv');local=pd.read_csv(OUT/'MR-local-block-tests.csv');trial=pd.read_csv(OUT/'phase-contrasts.csv')
    for r in dtests.itertuples():
        if r.p_holm<.05:identify(stat.text(r.center,4,glyph(r.p_holm,3),ha='center',va='center',fontsize=theme.tick_labelsize),'H-D-'+r.block_10_name.replace(' ','-'),'annotation.statistical_label',ref(OUT/'D-interaction-tests.csv'))
    for r in local.itertuples():
        for field,y,label in [('mean_p_fdr',3,'M'),('slope_p_raw',2,'R-raw'),('slope_p_fdr',1,'R-FDR')]:
            p=getattr(r,field);text='n/a' if not np.isfinite(p) else glyph(p,3)
            if text:identify(stat.text(r.center,y,text,ha='center',va='center',fontsize=theme.tick_labelsize),'H-'+label+'-'+r.block_10_name.replace(' ','-'),'annotation.statistical_label',{'artifact':ref(OUT/'MR-local-block-tests.csv'),'field':field,'block':r.block_10_name})
    sig=trial.loc[trial.change_p_fdr.lt(.05)]
    identify(stat.scatter(sig.trial_number,np.zeros(len(sig)),marker='*',s=6,color='black',lw=0),'H-trial-stars','annotation.statistical_label',{'artifact':ref(OUT/'phase-contrasts.csv'),'field':'change_p_fdr','selected_trials':sig.trial_number.astype(int).tolist()})
    for i,label in enumerate(['Pre','Tr1','Tr2','Tr3','Tr4','Tr5','Te1','Te2','Te3']):
        identify(stat.text(9.5+10*i,5.17118,label,ha='center',va='center',fontsize=theme.tick_labelsize),'H-phase-header-'+label,'title.phase',{'trial_span':[5+i*10,14+i*10]})
    letter=fig.text(1.5/width,1-1.1/height,'h',va='top',weight='bold',fontsize=profile.contract['styles']['panel_letter']['font_size_pt'])
    identify(letter,'H-letter','panel.letter',{'panel':'H'});extra.append('H-letter')
    return fig,['H-observed','H-statistics'],mappings,extra,[width,height]


def panel_export(panel,result,profile):
    fig,ids,mappings,extra,size=result
    spec={'scope':'panel','final_size_mm':size,'inset_mm':load_policy()[0]['selected_inset_mm'],
          'allocations':[{'id':panel,'kind':'panel','box_mm':[0,0,*size],'group_ids':['axes__'+x.lower()+'__main' for x in ids]+extra}],
          'background_ids':['canvas-'+panel],'alignments':[],'data_axes':[]}
    provenance=FigureProvenance(figure_id='Figure 2 '+panel,analysis_recipe='matched-selected-Delay-statistics',source_file=str(Path(__file__)),source_symbol='block_panel' if panel in 'DE' else 'trajectory_panel',source_hash=sha256_file(Path(__file__)),reproduction_snippet='.venv-trace/Scripts/python.exe scripts/assemble_figure2_trace3_matched.py',input_artifacts=tuple([ref(OUT/'plan.json'),ref(OUT/'completion.json')]),artist_mappings=mappings,analysis_identity={'cohort_counts':{'control':90,'trace':89} if panel!='D' else {'control':28,'delay':29},'excluded_fish':EXCLUDED,'presentation_profile_inputs':profile.inputs,'status':'unfrozen scientific draft'})
    export_matplotlib_figure(fig,OUT/('panel-'+panel),provenance,mode=FigureMode.PUBLICATION,panel_ids=ids,allow_dirty_publication=True,formats=('svg',),layout_frames=spec)
    plt.close(fig)


def namespace(root,prefix):
    names={n.get('id'):prefix+n.get('id') for n in root.iter() if n.get('id')}
    for n in root.iter():
        if n.get('id'):n.set('id',names[n.get('id')])
        for k,v in list(n.attrib.items()):
            if k=='id':continue
            v=re.sub(r'url\(#([^)]*)\)',lambda m:'url(#'+names.get(m[1],m[1])+')',v)
            if k.endswith('href') and v.startswith('#'):v='#'+names.get(v[1:],v[1:])
            n.set(k,v)
    return names


def build():
    ticket=load(OUT/'starting-ticket.json')
    for p,h in ticket.items():
        if p.endswith('render_figure2_delay_phase_lmm.py'):continue # documented backwards-compatible condition argument
        if sha256_file(Path(p))!=h:raise ValueError('Starting figure/input changed: '+p)
    completion=load(OUT/'completion.json')
    if not completion['primary_phase_available'] or completion['failures']:raise ValueError('Unresolved statistical fit')
    profile=load_presentation_profile('nature-main');tests=pd.read_csv(OUT/'DE-statistics-Holm36.csv')
    registry=load(REGISTRY);layout=registry['presentation_layout']['allocations'];alloc={a['id']:a for a in layout}
    def staged(p):
        if not (OUT/('panel-'+p+'.svg')).exists():return False
        path=OUT/('verified-current-staged-panels.json' if (OUT/'verified-current-staged-panels.json').exists() else 'verified-staged-panels.json')
        bound=load(path)[p]
        for a in [bound['svg'],bound['sidecar']]:
            if sha256_file(Path(a['path']))!=a['sha256']:raise ValueError('Edited staged panel '+p)
        return True
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        for p in ['D','E']:
            if not staged(p):panel_export(p,block_panel(p,alloc[p]['box_mm'][2],profile,theme,tests),profile)
        if not staged('H'):panel_export('H',trajectory_panel(profile,theme),profile)
    root=ET.fromstring(SOURCE.read_bytes());lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    protected={p:sha(ET.tostring(lookup[p])) for p in ['layout-A','layout-C','panel-F','panel-G','panel-I']}
    pooled=pd.read_parquet(OUT/'B-pooled-bins.parquet');raster_checks=[]
    cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black');norm=Normalize(-.25,.25,clip=False)
    for i,c in enumerate(['control','trace']):
        group=lookup[f'fig2__b{i}__heatmap__vigor-cells'];image=next(n for n in group.iter() if n.tag==tag('image'))
        matrix=pooled.loc[pooled.condition_id.eq(c)].pivot(index='acquisition_trial',columns='bin_index',values='mean_delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
        old=Image.open(io.BytesIO(base64.b64decode(image.get(XLINK).split(',',1)[1])))
        rgba=np.asarray(cmap(norm(np.ma.masked_invalid(matrix)))*255,dtype=np.uint8)
        raster=Image.fromarray(rgba[::-1]).resize(old.size,resample=Image.Resampling.NEAREST);buffer=io.BytesIO();raster.save(buffer,format='PNG')
        image.set(XLINK,'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
        group.set('data-scientific-role','heatmap.vigor');group.set('data-condition',c);group.set('data-pooled-values-sha256',sha(np.asarray(matrix,dtype='<f8').tobytes()))
        raster_checks.append({'condition':c,'matrix_shape':list(matrix.shape),'source':ref(OUT/'B-pooled-bins.parquet'),'rgba_sha256':sha(rgba.tobytes()),'display_flip':'same vertical flip and nearest-cell export as selected A','raster_size':list(old.size)})
    vb=list(map(float,root.get('viewBox').split()));scale=vb[2]/136
    meta=next(n for n in root.iter() if n.get('id')=='layout-frames');frames=json.loads(meta.text)
    new_axes=[];new_clips=[];backgrounds=[]
    for p in ['D','E','H']:
        container=lookup['panel-'+p]
        for child in list(container):container.remove(child)
        container.attrib.clear();container.set('id','panel-'+p)
        svg=ET.fromstring((OUT/('panel-'+p+'.svg')).read_bytes());names=namespace(svg,'matched-'+p+'__')
        for m in list(svg):
            if m.get('id')=='matched-'+p+'__layout-frames':svg.remove(m)
        x,y,w,h=alloc[p]['box_mm']
        adjustments=load(OUT/'placement-adjustments.json') if (OUT/'placement-adjustments.json').exists() else {}
        x+=adjustments.get(p,{}).get('dx_mm',0)
        for k,v in {'x':x*scale,'y':y*scale,'width':w*scale,'height':h*scale,'overflow':'visible'}.items():svg.set(k,str(v))
        container.append(svg);backgrounds.append('matched-'+p+'__canvas-'+p)
        for n in svg.iter():
            if n.tag==tag('clipPath'):new_clips.append({'id':n.get('id'),'evidence':{'purpose':'Renderer data-axis bounds; all plotted values checked inside recorded numerical limits','definition_sha256':sha(ET.tostring(n)),'source_svg':ref(OUT/('panel-'+p+'.svg'))}})
            if n.get('id','').endswith('data-rectangle'):
                axis_id=p if n.get('id').endswith(p+'-control-data-rectangle') else p+'-'+str(len(new_axes))
                new_axes.append({'id':axis_id,'element_id':n.get('id')})
    for p,h in protected.items():
        if sha(ET.tostring(lookup[p]))!=h:raise ValueError('Unrequested source subtree changed: '+p)
    existing={n.get('id') for n in root.iter() if n.get('id')}
    frames['data_axes']=[a for a in frames.get('data_axes',[]) if all(s in existing for s in ([a['element_id']] if 'element_id'in a else a.get('element_ids',[])))]+new_axes
    frames['background_ids']+=backgrounds
    frames['scientific_clips']=[c for c in frames.get('scientific_clips',[]) if c['id'] in existing]+new_clips
    meta.text=json.dumps(frames,sort_keys=True,separators=(',',':'),allow_nan=False)
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    legend=('Figure 2 | Population conditioning measures and exploratory comparisons.\n\n'
       '(a, b) Matched control/conditioned population heatmaps: Delay, 28/29 fish; 3sTrace, 90/89 fish. Direct 1-s median log positive detected-bout vigor, centred within fish/trial by finite baseline-bin medians over [−15,0) s, then averaged with equal fish weight. Colours saturate at ±0.25; black is missing. Green/violet guides show CS/expected US. Heatmap trials 1–90 are acquisition trials 5–94. Four timing-flagged fish are omitted from all B/E/H panels and retained in the source record.\n\n'
       '(d, e) Fish/block median response-minus-baseline median log positive valid bout-frame vigor. Baseline [−15,0) s; response [0,9) s (Delay) or [0,13) s (3sTrace). PT/ET/LT: trials 10–14/65–69/90–94; minimum eligible trials 3/1 for Delay/3sTrace. Thin paired fish; black median/IQR. Exact two-sided sign tests (zero and paired changes) and Brunner–Munzel t tests (independent blocks and paired changes), joint Holm36. D data are unchanged; adjusted annotations were recomputed jointly with E. Dagger: versus zero; stars: adjusted p<0.05/0.01/0.001/0.0001.\n\n'
       '(g, h) Observed across-fish median log-vigor differences with pointwise 95% intervals from 5,000 whole-fish resamples (seed 10). G/H use the same model families: baseline-adjusted block interactions (D, Holm8), local condition mean differences (M, BH9), local slope differences (R, raw/BH9), and phase-aware random-intercept/slope LMM trial change from average Pre5–14 (BH90). Statistical row labels in G also apply to H; H uses at most three stars. Unavailable fits show n/a. These exploratory results do not establish learning onset.\n\n'
       '(c, f, i) 10sTrace remains pending. Updated B/E/H and D annotations are unfrozen drafts.')
    write_new(OUT/'figure2-matched.svg',svg);write_new(OUT/'figure2-matched-legend.md',legend.encode())
    page=render_page([{'id':'fig2','svg_bytes':svg,'legend_text':legend,'svg':ref(OUT/'figure2-matched.svg'),'legend':ref(OUT/'figure2-matched-legend.md'),'whole_figure_frozen':False}])[0]
    write_new(OUT/'figure2-matched.html',page)
    evidence={'candidate':ref(OUT/'figure2-matched.svg'),'review':ref(OUT/'figure2-matched.html'),'legend':ref(OUT/'figure2-matched-legend.md'),'source':ref(SOURCE),'statistics':ref(OUT/'completion.json'),'common_cohort':{'control':90,'trace':89},'excluded_recordings':EXCLUDED,'protected_subtrees':protected,'B_raster_checks':raster_checks,'profile_inputs':profile.inputs,'generator':ref(Path(__file__)),'data_axis_geometry_mm':{'DE_height':13.782038393992613,'DE_width_each':13.04813,'GH_width':35.55614768162458,'GH_height':17.22759038333434},'DE_common_y_limits':[-.75,.75],'GH_common_y_limits':[-.21,.16],'verification':'candidate; exact browser/frame review pending'}
    write_new(OUT/'assembly-evidence.json',encode(evidence))
    print(OUT/'figure2-matched.html')


def publish(note):
    evidence=load(OUT/'assembly-evidence.json');frame=load(OUT/'layout-evidence.json')
    if frame['status']!='passed' or frame['artifact_sha256']!=evidence['candidate']['sha256']:raise ValueError('Fresh passing layout evidence required')
    for k in ['candidate','review','legend','source','statistics','generator']:
        a=evidence[k]
        actual=Path(a['path'])
        if k=='generator' and sha256_file(actual)!=a['sha256']:
            actual=OUT/'assembly-build-source.py'
        if sha256_file(actual)!=a['sha256']:raise ValueError('Edited bound artifact: '+a['path'])
    ticket=load(OUT/'starting-ticket.json');snapshot=Snapshot([]);snapshot.hashes={p:h for p,h in ticket.items() if not p.endswith('render_figure2_delay_phase_lmm.py')}
    snapshot.check()
    state=load(STATE);old=copy.deepcopy(state)
    history=OUT/'previous-selected';history.mkdir(exist_ok=True)
    for name,p in [('state.json',STATE),('figure2.svg',SOURCE),('figure2.html',Path(str(SOURCE).replace('.svg','.html'))),('legend.md',LEGEND),('semantic-registry.json',REGISTRY)]:write_new(history/name,p.read_bytes())
    registry=load(REGISTRY);registry['B_E_H_matched_expanded_cohort']={**evidence,'layout_frames':ref(OUT/'layout-evidence.json'),'visual_review':note,'status':'scoped draft; no new freeze','DE_adjusted_family_recomputed':True}
    registry['B_E_H_matched_expanded_cohort']['implementation']={'build':ref(OUT/'assembly-build-source.py'),'publish':ref(Path(__file__)),'placement_adjustments':ref(OUT/'placement-adjustments.json')}
    registry['matched_panel_registries']={p:ref(OUT/('panel-'+p+'.figure.json')) for p in ['D','E','H']}
    active_root=ET.fromstring(Path(evidence['candidate']['path']).read_bytes())
    active_ids={n.get('id') for n in active_root.iter() if n.get('id')}
    elements=registry.setdefault('element_registry',{})
    for key in list(elements):
        if key not in active_ids and str(elements[key].get('panel_id','')).lower() in {'d','e','h'}:del elements[key]
    axes=registry.setdefault('axis_inventory',{}).setdefault('axes',[])
    axes[:]=[a for a in axes if a.get('panel') not in ['D','E','H']]
    registry['axis_inventory']['source_svg_sha256']=evidence['candidate']['sha256']
    headings=registry.setdefault('heading_semantics',{}).setdefault('condition_headings',[])
    headings[:]=[a for a in headings if a.get('panel') not in ['D','E','H']]
    legends=registry.setdefault('plot_legends',{}).setdefault('groups',[])
    legends[:]=[a for a in legends if a.get('panel') not in ['D','E','H']]
    for p in ['D','E','H']:
        side=load(OUT/('panel-'+p+'.figure.json'));prefix='matched-'+p+'__'
        for key,row in side['artist_registry'].items():
            identity=prefix+key
            if identity in active_ids:elements[identity]={**row,'element_id':identity,'panel_id':p.lower(),'source_panel_registry':registry['matched_panel_registries'][p]}
        axis_panels=['d-control','d-delay'] if p=='D' else ['e-control','e-trace'] if p=='E' else ['h-observed','h-statistics']
        for ap in axis_panels:
            if ap.endswith('statistics'):continue
            condition='trace / control' if p=='H' else ap.split('-',1)[1]
            for dim in ['x','y']:
                axes.append({'id':p+'-'+condition+'-'+dim,'panel':p,'orientation':dim,'group_ids':[prefix+'axis__'+ap+'__'+dim],
                    'title_ids':[] if (dim=='y' and (p!='D' or condition!='control')) or (dim=='x' and p!='H') else [prefix+'axis-title__'+ap+'__'+dim],
                    'condition':condition,'quantity':('global CS trial' if p=='H' else 'selected session block') if dim=='x' else 'median log positive bout-frame response minus baseline',
                    'units':('trial' if p=='H' else 'categorical block') if dim=='x' else 'dimensionless natural-log difference',
                    'limits':([4,95] if p=='H' else [-.15,2.15]) if dim=='x' else ([-.21,.16] if p=='H' else [-.75,.75]),
                    'scientific_status':'author-requested expanded-cohort draft; exact definitions and data in matched panel registry',
                    'data_registry':registry['matched_panel_registries'][p]})
            if p in ['D','E']:headings.append({'panel':p,'element_id':prefix+p+'-'+condition+'-heading','condition':condition,'reference_axes':prefix+'axes__'+ap+'__main'})
        if p=='H':legends.append({'panel':'H','semantic_role':'legend.key','element_id':prefix+'legend__h-observed__main','reference_axes':prefix+'axes__h-observed__main','alignment':'right','series':['control','trace']})
    for i,c in enumerate(['control','trace']):
        key=f'fig2__b{i}__heatmap__vigor-cells'
        elements[key]={'element_id':key,'panel_id':'b','scientific_role':'heatmap.vigor','data':ref(OUT/'B-pooled-bins.parquet'),'condition':c,
            'quantity':'equal-fish mean of baseline-centred direct 1-s median positive bout-frame log vigor','array_shape':[90,40],
            'baseline_s':[-15,0],'time_s':[-20,20],'acquisition_trials':[5,94],'display_trials':[1,90],
            'colormap':'managua_r','limits':[-.25,.25],'missing':'NaN black','clipping':'colour saturation only; numerical values unchanged',
            'common_cohort':ref(OUT/'plan.json'),'coverage_field':'contributing_fish'}
    for item in registry.get('source_panels',[]):
        if item.get('id') in ['B','D','E','H']:item['active_scientific_override']='B_E_H_matched_expanded_cohort; earlier source is retained as history only'
    registry['axis_inventory_status']='Active A/C/F/G/I axes inherited; B follows selected A pipeline; D/E/H mappings replaced with explicit new renderer identities. Production glyph/embedding checks remain separate.'
    regpath=OUT/'matched-semantic-registry.json';write_new(regpath,encode(registry))
    draft=state['draft'];draft['working_svg']={'path':str(SOURCE),'sha256':evidence['candidate']['sha256']};htmlpath=Path(str(SOURCE).replace('.svg','.html'));draft['review']={'path':str(htmlpath),'sha256':evidence['review']['sha256']}
    draft['legend']={'path':str(LEGEND),'sha256':evidence['legend']['sha256'],'panel_ids':list('ABCDEFGHI')};draft['semantic_registry']=ref(regpath)
    draft['status']='179 common 3sTrace fish in B/E/H; matched Delay statistical families; D/E joint adjusted annotations updated; unfrozen scientific drafts'
    draft.setdefault('consistency_requests',[]).append({'id':'fig2-trace3-matched-delay-20261019','scope':['A/B','D/E','G/H'],'author_direction':'Match current Delay panels; same statistical tests; omit four timing-flagged fish from all Trace panels and retain their flags','status':'verified numerical methods, selected rectangular geometry and frame containment; scientific drafts, production checks remain open','evidence':ref(OUT/'assembly-evidence.json'),'layout_frames':ref(OUT/'layout-evidence.json')})
    draft['matched_trace3_scientific_draft']={'data':ref(OUT/'plan.json'),'statistics':ref(OUT/'completion.json'),'scope':['B','E','H','D joint correction only'],'frozen':False}
    draft['verification']={'layout_frames':ref(OUT/'layout-evidence.json'),'scientific_data':ref(OUT/'plan.json'),'model_diagnostics':ref(OUT/'completion.json'),'visual_review':{'status':'performed','note':note},'coverage':'Exact painted/unclipped frame containment, recorded equal spacing and native-size visual review; production font/PDF/accessibility and influence audits remain separate; no freeze.'}
    draft['reproduction']={'generator':ref(Path(__file__)),'statistical_analysis':ref(REPO/'scripts/analyze_figure2_trace3_matched.py'),'sealed_data':ref(OUT/'plan.json'),'calculation_cache':str(OUT),'publication_command':'.venv-trace/Scripts/python.exe scripts/assemble_figure2_trace3_matched.py --publish --visual-review-note <actual review>','policy':'Preserve selected bytes and prior candidates; authenticate existing caches and capture a fresh state ticket before subsequent edits.'}
    updates=[(SOURCE,Path(evidence['candidate']['path']).read_bytes()),(htmlpath,Path(evidence['review']['path']).read_bytes()),(LEGEND,Path(evidence['legend']['path']).read_bytes())]
    snapshot.hashes.update(Snapshot([OUT/'assembly-evidence.json',OUT/'layout-evidence.json',regpath,Path(__file__)]).hashes)
    publish_state(STATE,state,snapshot,evidence_updates=updates)
    print(htmlpath)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');parser.add_argument('--visual-review-note');args=parser.parse_args()
    if args.publish:
        if not args.visual_review_note:raise ValueError('Actual review note required')
        publish(args.visual_review_note)
    else:build()
