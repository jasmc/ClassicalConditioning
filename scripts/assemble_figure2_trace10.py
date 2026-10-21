"""Build/publish the author-approved C/F/I and D/E correction scientific draft."""
from pathlib import Path
import argparse,base64,copy,inspect,io,json,re,sys,xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
from classical_conditioning.figure_workspace import Snapshot,publish_state,tag,sha
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme
from classical_conditioning.figures.export import FigureProvenance,FigureMode,export_matplotlib_figure
from classical_conditioning.figures.layout_frames import assess
from analyze_figure2_trace10 import OUT,STATE,load,ref,save,check
from main_paper_figures import render as render_page
import assemble_figure2_trace3_matched as reference

BUILD=OUT/'assembly';MODELS=OUT/'I-models';DE=OUT/'DE-corrections'
XLINK='{http://www.w3.org/1999/xlink}href'
def write(p,b):
    with Path(p).open('xb') as f:f.write(b)
def enc(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()

def numerical_inputs():
    roster=load(OUT/'common-roster.json');assert roster['status']=='author-approved' and roster['counts']=={'control':32,'trace':29}
    calc=load(OUT/'calculation.json');check(calc['outputs']+calc['fish_records'])
    completion=load(MODELS/'completion.json');check(completion['outputs'])
    assert completion['primary_phase_available'] and not completion['failures']
    corrections=load(DE/'calculation.json');check(corrections['outputs'])
    # The original D/E calculation script is preserved byte-for-byte after its later import fix.
    old=next(r for r in corrections['inputs'] if r['path'].endswith('trace10_panel_statistics.py'))
    assert ref(DE/'trace10_panel_statistics-calculation-source.py')['sha256']==old['sha256']
    return roster,completion

def render_panel(p,profile,theme):
    env=dict(vars(reference));env.update(OUT=OUT,MODELS=MODELS)
    # Reuse the selected renderer with only panel/source/family identifiers changed.
    if p=='F':
        source=inspect.getsource(reference.block_panel).replace('def block_panel(','def adapted(')
        source=source.replace("'E-fish-blocks.parquet'","'F-fish-blocks.parquet'").replace("'DE-statistics-Holm36.csv'","'F-statistics-Holm18.csv'").replace('p_holm36','p_holm18').replace("'3sTrace'","'10sTrace'")
        exec(source,env);width=132.334196*25.4/72
        result=env['adapted']('F',width,profile,theme,pd.read_csv(OUT/'F-statistics-Holm18.csv'),y_limits=(-.25,.25),y_ticks=[-.2,-.1,0,.1,.2])
        inputs=[ref(OUT/'F-fish-blocks.parquet'),ref(OUT/'F-statistics-Holm18.csv')]
    else:
        source=inspect.getsource(reference.trajectory_panel).replace('def trajectory_panel(','def adapted(')
        source=source.replace("'H","'I").replace("'h'","'i'").replace("'3sTrace'","'10sTrace'")
        for name in ['D-interaction-tests.csv','MR-local-block-tests.csv','phase-contrasts.csv']:
            source=source.replace("OUT/'"+name+"'","MODELS/'"+name+"'")
        exec(source,env);result=env['adapted'](profile,theme)
        inputs=[ref(OUT/'I-trajectory.parquet'),ref(MODELS/'completion.json')]
    fig,ids,mapping,extra,size=result
    spec={'scope':'panel','final_size_mm':size,'inset_mm':1.,'allocations':[{'id':p,'kind':'panel','box_mm':[0,0,*size],'group_ids':['axes__'+i.lower()+'__main' for i in ids]+extra}],'background_ids':['canvas-'+p],'data_axes':[],'alignments':[]}
    provenance=FigureProvenance(figure_id='Figure 2 '+p,analysis_recipe='fixed10sTrace-selected-methods',source_file=str(Path(__file__)),source_symbol='render_panel',source_hash=ref(Path(__file__))['sha256'],reproduction_snippet='.venv-trace/Scripts/python.exe scripts/assemble_figure2_trace10.py build',input_artifacts=tuple([ref(OUT/'common-roster.json'),*inputs]),artist_mappings=mapping,analysis_identity={'cohort_counts':{'control':32,'trace':29},'response_s':[0,20],'baseline_s':[-15,0],'status':'author-requested scientific draft; no freeze','renderer_reuse':'selected 3sTrace renderer, only panel IDs, labels, file sources and Holm18 field adapted','adapted_renderer_sha256':sha(source.encode()),'profile_inputs':profile.inputs})
    export_matplotlib_figure(fig,BUILD/('panel-'+p),provenance,mode=FigureMode.PUBLICATION,panel_ids=ids,allow_dirty_publication=True,formats=('svg',),layout_frames=spec);plt.close(fig)
    write(BUILD/('panel-'+p+'-renderer.py'),source.encode())
    return size

def build():
    roster,completion=numerical_inputs();BUILD.mkdir(exist_ok=True)
    state=load(STATE);d=state['draft'];check([d[k] for k in ['working_svg','review','legend','semantic_registry']])
    corrected=DE/'render';evidence=load(corrected/'render-evidence.json');check([evidence[k] for k in ['svg','html','legend','source']])
    assert evidence['source']['sha256']==d['working_svg']['sha256'],'Current artwork changed; reconcile D/E against it first'
    inputs=[STATE,*[Path(d[k]['path']) for k in ['working_svg','review','legend','semantic_registry']],Path(__file__),REPO/'scripts/assemble_figure2_trace3_matched.py',OUT/'common-roster.json',OUT/'C-pooled-bins.parquet',OUT/'F-statistics-Holm18.csv',OUT/'F-fish-blocks.parquet',OUT/'I-trajectory.parquet',MODELS/'completion.json']
    ticket=Snapshot(inputs);save(BUILD/'starting-ticket.json',ticket.hashes)
    root=ET.fromstring((corrected/'figure2-DE-corrected.svg').read_bytes());lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    protected={k:sha(ET.tostring(lookup[k])) for k in ['layout-A','layout-B','panel-G','panel-H','fig2__shared__colorbar__axes']}
    profile=load_presentation_profile('nature-main');theme=None
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        for p in ['F','I']:render_panel(p,profile,theme)
    # Replace C's placeholder cells in their exact selected rectangles.
    c=lookup['layout-C'];parents={n:p for p in c.iter() for n in p}
    for n in list(c.iter()):
        if n.tag==tag('text') and ''.join(n.itertext()).strip().lower()=='pending':parents[n].remove(n)
    pooled=pd.read_parquet(OUT/'C-pooled-bins.parquet');cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black');norm=Normalize(-.25,.25)
    c_mapping={}
    for i,condition in enumerate(['control','trace']):
        group=lookup[f'fig2__c{i}__heatmap__background'];path=next(group.iter(tag('path')))
        vals=list(map(float,re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',path.get('d'))));xs,ys=vals[::2],vals[1::2];x,y=min(xs),min(ys);w,h=max(xs)-x,max(ys)-y
        matrix=pooled.loc[pooled.condition_id.eq(condition)].pivot(index='acquisition_trial',columns='bin_index',values='mean_delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
        rgba=np.asarray(cmap(norm(np.ma.masked_invalid(matrix)))*255,dtype=np.uint8);buf=io.BytesIO();Image.fromarray(rgba).save(buf,format='PNG')
        key=f'fig2__c{i}__heatmap__vigor-cells'
        ET.SubElement(group,tag('image'),{'id':key,'x':str(x),'y':str(y),'width':str(w),'height':str(h),'preserveAspectRatio':'none','style':'image-rendering:pixelated',XLINK:'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()})
        axes=lookup[f'fig2__c{i}__heatmap__axes']
        for t,dash,label in [(0,None,'CS-onset'),(10,'2,2','CS-offset')]:
            a=x+(t+20)*w/40;attrs={'id':key+'-'+label,'x1':str(a),'x2':str(a),'y1':str(y),'y2':str(y+h),'stroke':theme.cs_color,'stroke-width':str(theme.cs_linewidth)}
            if dash:attrs['stroke-dasharray']=dash
            ET.SubElement(axes,tag('line'),attrs)
        if condition=='trace':ET.SubElement(axes,tag('line'),{'id':key+'-expected-training-US20','x1':str(x+w),'x2':str(x+w),'y1':str(y+10*h/90),'y2':str(y+60*h/90),'stroke':theme.us_color,'stroke-width':str(theme.us_linewidth),'stroke-dasharray':'1,2'})
        for trial in [10,60]:ET.SubElement(axes,tag('line'),{'id':key+'-phase-'+str(trial),'x1':str(x),'x2':str(x+w),'y1':str(y+trial*h/90),'y2':str(y+trial*h/90),'stroke':'white','stroke-width':str(theme.axes_linewidth)})
        c_mapping[key]={'element_id':key,'panel_id':'c','scientific_role':'heatmap.vigor','data':ref(OUT/'C-pooled-bins.parquet'),'condition':condition,'matrix_sha256':sha(np.asarray(matrix,dtype='<f8').tobytes()),'rgba_sha256':sha(rgba.tobytes()),'shape':[90,40],'display_orientation':'matrix row0 at top; trials5-94 displayed1-90','baseline_s':[-15,0],'colormap':'managua_r','limits':[-.25,.25],'missing':'black','cohort':ref(OUT/'common-roster.json'),'coverage_field':'contributing_fish'}
    scale=float(root.get('viewBox').split()[2])/136
    frames=json.loads(lookup['layout-frames'].text);alloc={a['id']:a for a in frames['allocations']};new_clips=[]
    # I needs the same native data width as G/H. Expand only the canvas, not existing panels.
    final_width=156.;vb=list(map(float,root.get('viewBox').split()));vb[2]=scale*final_width;root.set('viewBox',' '.join(map(str,vb)));root.set('width',str(final_width)+'mm');root.set('aria-label','Figure 2 with authenticated fixed10sTrace panels C F I')
    frames['final_size_mm'][0]=final_width
    alloc['ABC-colorbar']['box_mm'][2]=final_width-alloc['ABC-colorbar']['box_mm'][0]
    alloc['F']['box_mm'][2]=final_width-alloc['F']['box_mm'][0]
    alloc['I']['box_mm'][2]=final_width-alloc['I']['box_mm'][0]
    for p in ['F','I']:
        container=lookup['panel-'+p]
        for child in list(container):container.remove(child)
        container.attrib.clear();container.set('id','panel-'+p)
        svg=ET.fromstring((BUILD/('panel-'+p+'.svg')).read_bytes());reference.namespace(svg,'trace10-'+p+'__')
        for child in list(svg):
            if child.get('id')=='trace10-'+p+'__layout-frames':svg.remove(child)
        size=list(map(float,svg.get('viewBox').split()[2:]));w,h=[v*25.4/72 for v in size]
        if p=='F':
            pair_center=(90.7268606127332+119.7125907475116)/2
            x=pair_center-(13.84712433660092+29.2439147441586+13.04813)/2;y=52.70646260309728
            letter=next(n for n in svg.iter() if n.get('id')=='trace10-F__F-letter');letter.set('transform',f'translate({(87.16934996437332-x-1.5)*72/25.4} 0)')
        else:x,y=106.45448825746593,95.97294097147432
        svg.attrib.update({k:str(v) for k,v in {'x':x*scale,'y':y*scale,'width':w*scale,'height':h*scale,'overflow':'visible'}.items()});container.append(svg)
        frames['background_ids'].append('trace10-'+p+'__canvas-'+p)
        for n in svg.iter():
            if n.tag==tag('clipPath'):new_clips.append({'id':n.get('id'),'evidence':{'purpose':'Preserved selected display limits; outside-view values retained and counted in numerical evidence','source_svg':ref(BUILD/('panel-'+p+'.svg')),'definition_sha256':sha(ET.tostring(n))}})
    present={n.get('id') for n in root.iter() if n.get('id')}
    frames['data_axes']=[a for a in frames.get('data_axes',[]) if all(k in present for k in ([a['element_id']] if 'element_id' in a else a.get('element_ids',[])))]
    frames['data_axes'] += [{'id':'F-pair','element_ids':['trace10-F__F-control-data-rectangle','trace10-F__F-trace-data-rectangle']},{'id':'I-observed','element_id':'trace10-I__I-observed-data-rectangle'}]
    frames['scientific_clips']=[v for v in frames.get('scientific_clips',[]) if v['id'] in present]+new_clips
    for r in frames.get('panel_relationships',[]):
        if r.get('id')=='fig2-E-B-F-C-column-centres-20261019':r['properties']=[v.replace('F placeholder centre','F data-pair centre') for v in r['properties']]
    lookup['layout-frames'].text=json.dumps(frames,sort_keys=True,separators=(',',':'))
    for key,h in protected.items():assert sha(ET.tostring(lookup[key]))==h
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    legend=('Figure 2 | Population conditioning measures and exploratory comparisons.\n\n'
      '(a–c) Control/conditioned population heatmaps: Delay28/29, 3sTrace90/89, 10sTrace32/29 fish. Direct1-s median log positive detected-bout vigor, centred within fish/trial by finite baseline-bin medians [−15,0) s, averaged with equal fish weight. managua_r saturates at ±0.25; black is missing. Green CS onset/offset and violet expected-US guides; 10sTrace US at20 s during training. Display trials1–90 correspond to acquisition5–94. Timing/coverage exclusions are recorded.\n\n'
      '(d–f) Fish/block median response-minus-baseline median log positive valid moving-frame vigor. Baseline[−15,0) s; response[0,9)/[0,13)/[0,20) s for Delay/3sTrace/10sTrace. PT/ET/LT:10–14/65–69/90–94, minimum3 eligible trials per5-trial block in every assay. Thin paired fish; black median/IQR. Display±0.25; outside-view values remain in analysis. Two-sided exact sign tests (zero/paired changes) and Brunner–Munzel t tests (independent blocks/paired changes), separate Holm18 per panel. D values unchanged; E eligibility and D/E annotations corrected. Dagger: versus zero; stars: adjusted p<0.05/0.01/0.001/0.0001.\n\n'
      '(g–i) Observed across-fish median log-vigor differences with pointwise95% intervals from5,000 whole-fish resamples(seed10). Separate within-panel families: baseline-adjusted block interaction(Holm8), local condition mean(BH9), local slope(raw/BH9), phase-aware random-intercept/slope LMM change from average Pre5–14(BH90). Row labels in G also apply to H/I; H/I maximum3 stars. Unavailable fits show n/a. Exploratory comparisons do not establish learning onset.\n\n'
      'C/F/I share the approved61-fish roster; 20230315_04(timing flag) and20250715_08(short recording) remain in audit/sensitivity records. The32 controls share historical3sTrace acquisition identities and are not independent new acquisitions across assays. Updated panels are unfrozen scientific drafts.')
    write(BUILD/'figure2-trace10.svg',svg);write(BUILD/'legend.md',legend.encode())
    page=render_page([{'id':'fig2','svg_bytes':svg,'legend_text':legend,'svg':ref(BUILD/'figure2-trace10.svg'),'legend':ref(BUILD/'legend.md'),'whole_figure_frozen':False}])[0];write(BUILD/'figure2-trace10.html',page)
    save(BUILD/'assembly-evidence.json',{'status':'candidate; saved-render and scientific consistency verification pending','svg':ref(BUILD/'figure2-trace10.svg'),'html':ref(BUILD/'figure2-trace10.html'),'legend':ref(BUILD/'legend.md'),'source':d['working_svg'],'registry':d['semantic_registry'],'protected_subtrees':protected,'C_mappings':c_mapping,'panel_registries':{p:ref(BUILD/('panel-'+p+'.figure.json')) for p in ['F','I']},'DE_correction':ref(corrected/'render-evidence.json'),'roster':ref(OUT/'common-roster.json'),'models':ref(MODELS/'completion.json'),'final_width_mm':final_width,'canvas_change':'136 to156 mm to fit I at unchanged G/H native scale; A/B/G/H subtree bytes unchanged','generator':ref(Path(__file__)),'profile_inputs':profile.inputs})
    ticket.check();print(BUILD/'figure2-trace10.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['build']);a=p.parse_args();build()
