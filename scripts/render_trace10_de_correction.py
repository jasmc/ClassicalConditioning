"""Reviewable D/E correction in the current Figure 2 geometry; no publication."""
from pathlib import Path
import copy,json,sys,xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from classical_conditioning.figure_workspace import Snapshot,tag,sha
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme
from classical_conditioning.figures.export import FigureProvenance,FigureMode,export_matplotlib_figure
from analyze_figure2_trace10 import OUT,STATE,load,ref,save,check
from main_paper_figures import render as render_page
import assemble_figure2_trace3_matched as selected

def main():
    data_dir=OUT/'DE-corrections';directory=data_dir/'render';directory.mkdir(exist_ok=True)
    report=load(data_dir/'calculation.json');check(report['outputs'])
    state=load(STATE);draft=state['draft'];check([draft[k] for k in ['working_svg','review','legend','semantic_registry']])
    source=Path(draft['working_svg']['path']);root=ET.fromstring(source.read_bytes())
    ticket=Snapshot([STATE,source,Path(draft['review']['path']),Path(draft['legend']['path']),Path(draft['semantic_registry']['path']),Path(__file__),REPO/'scripts/assemble_figure2_trace3_matched.py'])
    save(directory/'starting-ticket.json',ticket.hashes)
    lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    protected={k:sha(ET.tostring(lookup[k])) for k in ['layout-A','layout-B','layout-C','panel-F','panel-G','panel-H','panel-I','fig2__shared__colorbar__axes']}
    tests=pd.concat([pd.read_csv(data_dir/(p+'-statistics-Holm18.csv')) for p in ['D','E']])
    # Compatibility field used solely by the authenticated renderer's lookup.
    # Its values are the new separate-panel Holm18 values, not old Holm36 data.
    tests['p_holm36']=tests.p_holm18
    selected.OUT=data_dir
    base_ref=selected.ref
    def renderer_ref(p):
        p=Path(p)
        if p.name=='DE-statistics-Holm36.csv':return {'panel_statistics':[ref(data_dir/(q+'-statistics-Holm18.csv')) for q in ['D','E']],'family':'separate Holm18 per panel','compatibility_lookup_field':'p_holm36 aliases p_holm18 inside reference renderer only'}
        return base_ref(p)
    selected.ref=renderer_ref
    profile=load_presentation_profile('nature-main');frames=json.loads(lookup['layout-frames'].text)
    clips=[];maps={};outside={}
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        for p in ['D','E']:
            container=lookup['panel-'+p];old=next(n for n in container if n.tag==tag('svg'))
            width=float(old.get('viewBox').split()[2])*25.4/72
            fig,ids,mapping,extra,size=selected.block_panel(p,width,profile,theme,tests,y_limits=(-.25,.25),y_ticks=[-.2,-.1,0,.1,.2])
            spec={'scope':'panel','final_size_mm':size,'inset_mm':1.,'allocations':[{'id':p,'kind':'panel','box_mm':[0,0,*size],'group_ids':['axes__'+i.lower()+'__main' for i in ids]+extra}],'background_ids':['canvas-'+p],'data_axes':[],'alignments':[]}
            provenance=FigureProvenance(figure_id='Figure 2 '+p,analysis_recipe='author-separate-Holm18-min3',source_file=str(Path(__file__)),source_symbol='main',source_hash=ref(Path(__file__))['sha256'],reproduction_snippet='.venv-trace/Scripts/python.exe scripts/render_trace10_de_correction.py',input_artifacts=(ref(data_dir/(p+'-fish-blocks.parquet')),ref(data_dir/(p+'-statistics-Holm18.csv'))),artist_mappings=mapping,analysis_identity={'status':'unfrozen draft; original current source preserved','family':'Holm18 within '+p,'minimum_trials':3,'D_values_unchanged':p=='D','profile_inputs':profile.inputs})
            export_matplotlib_figure(fig,directory/('panel-'+p),provenance,mode=FigureMode.PUBLICATION,panel_ids=ids,allow_dirty_publication=True,formats=('svg',),layout_frames=spec);plt.close(fig)
            svg=ET.fromstring((directory/('panel-'+p+'.svg')).read_bytes());selected.namespace(svg,'matched-'+p+'__')
            for n in list(svg):
                if n.get('id')=='matched-'+p+'__layout-frames':svg.remove(n)
            attrs={k:old.get(k) for k in ['x','y','width','height','overflow']};svg.attrib.update({k:v for k,v in attrs.items() if v is not None})
            # Preserve the current column-aligned letter placement exactly.
            old_letter=lookup.get('matched-'+p+'__'+p+'-letter')
            new_letter=next(n for n in svg.iter() if n.get('id')=='matched-'+p+'__'+p+'-letter')
            if old_letter is not None and old_letter.get('transform'):new_letter.set('transform',old_letter.get('transform'))
            for child in list(container):container.remove(child)
            container.append(svg);maps[p]=ref(directory/('panel-'+p+'.figure.json'))
            for n in svg.iter():
                if n.tag==tag('clipPath'):clips.append({'id':n.get('id'),'evidence':{'purpose':'Preserved author display range [-0.25,0.25]; all values retained for statistics','source_svg':ref(directory/('panel-'+p+'.svg')),'definition_sha256':sha(ET.tostring(n))}})
            values=pd.read_parquet(data_dir/(p+'-fish-blocks.parquet'));values=values.loc[values.Eligible,'Fish median response / baseline']
            outside[p]={'below_view':int((values<-.25).sum()),'above_view':int((values>.25).sum())}
    existing={n.get('id') for n in root.iter() if n.get('id')}
    frames['scientific_clips']=[c for c in frames.get('scientific_clips',[]) if c['id'] in existing]+clips
    lookup['layout-frames'].text=json.dumps(frames,sort_keys=True,separators=(',',':'))
    for k,h in protected.items():assert sha(ET.tostring(lookup[k]))==h
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    legend=Path(draft['legend']['path']).read_text(encoding='utf-8')
    legend=legend.replace('minimum eligible trials 3/1 for Delay/3sTrace','minimum three eligible trials per five-trial block for both Delay and 3sTrace').replace('joint Holm36','separate within-panel Holm18').replace('D data are unchanged; adjusted annotations were recomputed jointly with E.','D data are unchanged; D and E adjusted annotations were recomputed within each panel. E contributing blocks were recomputed with the common three-trial minimum.')
    for p,b in [(directory/'figure2-DE-corrected.svg',svg),(directory/'legend.md',legend.encode())]:
        with p.open('xb') as f:f.write(b)
    page=render_page([{'id':'fig2','svg_bytes':svg,'legend_text':legend,'svg':ref(directory/'figure2-DE-corrected.svg'),'legend':ref(directory/'legend.md'),'whole_figure_frozen':False}])[0]
    with (directory/'figure2-DE-corrected.html').open('xb') as f:f.write(page)
    save(directory/'render-evidence.json',{'status':'rendered candidate; exact saved-render verification pending; not published',
        'svg':ref(directory/'figure2-DE-corrected.svg'),'html':ref(directory/'figure2-DE-corrected.html'),'legend':ref(directory/'legend.md'),
        'source':ref(source),'statistics':ref(data_dir/'calculation.json'),'protected_subtrees':protected,'panel_registries':maps,
        'outside_view':outside,'profile_inputs':profile.inputs,'generator':ref(Path(__file__)),
        'compatibility_adapter':'reference renderer uses field p_holm36; aliases only new per-panel p_holm18 with explicit mappings, no old correction reused'})
    ticket.check();print(directory/'figure2-DE-corrected.html')

if __name__=='__main__':main()
