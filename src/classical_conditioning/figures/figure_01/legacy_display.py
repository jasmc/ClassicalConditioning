"""Historical Figure 1 display assembly; preserves source scientific mappings."""
import base64
import json
import re
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import tag, sha as digest
from classical_conditioning.figures.review_html import figure_font_paths

def embed_consistent_svg_fonts(data):
    """Keep labels as text; bind regular/bold faces so fallback cannot vary by panel."""
    root=ET.fromstring(data)
    for pid in ('D','E'):
        group=next(g for g in root if g.get('id')=='panel-'+pid)
        for node in group.iter(tag('text')):
            if node.get('data-source-id','').startswith('trial-') and node.get('data-source-id','').endswith('-name'):
                trial=int(node.get('data-source-id').split('-')[1]);node.text=f'Trial {trial-4}'
                node.set('data-acquisition-trial',str(trial));node.set('data-display-trial',str(trial-4))
    for node in root.iter(tag('g')):
        if any(node.get('data-source-id','')==f'fig1__{pid}__heatmap__axes' for pid in ('f','g','h')):
            node.set('data-trial-display-offset','4');node.set('data-first-display-row','1');node.set('data-last-display-row','90')
    # Match raw-data body viewports; the E scale has its own independent viewport.
    for pid in ('D','E'):
        group=next(g for g in root if g.get('id')=='panel-'+pid)
        panel=group.find(tag('svg'))
        panel.set('viewBox','0 0 750 550');panel.set('width','750')
        if pid=='E':
            scale=ET.SubElement(group,tag('svg'),{'id':'edit-E-independent-colourbar','x':str(float(panel.get('x'))+750),'y':panel.get('y'),'width':'135','height':'550','viewBox':'750 0 135 550','overflow':'hidden'})
            for node in list(panel):
                if node.get('id')=='edit-E-colour-scale' or node.get('id','').startswith('edit-E-colour__ytick') or node.get('id')=='edit-E-colour-label':
                    panel.remove(node);scale.append(node)
    for node in root.iter():
        if node.tag not in {tag('text'),tag('tspan')}:continue
        node.set('font-family','DejaVu Sans, sans-serif');node.set('font-style','normal')
        style=node.get('style','')
        style=re.sub(r'(?:font-family|font-style)\s*:[^;]*(?:;|$)','',style)
        node.set('style',style+';font-family:"DejaVu Sans",sans-serif !important;font-style:normal !important')
    defs=ET.SubElement(root,tag('defs'),{'id':'figure-font-resources'})
    css=[];resources=[]
    for weight,path in zip((400,700),figure_font_paths()):
        raw=path.read_bytes();encoded=base64.b64encode(raw).decode()
        css.append('@font-face{font-family:"DejaVu Sans";font-style:normal;font-weight:'+str(weight)+';src:url("data:font/ttf;base64,'+encoded+'") format("truetype");}')
        resources.append(dict(path=str(path),sha256=digest(raw),weight=weight))
    ET.SubElement(defs,tag('style'),{'type':'text/css'}).text=''.join(css)
    meta=next(n for n in root if n.get('id')=='editable-copy-provenance')
    record=json.loads(meta.text)
    for entry in record['scientific_registry'].values():
        if entry.get('panel_id') in {'d','e'} and 'row_label' in entry:
            source_trial=entry['data_mapping']['trial'];entry['row_label']['trial']=f'Trial {source_trial-4}';entry['row_label'].update(acquisition_trial=source_trial,display_trial=source_trial-4)
    record['trial_numbering']={'display_trial_offset':4,'source_acquisition_trial_ids_preserved':True,'scope':['D','E','F','G','H']}
    record['embedded_font_resources']=resources;record['font_policy']='All labels remain native SVG text; common DejaVu Sans regular/bold faces and sans-serif fallback';meta.text=json.dumps(record)
    return svg_bytes(root),resources
