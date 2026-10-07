"""Render the user's specified trial-local centring and managua_r heatmaps."""
from pathlib import Path
import sys,json,copy
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT,FISH,digest,draw,q,text
from assemble_svg_figure import render,export_with_inkscape
OUT=ROOT/'trial-centred-managua-review-20261007'
SOURCE=ROOT/'cadence-review-v5-20261007'
OUT.mkdir(parents=True,exist_ok=True)
combined=ET.Element(q('svg'),{'width':'1650','height':'460','viewBox':'0 0 1650 460','font-family':'DejaVu Sans'})
text(combined,'Each trial centred on its own pre-CS baseline · managua_r',25,27,size=20,weight='bold')
text(combined,'Median ln(vigor) baseline [−15, 0) s; 0.5 s bout-median bins; colour limits ±0.25; missing bins black',25,51,size=14)
outputs={};reports=[]
for i,spec in enumerate(FISH):
    panel,name,fish,*_=spec
    source=SOURCE/f'Fig1_Panel{panel}_{name.replace(" ","")}_presumed-cadence_v5.svg'
    meta=json.loads(source.with_suffix('.svg.json').read_text())
    assert digest(source)==meta['svg_sha256']
    data=Path(meta['panel_data']);assert digest(data)==meta['panel_data_sha256']
    bins=pd.read_parquet(data)
    assert len(bins)==7200 and bins['Baseline start (s)'].eq(-15).all() and bins['Baseline end (s)'].eq(0).all()
    tree=draw(spec,bins)
    tree.set('data-time-source','presumed-acquisition-cadence')
    tree.set('data-reference','each-trial-own-median-ln-vigor-pre-CS-minus15-to-0')
    svg=OUT/f'Fig1_Panel{panel}_{name.replace(" ","")}_trial-centred_managua-r.svg'
    ET.ElementTree(tree).write(svg,encoding='utf-8',xml_declaration=True)
    export_with_inkscape(svg,['png','pdf'],font_directory=ROOT/'fonts')
    report={**meta,'palette':'managua_r','display_range':[-.25,.25],
        'baseline_s':[-15,0],'reference_scope':'this trial only',
        'normalisation':'subtract trial median eligible frame ln(vigor); no range division',
        'selection_status':'user-specified rendering; preprocessing confirmation pending',
        'svg':str(svg),'svg_sha256':digest(svg),'verified_saved_data':str(data),
        'verified_saved_data_sha256':digest(data)}
    svg.with_suffix('.svg.json').write_text(json.dumps(report,indent=2))
    reports.append(report);outputs[panel]=svg
    group=ET.SubElement(combined,q('g'),{'transform':f'translate({i*550},65)'})
    for child in tree:group.append(copy.deepcopy(child))
    text(combined,panel,i*550+12,95,size=22,weight='bold')
joined=OUT/'Fig1_F-G-H_trial-centred_managua-r.svg'
ET.ElementTree(combined).write(joined,encoding='utf-8',xml_declaration=True)
export_with_inkscape(joined,['png','pdf'],font_directory=ROOT/'fonts')
config=json.loads((REPO/'configs/paper-figures/figure1-cadence-review-v5.json').read_text())
assembly=OUT/'figure1_trial-centred_managua-r.svg'
config['output']=str(assembly)
config['title']='Figure 1 · each trial centred on its own [-15,0) baseline · managua_r'
for p in config['panels']:
    if p['id'] in outputs:
        p['source']=str(outputs[p['id']]);p['selection_status']='user-specified trial centring and managua_r; preprocessing confirmation pending'
configpath=Path(__file__).parent/'trial-centred-managua-assembly.json'
configpath.write_text(json.dumps(config,indent=2))
render(configpath,assembly,strict=True)
export_with_inkscape(assembly,['png','pdf'],font_directory=ROOT/'fonts')
# Source artifacts are checked again after rendering; none were overwritten.
for report in reports:
    assert digest(Path(report['verified_saved_data']))==report['verified_saved_data_sha256']
manifest={'baseline_s':[-15,0],'reference_scope':'per trial','palette':'managua_r','display_range':[-.25,.25],
    'panels':reports,'assembly':str(assembly),'combined_heatmaps':str(joined),
    'builder':str(Path(__file__)),'builder_sha256':digest(Path(__file__)),
    'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()]}
(OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
print(joined);print(assembly)
