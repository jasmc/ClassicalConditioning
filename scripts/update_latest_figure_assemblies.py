"""Compose the latest selected panels into one self-contained review HTML.

No scientific recomputation or new freeze; all original panel bytes are embedded.
"""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[1] / "src"))
from classical_conditioning.external_artifacts import resolve_artifact, external_output
from pathlib import Path
import base64, copy, hashlib, html, json, re
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
SSD = Path('/Volumes/JOAQUIM')
OLD = SSD / 'ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs'
PC = SSD / 'ClassicalConditioning-Mac-20261009/pc-outputs'
OUT = external_output('reviews/latest_figure_assemblies_20261010.html')
SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
def tag(s): return '{'+SVG+'}'+s
def sha(b): return hashlib.sha256(b).hexdigest()
def mapped(s):
    s=s.replace('\\','/')
    for old,new in [('J:',SSD),('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs',PC),('C:/Users/joaquim/Documents/ClassicalConditioning',REPO)]:
        if s.startswith(old+'/'):return new/s[len(old)+1:]
    return Path(s)
def archive(path, script, entry, expected):
    text=path.read_text()
    m=re.search(r'<script\b[^>]*id="'+re.escape(script)+r'"[^>]*>(.*?)</script>',text,re.S)
    if not m:raise ValueError(f'Missing archive {script}')
    item=json.loads(m[1])['files'][entry]
    b=base64.b64decode(item['base64'])
    assert sha(b)==item['sha256']==expected
    return b, {'path':str(path),'script_id':script,'entry':entry,'sha256':expected}
def source(path, expected=None):
    b=path.read_bytes(); h=sha(b)
    if expected:assert h==expected,(path,h,expected)
    return b,{'path':str(path),'sha256':h}
def panel(root,p,b,records,prov):
    x,y,w,h=p['box']; cx,cy,cw,ch=p.get('content_box',p['box'])
    assert x>=0 and y>=0 and w>0 and h>0
    ET.fromstring(b) if b else None
    g=ET.SubElement(root,tag('g'),{'id':'panel-'+p['id']})
    if b:
        ET.SubElement(g,tag('image'),{'x':str(cx),'y':str(cy),'width':str(cw),'height':str(ch),'preserveAspectRatio':'xMidYMid meet','href':'data:image/svg+xml;base64,'+base64.b64encode(b).decode()})
    else:
        ET.SubElement(g,tag('rect'),{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'fill':'#fafafa','stroke':'#aaa','stroke-dasharray':'7 5'})
        lines=p.get('placeholder_lines',[p['id']+': pending'])
        for i,line in enumerate(lines):
            ET.SubElement(g,tag('text'),{'x':str(x+w/2),'y':str(y+h/2+(i-len(lines)/2)*32),'text-anchor':'middle','font-size':'20'}).text=line
    if p.get('label'):
        l=p['label']; ET.SubElement(g,tag('text'),{'x':str(l['x']),'y':str(l['y']),'font-size':str(l['size']),'font-weight':'bold'}).text=l['text']
    records.append({'id':p['id'],'box':p['box'],'selection':p.get('selection_status'),'source':prov,'status':'embedded source' if b else 'placeholder'})
def build(n):
    layout=json.loads((REPO/f'configs/paper-figures/figure{n}-assembly.json').read_text())
    panels=copy.deepcopy(layout['panels']); sources={}; provenance={}
    for p in panels:
        if p.get('source'):
            path=mapped(p['source']) if ':' in p['source'] else OLD/f'figure{n}-assembly'/p['source']
            sources[p['id']],provenance[p['id']]=source(path,p.get('frozen_sha256') or p.get('source_provenance',{}).get('svg_sha256'))
    if n==1:
        # Wide selected E and F/G/H row keep their approved source aspect ratios.
        panels=panels[:4]
        panels[3]['box']=[55,940,1690,480];panels[3]['label'].update(x=52,y=975)
        ep=json.loads((REPO/'configs/paper-figures/selections/figure1-panel-e-heatmap-rows-freeze-20261009.json').read_text())
        fp=json.loads((REPO/'configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json').read_text())
        sources['E'],provenance['E']=archive(mapped(ep['container']['path']),'panel-e-frozen-archive',ep['export']['embedded_entry'],ep['export']['sha256'])
        sources['FGH'],provenance['FGH']=archive(resolve_artifact('reviews/fgh_candidate_palette_versions_20261009/index.html'),'version12-frozen-archive','frozen-candidate.svg',fp['exports'][0]['sha256'])
        panels.extend([{'id':'E','box':[0,1460,1800,1200],'label':{'text':'E','x':45,'y':1500,'size':32},'selection_status':'selected frozen Panel E, 2026-10-09'}, {'id':'FGH','box':[0,2700,1800,1020],'selection_status':'selected frozen F/G/H Version 12, 2026-10-09'}])
        canvas=[1800,3750];annotations=[]
    else:
        de=json.loads((REPO/'configs/paper-figures/selections/figure2-DE-B-freeze-20261009.json').read_text())
        gs=json.loads((REPO/'configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json').read_text())
        for pid in 'DE':
            p=next(p for p in panels if p['id']==pid)
            item=de['current_panels'][pid]['svg'];sources[pid],provenance[pid]=source(resolve_artifact(f'reviews/figure2_B_freeze_20261009/Fig2_{pid}_B.svg', expected_sha256=item['sha256']),item['sha256'])
            p['box'][3]=560;p['content_box']=[p['box'][0],830,540,523];p['label']={'text':pid,'x':p['box'][0]+18,'y':p['box'][1]+35,'size':32};p['selection_status']='selected frozen Version B, 2026-10-09'
        panels[5]['box'][3]=560
        g=panels[6];g['box']=[0,1400,1800,964];g.pop('content_box',None);g.pop('label',None);g['selection_status']=gs['selected_version']
        sources['G'],provenance['G']=source(mapped(gs['svg']['path']),gs['svg']['sha256'])
        for p,x in [(panels[7],330),(panels[8],930)]:
            p['box']=[x,2430,540,390];p['content_box']=[x+18,2505,504,290];p['label'].update(x=x+18,y=2473)
        canvas=[1800,2920]
        annotations=[a for a in layout['annotations'] if not a.get('panel_note') and a['y']<160]
        annotations[1]['text']='Latest selected D/E Version B and G LogMedian; other populated panels retain their previous sources'
        annotations += [{'x':45,'y':2890,'size':20,'text':'Assembly review; whole figure not frozen. C/F/I pending. A/B/H retain their earlier scientific definitions.'}]
    root=ET.Element(tag('svg'),{'viewBox':f'0 0 {canvas[0]} {canvas[1]}','width':str(canvas[0]),'height':str(canvas[1]),'font-family':'DejaVu Sans, sans-serif','role':'img','aria-label':f'Figure {n} latest selected panel assembly'})
    ET.SubElement(root,tag('rect'),{'width':str(canvas[0]),'height':str(canvas[1]),'fill':'white'})
    records=[]
    for p in panels:
        assert p['box'][0]+p['box'][2]<=canvas[0] and p['box'][1]+p['box'][3]<=canvas[1]
        panel(root,p,sources.get(p['id']),records,provenance.get(p['id']))
    for a in annotations:
        ET.SubElement(root,tag('text'),{'x':str(a['x']),'y':str(a['y']),'font-size':str(a['size']),'font-weight':a.get('weight','normal'),'text-anchor':a.get('anchor','start')}).text=a['text']
    b=ET.tostring(root,encoding='utf-8')
    return b,{'figure':n,'canvas':canvas,'svg_sha256':sha(b),'panels':records,'layout':{'panels':panels,'annotations':annotations},'scope':'assembly review; no new freeze or scientific recomputation'}
def main():
    assembled=[build(n) for n in (1,2)]
    text='<!doctype html><html><head><meta charset="utf-8"><title>Latest Figures 1 and 2 — 10 October 2026</title><style>body{font-family:system-ui;margin:24px;background:#eee;color:#222}main{max-width:1200px;margin:auto}img.figure{width:100%;display:block;background:white}section{margin:30px 0}details{background:white;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}nav{display:flex;gap:24px}</style></head><body><main><h1>Latest Figures 1 and 2</h1><p>10 October 2026. Selected frozen panels embedded with their exact source bytes. Updated assembly reviews; whole figures are not newly frozen.</p><nav><a href="#figure1">Figure 1</a><a href="#figure2">Figure 2</a></nav>'
    for n,(b,meta) in enumerate(assembled,1):
        note='A–D retained; E replaced by frozen raw-trace/heatmap rows; F/G/H replaced by frozen Version 12.' if n==1 else 'D/E replaced by frozen Version B; G replaced by frozen LogMedian with its statistics. A/B/H retained; C/F/I pending. The first row has not yet been recalculated using V12.'
        text+=f'<section id="figure{n}"><h2>Figure {n}</h2><p>{note}</p><img class="figure" alt="Figure {n} assembly" src="data:image/svg+xml;base64,{base64.b64encode(b).decode()}"></section>'
    metadata={'created':'2026-10-10','generator':str(Path(__file__).resolve()),'generator_sha256':sha(Path(__file__).read_bytes()),'figures':[m for b,m in assembled]}
    text+='<details><summary>Panel provenance and assembly layouts</summary><pre>'+html.escape(json.dumps(metadata,indent=2))+'</pre></details><script type="application/json" id="assembly-provenance">'+json.dumps(metadata).replace('</','<\\/')+'</script></main></body></html>'
    OUT.write_text(text);print(OUT)
if __name__=='__main__':main()
