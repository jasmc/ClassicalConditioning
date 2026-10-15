"""Read-only frame checks, browser review generation and bound evidence collection.

--review writes an explicit unmanaged external HTML review; browser output is
pasted/saved as JSON then checked with --browser-report. No current-state writer.
"""
from __future__ import annotations
import argparse
import base64
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from classical_conditioning.figures.layout_frames import COLLECTOR, assess, sha


def review_page(cases, title='Invisible frame review'):
    """cases: [{name, svg_base64}]; keep exact SVG bytes for hashing."""
    for case in cases:
        root = ET.fromstring(base64.b64decode(case['svg_base64']))
        for node in root.iter():
            if node.tag.rsplit('}',1)[-1] in {'script', 'foreignObject', 'animate', 'animateTransform', 'set'}:
                raise ValueError('Active SVG content is not supported in a frame review')
            for key, value in node.attrib.items():
                if key.lower().startswith('on'):
                    raise ValueError('SVG event attributes are not supported')
                if key.rsplit('}',1)[-1] == 'href' and not value.startswith(('#','data:')):
                    raise ValueError('Frame reviews require self-contained resources')
            css = (node.text or '') if node.tag.endswith('}style') else node.get('style','')
            if '@import' in css:
                raise ValueError('External CSS is not supported')
            for target in re.findall(r'url\(([^)]+)\)', css):
                if not target.strip().strip('\"\'').startswith(('#','data:')):
                    raise ValueError('External CSS resources are not supported')
    source = COLLECTOR.read_text()
    payload = json.dumps(cases).replace('<', '\\u003c')
    return '''<!doctype html><meta charset="utf-8"><title>'''+html.escape(title)+'''</title>
<style>body{font:15px/1.5 system-ui;margin:24px;color:#20313b;background:#eceff1}section{background:white;padding:20px;margin:20px 0;overflow:auto}h2{font-size:18px}svg{display:block;background:white}button{padding:10px 16px;margin:8px 8px 8px 0}pre{white-space:pre-wrap}textarea{width:95%;height:140px}@media print{body{margin:0;background:white}button,textarea{display:none}section{break-inside:avoid;padding:0}}</style>
<h1>'''+html.escape(title)+'''</h1><p>Comparison only. No current draft is changed. Dimensions refer to export/print size; screen size depends on display zoom. Choose a preset after visual review; no default is activated here.</p>
<button id="measure">Measure saved SVGs in this browser</button><button id="save" disabled>Save browser measurement JSON</button><p id="status">Not measured</p><main></main><textarea id="result" aria-label="Browser measurement JSON" readonly></textarea>
<script>'''+source+'''</script><script>
const cases='''+payload+''';
const collectorHash='''+json.dumps(sha(COLLECTOR.read_bytes()))+''';
const decode=b=>new TextDecoder().decode(Uint8Array.from(atob(b),c=>c.charCodeAt(0)));
for(const item of cases){const parsed=new DOMParser().parseFromString(decode(item.svg_base64),'image/svg+xml');
const fontStyle=document.createElement('style');fontStyle.textContent=[...parsed.querySelectorAll('style')].flatMap(n=>[...n.textContent.matchAll(/@font-face\\s*\\{[^}]*\\}/g)].map(m=>m[0])).join('\\n');document.head.append(fontStyle);
const section=document.createElement('section'),h=document.createElement('h2');h.textContent=item.name;section.append(h);
const holder=document.createElement('div');holder.style.font='initial';holder.attachShadow({mode:'open'}).append(document.importNode(new DOMParser().parseFromString(decode(item.svg_base64),'image/svg+xml').documentElement,true));section.append(holder);document.querySelector('main').append(section);}
document.querySelector('#measure').onclick=async()=>{
const results={};document.querySelector('#measure').disabled=true;
try{for(const item of cases){document.querySelector('#status').textContent='Measuring '+item.name;results[item.name]=await collectLayoutFrames(decode(item.svg_base64),collectorHash);}
document.querySelector('#result').value=JSON.stringify(results);document.querySelector('#save').disabled=false;document.querySelector('#status').textContent='Browser measurements complete; run the Python checker to evaluate them.';
}catch(e){document.querySelector('#status').textContent='Measurement unresolved: '+e;}finally{document.querySelector('#measure').disabled=false;}};
document.querySelector('#save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([document.querySelector('#result').value],{type:'application/json'}));a.download='layout-frame-browser-measurements.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
</script>'''


def external_destination(path):
    path=Path(path).resolve()
    if path.is_relative_to(ROOT) or not path.parent.is_dir():
        raise ValueError('Use an existing explicit external directory')
    if path.exists():
        raise FileExistsError(path)
    if path.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():
        raise ValueError('External storage is not mounted')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('svg',type=Path);p.add_argument('--browser-report',type=Path)
    p.add_argument('--review',type=Path);p.add_argument('--evidence',type=Path)
    args=p.parse_args()
    from classical_conditioning.figure_workspace import Snapshot, state_lock
    from classical_conditioning.artifacts import artifact_staging, publish_transaction
    from classical_conditioning.figures.layout_frames import POLICY
    snapshot=Snapshot([args.svg, POLICY, COLLECTOR, COLLECTOR.with_name('layout_frames.py'), Path(__file__),
                       *([args.browser_report] if args.browser_report else [])])
    data=args.svg.read_bytes();outputs=[]
    if args.review:
        destination=external_destination(args.review)
        outputs.append((destination,review_page([{'name':args.svg.name,'svg_base64':base64.b64encode(data).decode()}]).encode()))
    browser=json.loads(args.browser_report.read_text()) if args.browser_report else None
    if browser and 'artifact_sha256' not in browser:
        browser=browser[args.svg.name]
    report=assess(data,browser)
    if args.evidence:
        outputs.append((external_destination(args.evidence),(json.dumps(report,indent=2)+'\n').encode()))
    if outputs:
        if len({p for p,_ in outputs})!=len(outputs):raise ValueError('Review and evidence need distinct destinations')
        with artifact_staging(outputs[0][0].parent,prefix='.frame-check-') as staging:
            units=[]
            for index,(destination,payload) in enumerate(outputs):
                staged=staging/str(index);staged.write_bytes(payload);units.append((staged,destination))
            def verify():
                snapshot.check()
                for destination,payload in outputs:
                    if destination.read_bytes()!=payload:raise ValueError('Saved frame-check artifact mismatch')
            with state_lock(outputs[0][0]):
                snapshot.check();publish_transaction(units,staging,overwrite=False,verify_published=verify)
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':
    raise SystemExit(main())
