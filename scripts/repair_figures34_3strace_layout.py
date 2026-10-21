"""Guarded translation of the final profile row into its physical inset.

All scientific artists/labels and data coordinates remain byte-equivalent
inside each translated axes group; the canvas and all other rows are fixed.
"""
import json
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from classical_conditioning.artifacts import artifact_staging,publish_transaction
from analyze_figures34_3strace import ROOT,OUT,read,write,bound,verify

NS='{http://www.w3.org/2000/svg}'
ET.register_namespace('',NS[1:-1]);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def main():
    lock=OUT/'layout-repair.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        prior=read(OUT/'render-complete.json')
        for item in prior['artifacts']:verify(item)
        for item in read(OUT/'render-start.json')['inputs']:verify(item)
        before=bound(OUT/'render-complete.json');code=bound(Path(__file__))
        history=(OUT/'render-history/attempt-2-bottom-margin').resolve()
        assert history.is_relative_to(OUT.resolve())
        history.mkdir(parents=True)
        shutil.copy2(OUT/'render-complete.json',history/'render-complete.json')
        changes=[];new_hashes={};units=[]
        with artifact_staging(OUT,prefix='.layout-repair-') as staging:
            for name,start in [('figure4-condition-profiles',27),('figure4-individual-catches',12)]:
                svg=OUT/(name+'.svg');page=OUT/(name+'.html');old=svg.read_bytes();oldhtml=page.read_text(encoding='utf-8')
                root=ET.fromstring(old);view=list(map(float,root.attrib['viewBox'].split()))
                height=float(root.get('height').removesuffix('mm'))
                delta=1.25*view[3]/height
                ids=[]
                for index in range(start,start+3):
                    identity=name+f'_axes_{index}';group=root.find('.//'+NS+f'g[@id="{identity}"]')
                    assert group is not None and not group.get('transform')
                    original_node=ET.fromstring(ET.tostring(group));original_node.tail=None
                    original=ET.tostring(original_node)
                    group.set('transform',f'translate(0 {-delta:.12g})')
                    check=ET.fromstring(ET.tostring(group));check.attrib.pop('transform');check.tail=None
                    assert ET.tostring(check)==original
                    ids.append(identity)
                new=ET.tostring(root,encoding='utf-8',xml_declaration=True)
                assert old.decode('utf-8') in oldhtml
                newhtml=oldhtml.replace(old.decode('utf-8'),new.decode('utf-8'))
                for p in [svg,page]:shutil.copy2(p,history/p.name)
                stagedsvg=staging/svg.name;stagedsvg.write_bytes(new)
                stagedhtml=staging/page.name;stagedhtml.write_text(newhtml,encoding='utf-8')
                units.extend([(stagedsvg,svg),(stagedhtml,page)])
                new_hashes[str(svg)]={'path':str(svg),'sha256':bound(stagedsvg)['sha256']}
                changes.append({'source_svg':bound(svg),'source_html':bound(page),'output_svg':new_hashes[str(svg)],
                    'output_html':{'path':str(page),'sha256':bound(stagedhtml)['sha256']},
                    'translated_axes':ids,'translation_mm':[0,-1.25],
                    'scientific_artist_check':'all group children and attributes unchanged except recorded owning-group translation',
                    'reason':'measured xlabel envelope extended about 1.048 mm beyond the required bottom inset'})
            evidence={'code':code,'source_manifest':before,'changes':changes,'status':'layout-only; no scientific selection or freeze'}
            evidence_path=staging/'layout-repair.json';write(evidence_path,evidence)
            prior['artifacts']=[new_hashes.get(x['path'],x) for x in prior['artifacts']]
            prior['layout_repair']={'path':str(OUT/'layout-repair.json'),'sha256':bound(evidence_path)['sha256']}
            manifest=staging/'render-complete.json';write(manifest,prior)
            verify(before);verify(code)
            for change in changes:verify(change['source_svg']);verify(change['source_html'])
            def verify_new():
                verify(code)
                for change in changes:verify(change['output_svg']);verify(change['output_html'])
                for item in read(OUT/'render-start.json')['inputs']:verify(item)
            publish_transaction(tuple(units+[(evidence_path,OUT/'layout-repair.json'),(manifest,OUT/'render-complete.json')]),
                staging,overwrite=True,verify_published=verify_new)
        for name,_ in [('figure4-condition-profiles',27),('figure4-individual-catches',12)]:
            for suffix in ['.json','.png']:
                p=OUT/'browser-verification'/(name+suffix)
                if p.exists():p.rename(history/p.name)
        print(json.dumps(evidence,indent=2))
    finally:lock.unlink()

if __name__=='__main__':main()
