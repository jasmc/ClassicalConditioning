"""Move scoped D/E/F sample counts and method notes to the bound legend."""
from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from classical_conditioning.figure_workspace import Snapshot,tag,sha,file_hash,publish_state,state_lock
from build_figure_legend_review import svg_editing_page
STATE=ROOT/'docs/paper/figures/figure-02/current-state.json';ASTATE=STATE.parent/'panel-a/current-state.json';EVIDENCE=ROOT/'records/figure-reviews/figure-02/block-notes-to-legend-20261015.json';TEMP=Path('/tmp/fig2-block-notes.svg')
def enc(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def prepare():
    snap=Snapshot([]);snap.hashes=json.loads(Path('/tmp/fig2-block-notes-ticket.json').read_text());snap.hashes[str(Path(__file__).resolve())]=file_hash(__file__);snap.check()
    s=json.loads(STATE.read_text());d=s['draft']
    for key in ('working_svg','review','legend','semantic_registry'):assert file_hash(d[key]['path'])==d[key]['sha256'],'Preserve edited input'
    root=ET.parse(d['working_svg']['path']).getroot();before={n.get('id'):sha(ET.tostring(n)) for n in root if n.get('id') not in ('panel-D','panel-E','panel-F')};removed=[]
    for panel in root:
        if panel.get('id') not in ('panel-D','panel-E','panel-F'):continue
        parents={c:p for p in panel.iter() for c in p}
        for text in list(panel.iter(tag('text'))):
            value=text.text or ''
            if value.startswith('n=') or value in ('Fish median/IQR · Holm36','† versus 0 · sign / BM'):
                parent=parents[text];target=parent if parent.get('data-source-id','').endswith(('sample-count','summary-note','statistics-note')) else text
                parents[target].remove(target)
                removed.append({'panel':panel.get('id')[-1],'element_id':target.get('id'),'text':value,'destination':'D/E block-count and methods paragraphs in associated legend'})
    assert len(removed)==8
    after={n.get('id'):sha(ET.tostring(n)) for n in root if n.get('id') not in ('panel-D','panel-E','panel-F')};assert before==after
    legend=Path(d['legend']['path']).read_text();needle='Displayed block counts identify contributing fish.'
    assert needle in legend
    legend=legend.replace(needle,'Contributing fish counts in PT/ET/LT order are: D, Delay 29/29/29 and control 28/24/26; E, 3 s Trace 40/40/40 and control 19/19/18.')
    svg=ET.tostring(root);TEMP.write_bytes(svg);TEMP.with_suffix('.md').write_text(legend);TEMP.with_suffix('.html').write_text(svg_editing_page(svg,scope_label='Figure 2',legend=legend))
    evidence=json.loads(Path(d['semantic_registry']['path']).read_text())
    evidence['block_notes_to_legend']={'scope':['D','E','F'],'author_direction':'2026-10-15: move n=29/29/29-like counts and Fish median/IQR · Holm36; dagger versus zero sign/BM methods to legend','source_svg':d['working_svg'],'generator':{'path':str(Path(__file__).resolve()),'sha256':file_hash(__file__)},'removed':removed,'other_panels_unchanged':before,'candidate_sha256':sha(svg),'legend_sha256':sha(legend.encode()),'scientific_values_and_statistical_marks':'unchanged; only named count and methods text removed','layout_policy':{'path':str(ROOT/'configs/paper-figures/layout-frames.json'),'sha256':file_hash(ROOT/'configs/paper-figures/layout-frames.json'),'coverage':'No geometry changed; prior incomplete frame/browser coverage retained; no new freeze'},'visual_review':{'status':'pending'}}
    for item in removed:
        record=evidence.get('element_registry',{}).get(item['element_id'])
        if record:record.update(required_in_svg=False,presentation_location='associated legend',status='moved by author direction 2026-10-15')
    TEMP.with_suffix('.json').write_bytes(enc(evidence));Path('/tmp/fig2-block-notes-publish-ticket.json').write_bytes(enc(snap.hashes));print(TEMP)
def publish():
    snap=Snapshot([]);snap.hashes=json.loads(Path('/tmp/fig2-block-notes-publish-ticket.json').read_text());snap.check();s=json.loads(STATE.read_text());d=s['draft'];e=json.loads(TEMP.with_suffix('.json').read_text());change=e['block_notes_to_legend'];svg=TEMP.read_bytes();html=TEMP.with_suffix('.html').read_bytes();legend=TEMP.with_suffix('.md').read_bytes()
    assert change['visual_review']['status']=='passed' and sha(svg)==change['candidate_sha256'] and sha(legend)==change['legend_sha256'] and svg in html and file_hash(__file__)==change['generator']['sha256']
    history=Path(d['working_svg']['path']).parent/'history/figure2-before-block-notes-to-legend-20261015';history.mkdir(parents=True,exist_ok=True)
    for key in ('working_svg','review','legend'):
        src=Path(d[key]['path']);dst=history/src.name
        if dst.exists():assert dst.read_bytes()==src.read_bytes()
        else:
            with dst.open('xb') as f:f.write(src.read_bytes())
    for src in (STATE,ASTATE):
        dst=history/('panel-a-state.json' if src==ASTATE else 'current-state.json')
        if not dst.exists():dst.write_bytes(src.read_bytes())
    a=json.loads(ASTATE.read_text());a[a['default_display']]['legend']['sha256']=sha(legend)
    # Scoped A paragraph/review bytes remain identical; only the shared legend
    # document hash changes because D/E paragraphs acquired their moved counts.
    eb=enc(e);d['working_svg']['sha256']=sha(svg);d['review']['sha256']=sha(html);d['legend']['sha256']=sha(legend);d['semantic_registry']={'path':str(EVIDENCE),'sha256':sha(eb)}
    d['consistency_requests'].append({'id':'fig2-def-counts-and-methods-in-legend','scope':['D','E','F'],'author_direction':change['author_direction'],'status':'verified','properties':['cohort block counts and method captions in legend','all scientific marks, scales, fonts and geometry retained'],'evidence':{'path':str(EVIDENCE),'sha256':sha(eb)}})
    d['reproduction']={'generator':change['generator'],'command':'.venv/bin/python scripts/move_figure2_block_notes.py; review saved candidate; --publish with captured ticket'}
    with state_lock(ASTATE):publish_state(STATE,s,snap,artifacts=[(EVIDENCE,eb)],evidence_updates=[(Path(d['working_svg']['path']),svg),(Path(d['review']['path']),html),(Path(d['legend']['path']),legend),(ASTATE,enc(a))])
    print(d['review']['path'])
if __name__=='__main__':publish() if '--publish' in sys.argv else prepare()
