"""Reconcile final I spacing through exact measured bounds; preserve first candidate."""
from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import finish_figure2_trace10 as finish
from analyze_figure2_trace10 import OUT,load,ref,save
from assemble_figure2_trace10 import write,enc
from classical_conditioning.figure_workspace import Snapshot,tag
from main_paper_figures import render
FINAL=OUT/'final-spacing'
TEMP=Path(r'C:\Users\joaquim\.codex\visualizations\2026\10\18\01a14e0b-0191-7822-b4d7-47417feb0eff\trace10-final-v2')

def prepare():
    old=OUT/'final';report=load(finish.TEMP/'browser.json');a=report['allocations']
    gap=a['H']['painted_bounds_mm'][0]-sum(a['G']['painted_bounds_mm'][::2])
    current=a['I']['painted_bounds_mm'][0]-sum(a['H']['painted_bounds_mm'][::2])
    dx=gap-current
    root=ET.fromstring((old/'figure2.svg').read_bytes());scale=float(root.get('viewBox').split()[2])/156
    panel=next(n for n in root.iter() if n.get('id')=='panel-I');nested=next(n for n in panel if n.tag==tag('svg'))
    nested.set('x',str(float(nested.get('x'))+dx*scale))
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    FINAL.mkdir(exist_ok=True);write(FINAL/'figure2.svg',svg);write(FINAL/'legend.md',(old/'legend.md').read_bytes())
    legend=(FINAL/'legend.md').read_text(encoding='utf-8');page=render([{'id':'fig2','svg_bytes':svg,'legend_text':legend,'svg':ref(FINAL/'figure2.svg'),'legend':ref(FINAL/'legend.md'),'whole_figure_frozen':False}])[0];write(FINAL/'figure2.html',page)
    ticket=load(old/'starting-ticket.json');ticket.update(Snapshot([Path(__file__),old/'figure2.svg',old/'candidate.json',finish.TEMP/'browser.json']).hashes)
    save(FINAL/'starting-ticket.json',ticket)
    candidate=load(old/'candidate.json');candidate.update(svg=ref(FINAL/'figure2.svg'),html=ref(FINAL/'figure2.html'),legend=ref(FINAL/'legend.md'),generator=ref(Path(__file__)),spacing_reconciliation={'I_dx_mm':dx,'target_equal_GH_HI_content_gap_mm':gap,'prior_HI_gap_mm':current,'prior_candidate':ref(old/'candidate.json')})
    save(FINAL/'candidate.json',candidate);print(dx)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','verify','publish']);p.add_argument('--visual-review-note');a=p.parse_args()
    if a.operation=='prepare':prepare()
    else:
        finish.FINAL=FINAL;finish.TEMP=TEMP
        if a.operation=='verify':finish.verify()
        elif a.visual_review_note:finish.publish(a.visual_review_note)
        else:p.error('Actual visual note required')
