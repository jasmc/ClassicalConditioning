"""Reconcile final I spacing through exact measured bounds; preserve first candidate."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src'), str(_COMMAND_REPOSITORY / 'scripts')]

from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
REPO=_COMMAND_REPOSITORY
sys.path.insert(0,str(REPO/'src'))
from scripts.paper.figure_02.trace10 import finalize as finish
from scripts.paper.figure_02.trace10.analyze_figure2_trace10 import OUT,load,ref,save
from scripts.paper.figure_02.trace10.assemble_figure2_trace10 import write,enc
from classical_conditioning.operations.candidate_paths import candidate_directory, evidence_file
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot,tag
from scripts.main_paper_figures import render
DEFAULT_FINAL=OUT/'final-spacing'

def prepare(final_dir=DEFAULT_FINAL, *, previous_browser_evidence):
    final_dir=candidate_directory(final_dir)
    previous_browser_evidence=evidence_file(previous_browser_evidence)
    old=OUT/'final';report=load(previous_browser_evidence);a=report['allocations']
    gap=a['H']['painted_bounds_mm'][0]-sum(a['G']['painted_bounds_mm'][::2])
    current=a['I']['painted_bounds_mm'][0]-sum(a['H']['painted_bounds_mm'][::2])
    dx=gap-current
    root=ET.fromstring((old/'figure2.svg').read_bytes());scale=float(root.get('viewBox').split()[2])/156
    panel=next(n for n in root.iter() if n.get('id')=='panel-I');nested=next(n for n in panel if n.tag==tag('svg'))
    nested.set('x',str(float(nested.get('x'))+dx*scale))
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    final_dir.mkdir(exist_ok=True);write(final_dir/'figure2.svg',svg);write(final_dir/'legend.md',(old/'legend.md').read_bytes())
    legend=(final_dir/'legend.md').read_text(encoding='utf-8');page=render([{'id':'fig2','svg_bytes':svg,'legend_text':legend,'svg':ref(final_dir/'figure2.svg'),'legend':ref(final_dir/'legend.md'),'whole_figure_frozen':False}])[0];write(final_dir/'figure2.html',page)
    ticket=Snapshot([]);ticket.bind_hashes(load(old/'starting-ticket.json'));ticket.include([*command_sources(__file__),old/'figure2.svg',old/'candidate.json',Path(previous_browser_evidence)]);ticket.check()
    save(final_dir/'starting-ticket.json',ticket.hashes)
    candidate=load(old/'candidate.json');candidate.update(svg=ref(final_dir/'figure2.svg'),html=ref(final_dir/'figure2.html'),legend=ref(final_dir/'legend.md'),generator=ref(Path(__file__)),spacing_reconciliation={'I_dx_mm':dx,'target_equal_GH_HI_content_gap_mm':gap,'prior_HI_gap_mm':current,'prior_candidate':ref(old/'candidate.json')})
    save(final_dir/'candidate.json',candidate);print(dx)

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','verify','publish'])
    p.add_argument('--candidate-dir',type=Path,default=DEFAULT_FINAL)
    p.add_argument('--previous-browser-evidence',type=Path)
    p.add_argument('--browser-evidence',type=Path)
    p.add_argument('--visual-review-note')
    a=p.parse_args()
    if a.operation=='prepare' and a.previous_browser_evidence:
        prepare(a.candidate_dir,previous_browser_evidence=a.previous_browser_evidence)
    elif a.operation=='verify' and a.browser_evidence:
        finish.verify(a.candidate_dir,browser_evidence=a.browser_evidence)
    elif a.operation=='publish' and a.visual_review_note:
        finish.publish(a.visual_review_note,a.candidate_dir)
    else:p.error('prepare/verify require their explicit browser evidence; publish requires an actual visual-review note')

if __name__=='__main__':cli()
