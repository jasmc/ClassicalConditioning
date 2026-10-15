"""Guarded update of C box sizes and E colourbar title clearance only."""
from pathlib import Path
import argparse,copy,json,sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,file_hash,sha,tag,publish_state,measured_presentation
from build_figure_legend_review import svg_editing_page
from figure1_layout_spacing import apply_layout_spacing


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--publish',action='store_true');parser.add_argument('--visual-evidence',type=Path);args=parser.parse_args()
    ticket=json.loads(args.baseline.read_text());state_path=Path(ticket['state_path']);snap=Snapshot([]);snap.hashes=ticket['starting_hashes'];snap.check()
    state=json.loads(state_path.read_text());draft=state['draft'];svg=Path(draft['working_svg']['path']);html=Path(draft['review']['path']);legend=Path(draft['legend']['path']);measure=Path(draft['effective_presentation']['path']);epage=Path(draft['panel_views']['E']['path'])
    snap.hashes.update(Snapshot([Path(__file__),ROOT/'scripts/figure1_layout_spacing.py',ROOT/'scripts/build_figure_legend_review.py',ROOT/'scripts/update_figure1_panel_e_display.py',epage]).hashes)
    raw=svg.read_bytes();after,details=apply_layout_spacing(raw);repeated,_=apply_layout_spacing(after);assert repeated==after
    before=ET.fromstring(raw);root=ET.fromstring(after)
    preserved={}
    for p in before:
        if p.get('id','').startswith('panel-') and p.get('id') not in ['panel-C','panel-E']:
            new=next(n for n in root if n.get('id')==p.get('id'));assert ET.tostring(p)==ET.tostring(new);preserved[p.get('id')]=sha(ET.tostring(new))
    for panel in ['C','E']:
        a=next(n for n in before if n.get('id')=='panel-'+panel);b=next(n for n in root if n.get('id')=='panel-'+panel)
        assert [n.text for n in a.iter(tag('text'))]==[n.text for n in b.iter(tag('text'))],'Text/values changed'
    # E is preserved byte-for-byte outside the single colourbar subtree.
    ae=copy.deepcopy(next(n for n in before if n.get('id')=='panel-E'));be=copy.deepcopy(next(n for n in root if n.get('id')=='panel-E'))
    for branch in [ae,be]:
        for parent in branch.iter():
            for child in list(parent):
                if child.get('id')=='edit-E-colour-scale':parent.remove(child)
    for branch in [ae,be]:
        viewport=next(n for n in branch.iter() if n.get('id')=='edit-E-independent-colourbar')
        for key in ['x','width','viewBox']:viewport.set(key,'permitted colourbar viewport update')
    assert ET.tostring(ae)==ET.tostring(be)
    page=svg_editing_page(after,legend=legend.read_text()).encode();preview=Path('/tmp/figure1-CE-spacing-preview.html')
    if not args.publish:preview.write_bytes(page);print(json.dumps(dict(preview=str(preview),svg_sha256=sha(after),details=details)));return
    assert args.visual_evidence,'Actual browser measurements required for publication'
    visual=json.loads(args.visual_evidence.read_text());assert visual['svg_sha256']==sha(after)
    assert visual['checks_passed']
    snap.hashes.update(Snapshot([args.visual_evidence]).hashes)
    evidence=ROOT/'records/figure-reviews/figure-01/CE-layout-spacing-2026-10-15.json'
    report=dict(scope=['C','E'],starting_hashes=snap.hashes,source_code_before_edit=json.loads(Path('/tmp/figure1-CE-code-start.json').read_text()),source_svg_sha256=sha(raw),svg_sha256=sha(after),details=details,visual_review=visual,preserved_panels=preserved,E_outside_colourbar_preserved=True,F_arrowheads_preserved=True,all_text_and_scientific_values_preserved=True,frozen=False)
    record=(json.dumps(report,indent=2)+'\n').encode();ref=dict(path=str(evidence),sha256=sha(record))
    scoped=copy.deepcopy(root)
    for n in list(scoped):
        if n.get('id')!='panel-E' and n.tag not in [tag('defs'),tag('rect')]:scoped.remove(n)
    bounds=draft['panel_views']['E']['source_bounds'];scoped.attrib.update(viewBox=' '.join(map(str,bounds)),width='900',height='635',id='panel-E-scoped-view')
    text=legend.read_text();panel_legend=text[text.index('(E)'):text.index('\n\n(F–H)')]
    ep=svg_editing_page(ET.tostring(scoped),scope_label='Figure 1 · Panel E',legend=panel_legend).encode()
    measurement=json.loads(measure.read_text());measurement.update(measured_presentation(after));measurement['CE_layout_spacing']=report
    mb=(json.dumps(measurement,indent=2)+'\n').encode()
    draft['working_svg']['sha256']=sha(after);draft['review']['sha256']=sha(page);draft['effective_presentation']['sha256']=sha(mb)
    draft['panel_views']['E'].update(sha256=sha(ep),source_svg_sha256=sha(after),source_svg_path=str(svg))
    draft['layout_spacing_override']=ref;draft['semantic_registry']['CE_layout_spacing']=ref
    draft['consistency_requests'].append(dict(id='fig1-CE-layout-spacing-20261015',scope=['C','E'],property='C box/text containment and E colourbar title clearance',expected='Four equal 350 by 210-unit cards; positive interior text padding, no clipping; E tick/title gap same as H at assembly size',status='verified',evidence_reference=ref,affected_artifact_hashes=dict(svg=sha(after),review=sha(page))))
    state['verification']['visual_review']=dict(svg_sha256=sha(after),scope=['C','E'],observations='Browser rendering inspected; measured all C card text bounds and E/H colourbar spacing. F arrowheads unchanged.')
    publish_state(state_path,state,snap,artifacts=[(evidence,record)],evidence_updates=[(svg,after),(html,page),(measure,mb),(epage,ep)])
    print(json.dumps(dict(published=str(html),evidence=str(evidence))))


if __name__=='__main__':main()
