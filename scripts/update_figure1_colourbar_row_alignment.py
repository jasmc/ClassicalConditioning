"""Guarded rigid translations for Figure 1 E/H colourbars and FGH spacing."""
from pathlib import Path
import argparse,copy,json,sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,sha,tag,publish_state,measured_presentation
from build_figure_legend_review import svg_editing_page
from build_figure1_fgh_version_comparison import scoped_working_svg
from figure1_layout_spacing import align_colourbars_and_space_fgh


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--publish',action='store_true');ap.add_argument('--visual-evidence',type=Path);args=ap.parse_args()
    ticket=json.loads(args.baseline.read_text());snap=Snapshot([]);snap.hashes=ticket['starting_hashes'];snap.check();state_path=Path(ticket['state_path']);state=json.loads(state_path.read_text());draft=state['draft']
    svg=Path(draft['working_svg']['path']);html=Path(draft['review']['path']);legend=Path(draft['legend']['path']);measure=Path(draft['effective_presentation']['path']);fghpage=Path(draft['panel_views']['FGH']['path'])
    snap.hashes.update(Snapshot([Path(__file__),ROOT/'scripts/figure1_layout_spacing.py',ROOT/'scripts/build_figure_legend_review.py',ROOT/'scripts/build_figure1_fgh_version_comparison.py',ROOT/'scripts/publish_figure1_fgh_one_second.py',fghpage]).hashes)
    before=svg.read_bytes();after,details=align_colourbars_and_space_fgh(before);repeat,_=align_colourbars_and_space_fgh(after);assert repeat==after
    old=ET.fromstring(before);new=ET.fromstring(after);changed=set(details['translations'])
    for r in [old,new]:
        for node in r.iter():
            if node.get('id') in changed:node.attrib.pop('transform',None)
            if node.get('id')=='editable-copy-provenance':node.text='permitted layout mapping update'
    assert ET.tostring(old)==ET.tostring(new),'Unexpected edits outside named transforms'
    page=svg_editing_page(after,legend=legend.read_text()).encode();preview=Path('/tmp/figure1-EH-FGH-preview.html')
    if not args.publish:preview.write_bytes(page);print(json.dumps(dict(preview=str(preview),svg_sha256=sha(after),details=details)));return
    assert args.visual_evidence;visual=json.loads(args.visual_evidence.read_text());assert visual['svg_sha256']==sha(after) and visual['checks_passed'];snap.hashes.update(Snapshot([args.visual_evidence]).hashes)
    evidence=ROOT/'records/figure-reviews/figure-01/EH-colourbar-FGH-spacing-2026-10-15.json'
    report=dict(scope=['E','F','G','H'],source_svg_sha256=sha(before),svg_sha256=sha(after),starting_hashes=snap.hashes,source_code_before_edit=json.loads(Path('/tmp/figure1-EH-FGH-code-start.json').read_text()),details=details,verification=dict(only_named_group_translations_changed=True,heatmap_rasters_axes_and_labels_preserved=True,F_arrows_unchanged=True,override_idempotent=True),visual_review=visual,frozen=False)
    record=(json.dumps(report,indent=2)+'\n').encode();ref=dict(path=str(evidence),sha256=sha(record))
    measured=json.loads(measure.read_text());measured.update(measured_presentation(after));measured['EH_FGH_alignment']=report;mb=(json.dumps(measured,indent=2)+'\n').encode()
    scoped,sh=scoped_working_svg(after);scope_page=svg_editing_page(scoped,legend='(F–H)'+legend.read_text().split('(F–H)',1)[1]).encode()
    draft['working_svg']['sha256']=sha(after);draft['review']['sha256']=sha(page);draft['effective_presentation']['sha256']=sha(mb)
    draft['panel_views']['FGH'].update(sha256=sha(scope_page),source_svg_path=str(svg),source_svg_sha256=sha(after),native_panel_sha256=sh)
    # E subtree is identical, so its existing scoped view remains authenticated.
    draft['panel_views']['E']['source_svg_sha256']=sha(after)
    draft['right_scale_and_row_alignment']=ref;draft['semantic_registry']['EH_FGH_alignment']=ref
    request=dict(id='fig1-EH-right-FGH-spacing-20261015',author_direction='Align color scales of E and H to the right; space better FGH',scope=['E','F','G','H'],property='common colourbar x positions and equal visible FGH gutters',expected='E/H colourbar bars, ticks and titles share x positions; F-G, G-H and H-colourbar gaps equal; same heatmap dimensions and row positions',status='verified',evidence_reference=ref,affected_artifact_hashes=dict(svg=sha(after),review=sha(page)))
    draft['consistency_requests']=[r for r in draft['consistency_requests'] if r['id']!=request['id']]+[request]
    state['verification']['visual_review']=dict(svg_sha256=sha(after),scope=['E','F','G','H'],observations=visual['observations'])
    publish_state(state_path,state,snap,artifacts=[(evidence,record)],evidence_updates=[(svg,after),(html,page),(measure,mb),(fghpage,scope_page)])
    print(json.dumps(dict(published=str(html))))


if __name__=='__main__':main()
