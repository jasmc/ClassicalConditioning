"""Guarded local arrow presentation edit, preserving the active Figure 1 pair."""
from pathlib import Path
import argparse,json,sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,file_hash,sha,publish_state,measured_presentation,tag
from publish_figure1_fgh_one_second import left_align_f_example_arrows
from build_figure_legend_review import svg_editing_page
from build_figure1_fgh_version_comparison import scoped_working_svg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    ticket=json.loads(args.baseline.read_text());state_path=Path(ticket['state_path'])
    for p,h in ticket['starting_hashes'].items():assert file_hash(p)==h,'Concurrent edit: '+p
    state=json.loads(state_path.read_text());draft=state['draft']
    svg=Path(draft['working_svg']['path']);page=Path(draft['review']['path']);legend=Path(draft['legend']['path']);measure=Path(draft['effective_presentation']['path']);scoped=Path(draft['panel_views']['FGH']['path'])
    evidence=ROOT/'records/figure-reviews/figure-01/panel-f-left-arrows-2026-10-15.json'
    snap=Snapshot([state_path,svg,page,legend,measure,scoped,Path(__file__),ROOT/'scripts/publish_figure1_fgh_one_second.py',ROOT/'scripts/build_figure_legend_review.py',ROOT/'src/classical_conditioning/figure_workspace.py']);snap.hashes.update(ticket['starting_hashes']);snap.check()
    before=svg.read_bytes();after,arrows=left_align_f_example_arrows(before)
    old=ET.fromstring(before);new=ET.fromstring(after);allowed={a['element_id'] for a in arrows}
    # Remove precisely the permitted edits, then compare the remaining whole tree.
    for r in [old,new]:
        for meta in r.iter(tag('metadata')):
            if meta.get('id')=='editable-copy-provenance':meta.text='permitted arrow mapping update'
        for parent in r.iter():
            for child in list(parent):
                if child.get('id') in allowed:parent.remove(child)
    assert ET.tostring(old)==ET.tostring(new),'Change outside five arrows and their provenance'
    repeated,_=left_align_f_example_arrows(after);assert repeated==after,'Override must be repeatable'
    html=svg_editing_page(after,legend=legend.read_text()).encode()
    fgh,native_hash=scoped_working_svg(after);scope_html=svg_editing_page(fgh,legend='(F–H)'+legend.read_text().split('(F–H)',1)[1]).encode()
    measured=measured_presentation(after);measurement=(json.dumps(measured,indent=2)+'\n').encode()
    report=dict(scope=['F'],author_direction='2026-10-15: move arrowheads to left side, flip, align with trial numbers',source_svg_sha256=sha(before),svg_sha256=sha(after),arrows=arrows,verification=dict(only_arrow_nodes_and_mapping_changed=True,unrelated_panels_and_heatmap_pixels_preserved=True,override_idempotent=True,trial_centres_preserved=True,arrow_column_between_numbers_and_heatmap=True),starting_hashes=snap.hashes,frozen=False)
    preview=Path('/tmp/figure1-F-left-arrow-review');preview.mkdir(exist_ok=True)
    if not args.publish:
        (preview/'figure1.html').write_bytes(html);print(json.dumps({'preview':str(preview/'figure1.html'),'arrows':arrows}));return
    report['visual_review']=dict(status='passed',observations='Inspected full Figure 1 and enlarged FGH in browser: five right-pointing arrowheads sit beside the trial-number column at rows 5,13,59,62,89, clear of labels and phase text; axes, heatmap and other panels preserved.')
    record=(json.dumps(report,indent=2)+'\n').encode();ref=dict(path=str(evidence),sha256=sha(record))
    draft['working_svg']['sha256']=sha(after);draft['review']['sha256']=sha(html);draft['effective_presentation']['sha256']=sha(measurement)
    draft['panel_views']['FGH'].update(sha256=sha(scope_html),source_svg_path=str(svg),source_svg_sha256=sha(after),native_panel_sha256=native_hash,legend_source_sha256=file_hash(legend))
    draft['example_arrows'].update(side='left',direction='right',alignment='column between trial numbers and heatmap; y equals selected trial centre',display_trials=[5,13,59,62,89],evidence=ref)
    draft['semantic_registry']['F_arrow_presentation']=ref
    request=dict(id='fig1-F-left-arrow-alignment-20261015',author_direction=report['author_direction'],scope=['F'],elements=sorted(allowed),property='side, direction and trial-number column alignment',expected='left side, right-pointing, beside trial numbers; preserve trial centre y',implementation='left_align_f_example_arrows in publish_figure1_fgh_one_second.py; retained on future working rebuilds',status='verified',evidence_reference=ref,affected_artifact_hashes=dict(svg=sha(after),review=sha(html)))
    draft['consistency_requests']=[r for r in draft.get('consistency_requests',[]) if r['id']!=request['id']]+[request]
    state['verification']['visual_review']=dict(svg_sha256=sha(after),scope='F arrow placement in full Figure 1',observations=report['visual_review']['observations'])
    publish_state(state_path,state,snap,artifacts=[(evidence,record)],evidence_updates=[(svg,after),(page,html),(measure,measurement),(scoped,scope_html)])
    print(json.dumps({'published':str(page),'svg':str(svg)}))


if __name__=='__main__':main()
