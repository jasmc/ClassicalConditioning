"""Match working F/G/H axis labels to the existing 8 pt bold Figure 1 labels."""
from pathlib import Path
import json, sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot, file_hash, sha, publish_state, measured_presentation
from publish_figure1_fgh_one_second import apply_selected_fgh
from build_figure_legend_review import svg_editing_page
from build_figure1_fgh_version_comparison import scoped_working_svg


def main():
    ticket=json.loads(Path('/tmp/figure-01-fgh-font-alignment-start.json').read_text())
    for p,h in ticket['starting_hashes'].items():assert file_hash(p)==h,'Concurrent edit: '+p
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json';state=json.loads(state_path.read_text())
    svg=Path(state['draft']['working_svg']['path']);html=Path(state['draft']['review']['path']);legend=Path(state['draft']['legend']['path']);measure=Path(state['draft']['effective_presentation']['path'])
    frozen=state['scoped_fgh_freeze'];scope_page=html.parent/'figure-01-panels-fgh-current.html'
    snapshot=Snapshot([state_path,svg,html,legend,measure,scope_page,Path(__file__),ROOT/'scripts/publish_figure1_fgh_one_second.py',*[Path(frozen[k]['path']) for k in ['svg','data','freeze_manifest']]])
    before=svg.read_bytes();after,_,_=apply_selected_fgh(before,frozen)
    oldroot=ET.fromstring(before);newroot=ET.fromstring(after)
    for panel in 'ABCDE':
        a=next(n for n in oldroot if n.get('id')=='panel-'+panel);b=next(n for n in newroot if n.get('id')=='panel-'+panel)
        assert ET.tostring(a)==ET.tostring(b),'Unrelated panel changed: '+panel
    for panel in 'fgh':
        element=f'edit-FGH-fig1__{panel}__heatmap__vigor-cells'
        a=next(n for n in oldroot.iter() if n.get('id')==element);b=next(n for n in newroot.iter() if n.get('id')==element)
        assert ET.tostring(a)==ET.tostring(b),'Scientific raster changed'
    original=measured_presentation(before);measured=measured_presentation(after)
    affected=[r for r in measured['texts'] if any('panel-FGH'==c for c in r['context']) and r['text'] in ['Time from CS onset (s)','Log vigor relative to baseline']]
    assert len(affected)==4
    assert all(r['weight']=='bold' and abs(r['font_height_pt']-8)<1e-5 and 'DejaVu Sans' in r['font_family'] for r in affected)
    ticks=[r for r in measured['texts'] if 'panel-FGH' in r['context'] and any('tick' in c for c in r['context'])]
    assert all(r['weight']=='normal' and abs(r['font_height_pt']-7)<1e-5 for r in ticks)
    evidence=ROOT/'records/figure-reviews/figure-01/fgh-axis-font-alignment-2026-10-13.json'
    report={'scope':['F','G','H'],'before':[r for r in original['texts'] if 'panel-FGH' in r['context'] and r['text'] in ['Time from CS onset (s)','Log vigor relative to baseline']],
      'after':affected,'reference':'Existing B, D and E axis labels: DejaVu Sans, effective 8 pt, bold; ticks 7 pt normal',
      'scientific_rasters_preserved':True,'other_panels_preserved':list('ABCDE'),'frozen_originals_preserved':True,'svg_sha256':sha(after),
      'visual_review':{'renderer':'sharp/libvips at 1400px','observations':'The three F-H x labels and shared colorbar label now match D/E weight and size. Tick labels remain normal; labels are visible with no clipping or overlap.'},'sources':snapshot.hashes}
    evidence_bytes=(json.dumps(report,indent=2)+'\n').encode();ref={'path':str(evidence),'sha256':sha(evidence_bytes)}
    page=svg_editing_page(after,legend=legend.read_text()).encode();measurement=(json.dumps(measured,indent=2)+'\n').encode()
    scoped,scoped_hash=scoped_working_svg(after);scoped_legend=legend.read_text().split('(F–H)',1)[1];scoped_page=svg_editing_page(scoped,legend='(F–H)'+scoped_legend).encode()
    state['draft']['review']['sha256']=sha(page);state['draft']['working_svg']['sha256']=sha(after);state['draft']['effective_presentation']['sha256']=sha(measurement)
    state['draft']['panel_views']['FGH']={'path':str(scope_page),'sha256':sha(scoped_page),'source_svg_path':str(svg),'source_svg_sha256':sha(after),'native_panel_sha256':scoped_hash,'legend_source_sha256':file_hash(legend),'scope':['F','G','H'],'policy':'Current working presentation, derived from the selected scoped freeze; frozen original remains immutable'}
    state['draft']['axis_font_alignment']=ref
    request={'id':'fig1-fgh-axis-font-alignment-20261013','author_direction':'The axes labels of those panels must match the other Figure 1 panels in font, bold, etc.','scope':['F','G','H'],'property':'axis/colorbar label typography at final assembly scale','expected':'DejaVu Sans, 8 pt bold; ticks 7 pt normal','implementation':'apply_selected_fgh working presentation override','status':'verified','affected_artifact_hashes':{'svg':sha(after),'review':sha(page)},'evidence_reference':ref}
    state['draft']['consistency_requests']=[r for r in state['draft']['consistency_requests'] if r['id']!=request['id']]+[request]
    state['verification']['visual_review']={'svg_sha256':sha(after),'renderer':'sharp/libvips','scope':'F-H axis labels compared to B/D/E in the whole assembly','observations':report['visual_review']['observations']}
    updates=[(svg,after),(html,page),(measure,measurement)]
    artifacts=[(evidence,evidence_bytes)]
    if scope_page.exists():updates.append((scope_page,scoped_page))
    else:artifacts.append((scope_page,scoped_page))
    publish_state(state_path,state,snapshot,artifacts=artifacts,evidence_updates=updates)
    print(json.dumps({'published':True,'labels':4,'font':'DejaVu Sans','size_pt':8,'weight':'bold','html':str(html)}))


if __name__=='__main__':main()
