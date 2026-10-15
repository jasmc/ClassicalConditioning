"""Select and integrate the author-frozen historical-fish F/G/H row."""
from pathlib import Path
import json, sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot, file_hash, sha, publish_state, measured_presentation
from build_figure_legend_review import svg_editing_page
from publish_figure1_fgh_one_second import apply_selected_fgh, artifact


def main():
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json'
    ticket=json.loads(Path('/tmp/figure-01-historical-fgh-freeze-start.json').read_text())
    for p,h in ticket['starting_hashes'].items():assert file_hash(p)==h,'Concurrent edit: '+p
    state=json.loads(state_path.read_text());previous=state['scoped_fgh_freeze']
    stage=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-fgh-historical-one-second-managua-freeze-2026-10-13')
    record_root=ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-13-historical-fish-one-second-managua'
    manifest=record_root/'manifests/freeze.json';freeze=json.loads(manifest.read_text());assert freeze['freeze_check']['valid'] and not freeze['freeze_check']['issues']
    selection=json.loads((stage/'selection.json').read_text())
    pointer=dict(figure_id='fig1',panel_ids=['f','g','h'],status='author-frozen',selected_version=selection['revision'],freeze_scope='F/G/H only; not whole Figure 1',scientific_definition=selection['scientific_definition'],fish=selection['fish'],svg=artifact(stage/'frozen-candidate.svg'),data=artifact(stage/'dependency-figure-01-historical-fish-one-second-managua-2026-10-13.csv'),freeze_manifest=artifact(manifest),author_instruction=selection['author_instruction'],assembly_scale=freeze['assembly_scale'],superseded_selection=previous)
    pointer_path=ROOT/'configs/paper-figures/selections/figure1-fgh-historical-one-second-managua-freeze-20261013.json'
    svg=Path(state['draft']['working_svg']['path']);html=Path(state['draft']['review']['path']);legend=ROOT/'docs/paper/figures/figure-01/legend.md';measurement=Path(state['draft']['effective_presentation']['path'])
    docs=[ROOT/'docs/paper/figures/README.md',ROOT/'docs/paper/figures/figure-01/README.md',ROOT/'docs/paper/figures/figure-01/panels-f-g-h/README.md',ROOT/'records/frozen-analyses/selections.md',ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/README.md',ROOT/'docs/paper/figures/shared/iteration-workflow.md']
    snapshot=Snapshot([state_path,svg,html,legend,measurement,*docs,manifest,Path(pointer['svg']['path']),Path(pointer['data']['path']),Path(__file__),ROOT/'scripts/publish_figure1_fgh_one_second.py'])
    before=svg.read_bytes();updated,report,_=apply_selected_fgh(before,pointer)
    oldroot=ET.fromstring(before);newroot=ET.fromstring(updated)
    # F is unchanged, so E and all earlier panels remain exact.
    for pid in 'ABCDE':
        a=next(n for n in oldroot if n.get('id')=='panel-'+pid);b=next(n for n in newroot if n.get('id')=='panel-'+pid)
        assert ET.tostring(a)==ET.tostring(b),'Unrelated panel changed: '+pid
    legend_text=legend.read_text().replace('3 s Trace (20230519_11)','3 s Trace (20230307_12)')
    page=svg_editing_page(updated,legend=legend_text).encode();measured=measured_presentation(updated)
    evidence=ROOT/'records/figure-reviews/figure-01/historical-fgh-freeze-integration-2026-10-13.json'
    report.update(active_svg_sha256=sha(updated),active_html_sha256=sha(page),preserved_panels=list('ABCDE'),all_270_baselines_defined=True,visual_review={'status':'passed','evidence':'Approved historical-fish plot and candidate visual-review.png inspected. Same F/H and E rows; G now shows 20230307_12 with identical 1 s recipe, limits and guides. All 161 gate checks passed.'})
    evidence_bytes=(json.dumps(report,indent=2)+'\n').encode();evidence_ref={'path':str(evidence),'sha256':sha(evidence_bytes)}
    scoped=stage/'frozen-review.html';scoped_legend=legend_text[legend_text.index('(F–H)'):];scoped_page=svg_editing_page(Path(pointer['svg']['path']).read_bytes(),legend=scoped_legend).encode()
    pointer_bytes=(json.dumps(pointer,indent=2)+'\n').encode()
    state['scientific_selection']['records']=[r for r in state['scientific_selection']['records'] if 'figure1-fgh-one-second-managua-freeze-20261013' not in r['path']]+[{'path':str(pointer_path),'sha256':sha(pointer_bytes)}]
    state['scoped_fgh_freeze']=pointer;state['draft']['review']['sha256']=sha(page);state['draft']['working_svg']['sha256']=sha(updated);state['draft']['legend']['sha256']=sha(legend_text.encode())
    measure_bytes=(json.dumps(measured,indent=2)+'\n').encode();state['draft']['effective_presentation']['sha256']=sha(measure_bytes)
    state['draft']['scientific_changes']['FGH']='Current author-frozen historical examples: Delay 20221115_07, Trace 20230307_12, Control 20221115_09; direct original log-frame medians in 1 s bins, baseline [-15,0) bin median, no scaling, native managua_r +/-0.25. All 270 trial baselines defined.'
    state['draft']['fgh_integration_evidence']=evidence_ref;state['draft']['semantic_registry']['E_band_override']=evidence_ref;state['draft']['semantic_registry']['FGH_frozen_registry']=artifact(stage/'candidate.json')
    meta=next(n for n in newroot.iter() if n.get('id')=='editable-copy-provenance');state['draft']['semantic_registry']['scientific_records']=len(json.loads(meta.text)['scientific_registry'])
    state['draft']['panel_views']['FGH']={'path':str(scoped),'sha256':sha(scoped_page),'source':pointer['svg'],'legend_scope':'F–H'}
    for request in state['draft']['consistency_requests']:
        if request['id'] in ('fig1-annotation-20261013-e-fgh-scale','fig1-e-unbinned-trace-exact-f-band-20261013'):
            request['evidence_reference']=evidence_ref;request['affected_artifact_hashes']={'svg':sha(updated),'review':sha(page),'source_FGH':pointer['svg']['sha256']}
            if request['id']=='fig1-e-unbinned-trace-exact-f-band-20261013':request['verification']={'exact_F_rows':True,'bins_per_row':40,'rows':report['bands'],'source':pointer['data'],'F_raster_png_sha256':report['F_raster_png_sha256']}
    state['verification']={'current_sources_authenticated':True,'editable_source_structure':True,'visual_review':{'svg_sha256':sha(updated),'scope':'Historical-fish F/G/H freeze integrated; A-E verified unchanged'},'concurrency':'Starting ticket and source hashes checked under publication lock'}
    updates=[(svg,updated),(html,page),(legend,legend_text.encode()),(measurement,measure_bytes)]
    for p in docs:
        text=p.read_text().replace('figure1-fgh-one-second-managua-freeze-20261013.json','figure1-fgh-historical-one-second-managua-freeze-20261013.json')
        if p.name=='README.md' and 'panels-f-g-h' in str(p):
            if 'docs/paper' in str(p):
                text=text.replace('Trace 20230519_11','Trace 20230307_12').replace('six Trace trials are undefined','all 90 baseline references are defined in each fish')
            else:
                text=text.replace('[Current 1 s direct log-frame median freeze](2026-10-13-one-second-managua/README.md)','[Current historical-fish 1 s median freeze](2026-10-13-historical-fish-one-second-managua/README.md)\n- [Previous Trace 20230519_11 freeze, preserved](2026-10-13-one-second-managua/README.md)')
        text=text.replace('current F–H selection is the author-frozen 2026-10-13 1 s direct log-frame median managua_r version','current F–H selection is the author-frozen 2026-10-13 historical-fish 1 s direct log-frame median managua_r version (Trace 20230307_12)')
        updates.append((p,text.encode()))
    inventory={'current_freeze':artifact(manifest),'superseded_freeze':previous,'disposition':'Preserve all earlier freezes, candidate reviews and dependencies; no pruning authorized. Previous inventory remains valid and is supplemented by this record.','previous_inventory':artifact(ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-13-one-second-managua/manifests/previous-version-disposition.json')}
    readme=b'# Figure 1 F/G/H: historical fish, 1 s medians\n\nCurrent author-frozen fish: Delay 20221115_07, 3 s Trace 20230307_12, Control 20221115_09. [Freeze manifest](manifests/freeze.json). 161 checks passed. Direct eligible original natural-log frame medians in 1 s bins; subtract finite baseline-bin median over [-15,0); no amplitude scaling; native managua_r +/-0.25; zero centered and missing black. All 270 baseline references defined. Scope F/G/H only. Previous freezes and dependencies retained.\n'
    artifacts=[(pointer_path,pointer_bytes),(evidence,evidence_bytes),(scoped,scoped_page),(record_root/'README.md',readme),(record_root/'manifests/previous-version-disposition.json',(json.dumps(inventory,indent=2)+'\n').encode())]
    publish_state(state_path,state,snapshot,artifacts=artifacts,evidence_updates=updates)
    print(json.dumps({'published':True,'review':str(scoped),'figure1':str(html),'trace':'20230307_12'}))


if __name__=='__main__':main()
