"""Publish the checked E-only freeze into the guarded current Figure 1 display."""
import sys,json,copy,argparse
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,bind_loaded_state,publish_state,sha,tag,measured_presentation
from build_figure_legend_review import svg_editing_page
STAGE=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-panel-e-raw-frame-freeze-2026-10-13')
def encoded(v):return (json.dumps(v,indent=2)+'\n').encode()
def artifact(p):return dict(path=str(p),sha256=sha(p.read_bytes()))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);args=parser.parse_args();ticket=json.loads(args.baseline.read_text());start=Snapshot([]);start.hashes=ticket['starting_hashes'];start.check()
 sp=ROOT/'docs/paper/figures/figure-01/current-state.json';sb=sp.read_bytes();s=json.loads(sb);freeze=ROOT/'records/frozen-analyses/figure-01/panel-e/2026-10-13-raw-frame/manifests/freeze.json';manifest=json.loads(freeze.read_text());assert manifest['freeze_check']['valid'] and not manifest['freeze_check']['issues']
 for group in ('source_artifacts','data_artifacts','exports'):
  for i in manifest[group]:assert sha(Path(i['path']).read_bytes())==i['sha256']
 assert sha((ROOT/'configs/paper-figures/figure-elements.json').read_bytes())==manifest['specification_sha256']
 svg=Path(s['draft']['working_svg']['path']);html=Path(s['draft']['review']['path']);legend=Path(s['draft']['legend']['path']);mp=Path(s['draft']['effective_presentation']['path']);ep=ROOT/'records/figure-reviews/figure-01/panel-e-current-display.json';panelhtml=svg.parent/'figure-01-panel-e.html'
 selectionpath=ROOT/'configs/paper-figures/selections/figure1-panel-e-raw-frame-freeze-20261013.json';inventorypath=ROOT/'records/figure-reviews/figure-01/panel-e-freeze-disposition-inventory-20261013.json'
 docs=[ROOT/p for p in ('docs/paper/figures/README.md','docs/paper/figures/figure-01/README.md','records/frozen-analyses/selections.md','records/frozen-analyses/figure-01/panel-e/README.md')]
 snapshot=Snapshot([sp,svg,html,legend,mp,ep,panelhtml,selectionpath,inventorypath,freeze,STAGE/'frozen-candidate.svg',STAGE/'working-source.svg',ROOT/'configs/paper-figures/figure-elements.json',Path(__file__),*docs]);bind_loaded_state(snapshot,s,sb,sp)
 raw=svg.read_bytes();assert raw==(STAGE/'working-source.svg').read_bytes(),'Reconcile current work before integrating freeze'
 root=ET.fromstring(raw);frozenroot=ET.fromstring((STAGE/'frozen-candidate.svg').read_bytes());old=next(n for n in root if n.get('id')=='panel-E');new=copy.deepcopy(next(n for n in frozenroot if n.get('id')=='panel-E'));index=list(root).index(old);root.remove(old);root.insert(index,new)
 preserved={n.get('id'):sha(ET.tostring(n)) for n in root if n.get('id','').startswith('panel-') and n.get('id')!='panel-E'}
 meta=next(n for n in root if n.get('id')=='editable-copy-provenance');provenance=json.loads(meta.text)
 for k,r in manifest['element_registry'].items():
  if k.startswith('edit-E-'):
   prior=provenance['scientific_registry'].get(k,{})
   provenance['scientific_registry'][k]={**prior,**r,'source_element_id':k,'source_frozen_svg':artifact(STAGE/'frozen-candidate.svg'),'review_status':'author-frozen 2026-10-13'}
 pointer=dict(figure_id='fig1',panel_ids=['e'],status='author-frozen',selected_revision='unbinned-raw-baseline-scaled-0-5-F-one-second-bands',svg=artifact(STAGE/'frozen-candidate.svg'),freeze_manifest=artifact(freeze),candidate=artifact(STAGE/'candidate.json'),data=artifact(STAGE/'frame-and-band-mappings.json'),selection_record=dict(path=str(selectionpath)),assembly_scale=manifest['assembly_scale'],author_instruction=manifest['freeze_authorization'],scope='Panel E only; no whole-figure freeze')
 selected=json.loads((STAGE/'selection.json').read_text());selection={**pointer,'scientific_definition':{'black_trace':selected['scientific_definition'],'bands':s['scoped_fgh_freeze']['scientific_definition']},'trials':selected['acquisition_trials'],'display_trials':selected['display_trials'],'supersedes':artifact(ROOT/'configs/paper-figures/selections/figure1-panel-e-heatmap-rows-freeze-20261009.json')};selection.pop('selection_record');selectionbytes=encoded(selection);pointer['selection_record']['sha256']=sha(selectionbytes)
 provenance['current_scoped_E_selection']=pointer;meta.text=json.dumps(provenance,separators=(',',':'))
 updated=ET.tostring(root);leg=legend.read_text().replace('recorded in the draft evidence','recorded in the scoped E freeze evidence').replace('E’s black trace is an exploratory scientific draft; accepted frozen sources and F–H remain preserved.','Panel E is frozen in this scoped selection; earlier E sources and F–H remain preserved.').replace('The scoped F–H freeze does not freeze the whole figure or E’s black traces.','The scoped F–H and E freezes do not freeze the whole figure.')
 page=svg_editing_page(updated,legend=leg).encode();scope=copy.deepcopy(root)
 for n in list(scope):
  if n.get('id')!='panel-E' and n.tag not in (tag('defs'),tag('rect')):scope.remove(n)
 scope.attrib.update(viewBox='900 810 900 635',width='900',height='635',id='panel-E-scoped-view',**{'aria-label':'Figure 1 panel E: selected scoped frozen presentation'})
 panellegend=leg[leg.index('(E)'):leg.index('\n\n(F–H)')];panelpage=svg_editing_page(ET.tostring(scope),scope_label='Figure 1 · Panel E',legend=panellegend).encode()
 report=json.loads(ep.read_text());report.update(svg_sha256=sha(updated),scoped_freeze=pointer,preserved_panel_hashes=preserved,visual_review=manifest['verification']['visual_review']['evidence']);reportbytes=encoded(report)
 measure=json.loads(mp.read_text());measure.update(measured_presentation(updated));measure['panel_e_scoped_freeze']=pointer;measure['panel_e_display_update']=report;measurebytes=encoded(measure)
 s['scoped_e_freeze']=pointer;s['draft']['working_svg'].update(sha256=sha(updated),source_svg_sha256=sha(raw));s['draft']['review']['sha256']=sha(page);s['draft']['legend']['sha256']=sha(leg.encode());s['draft']['effective_presentation']['sha256']=sha(measurebytes)
 s['draft']['semantic_registry']['E_frozen_registry']=artifact(STAGE/'candidate.json');s['draft']['semantic_registry']['E_display_update']=dict(path=str(ep),sha256=sha(reportbytes));s['draft']['scientific_changes']['E_status']='Accepted scoped E freeze 2026-10-13; D and whole-figure draft approval separate.'
 s['draft']['panel_views']['E'].update(sha256=sha(panelpage),source_svg_sha256=sha(updated),legend_source_sha256=sha(leg.encode()),legend_scope_sha256=sha(panellegend.encode()),policy='Selected scoped E freeze integrated in sole working SVG')
 oldrecords=s['scientific_selection']['records'];s['scientific_selection']['records']=[r for r in oldrecords if 'figure1-panel-e-heatmap-rows-freeze-20261009' not in r['path']]+[dict(path=str(selectionpath),sha256=sha(selectionbytes))]
 s['verification']['visual_review']=dict(svg_sha256=sha(updated),observations=report['visual_review'])
 inventory=dict(scope='E',new_freeze=pointer,status='Pending author disposition; preserve all earlier versions and dependencies',recommendation='Keep the new freeze and shared active working pair. Retain old E freeze/history and approved v15 candidate as historical evidence until author decides.',prior_selected_records=[r for r in oldrecords if 'panel-e' in r['path']],preserved_approved_candidate=s.get('pending_freeze'),previous_working_source=artifact(STAGE/'working-source.svg'),historical_catalogue=dict(selection_record=selection['supersedes'],version_count=22,note='Prior E archive retains its 22-version catalogue and all byte-bound dependencies; new freeze grants no disposition authorization'),pending_dispositions='keep/archive/delete choices; none executed')
 inventorybytes=encoded(inventory);s['scoped_e_freeze']['disposition_inventory']=dict(path=str(inventorypath),sha256=sha(inventorybytes))
 docoutputs=[]
 for d in docs:
  text=d.read_text()
  if d.name=='selections.md':text=text.replace('| Figure 1 E |','| Figure 1 E historical |');text+='\nCurrent E selection: [unbinned raw-baseline-scaled black traces, 0–5, exact current F bands](../../configs/paper-figures/selections/figure1-panel-e-raw-frame-freeze-20261013.json). The earlier V12-row E freeze remains historical; E only is frozen, not the whole assembly.\n'
  elif d==ROOT/'docs/paper/figures/figure-01/README.md':
   text=text.replace('E’s active draft retains its black traces and now copies exact rows from the new frozen F heatmap.','E’s scoped 2026-10-13 freeze uses unbinned raw-baseline-scaled black traces (0–5) and exact rows from the selected frozen F heatmap.').replace('The active D/E draft has a separately recorded scientific change; use [state](current-state.json) rather than treating its recomputed E values as frozen.','E’s current scientific definition is accepted by its separate scoped freeze; D and whole-figure draft changes remain separately recorded in [state](current-state.json).').replace('matching frame-vigor draft','matching frozen frame-vigor')
  elif d==ROOT/'records/frozen-analyses/figure-01/panel-e/README.md':text='# Figure 1 panel E\n\nCurrent: [2026-10-13 unbinned raw-baseline-scaled black traces and current F bands](../../../../configs/paper-figures/selections/figure1-panel-e-raw-frame-freeze-20261013.json). Exact SVG, data mappings, dependencies and gate manifest are on JOAQUIM at the selected paths.\n\nHistorical: [2026-10-09 heatmap-row record collection](2026-10-09-heatmap-rows/README.md), its V12 definition and original approvals remain unchanged.\n'
  else:text+='\nCurrent E selection: [unbinned raw-baseline-scaled traces, 0–5, current F bands](../../../configs/paper-figures/selections/figure1-panel-e-raw-frame-freeze-20261013.json); E only is frozen.\n'
  docoutputs.append((d,text.encode()))
 outputs=[(svg,updated),(html,page),(legend,leg.encode()),(mp,measurebytes),(ep,reportbytes),(panelhtml,panelpage),(selectionpath,selectionbytes),(inventorypath,inventorybytes),*docoutputs]
 publish_state(sp,s,snapshot,artifacts=[(p,b) for p,b in outputs if not p.exists()],evidence_updates=[(p,b) for p,b in outputs if p.exists()]);print(json.dumps(dict(panel_view=str(panelhtml),selection=str(selectionpath),preserved_panels=list(preserved),freeze_valid=True)))
if __name__=='__main__':main()
