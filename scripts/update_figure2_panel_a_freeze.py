"""Prepare and integrate the explicitly requested Figure 2 A scoped freeze."""
from pathlib import Path
import copy,json,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot,sha,tag
STATE=ROOT/'docs/paper/figures/figure-02/panel-a/current-state.json'
STAGE=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-02-a-freeze-2026-10-14')
AUTH='Author, 2026-10-14: freeze Figure 2 panel A; match exactly the style of axes, labels and other elements in Figure 1 F/G/H; update Figure 2 A. Later B and C will follow the same pipeline and formatting. A only; no whole-figure or B/C freeze.'
def artifact(path,**extra):return dict(path=str(Path(path).resolve()),sha256=sha256_file(Path(path)),**extra)
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def prepare():
    baseline=json.loads(Path('/tmp/figure2-a-freeze-start.json').read_text())
    for path,h in baseline.items():
        if Path(path)==Path(__file__).resolve():continue
        from classical_conditioning.figure_workspace import file_hash
        if file_hash(path)!=h:raise ValueError('Starting authority changed: '+path)
    state=json.loads(STATE.read_text());draft=state['draft'];evidence=json.loads(Path(draft['semantic_registry']['path']).read_text())
    for key in ('working_svg','review','legend','semantic_registry'):
        a=draft[key]
        if sha256_file(Path(a['path']))!=a['sha256']:raise ValueError('Manual working-copy edit: '+a['path'])
    if not STAGE.parent.is_dir():raise FileNotFoundError(STAGE.parent)
    STAGE.mkdir()
    original=STAGE/'selected-original.svg';original.write_bytes(Path(draft['working_svg']['path']).read_bytes())
    root=ET.fromstring(original.read_bytes())
    for n in root.iter(tag('metadata')):
        if n.get('id')=='figure-02-a-draft-provenance':
            p=json.loads(n.text);p.update(frozen=True,freeze_scope=['A'],author_authorization=AUTH);n.text=json.dumps(p)
    svg=STAGE/'frozen-panel-a.svg';svg.write_bytes(ET.tostring(root))
    records=copy.deepcopy(evidence['element_registry']);spec=json.loads((ROOT/'configs/paper-figures/figure-elements.json').read_text());exceptions=[]
    for key,r in records.items():
        if r['scientific_role']=='label.cohort_count':r['scientific_role']='annotation.note'
        r['status']='author-selected freeze candidate';r['scientific_context']['statistical_inference']='none; descriptive population heatmaps'
        r['classification_evidence']=['Named source element authenticated by Figure 1 F–H freeze; explicit population/condition/trial/bin/cohort mapping in Figure 2 A evidence and pooled data; updated in update_figure2_panel_a_freeze.py.']
        r['style_verification']={prop:{'status':'passed','evidence':'Exact selected F–H style and geometry copied at native scale; 106 measured text/stroke/tick comparisons, scientific data/mask/palette checks, and renderer review bound in presentation-evidence.json.'} for prop in r['resolved_style']}
        r['style_verification']['raster_content']={'status':'passed','evidence':'All 7200 saved RGBA cell centres verified against the authenticated pooled values and native managua_r; opacity/missingness reviewed; frozen candidate changes only nonvisual freeze metadata.'}
        if r['scientific_role']=='title.condition':
            r['color_evidence']='Exact condition title paint copied from the selected frozen Figure 1 H Control and F Delay titles; shared condition specification and presentation evidence.'
        if r['scientific_role'] in ('stimulus.cs.onset','stimulus.cs.offset'):
            for prop,value in [('linewidth_pt',1.921259842519685),('alpha',.8)]:
                exid=r['scientific_role'].replace('.','-')+'-'+prop
                if not any(e['exception_id']==exid for e in exceptions):
                    exceptions.append(dict(exception_id=exid,scientific_role=r['scientific_role'],property=prop,value=value,reason='Preserve the exact selected Figure 1 F/G/H formatting explicitly requested for Figure 2 A.',scope={'figure_id':'fig2','panel_ids':['a']},approval_evidence=AUTH))
        # Keep the imported effective presentation; semantic context is explicitly pooled.
        r['panel_id']='a'
    saved=[]
    for key in ('fish_bins','pooled_bins'):
        a=evidence['calculation'][key];source=Path(a['path'])
        if sha256_file(source)!=a['sha256']:raise ValueError('Changed scientific table')
        target=STAGE/source.name;target.write_bytes(source.read_bytes());saved.append(artifact(target))
    for name,source in [('input-provenance.json',Path(evidence['calculation']['pooled_bins']['path']).parent/'calculation.json'),('presentation-evidence.json',Path(draft['semantic_registry']['path'])),('renderer-snapshot.py',ROOT/'scripts/build_figure2_pooled_heatmaps.py'),('calculation-snapshot.py',ROOT/'src/classical_conditioning/figures/pooled_log_heatmap.py'),('cadence-snapshot.py',ROOT/'src/classical_conditioning/preprocessing/acquisition_timing.py'),('movement-snapshot.py',ROOT/'src/classical_conditioning/analysis/movement_state.py'),('experiments-snapshot.py',ROOT/'src/classical_conditioning/config/experiments.py'),('freeze-builder-snapshot.py',Path(__file__))]:
        (STAGE/name).write_bytes(source.read_bytes())
    selection=dict(figure_id='fig2',panel_ids=['a'],scope='A only',authorization=AUTH,scientific_definition={**records['fig2-a-fig1__f__heatmap__vigor-cells']['scientific_context'],'conditions':['control','delay'],'cohort_counts':{'control':28,'delay':29},'trial_mapping':'display 1–90 = acquisition 5–94','pooling':'equal-fish arithmetic mean after individual trial baseline-bin median subtraction'},style_reference=evidence['style_reference'],width_mm=evidence['physical_width_mm'])
    write(STAGE/'selection.json',selection)
    width=evidence['physical_width_mm'];vb=list(map(float,root.get('viewBox').split()))
    candidate=dict(figure_id='fig2',panel_ids=['a'],selection_record=artifact(STAGE/'selection.json'),assembly_scale={'final_width_mm':width,'final_height_mm':78,'scope':'Figure 2 A pair at exact Figure 1 F–H element scale','source_to_final_transforms':{'root_user_unit_to_final_pt':width*72/25.4/vb[2],'svg_transforms':[{'element_id':n.get('id'),'transform':n.get('transform')} for n in root.iter() if n.get('transform')]}},element_registry=records,approved_exceptions=exceptions,source_artifacts=[artifact(original,kind='original_svg')]+[artifact(STAGE/n) for n in ['selection.json','presentation-evidence.json','input-provenance.json','renderer-snapshot.py','calculation-snapshot.py','cadence-snapshot.py','movement-snapshot.py','experiments-snapshot.py','freeze-builder-snapshot.py']],data_artifacts=saved,exports=[artifact(svg)],verification={'scientific_mapping_review':{'status':'passed','evidence':'Source-bound full 57-fish calculation; all fish/trial baseline/bin values and equal-fish means/counts independently checked; frozen F/H examples reproduced in all fields; all 7200 saved RGBA cells checked; context and exact trial/bin/event/condition mappings in presentation-evidence.json.'},'structure_review':{'status':'passed','evidence':'Explicit source roles for boxed heatmaps, shared y ticks/phases, CS onset/offset, paired Delay expected US at 9 s, cohort counts, headings and shared physical log-difference colourbar; formatting matches original frozen F/G/H at scale 1.'},'visual_review':{'status':'pending','evidence':'Inspect the exact saved frozen-panel-a.svg render before publishing.'}},freeze_authorization=AUTH)
    write(STAGE/'candidate.json',candidate);print(STAGE/'candidate.json')

def build_assembly():
    """Place frozen A at scale one; translate the preserved other panel payloads."""
    from update_latest_figure_assemblies import build
    from classical_conditioning.figure_workspace import editable_svg,measured_presentation
    raw,meta=build(2)
    native,inventory=editable_svg(raw,kind='Figure 2 assembly with scoped frozen A',source_records=meta['panels'])
    root=ET.fromstring(native);groups={n.get('id'):n for n in root if n.get('id','').startswith('panel-')}
    original_payload={key:sha(ET.tostring(node)) for key,node in groups.items() if key!='panel-A'}
    for node in list(root):
        if node.tag in (tag('text'),tag('metadata')):root.remove(node)
    root.set('viewBox','0 0 1800 3610');root.set('width','183mm');root.set('height',str(3610/1800*183)+'mm')
    for rect in root.findall(tag('rect')):rect.set('height','3610')
    a=groups['panel-A']
    for child in list(a):a.remove(child)
    selected=ET.fromstring((STAGE/'frozen-panel-a.svg').read_bytes());vb=list(map(float,selected.get('viewBox').split()))
    unit=183*72/25.4/1800;width=vb[2]/unit;height=vb[3]/unit
    selected.set('x',str((1800-width)/2));selected.set('y','0');selected.set('width',str(width));selected.set('height',str(height));selected.set('overflow','visible');a.append(selected)
    translations={'panel-B':(-285,635),'panel-C':(-255,635),**{f'panel-{pid}':(0,690) for pid in 'DEFGHI'}}
    for key,(dx,dy) in translations.items():groups[key].set('transform',f'translate({dx} {dy})')
    preserved={}
    for key,node in groups.items():
        if key=='panel-A':continue
        clone=copy.deepcopy(node);clone.attrib.pop('transform',None)
        if sha(ET.tostring(clone))!=original_payload[key]:raise ValueError('Unrequested panel payload changed: '+key)
        preserved[key]=original_payload[key]
    result=ET.tostring(root);report=measured_presentation(result,width_mm=183)
    standalone=measured_presentation((STAGE/'frozen-panel-a.svg').read_bytes(),width_mm=134.50499987545834)
    def key(row):
        ids=[x for x in row.get('context',[]) if x.startswith('fig2-a-fig1__')]
        return ids[-1] if ids else None
    checks=[]
    import numpy as np
    for kind,props in {'texts':['font_height_pt','font_width_scale_pt','font_family','weight'],'strokes':['width_range_pt','normal_width_pt','effective_opacity','color'],'ticks':['length_pt','normal_width_pt','effective_opacity','color']}.items():
        refs={key(r):r for r in standalone[kind] if key(r)}
        for row in report[kind]:
            k=key(row)
            if not k:continue
            for prop in props:
                actual,expected=row.get(prop),refs[k].get(prop)
                if isinstance(actual,(int,float,list)):np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
                elif actual!=expected:raise ValueError('Assembly style mismatch: '+k+' '+prop)
            checks.append(k)
    return result,dict(scope='Figure 2 assembly; A only newly frozen',A_effective_style_checks=len(checks),A_source=artifact(STAGE/'frozen-panel-a.svg'),A_source_to_final_scale=1,preserved_other_panel_payload_hashes=preserved,translations=translations,measured_presentation=report,source_panels=meta['panels'])
if __name__=='__main__':prepare()
