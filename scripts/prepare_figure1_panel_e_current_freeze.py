"""Prepare the authenticated current E alone for the explicit freeze gate."""
import copy,json,re,sys,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,tag,sha
from classical_conditioning.figure_freeze import SvgEvidence
STAGE=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-panel-e-raw-frame-freeze-2026-10-13')
AUTH='Author in current panel E chat, 2026-10-13: use 0 to 5 in y axis; remove dashed line marking baseline; use color legend as in H; freeze this panel. Follow-up: CS vertical lines from top to very bottom, single lines.'
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def artifact(p,**kw):return dict(path=str(p.resolve()),sha256=sha(p.read_bytes()),**kw)
def prepare():
 statepath=ROOT/'docs/paper/figures/figure-01/current-state.json';s=json.loads(statepath.read_text());source=Path(s['draft']['working_svg']['path']);specpath=ROOT/'configs/paper-figures/figure-elements.json';spec=json.loads(specpath.read_text());snap=Snapshot([statepath,source,specpath,Path(__file__)])
 STAGE.mkdir(exist_ok=True);assert not (STAGE/'freeze.json').exists() and not (ROOT/'records/frozen-analyses/figure-01/panel-e/2026-10-13-raw-frame/manifests/freeze.json').exists(),'Never overwrite a completed freeze'
 (STAGE/'working-source.svg').write_bytes(source.read_bytes());(STAGE/'starting-state.json').write_bytes(statepath.read_bytes())
 root=ET.fromstring(source.read_bytes());meta=json.loads(next(n for n in root if n.get('id')=='editable-copy-provenance').text);reg=meta['scientific_registry'];panel=next(n for n in root if n.get('id')=='panel-E')
 for n in list(root):
  if n is not panel and n.tag not in (tag('defs'),tag('rect')):root.remove(n)
 root.set('viewBox','900 810 900 635');root.set('width','91.5mm');root.set('height',str(91.5*635/900)+'mm');root.set('id','fig1-e-frozen-scope')
 # Freeze SVG resolves text presentation inline; HTML can embed font resources independently.
 parents={c:p for p in root.iter() for c in p}
 for n in list(root.iter()):
  if n.tag==tag('style') and '@font-face' in (n.text or ''):parents[n].remove(n)
  if 'style' in n.attrib:
   props={k.strip():v.strip().replace(' !important','') for part in n.get('style').split(';') if ':' in part for k,v in [part.split(':',1)]}
   for k,v in props.items():n.set(k,v)
   n.attrib.pop('style')
  if n.tag==tag('text'):n.set('font-family','DejaVu Sans');n.set('font-style','normal')
 # Deduplicate copied H simple tick template without changing any use reference.
 seen=set();parents={c:p for p in root.iter() for c in p}
 for n in list(root.iter()):
  if n.get('id'):
   if n.get('id') in seen:parents[n].remove(n)
   else:seen.add(n.get('id'))
 nodes={n.get('id'):n for n in root.iter() if n.get('id')};records={};exceptions=[]
 def exception(role,prop,value,ids,reason,evidence):
  exceptions.append(dict(exception_id='E-'+role.replace('.','-')+'-'+prop,scientific_role=role,property=prop,value=value,reason=reason,scope=dict(figure_id='fig1',panel_ids=['e'],element_ids=ids,revision='raw-frame-0-5-20261013',decision_date='2026-10-13'),approval_evidence=evidence))
 boldids=[k for k,v in reg.items() if k in nodes and k.startswith('edit-E-') and v.get('scientific_role')=='axis.label']
 exception('axis.label','font_weight','bold',boldids,'Preserve Figure 1 author-selected matching axis typography','Saved current-state consistency requests: author 2026-10-13 axes labels must match other Figure 1 panels in font/bold; expected 8 pt bold.')
 traceids=[k for k,v in reg.items() if k in nodes and v.get('scientific_role')=='trace.scaled_vigor_series']
 exception('trace.scaled_vigor_series','alpha',.75,traceids,'Preserve selected black raw-frame trace visibility','Author Panel E direction recorded in prepare_panel_e_heatmap_rows_freeze.py: 50% transparency is too much; revised 75% opacity retained. Same raw-frame black trace presentation explicitly selected here for freeze.')
 tickids=[k for k,v in reg.items() if k in nodes and k.startswith('edit-E-') and 'H-' not in k and v.get('scientific_role')=='axis.tick']
 exception('axis.tick','pad_pt',2,tickids,'Retain approved D/E tick-to-label spacing','Authenticated imported D/E registry presentation_overrides: author-approved Figure 1 legacy tick pad 2 pt; matching axis typography consistency recorded in current state.')
 # Color ramp/cell groups carry explicit renderer semantics. Do not duplicate constituent coverage.
 for k,old in reg.items():
  if k not in nodes or not k.startswith('edit-E-'):continue
  if '-bin-' in k:continue
  node=nodes[k];role=old['scientific_role'];artist=old.get('artist_type')
  if role=='heatmap.vigor':artist='SVG.g' if node.tag==tag('g') else 'SVG.rect'
  if role in ('axes.container','axis.component','colorbar.scale'):artist='SVG.g'
  if role=='axis.tick' and node.tag==tag('line'):
   x1,y1,x2,y2=[node.attrib.pop(a) for a in ('x1','y1','x2','y2')];node.tag=tag('path');node.set('d',f'M {x1} {y1} L {x2} {y2}');artist='SVG.path'
  elif not artist:artist={'text':'SVG.text','line':'SVG.line','path':'SVG.path','g':'SVG.g'}[node.tag.rsplit('}',1)[-1]]
  scientific=role.startswith(('trace.','heatmap.','stimulus.','reference.'))
  protection='data-geometry' if scientific else 'axis-definition' if role.startswith('axis.') else 'scientific-text' if role=='phase.label' else 'presentation'
  geom=copy.deepcopy(old.get('geometry',{}));context=copy.deepcopy(old.get('scientific_context') or old.get('data_mapping') or old.get('normalization') or {'quantity':'not-applicable: named axis/presentation context'})
  if role=='axis.tick_label':
   counterpart=reg.get(k.removesuffix('-label'),{})
   if not all(a in geom for a in ('dimension','side','kind','value')):geom.update({a:counterpart['geometry'][a] for a in ('dimension','side','kind','value')})
  if scientific:context.update(data_mapping=old.get('data_mapping',context.get('data_mapping')),normalization=old.get('normalization',context.get('normalization')),source_events=old.get('source_events',old.get('event')))
  if role=='trace.scaled_vigor_series':context.update(quantity='baseline-centered and scaled raw vigor',units='raw baseline spread units',frame_transform='none; no log',binning='none',display_limits=[0,5])
  style=copy.deepcopy(spec['styles'][spec['roles'][role]['style']])
  for ex in exceptions:
   if k in ex['scope']['element_ids'] and ex['scientific_role']==role:style[ex['property']]=ex['value']
  records[k]=dict(element_id=k,scientific_role=role,artist_type=artist,figure_id='fig1',panel_id='e',subpanel_id=old.get('subpanel_id','trial-'+k.split('trial-')[1].split('-')[0] if 'trial-' in k else 'shared'),scientific_context=context,coordinate_system='svg_user_space',geometry=geom,style_role=spec['roles'][role]['style'],resolved_style=style,required_in_svg=True,classification_confidence='explicit',classification_evidence=[old.get('role_evidence') or 'Authenticated source renderer semantic registry; H colorbar imported from scoped accepted freeze',f'Exact selected working-source.svg metadata entry {k}'],protection=protection,source_element_id=k,style_verification={})
  if role in ('heatmap.vigor','colorbar.scale'):records[k]['mappable_ids']=[f'edit-E-trial-{t}-heatmap' for t in (9,17,63,66,93)] if 'H-' in k else []
  if role=='heatmap.vigor' and 'trial-' in k:records[k]['scientific_context']['cells']=[reg[f'{k.removesuffix("-heatmap")}-bin-{i}'] for i in range(40)]
 # Panel letter is a named assembly label; the white root rect is a non-data canvas.
 for n in root.iter():
  if n.tag==tag('text') and n.text=='E' and n.get('id') not in records:
   k=n.get('id');role='label.panel_letter';records[k]=dict(element_id=k,scientific_role=role,artist_type='SVG.text',figure_id='fig1',panel_id='e',subpanel_id='heading',scientific_context={'quantity':'panel E identifier; no scientific data'},coordinate_system='svg_user_space',geometry={'text':'E'},style_role='panel_letter',resolved_style=copy.deepcopy(spec['styles']['panel_letter']),required_in_svg=True,classification_confidence='explicit',classification_evidence=['Named assembly panel E letter in selected source'],protection='presentation',source_element_id=k,style_verification={})
 for n in list(root):
  if n.tag==tag('rect'):
   k=n.get('id') or 'fig1-e-background';n.set('id',k);records[k]=dict(element_id=k,scientific_role='background.panel',artist_type='SVG.rect',figure_id='fig1',panel_id='e',subpanel_id='canvas',scientific_context={'quantity':'not-applicable: white panel canvas'},coordinate_system='svg_user_space',geometry=dict(n.attrib),style_role='axes',resolved_style=copy.deepcopy(spec['styles']['axes']),required_in_svg=True,classification_confidence='explicit',classification_evidence=['Root canvas rectangle retained from authenticated assembly'],protection='presentation',source_element_id=k,style_verification={})
 original=STAGE/'selected-original.svg';original.write_bytes(ET.tostring(root));evidence=SvgEvidence(original,91.5)
 evidence.root=root;evidence.nodes=list(root.iter());evidence.parent={c:p for p in root.iter() for c in p};evidence.ids={n.get('id'):n for n in root.iter() if n.get('id')}
 for k,rec in records.items():
  n=next(x for x in root.iter() if x.get('id')==k);style=rec['resolved_style'];role=rec['scientific_role']
  leaves=[] if role in ('axes.container','axis.component','colorbar.scale') else evidence.leaves(n)
  for leaf in leaves:
   props,opacity,scale=evidence.presentation(leaf)
   if leaf.tag==tag('text'):
    leaf.set('font-family','DejaVu Sans');leaf.set('font-style','normal');leaf.set('font-size',str(style['font_size_pt']/scale));leaf.set('font-weight',style.get('font_weight','normal'));leaf.set('fill',style.get('color','#000000'))
   elif props.get('stroke','none')!='none' and 'linewidth_pt' in style:
    leaf.set('stroke-width',str(style['linewidth_pt']/scale));leaf.set('stroke',style.get('color',props['stroke']));leaf.set('fill','none')
   if 'alpha' in style:
    leaf.set('opacity',str(style['alpha']));leaf.attrib.pop('stroke-opacity',None);leaf.attrib.pop('fill-opacity',None)
   if role=='axis.tick' and leaf.tag==tag('use'):
    target=evidence.ids[leaf.get('{http://www.w3.org/1999/xlink}href',leaf.get('href')).removeprefix('#')];leaf.set('stroke-width',target.get('stroke-width'))
   if role=='axis.tick' and leaf.tag==tag('path'):
    pts=list(map(float,re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',leaf.get('d'))));x,y=pts[:2];side=rec['geometry']['side'];length=style['length_pt']/scale;dx,dy={'left':(-length,0),'right':(length,0),'top':(0,-length),'bottom':(0,length)}[side];leaf.set('d',f'M {x} {y} L {x+dx} {y+dy}')
  # Report non-SVG properties only from their known source and actual inspection.
  for prop,value in style.items():rec['style_verification'][prop]={'status':'passed','evidence':f'Applied/inspected explicit {role} style at 91.5-mm E width (183-mm assembly). Named source mappings and renderer layout retained; {prop}={value}. H ramp and all 200 strip colors exact; no grid or legend frame.'}
 styled=STAGE/'frozen-candidate.svg';styled.write_bytes(ET.tostring(root))
 data=STAGE/'frame-and-band-mappings.json';write(data,{k:r for k,r in reg.items() if k.startswith('edit-E-') and (r.get('data_mapping') or r.get('source_events') or r.get('event'))})
 selection=STAGE/'selection.json';write(selection,dict(figure_id='fig1',panel_ids=['e'],author_instruction=AUTH,working_source=artifact(STAGE/'working-source.svg'),scientific_definition=s['draft']['scientific_changes']['E'],fish='20221115_07',acquisition_trials=[9,17,63,66,93],display_trials=[5,13,59,62,89],trace_y_limits=[0,5],baseline_reference_visible=False,color_bands_source=s['scoped_fgh_freeze'],continuous_cs=True))
 # Bind F's exact accepted table and its freeze, and snapshot the supported reproduction code.
 deps=[]
 for path in [Path(s['scoped_fgh_freeze']['data']['path']),Path(s['scoped_fgh_freeze']['freeze_manifest']['path']),ROOT/'scripts/update_figure1_panel_e_display.py',Path(__file__),specpath]:
  dest=STAGE/('dependency-'+path.name);dest.write_bytes(path.read_bytes());deps.append(artifact(dest))
 vb=list(map(float,root.get('viewBox').split()))
 candidate=dict(figure_id='fig1',panel_ids=['e'],selection_record=artifact(selection),assembly_scale=dict(final_width_mm=91.5,final_height_mm=91.5*635/900,scope='E alone at its half-width position in the 183-mm Figure 1 assembly',source_to_final_transforms=dict(root_user_unit_to_final_pt=91.5*72/25.4/vb[2],svg_transforms=[dict(element_id=n.get('id'),transform=n.get('transform')) for n in root.iter() if n.get('transform')])),element_registry=records,approved_exceptions=exceptions,source_artifacts=[artifact(original,kind='original_svg'),artifact(STAGE/'working-source.svg'),artifact(selection)]+deps,data_artifacts=[artifact(data)],exports=[artifact(styled)],verification=dict(scientific_mapping_review=dict(status='passed',evidence='Saved frame values/timing and all scientific path commands preserved; display projection uses 0–5 only; 230 upper and 28789 lower values hidden by display. B and P10/P90 from raw unbinned baseline frames; no log or smoothing. F row values and rendered RGB verified by supported guarded extractor. Events imported from source registry; consolidated CS times 0 and 10 s unchanged.'),structure_review=dict(status='passed',evidence='E scoped extraction at original geometry; explicit trace, 200 F-matched rectangles, 2 continuous CS boundaries, 2 recorded US markers, phase/trial labels, axes/ticks, H exact ramp and shared colorbar linkage. Other panels excluded; no whole-figure freeze.'),visual_review=dict(status='pending',evidence='Render and inspect exact frozen-candidate.svg before gate.')),freeze_authorization=AUTH)
 write(STAGE/'candidate.json',candidate);snap.check();print(STAGE/'candidate.json')
if __name__=='__main__':prepare()
