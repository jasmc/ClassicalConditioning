"""Extend existing draft registries with explicit axis identities; never edit SVGs.

Addresses without unique XML IDs are version-bound structural addresses, not a
claim of complete new-freeze coverage. Scientific roles use named renderer axis
artists plus authenticated legends/selections; appearance is never an identity.
"""
from __future__ import annotations
import argparse
import copy
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot, sha, file_hash, publish_state, state_lock, tag
from main_paper_figures import reference,encode,authenticate


def axis_definitions(number):
    axes=[]
    def add(identity,panel,orientation,quantity,units,groups,titles=(),condition=None,**extra):
        prefix=groups[0].rsplit('axis-',1)[0] if len(groups)==1 and 'axis-' in groups[0] else None
        if prefix and '__heatmap__' in prefix:
            extra['spine_ids']=[prefix+'spine-'+side for side in (('bottom','top') if orientation=='x' else ('left','right'))]
        if prefix and '__colorbar__' in prefix:extra['spine_ids']=[prefix+'spine-outline']
        axes.append(dict(id=identity,panel=panel,orientation=orientation,quantity=quantity,units=units,group_ids=list(groups),title_ids=list(titles),condition=condition,**extra))
    if number==1:
        for name in ('control','delay','trace-3s','trace-10s'):
            title='left' if name in ('control','trace-3s') else 'right'
            add('B-'+name+'-x','B','x','time relative to CS onset','s',['edit-B-time-axis-'+name],['edit-B-column-time-label-'+title],name,coordinate_origin='CS onset',scientific_status='protocol schematic; not measured responses')
        for panel in ('D','E'):
            ids=[f'edit-{panel}-time__spine-bottom',*[f'edit-{panel}-time__xtick_{i}' for i in range(5)],*[f'edit-{panel}-time__xtick_{i}-label' for i in range(5)]]
            add(panel+'-x',panel,'x','measured time relative to CS onset','s',ids,[f'edit-{panel}-time-label'],'Delay',coordinate_origin='measured CS onset',shared_by_acquisition_trials=[9,17,63,66,93])
            for trial in (9,17,63,66,93):
                ids=[f'edit-{panel}-trial-{trial}__spine-left',*[f'edit-{panel}-trial-{trial}__ytick_{i}' for i in range(3)],*[f'edit-{panel}-trial-{trial}__ytick_{i}-label' for i in range(3)]]
                add(f'{panel}-{trial}-y',panel,'y','baseline-centred distal cumulative tail angle' if panel=='D' else 'raw-frame baseline-centred and scaled vigor','degrees' if panel=='D' else 'baseline units',ids,[f'edit-{panel}-value-label'],'Delay',acquisition_trial=trial,displayed_trial=trial-4,fish='20221115_07',coordinate_origin='trial median baseline' if panel=='D' else 'eligible raw-frame baseline B; no logarithm')
        for panel,condition,fish in [('F','Delay','20221115_07'),('G','3 s Trace','20230307_12'),('H','Control','20221115_09')]:
            for orientation in ('x','y'):
                prefix=f'edit-FGH-fig1__{panel.lower()}__heatmap__'
                add(panel+'-'+orientation,panel,orientation,'cadence-reconstructed time relative to CS onset' if orientation=='x' else 'displayed trial number','s' if orientation=='x' else 'trial', [prefix+'axis-'+orientation],[prefix+'label-x'] if orientation=='x' else [],condition,fish=fish,coordinate_origin='CS onset' if orientation=='x' else 'acquisition trial minus 4',shared_axis_group='fig1-FGH-trial-y' if orientation=='y' else None)
        for identity,panel,prefix in [('E-band-colorbar','E','edit-E-H-fig1__h__colorbar__'),('FGH-colorbar','F/G/H','edit-FGH-fig1__h__colorbar__')]:
            add(identity,panel,'color','centred direct 1 s log-frame median vigor','natural-log vigor difference',[prefix+'axis-y'],[prefix+'label-y'],applies_to='E F-matched bands only; not E black traces' if panel=='E' else 'F/G/H heatmap colors',display_limits=[-.25,.25])
    else:
        for letter in ('a','b','c'):
            for index in (0,1):
                condition=('Control' if index==0 else {'a':'Delay','b':'3 s Trace','c':'10 s Trace'}[letter]);prefix=f'fig2__{letter}{index}__heatmap__'
                for orientation in ('x','y'):
                    add(f'{letter}{index}-{orientation}',letter.upper(),orientation,'time relative to CS onset' if orientation=='x' else 'displayed trial number','s' if orientation=='x' else 'trial',[prefix+'axis-'+orientation],[prefix+'label-x'] if orientation=='x' and index==0 else [],condition,coordinate_origin='CS onset' if orientation=='x' else 'acquisition trial minus 4',scientific_status='pending inputs; no data' if letter=='c' else 'accepted A estimator' if letter=='a' else 'retained B estimator; differs from A',shared_axis_group='fig2-ABC-trial-y' if orientation=='y' else None,control_identity='matched to '+{'a':'Delay','b':'3 s Trace','c':'10 s Trace'}[letter]+'; cohort equality requires selected roster hashes' if index==0 else None)
        add('ABC-colorbar','A/B/C','color','displayed population color value; estimator is panel-specific','see panel scientific definition',['fig2__shared__colorbar__axis-y'],['fig2__shared__colorbar__label-y'],display_limits=[-.25,.25],caution='Common color key does not establish a common estimator: B differs from A, C has no result')
        for panel,assay in [('D','d-delay'),('E','e-trace')]:
            for condition,key in [('paired',assay),('matched control',panel.lower()+'-control')]:
                add(f'{panel}-{condition}-x',panel,'x','selected session block','categorical block',[f'edit-{panel}-axis__{key}__x'],condition=('Delay' if panel=='D' else '3 s Trace') if condition=='paired' else condition,matched_assay='Delay' if panel=='D' else '3 s Trace',display_categories=['PT','ET','LT'],scientific_definition='Pre-training / early test / late test; trial windows and eligibility from scoped D/E freeze')
        add('DE-shared-y','D/E','y','median natural-log positive valid response-bout vigor minus baseline','natural-log vigor difference',['shared-DEF-edit-E-axis__e-trace__y'],['edit-D-axis-title__d-delay__y'],shared_axis_group='fig2-DE-response-minus-baseline-y',spine_ids=['edit-D-spine__d-delay__left'],not_members='F is a Pending placeholder and supplies no scientific values')
        for orientation in ('x','y'):
            add('G-'+orientation,'G',orientation,'global CS acquisition trial' if orientation=='x' else 'historical log-median response-minus-baseline trajectory','trial' if orientation=='x' else 'natural-log vigor difference',['matplotlib.axis_1' if orientation=='x' else 'matplotlib.axis_2'],['fig2__g__native__label-'+orientation],condition=['Delay','matched Control'],spine_ids=['fig2__g__native__spine-'+('bottom' if orientation=='x' else 'left')],scientific_definition='scoped G LogMedian selection')
            add('H-'+orientation,'H',orientation,'global CS trial' if orientation=='x' else 'response activity / pre-CS baseline activity','trial' if orientation=='x' else 'dimensionless ratio',['edit-H-matplotlib.axis_1' if orientation=='x' else 'edit-H-matplotlib.axis_2'],['edit-H-node-55' if orientation=='x' else 'edit-H-node-90'],'3 s Trace / matched control',caution='H ratio is distinct from D/E/G log differences; shared y semantics are not asserted')
        for orientation in ('x','y'):
            add('G-stats-'+orientation,'G',orientation,'auxiliary statistical annotation coordinate','categorical / annotation position',[f'edit-G-axis__g-stats__{orientation}'],scientific_status='hypotheses and correction families belong to scoped G evidence; annotation coordinates are not the main trajectory outcome')
    return axes


def build_inventory(svg,number,legend_ref,selections,existing):
    root=ET.fromstring(svg);paths={};ids={};parents={}
    def visit(node,path):
        paths[node]=path
        if node.get('id'):ids.setdefault(node.get('id'),[]).append(node)
        for i,child in enumerate(node):parents[child]=node;visit(child,path+[i])
    visit(root,[])
    title_roles=existing['presentation_styles']['axis_titles']['roles'];title_ids={r['element_id'] for r in title_roles}
    def address(node):
        identity=node.get('id');return {'element_id':identity,'unique_id':bool(identity) and len(ids.get(identity,[]))==1,'xml_index_path':paths[node],'svg_type':node.tag.rsplit('}',1)[-1],'source_element_id':node.get('data-source-id',node.get('data-comparison-source-id'))}
    records=[];unresolved=[]
    for definition in axis_definitions(number):
        record=copy.deepcopy(definition);groups=[];missing=[]
        for mapping in existing.get('condition_order_mapping',{}).get('mapping',[]):
            if mapping.get('source_panel')==definition['panel']:
                record['source_panel']=definition['panel'];record['panel']=mapping['display_panel']
                record['display_mapping_reference']='condition_order_mapping in this same semantic registry'
        for identity in definition['group_ids']:
            if len(ids.get(identity,[]))!=1:missing.append(identity)
            else:groups.append(ids[identity][0])
        texts=[];marks=[];spines=[]
        seen=set()
        for group in groups:
            for node in group.iter():
                if node in seen or node.tag in {tag('defs'),tag('clipPath')}:continue
                ancestry_nodes=[];parent=node
                while parent is not None:
                    ancestry_nodes.append(parent);parent=parents.get(parent)
                if any(n.tag in {tag('defs'),tag('clipPath'),tag('marker'),tag('symbol')} for n in ancestry_nodes):continue
                seen.add(node);identity=node.get('id','');source=node.get('data-source-id','')
                if node.tag==tag('text'):
                    ancestry=[];parent=node
                    while parent is not None:
                        ancestry.append(parent.get('id',''));parent=parents.get(parent)
                    if not any(i in title_ids or i.endswith(('__label-x','__label-y')) or 'axis-title__' in i or i in definition['title_ids'] for i in ancestry):
                        texts.append({**address(node),'scientific_role':'axis.tick_label','display_text':''.join(node.itertext())})
                elif node.tag.rsplit('}',1)[-1] in {'line','path','use'}:
                    if 'spine' in identity or 'spine' in source:spines.append({**address(node),'scientific_role':'axis.spine'})
                    elif any(re.search(r'(?:[xy]tick|tick[ml]|tickmark)',n.get('id','')+' '+n.get('data-source-id','')) for n in ancestry_nodes):marks.append({**address(node),'scientific_role':'axis.tick','classification':'mark under named renderer tick artist'})
                    else:marks.append({**address(node),'scientific_role':'axis.component','classification':'axis-owned mark; tick/spine subtype unresolved without explicit renderer mapping'})
        for identity in definition.get('spine_ids',[]):
            if len(ids.get(identity,[]))!=1:missing.append(identity)
            else:spines.append({**address(ids[identity][0]),'scientific_role':'axis.spine'})
        titles=[]
        for identity in definition['title_ids']:
            if len(ids.get(identity,[]))!=1:missing.append(identity);continue
            group=ids[identity][0]
            for node in ([group] if group.tag==tag('text') else group.iter(tag('text'))):titles.append({**address(node),'scientific_role':'axis.label','display_text':''.join(node.itertext()),'style_group':'main-paper-axis-titles'})
        record.update({'scientific_role':'axis.component','axis_artists':[address(n) for n in groups],'titles':titles,'tick_labels':texts,'axis_marks':marks,'explicit_spines':spines,'classification_evidence':{'kind':'named renderer axis artists plus associated legend/scoped selections','legend':legend_ref,'scientific_selections':selections},'status':'identified-in-current-draft' if not missing else 'unresolved','missing_artists':missing})
        if missing:unresolved.append({'axis':definition['id'],'reason':'Missing or ambiguous axis artist','elements':missing})
        records.append(record)
    addresses=[x for a in records for key in ('axis_artists','titles','tick_labels','axis_marks','explicit_spines') for x in a[key]]
    return {'schema_version':1,'figure_id':f'fig{number}','source_svg_sha256':sha(svg),'axes':records,'axis_count':len(records),'address_count':len(addresses),'distinct_element_address_count':len({tuple(a['xml_index_path']) for a in addresses}),'unique_id_address_count':sum(a['unique_id'] for a in addresses),'panels_without_numerical_axes':['A','C'] if number==1 else [],'placeholder_panels_without_result':['C','F','I'] if number==2 else [],'panel_relationships_reference':'panel_relationships in this same semantic registry','style_reference':'presentation_styles.axis_titles in this same semantic registry','scientific_identity_requirements':['Identify experiment/assay and condition independently of labels/colors','Bind paired and matched-control cohorts to roster/source hashes before asserting equality across panels or figures','Retain fish/trial/frame/block identities, coordinate origin, event definitions, eligibility, normalization and statistical-family mappings','Keep layout groups, shared-axis information and scientific data associations distinct'],'freeze_requirements':{'whole_figure_ready':False,'required':['Every visible element needs a unique stable XML ID and a semantic registry entry','Version-bound structural addresses do not replace stable IDs or scientific mappings','Complete required experiment/cohort/control links using authenticated selected data sources','Resolve missing role, source, scientific and final-scale coverage before freeze'],'remaining':['Inventory does not classify every non-axis scientific mark','Legacy anonymous axis descendants still require stable IDs for a new freeze','Do not infer matched-control roster equality from similar cohort counts','Legacy spine subtypes without explicit source mappings remain unresolved']},'unresolved':unresolved}


def publish(ticket,directory):
    baseline=json.loads(ticket.read_text());baseline.pop(str(Path(__file__).resolve()),None)
    snapshot=Snapshot([]);snapshot.hashes=baseline;snapshot.check();snapshot.hashes[str(Path(__file__).resolve())]=file_hash(__file__)
    if not directory.is_dir() or not Path('/Volumes/JOAQUIM').is_mount():raise ValueError('Existing mounted external registry directory required')
    prepared=[];summary={}
    for number in (1,2):
        statepath=ROOT/f'docs/paper/figures/figure-{number:02}/current-state.json';raw=statepath.read_bytes();state=json.loads(raw);draft=state['draft'];reg=draft['semantic_registry'];registry=json.loads(authenticate(reg,snapshot)) if 'path'in reg else copy.deepcopy(reg)
        svg=authenticate(draft['working_svg'],snapshot);authenticate(draft['legend'],snapshot)
        inventory=build_inventory(svg,number,draft['legend'],state['scientific_selection']['records'],registry)
        if inventory['unresolved']:raise ValueError(inventory['unresolved'])
        registry['axis_inventory']=inventory
        registry.setdefault('element_identification_requirements',{}).update({'axis_roles':['axes.container','axis.component','axis.tick','axis.tick_label','axis.label','axis.spine'],'typed_relationships':['layout','scientific','presentation-style'],'required_scientific_entities':['experiment/assay','condition','matched control','cohort','fish','trial','frame','block'],'freeze_readiness':'partial; do not claim all elements or scientific links registered'})
        evidencepath=directory/f'figure-{number:02}-axis-registration.json';eb=encode({'figure_id':f'fig{number}','source_state':reference(statepath,raw),'source_svg':draft['working_svg'],'legend':draft['legend'],'inventory':inventory,'artwork_changed':False,'generator':reference(Path(__file__))});evidence=reference(evidencepath,eb)
        registry['axis_inventory']['registration_evidence']=evidence
        artifacts=[(evidencepath,eb),(directory/f'figure-{number:02}-before-axis-registration-state.json',raw)]
        if 'path'in reg:
            dest=directory/f'figure-{number:02}-semantic-registry.json';rb=encode(registry);artifacts.append((dest,rb));draft['semantic_registry']=reference(dest,rb)
        else:draft['semantic_registry']=registry
        draft.setdefault('consistency_requests',[]).append({'id':f'fig{number}-axis-identification-registry-20261015','scope':[f'Figure {number}'],'author_direction':'Add missing axis/element identification and associated scientific-scope requirements to the registry','property':'registered axis identities, quantities/units, artists, titles, ticks, spines and typed panel relationships','status':'verified-for-listed-axis-artists; whole-figure scientific coverage partial','affected_artifact_hashes':{'svg':sha(svg)},'evidence':evidence})
        prepared.append((statepath,state,artifacts));summary[f'fig{number}']={k:inventory[k] for k in ('axis_count','address_count','unique_id_address_count','unresolved')}
    snapshot.check()
    for statepath,state,artifacts in prepared:
        publish_state(statepath,state,snapshot,artifacts=artifacts)
        snapshot.hashes[str(statepath.resolve())]=file_hash(statepath)
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ticket',required=True,type=Path);parser.add_argument('--directory',required=True,type=Path);args=parser.parse_args();publish(args.ticket,args.directory)
if __name__=='__main__':main()
