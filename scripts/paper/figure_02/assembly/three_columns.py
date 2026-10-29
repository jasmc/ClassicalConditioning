"""Align the selected Figure 2 in three columns without recomputing science.

Only whole-panel translations, panel-letter translations, the three statistical
insets' uniform size and frame metadata may change. Publication requires fresh
600-dpi browser frame/column evidence and preserves the previous working pair.
"""
from pathlib import Path as _Path
import sys as _sys
ROOT = next(p for p in _Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
_sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, sha, tag, measured_presentation, publish_state, state_lock
from classical_conditioning.artifacts import artifact_staging, publish_transaction
from classical_conditioning.figures.layout_frames import read_spec, embed_spec, assess, POLICY, COLLECTOR
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from scripts.paper.shared.collection import render, reference, encode
from scripts.paper.shared.correct_relationships import matrices
from scripts.paper.shared.reconcile_saved_draft import authenticated, verify_review
from scripts.paper.shared.verify_layout_frames import review_page
import base64

STATS = ('edit-G-axes__g-stats__main', 'matched-H__axes__h-statistics__main', 'trace10-I__axes__i-statistics__main')
CENTRES = (32., 80., 128.)
WIDTH = 159.
FACTOR = 5./6.


def find(root, identity):
    nodes = [n for n in root.iter() if n.get('id') == identity]
    if len(nodes) != 1:
        raise ValueError('Missing/ambiguous element: '+identity)
    return nodes[0]


def protection(source, candidate, changes):
    """Reject changes outside explicit element attributes; text/data stay exact."""
    a, b = ET.fromstring(source), ET.fromstring(candidate)
    # Frame metadata is separately validated from the candidate's exact bytes.
    for root in (a, b):
        root.remove(find(root, 'layout-frames'))
    old, new = list(a.iter()), list(b.iter())
    if len(old) != len(new):
        raise ValueError('Unrequested SVG structure change')
    allowed = {tuple(c['path']): set(c['attributes']) for c in changes}
    paths = {}
    def walk(node, path=()):
        paths[id(node)] = path
        for i, child in enumerate(node):
            walk(child, (*path, i))
    walk(a)
    for x, y in zip(old, new):
        aa, bb = dict(x.attrib), dict(y.attrib)
        for key in allowed.get(paths[id(x)], ()):
            aa.pop(key, None); bb.pop(key, None)
        if x.tag != y.tag or x.text != y.text or aa != bb:
            raise ValueError('Protected scientific/artwork payload changed: '+str(x.get('id')))
    return {'all_data_paths_rasters_text_and_scientific_clips_preserved': True,
            'allowed': 'named panel/letter translations; uniform statistical-inset scale; one phase-header font; frame canvas metadata',
            'changes': changes}


def prepare(directory, geometry_path):
    directory, geometry_path = Path(directory).resolve(), Path(geometry_path).resolve()
    ticket = json.loads((directory/'starting-ticket.json').read_bytes())
    snapshot = Snapshot([*command_sources(__file__), geometry_path, POLICY, COLLECTOR])
    for p, h in ticket['starting_hashes'].items():
        if snapshot.hashes.setdefault(p, h) != h:
            raise ValueError('Starting input changed: '+p)
    snapshot.check()
    state_path = Path(ticket['state_path']); state_raw = state_path.read_bytes()
    if sha(state_raw) != ticket['state_sha256']:
        raise ValueError('Selected figure changed')
    state = json.loads(state_raw); draft = state['draft']
    source = authenticated(draft['working_svg'], snapshot)
    legend = authenticated(draft['legend'], snapshot)
    verify_review(source, authenticated(draft['review'], snapshot), legend, 'fig2')
    profile = load_presentation_profile()
    if profile.inputs != ticket['profile_inputs']:
        raise ValueError('Presentation profile changed')
    geometry = json.loads(geometry_path.read_bytes())
    if geometry['artifact_sha256'] != sha(source):
        raise ValueError('Source geometry belongs to different artwork')
    before_frame = json.loads(authenticated(draft['layout_frames'], snapshot))['browser']
    root = ET.fromstring(source); spec = read_spec(source)
    # Paths exclude the metadata node, matching protection's traversal.
    stripped = copy.deepcopy(root); stripped.remove(find(stripped, 'layout-frames'))
    node_paths = {}
    def paths(node, path=()):
        node_paths[node.get('id')] = path
        for i, child in enumerate(node): paths(child, (*path, i))
    paths(stripped)
    # Anonymous panel letters receive an explicit path instead of a scientific ID.
    def index_path(node):
        parents = {c:p for p in root.iter() for c in p}
        result=[]
        while node is not root:
            parent=parents[node];result.append(list(parent).index(node));node=parent
        result.reverse()
        metadata_index=list(root).index(find(root,'layout-frames'))
        if result and result[0]>metadata_index:result[0]-=1
        return result
    changes = []
    def change(node, attribute, value, role):
        path=index_path(node)
        record=next((c for c in changes if c['path']==path), None)
        if record is None:
            record={'path':path,'element_id':node.get('id'),'role':role,'attributes':{}};changes.append(record)
        record['attributes'][attribute]={'before':node.get(attribute),'after':value}
        node.set(attribute,value)
    def parent_matrix(node):
        parent=next(p for p in root.iter() if node in list(p))
        m=matrices(root)[parent]
        if abs(m[1])+abs(m[2])>1e-8 or min(m[0],m[3])<=0:
            raise ValueError('Unsupported owner coordinate transform')
        return m
    unit=spec['final_size_mm'][0]/float(root.get('viewBox').split()[2])
    groups={g['id']:g['box'] for g in geometry['groups']}
    # Reduce the existing statistical insets, preserving every glyph and mark.
    # Anchor G at the observed-axis right edge; H/I at their observed centres.
    gsp=groups['fig2__g__native__spine-bottom']; gx=gsp[0]+gsp[2]
    bounds=before_frame['data_axes']
    hc=bounds['H-4'][0]+bounds['H-4'][2]/2
    ic=bounds['I-observed'][0]+bounds['I-observed'][2]/2
    for identity, anchor in zip(STATS,(gx,hc,ic)):
        node=find(root,identity);m=parent_matrix(node)
        cx=(anchor/unit-m[4])/m[0];cy=(groups[STATS[0]][1]/unit-m[5])/m[3]
        value=f'translate({cx:.15g} {cy:.15g}) scale({FACTOR:.15g}) translate({-cx:.15g} {-cy:.15g}) '+node.get('transform','')
        change(node,'transform',value,'statistical inset uniform size')
    # The inherited Pre header is 7 pt; the other statistical text is 6 pt.
    pre=find(root,'edit-G-node-53')
    change(pre,'style',pre.get('style','')+';font-size: '+str(6/7*float(next(t['local_size'] for t in geometry['texts'] if t['id']=='edit-G-node-53')))+'px !important','phase-header size')
    b=before_frame['data_axes']
    centre=lambda box:box[0]+box[2]/2
    old_centres={p:centre(b[p+'-pair']) for p in 'ABC'}
    old_centres['D']=(b['D'][0]+b['D-1'][0]+b['D-1'][2])/2
    old_centres.update(E=centre(b['E-pair']),F=centre(b['F-pair']),G=centre(gsp),H=centre(b['H-4']),I=centre(b['I-observed']))
    owners={a['id']:a['group_ids'][0] for a in spec['allocations']}
    translations={}
    for row in ('ABC','DEF','GHI'):
        for letter, target in zip(row,CENTRES):
            node=find(root,owners[letter]);dx=target-old_centres[letter];m=parent_matrix(node)
            change(node,'transform',f'translate({dx/unit/m[0]:.15g} 0) '+node.get('transform',''),'column owner translation')
            translations[letter]=dx
    # Move the colourbar with C, retaining its exact distance from the heatmaps.
    cb=find(root,owners['ABC-colorbar']);m=parent_matrix(cb)
    change(cb,'transform',f'translate({translations["C"]/unit/m[0]:.15g} 0) '+cb.get('transform',''),'colourbar translation')
    letters={x['text']:x['box'] for x in geometry['letters']}
    for row in ('ABC','DEF','GHI'):
        for letter, target in zip(row,(1.06,57.06,105.06)):
            owner=find(root,owners[letter]);matches=[n for n in owner.iter() if n.tag==tag('text') and n.text==letter.lower()]
            if len(matches)!=1:raise ValueError('Ambiguous panel letter '+letter)
            node=matches[0];m=parent_matrix(node);dx=target-letters[letter.lower()][0]-translations[letter]
            change(node,'transform',f'translate({dx/unit/m[0]:.15g} 0) '+node.get('transform',''),'panel-letter column alignment')
    # Reuse current invisible frames, with new column boundaries and 2 mm extra
    # canvas width for the unchanged colourbar and its physical outside margin.
    boundaries={'A':(0,56),'B':(56,48),'C':(104,40.7),'ABC-colorbar':(144.7,14.3),
                'D':(0,56),'E':(56,48),'F':(104,55),'G':(0,56),'H':(56,48),'I':(104,55)}
    for a in spec['allocations']:a['box_mm'][0],a['box_mm'][2]=boundaries[a['id']]
    spec['final_size_mm'][0]=WIDTH
    view=root.get('viewBox').split();view[2]=str(WIDTH/unit)
    change(root,'viewBox',' '.join(view),'canvas width')
    change(root,'width',f'{WIDTH}mm','canvas width')
    bg=find(root,'assembly-node-1');change(bg,'width',str(WIDTH/unit),'white canvas width')
    spec['spacing_groups']=[g for g in spec.get('spacing_groups',[]) if g['id']!='fig2-GHI-equal-spacing']
    for rel in spec.get('panel_relationships',[]):
        if rel['id']=='fig2-GHI-equal-spacing':
            rel['status']='superseded by author-requested three-column centres';rel['previous_spacing_group']=rel.pop('spacing_group',None)
    extra=[{'id':'D-pair','element_ids':['matched-D__D-control-data-rectangle','matched-D__D-delay-data-rectangle']},
           {'id':'G-observed','element_ids':['fig2__g__native__spine-left','fig2__g__native__spine-bottom']},
           {'id':'H-observed','element_id':'matched-H__H-observed-data-rectangle'}]
    spec['data_axes'].extend(extra)
    relation={'id':'fig2-three-column-centres-20261029','type':'layout','panels':list('ABCDEFGHI'),
              'properties':['A/D/G, B/E/H and C/F/I data-body centres aligned','panel letters share column left edges'],
              'columns':[['A-pair','D-pair','G-observed'],['B-pair','E-pair','H-observed'],['C-pair','F-pair','I-observed']],
              'centres_mm':list(CENTRES),'tolerance_mm':.02,'status':'requires fresh exact browser verification'}
    spec['panel_relationships'].append(relation)
    candidate=embed_spec(ET.tostring(root),spec)
    protected=protection(source,candidate,changes)
    page=render([{'id':'fig2','svg_bytes':candidate,'legend_text':legend.decode(),
                  'svg':reference(directory/'candidate.svg',candidate),'legend':draft['legend'],'whole_figure_frozen':False}])[0]
    evidence={'schema_version':1,'source_svg':draft['working_svg'],'source_state':reference(state_path,state_raw),
              'candidate_sha256':sha(candidate),'review_sha256':sha(page),
              'starting_ticket':reference(directory/'starting-ticket.json'),'source_hashes':snapshot.hashes,
              'profile_inputs':profile.inputs,'scoped_override':{'statistical_text_pt':5,'reason':'Author requested smaller last-row statistical indications'},
              'translations_mm':translations,'relationship':relation,'protection':protected}
    payloads={'candidate.svg':candidate,'review.html':page,
              'layout-review.html':review_page([{'name':'candidate.svg','svg_base64':base64.b64encode(candidate).decode()}]).encode(),
              'edit-evidence.json':encode(evidence)}
    with artifact_staging(directory,prefix='.columns-') as staging:
        units=[]
        for i,(name,raw) in enumerate(payloads.items()):
            staged=staging/str(i);staged.write_bytes(raw);units.append((staged,directory/name))
        def verify():
            snapshot.check()
            for name,raw in payloads.items():
                if (directory/name).read_bytes()!=raw:raise ValueError('Candidate publication byte mismatch')
        with state_lock(state_path):
            snapshot.check();publish_transaction(units,staging,overwrite=False,verify_published=verify)
    return {'candidate':str(directory/'review.html'),'svg_sha256':sha(candidate)}


def publish(directory, browser_path, note, check_only=False):
    directory=Path(directory).resolve();ep=directory/'edit-evidence.json'
    snapshot=Snapshot([*command_sources(__file__),ep,Path(browser_path),directory/'candidate.svg',directory/'review.html'])
    edit=json.loads(ep.read_bytes())
    for p,h in edit['source_hashes'].items():
        if snapshot.hashes.setdefault(p,h)!=h:raise ValueError('Implementation/source changed: '+p)
    snapshot.check();sp=Path(edit['source_state']['path']);original=authenticated(edit['source_state'],snapshot);state=json.loads(original);draft=state['draft']
    source=authenticated(draft['working_svg'],snapshot);oldhtml=authenticated(draft['review'],snapshot);legend=authenticated(draft['legend'],snapshot)
    candidate=(directory/'candidate.svg').read_bytes()
    if sha(candidate)!=edit['candidate_sha256'] or sha((directory/'review.html').read_bytes())!=edit['review_sha256']:
        raise ValueError('Prepared candidate was changed')
    protection(source,candidate,edit['protection']['changes'])
    # Replay expected attribute edits rather than permitting arbitrary new values.
    a,b=ET.fromstring(source),ET.fromstring(candidate)
    for r in (a,b):r.remove(find(r,'layout-frames'))
    for change in edit['protection']['changes']:
        node=b
        for i in change['path']:node=list(node)[i]
        for key,values in change['attributes'].items():
            if node.get(key)!=values['after']:raise ValueError('Candidate edit differs from prepared value')
    browser=json.loads(Path(browser_path).read_bytes())
    if 'artifact_sha256'not in browser:browser=browser['candidate.svg']
    frame=assess(candidate,browser)
    if frame['status']!='passed':raise ValueError(frame['issues'])
    # Containment/column centres alone cannot certify statistical registration.
    # This baseline recipe's inset scaling must never select a trial-misaligned
    # candidate again; the maintained correction restores the common trial map.
    from scripts.paper.figure_02.assembly.align_statistics import registration
    trial_checks=registration(ET.fromstring(candidate))
    if max(v['max_error_mm']for v in trial_checks.values())>.02:
        raise ValueError('Statistics no longer match trajectory trials; use align_statistics.py before publication')
    measured={}
    for ids,target in zip(edit['relationship']['columns'],CENTRES):
        centres=[browser['data_axes'][i][0]+browser['data_axes'][i][2]/2 for i in ids]
        if max(abs(v-target) for v in centres)>.02:raise ValueError('Column centres do not align: '+str(centres))
        measured['/'.join(ids)]=centres
    if not note:raise ValueError('Actual final visual review note required')
    verify_review(candidate,(directory/'review.html').read_bytes(),legend,'fig2')
    fb=encode({**frame,'column_centres_mm':measured,'visual_review':note,'protection':edit['protection']})
    fr=reference(directory/'layout-evidence.json',fb)
    registry=json.loads(authenticated(draft['semantic_registry'],snapshot))
    relation={**edit['relationship'],'status':'verified','evidence':fr,'artifact_sha256':sha(candidate)}
    registry.setdefault('panel_relationships',[]).append(relation)
    registry['three_column_presentation']={'relationship':relation,'statistics':edit['scoped_override'],'profile_inputs':edit['profile_inputs'],
        'evidence':fr,'coverage':'Presentation alignment only; inherited scientific mapping/approval gaps remain.'}
    if registry.get('axis_inventory'):registry['axis_inventory']['source_svg_sha256']=sha(candidate)
    rb=encode(registry)
    draft['working_svg']={**draft['working_svg'],'sha256':sha(candidate)}
    page=render([{'id':'fig2','svg_bytes':candidate,'legend_text':legend.decode(),'svg':draft['working_svg'],'legend':draft['legend'],'whole_figure_frozen':False}])[0]
    draft['review']={**draft['review'],'sha256':sha(page)};draft['semantic_registry']=reference(directory/'semantic-registry.json',rb);draft['layout_frames']=fr
    pb=encode(measured_presentation(candidate,width_mm=WIDTH));draft['effective_presentation']=reference(directory/'effective-presentation.json',pb)
    draft.setdefault('assembly_scale',{})['final_width_mm']=WIDTH
    draft.setdefault('consistency_requests',[]).append(relation);draft.setdefault('verification',{})['three_columns']=fr
    artifacts=[(directory/'before-state.json',original),(directory/'before.svg',source),(directory/'before.html',oldhtml),
               (directory/'legend.md',legend),(directory/'layout-evidence.json',fb),(directory/'semantic-registry.json',rb),(directory/'effective-presentation.json',pb)]
    for p,_ in artifacts:
        if p.exists():raise FileExistsError(p)
    snapshot.check()
    if not check_only:publish_state(sp,state,snapshot,artifacts=artifacts,evidence_updates=[(Path(draft['working_svg']['path']),candidate),(Path(draft['review']['path']),page)])
    return {'status':'checked'if check_only else'published','figure_id':'fig2','svg_sha256':sha(candidate),'column_centres_mm':measured}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','publish']);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--geometry',type=Path);parser.add_argument('--browser-report',type=Path);parser.add_argument('--visual-review-note');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    result=prepare(args.directory,args.geometry)if args.command=='prepare'else publish(args.directory,args.browser_report,args.visual_review_note,args.check_only)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
