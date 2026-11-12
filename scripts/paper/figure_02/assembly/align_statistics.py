"""Restore trial registration of G/H/I statistics and abbreviate row labels.

The original statistical payloads are restored from their authenticated saved
SVG. A common trial map is recovered from trajectory ticks and phase-header
anchors, never from pictures. Text glyphs retain their original proportions.
"""
from pathlib import Path as _Path
import sys as _sys
ROOT=next(p for p in _Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
_sys.path[:0]=[str(ROOT),str(ROOT/'src')]

import argparse
import base64
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from classical_conditioning.artifacts import artifact_staging, publish_transaction
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot,sha,tag,state_lock,publish_state,measured_presentation
from classical_conditioning.figures.layout_frames import read_spec,embed_spec,assess,POLICY,COLLECTOR
from scripts.paper.shared.correct_relationships import matrices
from scripts.paper.shared.collection import encode,reference,render
from scripts.paper.shared.reconcile_saved_draft import authenticated,verify_review
from scripts.paper.shared.verify_layout_frames import review_page

STATS={'G':'edit-G-axes__g-stats__main','H':'matched-H__axes__h-statistics__main','I':'trace10-I__axes__i-statistics__main'}
OBSERVED={'G':'fig2__g__native__axes','H':'matched-H__axes__h-observed__main','I':'trace10-I__axes__i-observed__main'}
LABELS={'edit-G-node-39':('D: block interaction','D'),
        'edit-G-node-42':('M: block mean','M'),
        'edit-G-node-45':('R: slope (raw)','R'),
        'edit-G-node-48':('R: slope (FDR)','R-FDR'),
        'edit-G-node-51':('Trial change (FDR)','T-FDR')}
PHASES=('Pre','Tr1','Tr2','Tr3','Tr4','Tr5','Te1','Te2','Te3')

def find(root,identity):
    found=[n for n in root.iter()if n.get('id')==identity]
    if len(found)!=1:raise ValueError('Missing/ambiguous element '+identity)
    return found[0]

def text(root,identity):
    n=find(root,identity)
    if n.tag==tag('text'):return n
    found=list(n.iter(tag('text')))
    if len(found)!=1:raise ValueError('Ambiguous text '+identity)
    return found[0]

def headers(root,panel):
    return [text(root,'edit-G-node-'+str(53+2*i))if panel=='G'else
            text(root,('matched-H__H-'if panel=='H'else'trace10-I__I-')+'phase-header-'+phase)
            for i,phase in enumerate(PHASES)]

def anchor_mm(root,node):
    m=matrices(root)[node];unit=float(root.get('width')[:-2])/float(root.get('viewBox').split()[2])
    return (m[0]*float(node.get('x','0'))+m[2]*float(node.get('y','0'))+m[4])*unit

def trial_map(root,panel):
    group=find(root,OBSERVED[panel]);ticks={}
    for value in (5,94):
        nodes=[n for n in group.iter(tag('text'))if n.text==str(value)]
        if len(nodes)!=1:raise ValueError('Ambiguous observed trial tick')
        ticks[value]=anchor_mm(root,nodes[0])
    slope=(ticks[94]-ticks[5])/89
    return slope,ticks[5]-5*slope

def registration(root):
    report={}
    for panel in STATS:
        slope,intercept=trial_map(root,panel)
        checks=[{'phase':phase,'trial':9.5+10*i,'stats_x_mm':anchor_mm(root,n),
                 'trajectory_x_mm':intercept+slope*(9.5+10*i)}
                for i,(phase,n)in enumerate(zip(PHASES,headers(root,panel)))]
        error=max(abs(c['stats_x_mm']-c['trajectory_x_mm'])for c in checks)
        report[panel]={'observed_trial_map_mm':[slope,intercept],'phase_checks':checks,'max_error_mm':error}
    return report

def outside_statistics(raw):
    root=ET.fromstring(raw)
    root.remove(find(root,'layout-frames'))
    for identity in STATS.values():
        n=find(root,identity);n.clear();n.set('id',identity)
    return ET.tostring(root)

def check_payload(source,candidate,original,edits):
    if outside_statistics(source)!=outside_statistics(candidate):
        raise ValueError('Artwork outside statistical insets changed')
    a,b=ET.fromstring(original),ET.fromstring(candidate)
    for panel,identity in STATS.items():
        expected=copy.deepcopy(find(a,identity));actual=copy.deepcopy(find(b,identity))
        spec=edits[panel]
        if actual.get('transform')!=spec['after_transform']:
            raise ValueError('Unexpected statistical coordinate transform')
        old=expected.get('transform')
        if old is None:actual.attrib.pop('transform',None)
        else:actual.set('transform',old)
        old_text=list(expected.iter(tag('text')));new_text=list(actual.iter(tag('text')))
        if len(old_text)!=len(new_text):raise ValueError('Statistical mark count changed')
        for i,(x,y)in enumerate(zip(old_text,new_text)):
            if y.get('transform')!=spec['text_transforms'][i]['after']:
                raise ValueError('Unexpected statistical glyph transform')
            if x.get('transform')is None:y.attrib.pop('transform',None)
            else:y.set('transform',x.get('transform'))
            if x.get('id')in LABELS:
                before,after=LABELS[x.get('id')]
                if x.text!=before or y.text!=after:raise ValueError('Unapproved row-label change')
                y.text=before
        if ET.tostring(expected)!=ET.tostring(actual):
            raise ValueError('Original statistical values/mark geometry changed')
    return {'outside_statistics_unchanged':True,'original_statistical_values_and_raw_mark_coordinates_preserved':True,
            'row_label_aliases':LABELS,'glyph_proportions_preserved':True}

def prepare(directory,original_path):
    directory=Path(directory).resolve();ticket_path=directory/'starting-ticket.json'
    ticket=json.loads(ticket_path.read_bytes());snapshot=Snapshot([*command_sources(__file__),ticket_path,Path(original_path),POLICY,COLLECTOR])
    snapshot.bind_hashes(ticket['starting_hashes']);snapshot.check()
    sp=Path(ticket['state_path']);state_raw=sp.read_bytes()
    if sha(state_raw)!=ticket['state_sha256']:raise ValueError('Selected state advanced')
    state=json.loads(state_raw);d=state['draft'];source=authenticated(d['working_svg'],snapshot);oldlegend=authenticated(d['legend'],snapshot)
    verify_review(source,authenticated(d['review'],snapshot),oldlegend,'fig2')
    original=Path(original_path).read_bytes()
    if sha(original)!='b480a26eead5d7a55380dcc668ae02f072e37be16e056662f4c4003c8baa7ad9':
        raise ValueError('Use the authenticated pre-shrink statistical source')
    root=ET.fromstring(source);prior=ET.fromstring(original)
    for identity in STATS.values():
        node=find(root,identity);parent=next(p for p in root.iter()if node in list(p));i=list(parent).index(node)
        parent.remove(node);parent.insert(i,copy.deepcopy(find(prior,identity)))
    unit=float(root.get('width')[:-2])/float(root.get('viewBox').split()[2]);edits={}
    for panel,identity in STATS.items():
        slope,intercept=trial_map(root,panel);nodes=headers(root,panel)
        x0,x8=anchor_mm(root,nodes[0]),anchor_mm(root,nodes[-1]);stat_slope=(x8-x0)/80
        sx=slope/stat_slope;tx_mm=intercept+slope*9.5-sx*x0
        node=find(root,identity);parent=next(p for p in root.iter()if node in list(p));m=matrices(root)[parent]
        if abs(m[1])+abs(m[2])>1e-8:raise ValueError('Rotated statistical parent')
        dx=(sx*m[4]+tx_mm/unit-m[4])/m[0]
        old=node.get('transform','');node.set('transform',f'matrix({sx:.16g} 0 0 1 {dx:.16g} 0) '+old)
        transforms=[]
        for t in node.iter(tag('text')):
            before=t.get('transform');x=float(t.get('x','0'))
            # Inverse text-only X scale keeps the original glyph dimensions;
            # its anchor follows the corrected shared trial coordinate.
            after=(before or'')+f' translate({x:.16g} 0) scale({1/sx:.16g} 1) translate({-x:.16g} 0)'
            t.set('transform',after);transforms.append({'before':before,'after':after})
        edits[panel]={'x_scale':sx,'x_shift_mm':tx_mm,'after_transform':node.get('transform'),'text_transforms':transforms}
    for identity,(before,after)in LABELS.items():
        node=find(root,identity)
        if node.text!=before:raise ValueError('Unexpected source row label')
        node.text=after
    measured=registration(root)
    if max(r['max_error_mm']for r in measured.values())>.001:raise ValueError('Prepared trial registration failed')
    spec=read_spec(source);relation={'id':'fig2-statistics-trial-registration-20261029','type':'layout','panels':['G','H','I'],
        'properties':['statistical marks and phase headers use the trajectory global-CS-trial mapping','row explanations moved to matching legend'],
        'source_definition':'Observed SVG trial ticks 5/94; saved producer phase centres 9.5+10*i and shared (4,95) trial limits',
        'tolerance_mm':.02,'status':'requires fresh browser verification'}
    spec['panel_relationships'].append(relation);candidate=embed_spec(ET.tostring(root),spec)
    protected=check_payload(source,candidate,original,edits)
    legend=oldlegend.decode();old='Separate within-panel families: baseline-adjusted block interaction (Holm8), local condition mean (BH9), local slope (raw/BH9), phase-aware random-intercept/slope LMM change from average Pre5–14 (BH90).'
    new='Shared statistical row labels: D, baseline-adjusted block interaction (Holm8); M, local condition mean (BH9); R, local slope (raw); R-FDR, local slope (BH9); T-FDR, phase-aware random-intercept/slope LMM change from average Pre5–14 (BH90).'
    if legend.count(old)!=1:raise ValueError('Canonical legend wording advanced')
    legend=legend.replace(old,new).encode()
    page=render([{'id':'fig2','svg_bytes':candidate,'legend_text':legend.decode(),'svg':reference(directory/'candidate.svg',candidate),
                  'legend':reference(directory/'legend.md',legend),'whole_figure_frozen':False}])[0]
    evidence={'source_state':reference(sp,state_raw),'source_svg':d['working_svg'],'original_statistics':reference(Path(original_path)),
              'source_hashes':snapshot.hashes,'candidate_sha256':sha(candidate),'review_sha256':sha(page),'legend_sha256':sha(legend),
              'statistics_edits':edits,'protection':protected,'trial_registration':measured,'relationship':relation,
              'scientific_selection_changed':False,'freeze_created':False}
    files={'candidate.svg':candidate,'review.html':page,'legend.md':legend,'edit-evidence.json':encode(evidence),
           'layout-review.html':review_page([{'name':'candidate.svg','svg_base64':base64.b64encode(candidate).decode()}]).encode()}
    with artifact_staging(directory,prefix='.statistics-')as staging:
        units=[]
        for i,(name,raw)in enumerate(files.items()):p=staging/str(i);p.write_bytes(raw);units.append((p,directory/name))
        def verify():
            snapshot.check()
            for name,raw in files.items():
                if (directory/name).read_bytes()!=raw:raise ValueError('Candidate publication mismatch')
        with state_lock(sp):snapshot.check();publish_transaction(units,staging,overwrite=False,verify_published=verify)
    return {'candidate':str(directory/'review.html'),'registration':measured}

def publish(directory,browser_path,note,check_only=False):
    directory=Path(directory).resolve();ep=directory/'edit-evidence.json';edit=json.loads(ep.read_bytes())
    snapshot=Snapshot([*command_sources(__file__),ep,Path(browser_path),directory/'candidate.svg',directory/'review.html',directory/'legend.md'])
    snapshot.bind_hashes(edit['source_hashes']);snapshot.check();sp=Path(edit['source_state']['path']);oldstate=authenticated(edit['source_state'],snapshot)
    state=json.loads(oldstate);draft=state['draft'];source=authenticated(draft['working_svg'],snapshot);oldhtml=authenticated(draft['review'],snapshot);oldlegend=authenticated(draft['legend'],snapshot)
    candidate=(directory/'candidate.svg').read_bytes();legend=(directory/'legend.md').read_bytes()
    if sha(candidate)!=edit['candidate_sha256']or sha(legend)!=edit['legend_sha256']or sha((directory/'review.html').read_bytes())!=edit['review_sha256']:
        raise ValueError('Prepared candidate changed')
    check_payload(source,candidate,authenticated(edit['original_statistics'],snapshot),edit['statistics_edits'])
    browser=json.loads(Path(browser_path).read_bytes())
    if 'artifact_sha256'not in browser:browser=browser['candidate.svg']
    frame=assess(candidate,browser)
    if frame['status']!='passed':raise ValueError(frame['issues'])
    measured=registration(ET.fromstring(candidate))
    if max(v['max_error_mm']for v in measured.values())>.001:raise ValueError('Trial registration failed')
    if not note:raise ValueError('Actual final visual review required')
    fb=encode({**frame,'trial_registration':measured,'protection':edit['protection'],'visual_review':note});fr=reference(directory/'layout-evidence.json',fb)
    registry=json.loads(authenticated(draft['semantic_registry'],snapshot));relation={**edit['relationship'],'status':'verified','artifact_sha256':sha(candidate),'evidence':fr}
    registry.setdefault('panel_relationships',[]).append(relation);registry['statistics_trial_registration']={'relationship':relation,'row_aliases':LABELS,'evidence':fr,
        'precedence':'Restored shared trial coordinates supersede whole-statistical-table scaling in the preceding three-column draft.'}
    rb=encode(registry);draft['working_svg']={**draft['working_svg'],'sha256':sha(candidate)};draft['legend']={**draft['legend'],'sha256':sha(legend)}
    page=render([{'id':'fig2','svg_bytes':candidate,'legend_text':legend.decode(),'svg':draft['working_svg'],'legend':draft['legend'],'whole_figure_frozen':False}])[0]
    verify_review(candidate,page,legend,'fig2');draft['review']={**draft['review'],'sha256':sha(page)};draft['semantic_registry']=reference(directory/'semantic-registry.json',rb);draft['layout_frames']=fr
    pb=encode(measured_presentation(candidate,width_mm=read_spec(candidate)['final_size_mm'][0]));draft['effective_presentation']=reference(directory/'effective-presentation.json',pb)
    draft.setdefault('consistency_requests',[]).append(relation);draft.setdefault('verification',{})['statistics_trial_registration']=fr
    artifacts=[(directory/'before-state.json',oldstate),(directory/'before.svg',source),(directory/'before.html',oldhtml),(directory/'before-legend.md',oldlegend),
               (directory/'layout-evidence.json',fb),(directory/'semantic-registry.json',rb),(directory/'effective-presentation.json',pb)]
    for p,_ in artifacts:
        if p.exists():raise FileExistsError(p)
    snapshot.check()
    if not check_only:publish_state(sp,state,snapshot,artifacts=artifacts,evidence_updates=[(Path(draft['working_svg']['path']),candidate),(Path(draft['review']['path']),page),(Path(draft['legend']['path']),legend)])
    return {'status':'checked'if check_only else'published','figure_id':'fig2','svg_sha256':sha(candidate),'maximum_trial_registration_error_mm':max(v['max_error_mm']for v in measured.values())}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','publish']);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--original-statistics',type=Path);p.add_argument('--browser-report',type=Path);p.add_argument('--visual-review-note');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    result=prepare(a.directory,a.original_statistics)if a.command=='prepare'else publish(a.directory,a.browser_report,a.visual_review_note,a.check_only)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
