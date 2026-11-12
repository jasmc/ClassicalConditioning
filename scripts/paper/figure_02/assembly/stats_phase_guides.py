"""Audit saved G/H/I annotation trial positions and restore phase guides only."""
from pathlib import Path as _Path
import sys as _sys
ROOT=next(p for p in _Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
_sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse,base64,copy,json,re
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from classical_conditioning.figure_workspace import Snapshot,sha,tag,publish_state,measured_presentation
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figures.layout_frames import read_spec,embed_spec,assess,POLICY,COLLECTOR
from classical_conditioning.analysis.review.saved_statistics_audit import annotation_labels,star_text
from scripts.paper.figure_02.assembly.align_statistics import STATS,LABELS,find,text,anchor_mm,trial_map,registration
from scripts.paper.shared.correct_relationships import matrices
from scripts.paper.shared.collection import encode,reference,render
from scripts.paper.shared.reconcile_saved_draft import authenticated,verify_review
from scripts.paper.shared.verify_layout_frames import review_page

BOUNDARIES=(14.5,64.5)

def audit(raw,sources):
    root=ET.fromstring(raw);report={}
    for panel,refs in sources.items():
        frames={}
        for name,ref in refs.items():
            data=Path(ref['path']).read_bytes()
            if sha(data)!=ref['sha256']:raise ValueError('Saved statistics changed')
            frames[name]=pd.read_csv(ref['path'])
        d,m,t=(frames[n]for n in ('D-interaction-tests.csv','MR-local-block-tests.csv','phase-contrasts.csv'))
        labels=annotation_labels(raw,panel,d,m,t);a,b=trial_map(root,panel);checks=[]
        prefix={'G':'edit-G-fig2__g__stats__','H':'matched-H__H-','I':'trace10-I__I-'}[panel]
        def check(identity,trial,node=None):
            node=text(root,identity)if node is None else node
            actual=anchor_mm(root,node);checks.append({'id':identity,'trial':float(trial),'stats_x_mm':actual,'trajectory_x_mm':a*trial+b})
        for idx,row in d.iterrows():
            if star_text(row.p_holm):check(prefix+'D-'+(str(idx)if panel=='G'else row.block_10_name.replace(' ','-')),row.center)
        for idx,row in m.iterrows():
            for field,label in [('mean_p_fdr','M'),('slope_p_raw','R-raw'),('slope_p_fdr','R-FDR')]:
                if star_text(row[field]):check(prefix+(field+'-'+str(idx)if panel=='G'else label+'-'+row.block_10_name.replace(' ','-')),row.center)
        significant=sorted(labels['significant_trials'])
        if panel=='G':
            for trial in significant:check(prefix+'trial-'+str(trial),trial)
        else:
            nodes=sorted(find(root,prefix+'trial-stars').iter(tag('use')),key=lambda n:anchor_mm(root,n))
            for idx,(trial,node)in enumerate(zip(significant,nodes)):check(prefix+'trial-stars/use-'+str(idx),trial,node)
        error=max((abs(c['stats_x_mm']-c['trajectory_x_mm'])for c in checks),default=0.)
        if error>.02:raise ValueError(f'{panel} annotation trial mismatch: {error}')
        report[panel]={'labels':labels,'marker_checks':checks,'max_error_mm':error,'phase_headers':registration(root)[panel]}
    return {'panels':report,'saved_sources':sources,'no_model_refits':True,'scope':'Every named block-statistic glyph and every displayed per-trial significance marker; saved tables authenticated.'}

def xy(root,node,x=0,y=0):
    m=matrices(root)[node];u=float(root.get('width')[:-2])/float(root.get('viewBox').split()[2]);return ((m[0]*x+m[2]*y+m[4])*u,(m[1]*x+m[3]*y+m[5])*u)

def local(root,node,x,y):
    m=matrices(root)[node];u=float(root.get('width')[:-2])/float(root.get('viewBox').split()[2]);det=m[0]*m[3]-m[1]*m[2];dx=x/u-m[4];dy=y/u-m[5]
    return ((m[3]*dx-m[2]*dy)/det,(-m[1]*dx+m[0]*dy)/det)

def guide_id(panel,trial):return f'fig2-{panel}-stats-phase-boundary-{trial}'

def restore_guides(raw):
    root=ET.fromstring(raw);edits=[]
    style=find(root,'edit-G-node-22').get('style')
    # Existing G labels set the five common statistical row baselines.
    baselines=[xy(root,find(root,k),float(find(root,k).get('x','0')),float(find(root,k).get('y','0')))[1]for k in LABELS]
    top,bottom=min(baselines)-2,max(baselines)+1
    for panel in STATS:
        a,b=trial_map(root,panel);group=find(root,STATS[panel])
        for trial in BOUNDARIES:
            if panel=='G':
                owner=find(root,f'edit-G-fig2__g__stats__boundary-{trial}');node=next(owner.iter(tag('path')));edits.append({'id':node.get('id'),'before':dict(node.attrib),'added':False})
            else:
                node=ET.SubElement(group,tag('path'),{'id':guide_id(panel,trial),'style':style});edits.append({'id':node.get('id'),'added':True})
            x=a*trial+b;p=local(root,node,x,top);q=local(root,node,x,bottom)
            node.set('d',f'M {p[0]:.14g} {p[1]:.14g} L {q[0]:.14g} {q[1]:.14g}');node.attrib.pop('clip-path',None)
    return ET.tostring(root),edits,{'y_extent_mm':[top,bottom],'boundary_trials':BOUNDARIES,'style_source':'edit-G-node-22','style':style}

def protect(source,candidate,edits):
    a,b=ET.fromstring(source),ET.fromstring(candidate)
    for root in (a,b):root.remove(find(root,'layout-frames'))
    for edit in edits:
        node=find(b,edit['id'])
        if edit['added']:next(p for p in b.iter()if node in list(p)).remove(node)
        else:node.attrib.clear();node.attrib.update(edit['before'])
    if ET.tostring(a)!=ET.tostring(b):raise ValueError('Artwork outside declared phase-guide paths changed')
    return {'all_other_artwork_exactly_preserved':True,'scientific_data_and_statistical_marks_unchanged':True}

def guide_check(raw,edits,geometry):
    root=ET.fromstring(raw);checks=[]
    for panel in STATS:
        a,b=trial_map(root,panel)
        for trial in BOUNDARIES:
            node=next(find(root,f'edit-G-fig2__g__stats__boundary-{trial}').iter(tag('path')))if panel=='G'else find(root,guide_id(panel,trial))
            v=list(map(float,re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',node.get('d'))));points=[xy(root,node,*v[:2]),xy(root,node,*v[2:])]
            error=max(abs(p[0]-(a*trial+b))for p in points)
            if error>.001 or max(abs(p[1]-y)for p,y in zip(points,geometry['y_extent_mm']))>.001:raise ValueError('Phase guide registration failed')
            checks.append({'panel':panel,'trial':trial,'id':node.get('id'),'endpoints_mm':points,'error_mm':error})
    return checks

def prepare(directory):
    ticket_path=directory/'starting-ticket.json';ticket=json.loads(ticket_path.read_bytes());snap=Snapshot([ticket_path,*command_sources(__file__),POLICY,COLLECTOR]);snap.bind_hashes(ticket['starting_hashes']);snap.check()
    sp=Path(ticket['state_path']);state=json.loads(sp.read_bytes());d=state['draft'];source=authenticated(d['working_svg'],snap);legend=authenticated(d['legend'],snap);verify_review(source,authenticated(d['review'],snap),legend,'fig2')
    before=audit(source,ticket['statistical_sources']);candidate,edits,geometry=restore_guides(source)
    spec=read_spec(source);relation={'id':'fig2-statistical-trial-markers-and-phase-guides-20261029','type':'layout','panels':['G','H','I'],'properties':['all saved significance marks mapped to observed global-CS-trial coordinates','Pre/Train and Train/Test guides at 14.5 and 64.5 across all statistical rows'],'source_definition':'Hash-bound saved D/MR/phase CSVs, observed ticks 5/94 and recorded trial phases','status':'requires fresh browser verification','tolerance_mm':.02};spec['panel_relationships'].append(relation);candidate=embed_spec(candidate,spec)
    protected=protect(source,candidate,edits);result=audit(candidate,ticket['statistical_sources']);guides=guide_check(candidate,edits,geometry)
    page=render([{'id':'fig2','svg_bytes':candidate,'legend_text':legend.decode(),'svg':reference(directory/'candidate.svg',candidate),'legend':d['legend'],'whole_figure_frozen':False}])[0]
    evidence={'source_state':reference(sp),'source_hashes':snap.hashes,'candidate_sha256':sha(candidate),'review_sha256':sha(page),'edits':edits,'guide_geometry':geometry,'guide_checks':guides,'alignment_audit':result,'protection':protected,'relationship':relation,'scientific_selection_changed':False,'freeze_created':False}
    files={'candidate.svg':candidate,'review.html':page,'edit-evidence.json':encode(evidence),'alignment-audit-before.json':encode(before),'layout-review.html':review_page([{'name':'candidate.svg','svg_base64':base64.b64encode(candidate).decode()}]).encode()}
    snap.check()
    for name,raw in files.items():
        with (directory/name).open('xb')as f:f.write(raw)
    return {p:{'markers':len(v['marker_checks']),'max_error_mm':v['max_error_mm']}for p,v in result['panels'].items()}

def publish(directory,note,check_only=False):
    ep=directory/'edit-evidence.json';edit=json.loads(ep.read_bytes());bp=directory/'browser.json';snap=Snapshot([ep,bp,directory/'candidate.svg',directory/'review.html',*command_sources(__file__)]);snap.bind_hashes(edit['source_hashes']);snap.check();sp=Path(edit['source_state']['path']);oldstate=authenticated(edit['source_state'],snap);state=json.loads(oldstate);d=state['draft'];source=authenticated(d['working_svg'],snap);oldhtml=authenticated(d['review'],snap);legend=authenticated(d['legend'],snap);candidate=(directory/'candidate.svg').read_bytes()
    if sha(candidate)!=edit['candidate_sha256']:raise ValueError('Candidate changed')
    protect(source,candidate,edit['edits']);audit(candidate,edit['alignment_audit']['saved_sources']);guide_check(candidate,edit['edits'],edit['guide_geometry'])
    browser=json.loads(bp.read_bytes())['candidate.svg'];frame=assess(candidate,browser)
    if frame['status']!='passed':raise ValueError(frame['issues'])
    if not note:raise ValueError('Final-scale visual review required')
    fb=encode({**frame,'alignment_audit':edit['alignment_audit'],'guide_checks':edit['guide_checks'],'protection':edit['protection'],'visual_review':note});fr=reference(directory/'layout-evidence.json',fb)
    registry=json.loads(authenticated(d['semantic_registry'],snap));relation={**edit['relationship'],'status':'verified','artifact_sha256':sha(candidate),'evidence':fr};registry.setdefault('panel_relationships',[]).append(relation);registry['statistics_phase_guide_and_marker_audit']={'relationship':relation,'evidence':fr,'source_data':edit['alignment_audit']['saved_sources']};rb=encode(registry)
    d['working_svg']={**d['working_svg'],'sha256':sha(candidate)};page=render([{'id':'fig2','svg_bytes':candidate,'legend_text':legend.decode(),'svg':d['working_svg'],'legend':d['legend'],'whole_figure_frozen':False}])[0];verify_review(candidate,page,legend,'fig2');d['review']={**d['review'],'sha256':sha(page)};d['semantic_registry']=reference(directory/'semantic-registry.json',rb);d['layout_frames']=fr;pb=encode(measured_presentation(candidate,width_mm=159));d['effective_presentation']=reference(directory/'effective-presentation.json',pb);d.setdefault('verification',{})['statistics_phase_guides_and_marker_positions']=fr;d.setdefault('consistency_requests',[]).append(relation)
    artifacts=[(directory/'before-state.json',oldstate),(directory/'before.svg',source),(directory/'before.html',oldhtml),(directory/'before-legend.md',legend),(directory/'layout-evidence.json',fb),(directory/'semantic-registry.json',rb),(directory/'effective-presentation.json',pb)]
    snap.check()
    if not check_only:publish_state(sp,state,snap,artifacts=artifacts,evidence_updates=[(Path(d['working_svg']['path']),candidate),(Path(d['review']['path']),page)])
    return {'status':'checked'if check_only else'published','sha256':sha(candidate),'guides':len(edit['guide_checks'])}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','publish']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--visual-review-note');p.add_argument('--check-only',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.directory)if a.command=='prepare'else publish(a.directory,a.visual_review_note,a.check_only),indent=2))
if __name__=='__main__':main()
