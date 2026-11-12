"""Presentation-only publication, independent of numerical reconstruction."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import (
    DRAWABLE, Snapshot, sha, publish_state, measured_presentation, native_presentation_width,
)
from classical_conditioning.figures.review_html import svg_editing_page


def verify_declared_changes(before, after, changes):
    """Reject every XML change not explicitly declared by stable element identity."""
    old, new = ET.fromstring(before), ET.fromstring(after)
    declarations = {r['element_id']: r for r in changes}
    if len(declarations) != len(changes): raise ValueError('Duplicate change declarations')
    seen = set()
    def compare(a, b):
        if a.tag != b.tag or len(a) != len(b) or a.tail != b.tail:
            raise ValueError('Unrequested SVG structure change')
        identity = a.get('id'); rule = declarations.get(identity, {})
        if identity and identity in seen: raise ValueError('Ambiguous SVG identity: '+identity)
        if identity: seen.add(identity)
        permitted = set(rule.get('attributes', []))
        # Scientific payload/reference attributes are never a presentation shortcut.
        if permitted - {'x','y','transform','font-size','font-weight','font-family','text-anchor'}:
            raise ValueError('Unsupported presentation attribute')
        for key in a.attrib.keys() | b.attrib.keys():
            if a.get(key) != b.get(key) and key not in permitted:
                raise ValueError('Undeclared SVG attribute change: '+str(identity)+' '+key)
            if a.get(key) != b.get(key) and key in {'x', 'y', 'transform'}:
                if any(n.tag.rsplit('}', 1)[-1] in DRAWABLE - {'text'} for n in a.iter()):
                    raise ValueError('Data-bearing coordinate changes require a mapping-aware layout route: '+str(identity))
        if a.text != b.text and not (rule.get('text') and a.tag.endswith(('}text','}tspan'))):
            raise ValueError('Undeclared SVG text/data change: '+str(identity))
        for x,y in zip(a,b): compare(x,y)
    compare(old,new)
    if set(declarations)-seen: raise ValueError('Unknown declared SVG identity')


def publish_presentation(state_path, baseline, *, candidate_svg=None, candidate_legend=None,
                         changes=(), visual_review=None, layout_evidence=None):
    from scripts.paper.shared.working_copy import display_target
    from classical_conditioning.figures.layout_frames import assess
    path=Path(state_path).resolve();raw=path.read_bytes();state=json.loads(raw)
    ticket=json.loads(Path(baseline).read_bytes())
    if Path(ticket['state_path']).resolve() != path: raise ValueError('Ticket belongs to another state')
    snapshot=Snapshot([]);snapshot.hashes=ticket['starting_hashes'];snapshot.check()
    if snapshot.hashes.get(str(path)) != sha(raw): raise ValueError('Stale state ticket')
    target=display_target(path);scope=state[state.get('default_display','draft')] if 'draft' in state else state
    old_svg=Path(target['svg']['path']).read_bytes();old_legend=Path(target['legend']['path']).read_bytes()
    inputs=[Path(p) for p in (candidate_svg,candidate_legend,layout_evidence) if p]
    from classical_conditioning.command_provenance import command_sources
    snapshot.include([*inputs,*command_sources(__file__)])
    svg=Path(candidate_svg).read_bytes() if candidate_svg else old_svg
    legend=Path(candidate_legend).read_bytes() if candidate_legend else old_legend
    for candidate,data in [(candidate_svg,svg),(candidate_legend,legend)]:
        if candidate and sha(data)!=snapshot.hashes[str(Path(candidate).resolve())]:
            raise ValueError('Candidate changed while being read')
    for key, data in [('svg',old_svg),('legend',old_legend)]:
        if sha(data)!=target[key]['sha256']:raise ValueError('Selected artwork advanced')
    from classical_conditioning.figures.preview import read_registry
    for data,directory in [(old_svg,Path(target['svg']['path']).parent),(svg,Path(candidate_svg).resolve().parent if candidate_svg else Path(target['svg']['path']).parent)]:
        registry=read_registry(data,directory)
        if registry.get('registry_ref'):
            ref=registry['registry_ref'];sidecar=(directory/ref['path']).resolve()
            snapshot.include([sidecar])
            if snapshot.hashes[str(sidecar)]!=ref['sha256']:raise ValueError('Registry advanced while being read')
    if svg != old_svg:
        verify_declared_changes(old_svg,svg,changes)
        if not visual_review or not layout_evidence:
            raise ValueError('SVG changes require actual visual review and new layout evidence')
        evidence_raw=Path(layout_evidence).read_bytes()
        if sha(evidence_raw)!=snapshot.hashes[str(Path(layout_evidence).resolve())]:
            raise ValueError('Layout evidence changed while being read')
        evidence=json.loads(evidence_raw)
        assessment=assess(svg,evidence.get('browser'))
        if assessment['status']!='passed' or evidence.get('artifact_sha256') != sha(svg):
            raise ValueError('Candidate layout evidence does not verify this SVG')
        for key in ('spec_sha256','policy_sha256','collector_sha256','helper_sha256'):
            if evidence.get(key)!=assessment.get(key):raise ValueError('Stale candidate layout evidence')
        scope['layout_frames']={'path':str(Path(layout_evidence).resolve()),'sha256':sha(evidence_raw)}
        report = json.dumps(measured_presentation(svg, width_mm=native_presentation_width(svg)), indent=2).encode()
    parent=Path(target['svg']['path']).parent
    history=parent/'presentation-history'/sha(raw)
    if history.exists():raise FileExistsError(history)
    history.mkdir(parents=True)
    artifacts=[(history/'state.json',raw),(history/'selected.svg',old_svg),
               (history/'selected.html',Path(target['html']['path']).read_bytes()),(history/'legend.md',old_legend)]
    html=svg_editing_page(svg,scope_label=state.get('figure_id',state.get('scope','Figure')),legend=legend.decode()).encode()
    updates=[(Path(target['html']['path']),html),(Path(target['svg']['path']),svg),(Path(target['legend']['path']),legend)]
    for key,data in [('review',html),('working_svg',svg),('legend',legend)]:
        scope[key]={**scope[key],'path':target[{'review':'html','working_svg':'svg','legend':'legend'}[key]]['path'],'sha256':sha(data)}
    if svg!=old_svg:
        report_path=history/'effective-presentation.json'
        artifacts.append((report_path,report));scope['effective_presentation']={'path':str(report_path),'sha256':sha(report)}
    scope.setdefault('verification',{})['presentation_edit']={'svg_sha256':sha(svg),'legend_sha256':sha(legend),
        'visual_review':visual_review if svg!=old_svg else 'SVG bytes unchanged; legend-only publication',
        'declared_changes':changes,'scientific_reconstruction':False}
    try:
        publish_state(path,state,snapshot,artifacts=artifacts,evidence_updates=updates)
    except BaseException:
        if history.exists() and not any(history.iterdir()):history.rmdir()
        raise
    return {'state':str(path),'scientific_reconstruction':False,'history':str(history)}
