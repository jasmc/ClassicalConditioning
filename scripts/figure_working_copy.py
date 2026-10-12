"""Prepare/verify current frozen and draft editable SVG copies without freezing.

Writes standalone SVGs only under an explicit external directory. Source review,
frozen selection/registry hashes and state are checked again before publication.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import (Snapshot,editable_svg,file_hash,measured_presentation,publish_state,sha,StaleFigureState,draft_axis_findings,bind_loaded_state)

REVIEW_ROOT=ROOT/'docs/analysis/figures/reviews'
STATE=REVIEW_ROOT/'figure-01/current-state.json'
SELECTIONS=[ROOT/'configs/paper-figures/selections'/p for p in (
    'figure1-freeze.json','figure1-panel-e-heatmap-rows-freeze-20261009.json','figure1-fgh-version12-freeze-20261009.json')]

def verify_state(path=STATE,expected_bytes=None):
    raw=path.read_bytes() if expected_bytes is None else expected_bytes
    state=json.loads(raw)
    records=state['scientific_selection']['records']+[state['draft']['review']]+list(state.get('preserved_figures',{}).values())
    for key in ('frozen_display','draft'):
        if state[key].get('working_svg'):records.append(state[key]['working_svg'])
        if state[key].get('effective_presentation'):records.append(state[key]['effective_presentation'])
    for item in records:
        if file_hash(item['path'])!=item['sha256']:raise StaleFigureState('Bound current artifact changed/unavailable: '+item['path'])
    if file_hash(path)!=sha(raw):raise StaleFigureState('Current state advanced while being read')
    return state

def begin(ticket):
    """Record candidate/source hashes before an editing turn, not after it."""
    raw=STATE.read_bytes();state=verify_state(expected_bytes=raw)
    paths=[STATE,*[Path(i['path']) for i in state['scientific_selection']['records']],
           Path(state['draft']['review']['path']),*[Path(state[k]['working_svg']['path']) for k in ('frozen_display','draft')]]
    snapshot=Snapshot(paths)
    bind_loaded_state(snapshot,state,raw,STATE)
    value=dict(schema_version=1,figure_id='fig1',state_path=str(STATE),starting_hashes=snapshot.hashes,
               created_at=datetime.now(ZoneInfo('Europe/Madrid')).isoformat())
    with ticket.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return value

def registry_sources():
    # Exact current selections, not whichever manifest filename sorts last.
    paths={'E':ROOT/'records/frozen-analyses/figure-01/panel-e/2026-10-09-heatmap-rows/manifests/candidate.json',
           'FGH':ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-09-version-12/manifests/candidate.json'}
    index=json.loads((ROOT/'records/frozen-analyses/index.json').read_text())
    known={str(ROOT/i['path']):i['sha256'] for i in index['files']}
    registries={}
    for pid,path in paths.items():
        if file_hash(path)!=known[str(path)]:raise ValueError('Frozen registry hash mismatch: '+str(path))
        registries[pid]=json.loads(path.read_text())['element_registry']
    return registries,list(paths.values())+[ROOT/'records/frozen-analyses/index.json']

def prepare(output_dir,revision,refresh=False):
    if not output_dir.is_dir():raise FileNotFoundError('Use an existing external destination: '+str(output_dir))
    if output_dir.resolve().is_relative_to(ROOT):raise ValueError('Permanent SVGs belong on external storage')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*',revision):raise ValueError('Revision must be lowercase letters/digits/hyphens')
    state_bytes=STATE.read_bytes() if STATE.exists() else None
    if state_bytes is not None:
        state=verify_state(expected_bytes=state_bytes)
        if not refresh and all(state[key].get('working_svg') for key in ('frozen_display','draft')):
            return state  # Authenticated reusable copies; never discard manual changes.
    legacy=REVIEW_ROOT/'legend-review-verification.json'
    legacy_bytes=legacy.read_bytes()
    verification=json.loads(legacy_bytes) if state_bytes is None else dict(review_path=state['draft']['review']['path'],review_sha256=state['draft']['review']['sha256'])
    review_path=Path(verification['review_path'])
    registries,registry_paths=registry_sources()
    spec=importlib.util.spec_from_file_location('legend_review',ROOT/'scripts/build_figure_legend_review.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Resolve authenticated source container before rendering, then snapshot it.
    from classical_conditioning.external_artifacts import resolve_artifact
    container=resolve_artifact('reviews/latest_figure_assemblies_20261010.html')
    dependencies=[STATE,legacy,review_path,container,*SELECTIONS,*registry_paths,
                  ROOT/'scripts/build_figure_legend_review.py',Path(__file__),ROOT/'src/classical_conditioning/figure_workspace.py']
    if STATE.exists():dependencies.extend(Path(state[k]['working_svg']['path']) for k in ('frozen_display','draft'))
    snapshot=Snapshot(dependencies)
    bind_loaded_state(snapshot,state if state_bytes is not None else None,state_bytes,STATE)
    snapshot.hashes[str(legacy.resolve())]=sha(legacy_bytes)
    snapshot.check()
    if file_hash(review_path)!=verification['review_sha256']:raise ValueError('Current review hash mismatch')
    sources,_=module.load_sources();source=sources[0]
    text=review_path.read_text();blobs=re.findall(r'<img class="figure"[^>]*src="data:image/svg\+xml;base64,([^"]+)"',text)
    if len(blobs)!=4:raise ValueError('Expected original/candidate Figures 1 and 2')
    original=base64.b64decode(blobs[1],validate=True);draft=base64.b64decode(blobs[0],validate=True)
    if original!=source['bytes']:raise ValueError('Review original does not match current scoped frozen sources')
    records={pid:info['provenance'] for pid,info in source['panels'].items()}
    frozen,frozen_map=editable_svg(original,kind='Figure 1: assembly of current frozen panels; editable copy, whole assembly not frozen',source_records=records,registries=registries)
    # Draft scientific labels changed during decluttering: do not blindly copy
    # the frozen registry as if it fully classified/reviewed this candidate.
    candidate,draft_map=editable_svg(draft,kind='Figure 1: current unfrozen presentation draft',source_records=records)
    artifacts=[];output_records={}
    for key,data in [('frozen',frozen),('draft',candidate)]:
        path=output_dir/f'figure-01-{key}-working-{revision}.svg'
        artifacts.append((path,data));output_records[key]=dict(path=str(path),sha256=sha(data),source_svg_sha256=sha(original if key=='frozen' else draft))
    frozen_measures=measured_presentation(frozen);draft_measures=measured_presentation(candidate)
    state=dict(schema_version=1,figure_id='fig1',updated_at=datetime.now(ZoneInfo('Europe/Madrid')).isoformat(),
               default_display='frozen_display',active_edit_scope=['Figure 1'],
               scientific_selection=dict(authority='scoped accepted freezes; this state file does not select new science',whole_figure_frozen=False,
                    records=[dict(path=str(p),sha256=file_hash(p)) for p in SELECTIONS]),
               frozen_display=dict(status='assembly-of-current-frozen-panels',working_svg=output_records['frozen'],
                    structural_elements=len(frozen_map['structural_inventory']),mapped_scientific_elements=len(frozen_map['scientific_registry']),
                    registry_limitations=frozen_map['semantic_limitations'],source_assembly_sha256=sha(original)),
               draft=dict(status='unfrozen-presentation-candidate',review=dict(path=str(review_path),sha256=file_hash(review_path)),
                    working_svg=output_records['draft'],source_assembly_sha256=sha(draft),
                    author_directions_record=str(legacy),preserved_other_figure='Figure 2 original/candidate unchanged'),
               measurement_policy='Version-bound measured reports are linked separately from frozen_display and draft; not freeze approval',
               outstanding_questions=['D uses Pre-Train while E uses Habituation for trial 9',
                    'G arrowheads are the 3 s Trace example; intended relationship to D/E Delay trials needs author review'],
               verification=dict(current_sources_authenticated=True,editable_source_structure=True,
                    visual_review='pending native-SVG/source rendering comparison',concurrency='starting file hashes checked under exclusive state publication lock'),
               reproduction=dict(command=f'.venv/bin/python scripts/figure_working_copy.py prepare --output-dir "{output_dir}" --revision {revision}',
                    generator_sha256=file_hash(__file__),inputs=snapshot.hashes))
    # Evidence reports are generated from actual final SVGs; no nominal property
    # declarations are promoted to a passed check.
    audit=dict(frozen=frozen_measures,draft=draft_measures,draft_axis_findings=draft_axis_findings(draft_measures),
               note='Use coordinate matrices/physical measurements for review. Glyph-edge padding, clipping and scientific role completeness require explicit review. Nonuniform strokes are ranges unless a straight-line normal can be measured.')
    audit_path=REVIEW_ROOT/f'figure-01/effective-presentation-audit-{revision}.json'
    snapshot.hashes[str(audit_path.resolve())]=file_hash(audit_path)
    # Publish report as part of the locked state operation without overwriting an
    # earlier report silently. This initial preparation creates it exactly once.
    audit_bytes=(json.dumps(audit,indent=2)+'\n').encode()
    artifacts.append((audit_path,audit_bytes))
    for key in ('frozen_display','draft'):state[key]['effective_presentation']=dict(path=str(audit_path),sha256=sha(audit_bytes))
    publish_state(STATE,state,snapshot,artifacts)
    return state

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--revision',required=True);p.add_argument('--refresh',action='store_true',help='Create new versioned copies; preserve existing authenticated copies')
    sub.add_parser('verify');p=sub.add_parser('begin');p.add_argument('--ticket',type=Path,required=True);args=parser.parse_args()
    if args.command=='begin':print(json.dumps(begin(args.ticket),indent=2));return
    result=prepare(args.output_dir.resolve(),args.revision,args.refresh) if args.command=='prepare' else verify_state()
    print(json.dumps({key:result[key] for key in ('frozen_display','draft')},indent=2))

if __name__=='__main__':main()
