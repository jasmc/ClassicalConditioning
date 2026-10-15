"""Archive author-disposed single-fish reviews using the verified transfer tools."""
from pathlib import Path
import argparse, json, os, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import repository_archive as archive
from classical_conditioning.external_artifacts import sha256
from classical_conditioning.figure_workspace import Snapshot, state_lock, publish_state

CONTROL=ROOT/'records/repository-archive/singlefish-heatmaps-20261013'
CODE_CONTROL=ROOT/'records/repository-archive/singlefish-heatmap-code-20261013'
SOURCE=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews')
DEST=Path('/Volumes/JOAQUIM/ClassicalConditioning-Archive/figure-01/singlefish-heatmaps/2026-10-13-superseded')
CODE_DEST=DEST.parent/'2026-10-13-candidate-code'
CODE=['build_figure1_fgh_palette_review.py','organize_figure1_fgh_comparison.py','show_figure1_fgh_version04.py','render_figure1_version04_trace_palettes.py','render_figure1_one_second_cyan_palettes.py','render_figure1_blue_orange_transition.py','prepare_figure1_fgh_one_second_freeze.py']


def manifest(paths,root,destination):
    return {'schema_version':1,'date':'2026-10-13','scope':'Previous Figure 1 single-fish F/G/H heatmap versions only','author_authorization':'archive the previous versions of these singlefish heatmaps','source_root':str(root),'archive_root':str(destination),'files':[{'path':str(p.relative_to(root)),'historical_paths':[str(p)],'sha256':sha256(p),'bytes':p.stat().st_size,'action':'archive_to_ssd','reason':'Superseded single-fish heatmap review, input cache, freeze dependency or candidate code'} for p in paths]}


def inventory():
    archive.validate_volume(DEST)
    if CONTROL.exists() or CODE_CONTROL.exists():raise ValueError('Existing control inventory; inspect/resume rather than overwrite')
    state=json.loads((ROOT/'docs/paper/figures/figure-01/current-state.json').read_text())
    selected_dir=Path(state['scoped_fgh_freeze']['svg']['path']).parent
    paths=[]
    for p in sorted(SOURCE.iterdir()):
        if p.is_file() and any(p.name.startswith(prefix) for prefix in ['figure-01-fgh-all-versions-','figure-01-fgh-magma-','figure-01-fgh-v12-','figure-01-fgh-version04-','figure-01-best-learners-version08-','figure-01-version04-trace-','figure-01-one-second-','figure-01-historical-fish-one-second-']):paths.append(p)
        if p.is_dir() and p.name in ['figure-01-best-learners-version08-2026-10-13-inputs','figure-01-fgh-one-second-managua-freeze-2026-10-13']:
            assert p!=selected_dir;paths.extend(sorted(x for x in p.rglob('*') if x.is_file()))
    protected={Path(state['draft'][key]['path']) for key in ['review','working_svg','legend','effective_presentation']}
    protected.update(Path(v['path']) for v in state['draft'].get('panel_views',{}).values() if 'path' in v)
    assert not protected.intersection(paths)
    assert all(not p.is_symlink() and not p.is_relative_to(selected_dir) for p in paths)
    CONTROL.mkdir();CODE_CONTROL.mkdir()
    archive.write_json(CONTROL/'transfer-inventory.json',manifest(paths,SOURCE,DEST))
    archive.write_json(CODE_CONTROL/'transfer-inventory.json',manifest([ROOT/'scripts'/name for name in CODE],ROOT,CODE_DEST))
    print(json.dumps({'phase':'inventoried','media_files':len(paths),'media_bytes':sum(p.stat().st_size for p in paths),'candidate_code_files':len(CODE),'protected_freeze':str(selected_dir)}),flush=True)


def transfer():
    for control,root,destination in [(CONTROL,SOURCE,DEST),(CODE_CONTROL,ROOT,CODE_DEST)]:
        m=json.loads((control/'transfer-inventory.json').read_text())
        if destination.exists():count=archive.verify_payload(m,destination)
        else:count=archive.transfer(m,root,destination)
        archive.write_json(control/'copy-verification.json',{'files':count,'archive_root':str(destination),'payload_hashes_verified':True})
        print(json.dumps({'phase':'copied-and-verified','files':count,'destination':str(destination)}),flush=True)


def finalize():
    locations=ROOT/'records/repository-archive/verified-artifact-locations.json'
    relocations=ROOT/'records/repository-archive/relocations.json'
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json'
    manifests=[json.loads((c/'transfer-inventory.json').read_text()) for c in [CONTROL,CODE_CONTROL]]
    for m,dest in zip(manifests,[DEST,CODE_DEST]):archive.verify_payload(m,dest)
    snapshot=Snapshot([state_path,locations,relocations])
    state=json.loads(state_path.read_text());selected_dir=Path(state['scoped_fgh_freeze']['svg']['path']).parent
    assert all(not (SOURCE/r['path']).is_relative_to(selected_dir) for r in manifests[0]['files'])
    loc=json.loads(locations.read_text());reloc=json.loads(relocations.read_text())
    for m,dest in zip(manifests,[DEST,CODE_DEST]):
        for row in m['files']:
            references=row['historical_paths']+[row['path']]
            loc['entries'].append({'references':references,'sha256':row['sha256'],'archive_root':str(dest),'relative_path':'payload/'+row['path'],'recovery_method':'Author-authorized single-fish archive; verified exact bytes before prune'})
    # Superseded local freeze records retain their immutable bytes under history.
    old=ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-13-one-second-managua'
    history=old.parent/'history'/old.name
    old_pointer=ROOT/'configs/paper-figures/selections/figure1-fgh-one-second-managua-freeze-20261013.json'
    pointer_history=old_pointer.parent/'history'/old_pointer.name
    moved=[]
    with state_lock(state_path):
        snapshot.check()
        for source,target in [(old,history),(old_pointer,pointer_history)]:
            if source.exists():
                if target.exists():raise ValueError('Historical record destination collision')
                if source.is_dir():
                    for p in source.rglob('*'):
                        if p.is_file():reloc['repository_moves'][str(p.relative_to(ROOT))]=str((target/p.relative_to(source)).relative_to(ROOT))
                else:reloc['repository_moves'][str(source.relative_to(ROOT))]=str(target.relative_to(ROOT))
                target.parent.mkdir(parents=True,exist_ok=True);os.rename(source,target);moved.append((source,target))
        try:
            archive.write_json(locations,loc);archive.write_json(relocations,reloc)
        except BaseException:
            for source,target in reversed(moved):os.rename(target,source)
            raise
    print(json.dumps({'phase':'recovery-aliases-registered','aliases':sum(len(m['files']) for m in manifests),'historical_record_moves':len(moved)}),flush=True)
    # Each prune independently verifies every archive and original byte again.
    for m,root,dest,control in zip(manifests,[SOURCE,ROOT],[DEST,CODE_DEST],[CONTROL,CODE_CONTROL]):
        result=archive.prune(m,root,dest);archive.write_json(control/'completed-transfer.json',result)
        print(json.dumps({'phase':'originals-pruned-after-verification','files':len(result['removed_paths']),'destination':str(dest)}),flush=True)
    for name in ['figure-01-best-learners-version08-2026-10-13-inputs','figure-01-fgh-one-second-managua-freeze-2026-10-13']:
        folder=SOURCE/name
        if folder.exists():
            for p in sorted((p for p in folder.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
            folder.rmdir()
    update_navigation(history,pointer_history)


def update_navigation(history,pointer_history):
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json';state=json.loads(state_path.read_text())
    docs=[ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/README.md',ROOT/'records/frozen-analyses/selections.md']
    snapshot=Snapshot([state_path,*docs]);updates=[]
    for p in docs:
        text=p.read_text().replace('(2026-10-13-one-second-managua/README.md)','(history/2026-10-13-one-second-managua/README.md)')
        text+='\nEarlier single-fish heatmap reviews, input cache and candidate code were archived on 2026-10-13. [Archive and recovery record]('+str(CONTROL/'README.md')+'). Current Figure 1 and its selected F/G/H freeze remain active.\n'
        updates.append((p,text.encode()))
    readme=CONTROL/'README.md';index=DEST/'index.html'
    entries=json.loads((CONTROL/'transfer-inventory.json').read_text())['files']
    links=''.join('<li><a href="payload/'+r['path']+'">'+r['path']+'</a></li>' for r in entries if Path(r['path']).suffix=='.html')
    archive_page=('<!doctype html><meta charset="utf-8"><title>Archived single-fish heatmaps</title><h1>Previous Figure 1 single-fish heatmaps</h1><p>Archived 2026-10-13. Current selected panels remain in Figure 1.</p><ul>'+links+'</ul>').encode()
    note=('# Single-fish heatmap archive · 2026-10-13\n\nAuthor authorization: archive previous single-fish heatmap versions.\n\n- Reviews, numerical files, old input cache and superseded Trace 20230519_11 freeze: '+str(DEST)+'\n- Superseded candidate code: '+str(CODE_DEST)+'\n- Older V1–V12 recipes were already archived in the 2026-10-10 repository archive and remain unchanged.\n- Current historical-fish freeze (Trace 20230307_12), working Figure 1, current panel view and shared supported code remain active.\n- Exact SHA-256 recovery aliases are registered; immutable old records and manifests were not rewritten. The preceding accepted freeze now lives under its scoped history folder.\n\n[Transfer inventory](transfer-inventory.json) · [Verified pruning record](completed-transfer.json)\n').encode()
    state['previous_singlefish_heatmaps_archive']={'review_archive':str(DEST),'code_archive':str(CODE_DEST),'record':str(readme),'date':'2026-10-13','status':'verified and archived; original historical paths registered for exact-byte recovery'}
    publish_state(state_path,state,snapshot,artifacts=[(readme,note),(index,archive_page)],evidence_updates=updates)
    print(json.dumps({'phase':'complete','archive_index':str(index),'selected_fish':state['scoped_fgh_freeze']['fish']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['inventory','transfer','finalize']);args=parser.parse_args();globals()[args.phase]()
