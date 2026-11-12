"""Append explicitly retained bundles without scanning unrelated archive trees."""
from pathlib import Path
import copy
import json
import re
import tempfile
from contextlib import ExitStack
from scripts.reviews.versions import Catalogue, figure_page
from classical_conditioning.figure_workspace import Snapshot, sha, publish_state, state_lock
from classical_conditioning.external_artifacts import relocated_external_directory, resolve_artifact


def build_changed(output_root, archive_root, repo, changed, scopes):
    outputs=relocated_external_directory(output_root).resolve()
    archive=relocated_external_directory(archive_root).resolve();repo=Path(repo).resolve()
    control=outputs/'review-library/catalogue-publication.json'
    if not control.is_file():raise ValueError('Incremental update requires an explicit full build first')
    raw=control.read_bytes();prior=json.loads(raw);publication=copy.deepcopy(prior)
    paths=[control,*(outputs/r['path'] for r in prior['generated_pages'])]
    snapshot=Snapshot(paths)
    if snapshot.hashes[str(control.resolve())]!=sha(raw):raise ValueError('Publication advanced')
    for r in prior['generated_pages']:
        if snapshot.hashes[str((outputs/r['path']).resolve())]!=r['sha256']:
            raise ValueError('Generated navigation has unrecorded edits')
    model_path=outputs/prior['versions']['path'];model=json.loads(model_path.read_bytes())
    known={r['path']:r for r in model['inventory'] if r.get('path')}
    sources=[]; advanced=set()
    for value in changed:
        p=Path(value).resolve()
        if not p.is_relative_to(outputs) or p.is_relative_to(outputs/'review-library'):
            raise ValueError('Changed bundles must be retained under the output root')
        if p.is_dir() and ('versions' not in p.relative_to(outputs).parts or p.name=='versions'):
            raise ValueError('Directory updates require one explicit retained version bundle; use file arguments otherwise')
        files=sorted(p.rglob('*')) if p.is_dir() else [p]
        for f in files:
            if f.is_symlink():raise ValueError('Source symlink requires review')
            if not f.is_file():continue
            if str(f) in known:
                if known[str(f)].get('sha256') != sha(f.read_bytes()):
                    expected=known[str(f)].get('sha256')
                    if not expected:raise ValueError('Registered source lacks a recovery identity; run full reconciliation')
                    # Only an authenticated retained original can support a new byte version.
                    original=resolve_artifact(f,expected_sha256=expected,repo=repo)
                    if sha(original.read_bytes())!=expected:raise ValueError('Historical source recovery failed')
                    snapshot.include([original]);advanced.add(str(f))
                else:continue
            sources.append(('outputs',f))
    if not sources:return {'status':'unchanged','coverage':'no global scan'}
    with tempfile.TemporaryDirectory(dir=outputs/'review-library',prefix='.incremental-') as tmp:
        discovery=Catalogue(outputs,archive,repo,Path(tmp),load_history=False)
        allowed={int(re.search(r'figure-0*(\d+)',s)[1]) for s in scopes if re.search(r'figure-0*(\d+)',s)}
        state_paths=[p for n in allowed for p in (repo/f'docs/paper/figures/figure-{n:02}').rglob('*state.json')]
        delta=discovery.discover(source_list=sources,include_states=True,state_paths=state_paths,reconcile=False)
        for entry in model['versions']:
            if entry['source']['path'] in advanced and entry.get('selected'):
                entry['selected']=False;entry['status']='Retained historical byte version'
        existing={r['id'] for r in model['versions']}
        additions=[r for r in delta['versions'] if r['id'] not in existing]
        figures={n for r in additions for n in r['figures']}
        allowed={int(re.search(r'figure-0*(\d+)',s)[1]) for s in scopes if re.search(r'figure-0*(\d+)',s)}
        if not allowed or figures-allowed:raise ValueError('Declare every affected figure using --scope figure-NN')
        model['versions'].extend(additions);model['inventory'].extend(delta['inventory']);model['issues'].extend(delta['issues'])
        model['incremental_coverage']={'changed':[str(p) for _,p in sources], 'scopes':list(scopes),
            'last_full_snapshot':prior.get('last_full_snapshot',prior['snapshot']),'global_scan_performed':False}
        generated={};artifacts=[]
        for item in discovery.payloads.values():
            target=outputs/'review-library'/item['path'];payload=Path(item['source']).read_bytes()
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target.read_bytes())!=sha(payload):raise ValueError('Payload collision')
            else:artifacts.append((target,payload))
            publication['payloads'].append({'path':target.relative_to(outputs).as_posix(),'sha256':sha(payload),'bytes':len(payload)})
        for n in figures:
            target=outputs/f'figures/figure-{n:02}/INDEX.html';old=target.read_text()
            match=re.search(r'(<h2>Studies and reading</h2>.*?)<details><summary>All retained files',old,re.S)
            generated[target]=figure_page(n,model,outputs,reading_html=match[1] if match else '',figure_links=[1,2,3,4])
        generated[model_path]=(json.dumps(model,indent=2,ensure_ascii=False)+'\n').encode()
        home=outputs/'START-HERE.html'
        generated[home]=re.sub(r'\d+ source-bound artwork views; \d+ recorded coverage exceptions',
            f"{len(model['versions'])} source-bound artwork views; {len(model['issues'])} recorded coverage exceptions",home.read_text()).encode()
        # Keep exact previous navigation bytes reachable through the existing recovery resolver.
        locations=repo/'records/repository-archive/verified-artifact-locations.json'
        snapshot.include([locations]);registry=json.loads(locations.read_bytes())
        history=archive/'repository-history/incremental-browser/navigation-originals';history.mkdir(parents=True,exist_ok=True)
        for target,payload in generated.items():
            old=target.read_bytes();h=sha(old);retained=history/(h+target.suffix)
            if not retained.exists():artifacts.append((retained,old))
            elif sha(retained.read_bytes())!=h:raise ValueError('Recovery collision')
            if not any(r['sha256']==h and str(target) in r['references'] for r in registry['entries']):
                registry['entries'].append({'references':[str(target)],'sha256':h,'archive_root':str(history),
                    'relative_path':retained.name,'recovery_method':'Exact previous incremental navigation'})
        generated[locations]=(json.dumps(registry,indent=2)+'\n').encode()
        # Extend, rather than replace, the full source identity ledger.
        guard_ref=prior['source_guards'];guard_path=outputs/guard_ref['path'];guard_raw=guard_path.read_bytes()
        if sha(guard_raw)!=guard_ref['sha256']:raise ValueError('Source ledger changed')
        guards=json.loads(guard_raw)+list(discovery.guards.values())
        guard_bytes=(json.dumps(guards,indent=2)+'\n').encode()
        guard_target=outputs/'review-library'/('incremental-'+sha(guard_bytes)[:16]+'.json')
        if not guard_target.exists():artifacts.append((guard_target,guard_bytes))
        publication['source_guards']={'path':guard_target.relative_to(outputs).as_posix(),'sha256':sha(guard_bytes)}
        publication['last_full_snapshot']=prior.get('last_full_snapshot',prior['snapshot'])
        publication['incremental_coverage']=model['incremental_coverage']
        publication['versions']['sha256']=sha(generated[model_path]);publication['html_sha256']=sha(generated[home])
        for row in publication['generated_pages']:
            target=outputs/row['path']
            if target in generated:row['sha256']=sha(generated[target])
        snapshot.include([guard_path,*[p for _,p in sources]])
        snapshot.bind_hashes({str(Path(p).resolve()):r['sha256'] for p,r in discovery.guards.items()})
        with ExitStack() as locks:
            for path in sorted(state_paths):locks.enter_context(state_lock(path))
            publish_state(control,publication,snapshot,artifacts=artifacts,evidence_updates=list(generated.items()))
    return {'status':'published','added_versions':len(additions),'coverage':publication['incremental_coverage']}
