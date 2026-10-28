"""Guarded publication of the ordered figure browser and its reading catalogue."""
from contextlib import ExitStack
from pathlib import Path
from collections import defaultdict
import hashlib
import html
import json
import os
import shutil
import tempfile
from urllib.parse import quote

from scripts.reviews.versions import Catalogue, figure_page, natural, packed, sha, scope_key, source_files, GENERATED
from scripts.maintenance.storage_migration import publish_controls, digest
from classical_conditioning.figure_workspace import state_lock
from classical_conditioning.external_artifacts import resolve_artifact, relocated_external_directory


def ordered_groups(catalogue):
    grouped=defaultdict(list);shared=[]
    import re
    for group in catalogue['groups']:
        match=re.match(r'figure-0*(\d+)',group['id'])
        for entry in group['entries']:
            figure=int(match[1]) if match else None
            if figure is None:
                m=re.search(r'(?:^|/)figures/figure-0*(\d+)(?:/|$)',entry['path'])
                if m:figure=int(m[1])
            (grouped[figure] if figure is not None else shared).append(entry)
    result=[]
    def key(entry):
        match=re.search(r'/(panels?-[a-z](?:-[a-z])*)/',entry['path'])
        return (0 if entry.get('kind')=='generated_navigation' else 1,
                scope_key(match[1] if match else 'assembly'),natural(entry['title']))
    for number in sorted(grouped):
        result.append(dict(id='figure-'+str(number),figure=number,title='Figure '+str(number),
                           context='Panels, retained versions, legends and studies.',entries=sorted(grouped[number],key=key)))
    if shared:result.append(dict(id='shared-studies',title='Shared discussions and recovery',
                                 context='Shared material has one maintained home.',entries=sorted(shared,key=key)))
    return result


def verify_package(outputs, *, sources=True):
    """Authenticate the generated browser and its exact declared source identities."""
    outputs=relocated_external_directory(outputs);control=outputs/'review-library/catalogue-publication.json'
    if not control.is_file():return []
    publication=json.loads(control.read_text(encoding='utf-8'));issues=[]
    for item in publication.get('generated_pages',[]):
        p=outputs/item['path']
        if not p.is_file() or digest(p)!=item['sha256']:issues.append('Generated browser changed: '+str(p))
    if publication.get('versions'):
        row=publication['versions'];p=outputs/row['path']
        if not p.is_file() or digest(p)!=row['sha256']:issues.append('Version catalogue changed: '+str(p));return issues
        data=json.loads(p.read_bytes())
        seen=set()
        for item in data.get('versions',[]):
            if item['id'] in seen:issues.append('Duplicate version identity: '+item['id'])
            seen.add(item['id'])
        for row in publication.get('payloads',[]):
            p=outputs/row['path']
            if not p.is_file() or digest(p)!=row['sha256']:issues.append('Generated artwork payload changed: '+str(p))
        if sources:
            guard=publication['source_guards'];p=outputs/guard['path']
            if not p.is_file() or digest(p)!=guard['sha256']:issues.append('Source identity ledger changed');return issues
            refs={r['path']:r for r in json.loads(p.read_bytes())}
            for version in data['versions']:
                for ref in [version.get('legend',{}),version.get('state') or {},*version.get('source_bindings',[])]:
                    if ref.get('path') and ref.get('sha256'):refs[ref['path']]=ref
            for item in refs.values():
                try:
                    p=resolve_artifact(item['path'],expected_sha256=item['sha256'])
                    if 'bytes' in item and p.stat().st_size!=item['bytes']:
                        raise ValueError('Source size mismatch')
                except (ValueError, FileNotFoundError) as error:
                    issues.append('Browser source advanced: '+item['path']+'; '+str(error))
    return issues


def verify_previous_sources(outputs, catalogue, repo):
    """Changing a historical source requires reconciliation, never blind rebinding.

    Explicit current-state or catalogue identities may advance through their
    supported authoring flows; discovery still authenticates those exact bytes.
    """
    control=outputs/'review-library/catalogue-publication.json'
    if not control.is_file():return
    publication=json.loads(control.read_bytes());guard=publication.get('source_guards')
    if not guard:return
    p=outputs/guard['path']
    if not p.is_file() or digest(p)!=guard['sha256']:raise ValueError('Source identity ledger changed')
    allowed={}
    for p in (repo/'docs/paper/figures').rglob('*state.json'):
        d=json.loads(p.read_bytes());active=d.get(d.get('default_display','draft'),{})
        if isinstance(active,dict):
            for key in ('working_svg','review','legend'):
                ref=active.get(key,{})
                if isinstance(ref,dict) and ref.get('path') and ref.get('sha256'):allowed[ref['path']]=ref['sha256']
    for group in catalogue['groups']:
        for entry in group['entries']:
            if entry.get('sha256') and entry.get('kind')!='generated_navigation' and entry['storage']=='outputs':allowed[str(outputs/entry['path'])]=entry['sha256']
    for row in json.loads((outputs/guard['path']).read_bytes()):
        target=resolve_artifact(row['path'])
        if not target.is_file():raise ValueError('Browser source missing: '+str(target))
        observed=digest(target)
        if observed!=row['sha256'] and observed!=allowed.get(str(target)):raise ValueError('Historical browser source changed; preserve and reconcile: '+str(target))


def build(catalogue, output_root, archive_root, repo, *, check, page):
    outputs=relocated_external_directory(output_root).resolve();archive=relocated_external_directory(archive_root).resolve();repo=Path(repo).resolve()
    if not outputs.is_dir() or outputs.is_relative_to(repo):raise ValueError('Use an existing external output root')
    if str(outputs).startswith('/Volumes/') and not Path('/Volumes/JOAQUIM').is_mount():raise ValueError('JOAQUIM unavailable')
    report=check(catalogue,repo,outputs,archive,external=True,check_browser_sources=False)
    if report['issues']:raise ValueError('\n'.join(report['issues']))
    review=outputs/'review-library';review.mkdir(exist_ok=True)
    control=review/'catalogue-publication.json';prior=json.loads(control.read_bytes()) if control.is_file() else {}
    for path,key in [(outputs/'START-HERE.html','html_sha256'),(outputs/'REVIEW-LIBRARY.md','markdown_sha256')]:
        if path.exists() and sha(path.read_bytes())!=prior.get(key):raise ValueError(str(path.name)+' has unrecorded edits; preserve and reconcile them')
    verify_previous_sources(outputs,catalogue,repo)
    groups=ordered_groups(catalogue)
    before_pages={p:p.read_bytes() if p.is_file() else None for p in
                  [outputs/'shared/INDEX.html',outputs/'START-HERE.html',outputs/'REVIEW-LIBRARY.md',control,review/'versions.json',
                   *[outputs/e['path'] for g in catalogue['groups'] for e in g['entries'] if e.get('kind')=='generated_navigation']]}
    for g in catalogue['groups']:
        for e in g['entries']:
            if e.get('kind')=='generated_navigation':
                observed=next((r['sha256'] for r in report['checked'] if r['id']==e['id']),None)
                if before_pages[outputs/e['path']] is None or sha(before_pages[outputs/e['path']])!=observed:raise ValueError('Navigation advanced before preparation')
    with tempfile.TemporaryDirectory(dir=review,prefix='.prepare-') as temp:
        temp=Path(temp);discovery=Catalogue(outputs,archive,repo,temp)
        # Bind supported implementation and catalogue inputs as well as artwork.
        for p in [Path(__file__),Path(__file__).with_name('library.py'),Path(__file__).with_name('versions.py'),Path(__file__).parents[1]/'maintenance/storage_migration.py',repo/'docs/paper/figures/review-library.json']:
            if p.is_file():discovery.read(p)
        data=discovery.discover()
        initial_sources=discovery.source_names
        roots={'repo':repo,'outputs':outputs,'archive':archive}
        snapshot_files={};reading={};cards=[];markdown=['# Figure and review library','',catalogue['purpose'],'']
        source_identities=[]
        # A generated navigation destination is an output, not its own source input.
        for group in groups:
            sections=[];markdown.extend(['## '+group['title'],''])
            for entry in group['entries']:
                target=roots[entry['storage']]/entry['path'];generated=entry.get('kind')=='generated_navigation'
                if generated:href=entry['path']
                else:
                    raw=discovery.read(target);identity=sha(raw)
                    if entry.get('sha256') and entry['sha256']!=identity:raise ValueError('Reading source changed: '+str(target))
                    source_identities.append(dict(id=entry['id'],path=str(target),sha256=identity))
                    if entry['storage']=='repo' or target.suffix=='.md':
                        key='reading/'+('' if entry['storage']=='repo' else entry['storage']+'/')+entry['path'];snapshot_files[key]=raw
                        if target.suffix=='.md':
                            # @SNAPSHOT@ is substituted only after the snapshot identity is known.
                            canonical=quote(Path(os.path.relpath(target,(outputs/'review-library/@SNAPSHOT@'/key).parent)).as_posix(),safe='/._-')
                            snapshot_files[key+'.html']=page(entry['title'],f'<h1>{html.escape(entry["title"])}</h1><p><a href="{canonical}">Canonical editable source</a> · <a href="{quote(target.name)}">Read-only Markdown snapshot</a></p><pre>{html.escape(raw.decode())}</pre>')
                            key+='.html'
                        href='review-library/@SNAPSHOT@/'+key
                    else:href=Path(os.path.relpath(target,outputs)).as_posix()
                escaped=quote(href,safe='/._-@');title=html.escape(entry['title']);description=html.escape(entry['description'])
                sections.append(f'<article data-search><span class="status">{html.escape(entry["status"])}</span><h3><a href="{escaped}">{title}</a></h3><p>{description}</p></article>')
                markdown.append(f'- [{entry["title"]}]({escaped}) — {entry["status"]}')
            reading[group.get('figure')]=''.join(sections)
            alias=''
            # Preserve previous topic anchors beneath their owning figure.
            for old in catalogue['groups']:
                import re
                m=re.match(r'figure-0*(\d+)',old['id'])
                if group.get('figure') is not None and m and int(m[1])==group['figure'] and old['id']!=group['id']:
                    alias+='<span id="'+html.escape(old['id'])+'"></span>'
            cards.append('<section id="'+group['id']+'">'+alias+'<h2>'+group['title']+'</h2>'+''.join(sections)+'</section>')
        numbers=[g['figure'] for g in groups if g.get('figure') is not None]
        # Pages have no snapshot link cycle: reading URLs use a stable generated
        # reading index, while immutable manifests bind every published output.
        viewer_pages={}
        for n in numbers:
            viewer_pages[outputs/f'figures/figure-{n:02}/INDEX.html']=figure_page(n,data,outputs,
                 reading_html='<h2>Studies and reading</h2>'+reading.get(n,'').replace('href="','href="../../'),figure_links=numbers)
        viewer_pages[outputs/'shared/INDEX.html']=figure_page(0,data,outputs,figure_links=numbers)
        model=packed(data)
        identity=sha(packed(dict(catalogue=catalogue,versions_sha256=sha(model),
                                sources=source_identities,viewers={p.relative_to(outputs).as_posix():sha(raw) for p,raw in viewer_pages.items()})))[:16]
        relative='review-library/'+identity;snapshot=outputs/relative
        # Catalogue metadata is embedded in each viewer. Only its selected artwork
        # is loaded, with plain relative URLs usable under file://.
        replace=lambda raw:raw.replace(b'@SNAPSHOT@',identity.encode())
        viewer_pages={p:replace(raw) for p,raw in viewer_pages.items()}
        snapshot_files={p:replace(raw) for p,raw in snapshot_files.items()}
        cards=[x.replace('@SNAPSHOT@',identity) for x in cards];markdown=[x.replace('@SNAPSHOT@',identity) for x in markdown]
        snapshot_files['versions.json']=model
        generated_assets=[]
        for item in discovery.payloads.values():
            target=review/item['path'];raw=Path(item['source']).read_bytes();target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.read_bytes()!=raw:raise ValueError('Generated payload has manual edits: '+str(target))
            else:
                with target.open('xb') as stream:stream.write(raw)
            generated_assets.append(dict(path=target.relative_to(outputs).as_posix(),sha256=item['sha256'],bytes=item['bytes']))
        # Complete file accounting is linked after the four figure sections.
        all_links=[]
        for item in data['inventory']:
            if item.get('path') and item.get('relative_path'):
                target=Path(item['path']);href=quote(Path(os.path.relpath(target,snapshot)).as_posix(),safe='/._-')
                all_links.append('<li data-search><a href="'+href+'">'+html.escape(item['relative_path'])+'</a> <small>'+html.escape(item['role'])+'</small></li>')
        snapshot_files['ALL-REVIEWS.html']=page('All retained files','<h1>All retained files and source roles</h1><p><a href="../../START-HERE.html">Figure library</a></p><label>Filter files <input id="search"></label><ul>'+''.join(all_links)+'</ul><script>document.querySelector("input").oninput=e=>document.querySelectorAll("[data-search]").forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>')
        snapshot_files['catalogue.json']=packed(catalogue)
        snapshot_files['source-guards.json']=packed([r for r in discovery.guards.values() if Path(r['path']).suffix.lower() in {'.svg','.png','.jpg','.jpeg','.gif','.webp','.pdf','.html','.zip'} and Path(r['path']).name not in GENERATED])
        snapshot_files['manifest.json']=packed(dict(catalogue_sha256=sha(packed(catalogue)),source_identities=source_identities,
            coverage=data['coverage'],files=[dict(path=p,sha256=sha(raw),bytes=len(raw)) for p,raw in sorted(snapshot_files.items())]))
        if snapshot.exists():
            if any(not (snapshot/p).is_file() or (snapshot/p).read_bytes()!=raw for p,raw in snapshot_files.items()):raise ValueError('Existing reading snapshot differs; preserve its edits')
        else:
            stage=temp/'snapshot';stage.mkdir()
            for name,raw in snapshot_files.items():p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            os.replace(stage,snapshot)
        navigation=' · '.join('<a href="#'+g['id']+'">'+g['title']+'</a>' for g in groups)
        shared='<section><h2>Shared inventory and recovery</h2><p><a href="shared/INDEX.html">Browse shared/reference artwork</a> · <a href="'+relative+'/ALL-REVIEWS.html">All retained files, source illustrations and recovery material</a> · <a href="'+relative+'/versions.json">Version coverage and exceptions</a></p><p>'+str(len(data['versions']))+' source-bound artwork views; '+str(len(data['issues']))+' recorded coverage exceptions. Viewing does not select science.</p></section>'
        home=page('Classical Conditioning — figures and versions','<h1>Classical Conditioning<br>Figures and versions</h1><p>'+html.escape(catalogue['purpose'])+'</p><nav>'+navigation+'</nav><label>Find a topic <input id="search" type="search"></label>'+''.join(cards)+shared+'<script>document.querySelector("input").oninput=e=>document.querySelectorAll("[data-search]").forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>')
        generated=dict(viewer_pages);generated[review/'versions.json']=model
        generated[outputs/'REVIEW-LIBRARY.md']=('\n'.join(markdown)+'\n').encode()
        generated[outputs/'START-HERE.html']=home
        publication=dict(html_sha256=sha(home),markdown_sha256=sha(generated[outputs/'REVIEW-LIBRARY.md']),snapshot=relative,
            manifest_sha256=sha(snapshot_files['manifest.json']),entries=report['entries'],coverage=data['coverage'],
            versions=dict(path='review-library/versions.json',sha256=sha(model)),
            source_guards=dict(path=relative+'/source-guards.json',sha256=sha(snapshot_files['source-guards.json'])),payloads=generated_assets,
            generated_pages=[dict(path=p.relative_to(outputs).as_posix(),sha256=sha(raw)) for p,raw in generated.items()])
        generated[control]=packed(publication)
        # Preserve replaced navigation identities for exact historical requests.
        locations=repo/'records/repository-archive/verified-artifact-locations.json';location_before=None
        if locations.is_file():
            location_before=locations.read_bytes();registry=json.loads(location_before);changed=False
            history=archive/'repository-history/ordered-figure-library-20261027/navigation-originals';history.mkdir(parents=True,exist_ok=True)
            for target,raw in generated.items():
                if target==control or not target.is_file():continue
                old=target.read_bytes()
                if old==raw:continue
                h=sha(old);retained=history/(h+target.suffix);retained.parent.mkdir(exist_ok=True)
                if retained.exists():
                    if retained.read_bytes()!=old:raise ValueError('Historical navigation collision')
                else:retained.write_bytes(old)
                if not any(e['sha256']==h and str(target) in e['references'] for e in registry['entries']):
                    registry['entries'].append(dict(references=[str(target)],sha256=h,archive_root=str(history),relative_path=retained.name,recovery_method='Exact prior generated navigation; no artwork selection'));changed=True
            if changed:generated[locations]=packed(registry)
        prepared=[]
        for i,(target,raw) in enumerate(generated.items()):
            target.parent.mkdir(parents=True,exist_ok=True);source=temp/('control-'+str(i));source.write_bytes(raw)
            prepared.append(dict(path=str(target),source=str(source),sha256=sha(raw),bytes=len(raw),expected_sha256=sha(before_pages[target]) if target in before_pages and before_pages[target] is not None else (None if target in before_pages else digest(target) if target.is_file() else None),role='state' if target in {outputs/'START-HERE.html',control} else 'control'))
        # Dependencies, viewer pages and evidence precede home/control pointers.
        prepared.sort(key=lambda r:(r['path'] in {str(outputs/'START-HERE.html'),str(control)},r['path']==str(control)))
        guards=list(discovery.guards.values())
        guards.extend(dict(path=str(outputs/r['path']),sha256=r['sha256'],bytes=r['bytes']) for r in generated_assets)
        guards.extend(dict(path=str(snapshot/name),sha256=sha(raw),bytes=len(raw)) for name,raw in snapshot_files.items())
        if location_before is not None:guards.append(dict(path=str(locations),sha256=sha(location_before),bytes=len(location_before)))
        actual={(storage,str(path)) for storage,path in source_files(outputs,archive) if path.name not in GENERATED}
        if actual!=initial_sources:raise ValueError('Unexpected source files appeared during preparation')
        with ExitStack() as locks:
            for path in sorted(p for p in discovery.guards if p.endswith('state.json')):locks.enter_context(state_lock(Path(path)))
            publish_controls(dict(lock=str(review/'browser-publication'),controls=prepared,guards=guards,guard_checkpoints='publication-phases'))
        return publication
