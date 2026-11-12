"""Task-owned provisional artwork; no scientific selection or model execution."""
from pathlib import Path
import json
import shutil
import re
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import sha, Snapshot, state_lock
from classical_conditioning.artifacts import artifact_staging, publish_transaction
from classical_conditioning.operations.candidate_paths import candidate_directory
from classical_conditioning.figures.review_html import svg_editing_page


def compact_svg(data):
    """Externalize bulky nonvisual registries, preserving visible SVG nodes."""
    root = ET.fromstring(data)
    registry = {}
    font_rules = set()
    def unique_font(match):
        rule=match.group(0)
        if rule in font_rules:return ''
        font_rules.add(rule);return rule
    for node in root.iter('{http://www.w3.org/2000/svg}style'):
        node.text=re.sub(r'@font-face\s*\{[^{}]*\}',unique_font,node.text or '')
    for node in root.iter('{http://www.w3.org/2000/svg}metadata'):
        if node.get('id') != 'editable-copy-provenance':
            continue
        record = json.loads(node.text)
        for key in ('scientific_registry', 'structural_inventory'):
            if key in record:
                registry[key] = record.pop(key)
        payload = (json.dumps(registry, sort_keys=True, allow_nan=False)+'\n').encode()
        if registry:
            record['registry_ref'] = {'path': 'registry.json', 'sha256': sha(payload)}
        node.text = json.dumps(record, allow_nan=False)
    return ET.tostring(root, encoding='utf-8'), registry


def read_registry(svg, directory):
    root = ET.fromstring(svg)
    node = root.find("{http://www.w3.org/2000/svg}metadata[@id='editable-copy-provenance']")
    if node is None:
        return {}
    record = json.loads(node.text)
    if 'registry_ref' not in record:
        return record
    ref = record['registry_ref']; base = Path(directory).resolve()
    path = (base/ref['path']).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Registry reference escapes bundle')
    raw = path.read_bytes()
    if sha(raw) != ref['sha256']:
        raise ValueError('Registry hash mismatch')
    return {**record, **json.loads(raw)}


def write_preview(svg, legend, directory, *, inputs, command, sources=(), parameters=None):
    """Save a fresh scratch bundle, authenticating all declared input identities."""
    directory = candidate_directory(directory)
    records = [*inputs, *sources]
    snapshot = Snapshot([r['path'] for r in records])
    for ref in records:
        if snapshot.hashes[str(Path(ref['path']).resolve())] != ref['sha256']:
            raise ValueError('Preview input changed: '+ref['path'])
    svg, registry = compact_svg(svg)
    files = {'preview.svg': svg, 'legend.md': legend.encode(),
             'preview.html': svg_editing_page(svg, scope_label='Provisional exploratory preview',
                                              legend='Provisional exploratory preview.\n\n'+legend).encode()}
    if registry:
        files['registry.json'] = (json.dumps(registry, sort_keys=True, allow_nan=False)+'\n').encode()
    for ref in sources:
        path = Path(ref['path']); raw=path.read_bytes()
        if sha(raw)!=ref['sha256']:raise ValueError('Renderer source changed while being read')
        files['sources/'+ref['sha256']+path.suffix] = raw
    manifest = {'schema_version': 1, 'status': 'provisional; no scientific or presentation selection',
                'inputs': inputs, 'sources': list(sources), 'command': command,
                'parameters': parameters or {}, 'artifacts': [
                    {'path': name, 'sha256': sha(raw)} for name, raw in files.items()]}
    files['preview.json'] = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    # Exclusively owned output: never overwrite previous iterations.
    directory.mkdir(exist_ok=False)
    try:
        for name, raw in files.items():
            p = directory/name; p.parent.mkdir(exist_ok=True); p.write_bytes(raw)
        snapshot.check()
    except BaseException:
        shutil.rmtree(directory)
        raise
    return directory/'preview.json'


def retain_preview(manifest_path, output_dir, scope):
    """Retain exact preview bytes without selecting artwork or refreshing navigation."""
    source = Path(manifest_path).resolve(); manifest = json.loads(source.read_bytes())
    if manifest.get('schema_version') != 1 or not manifest.get('status', '').startswith('provisional'):
        raise ValueError('Unsupported preview manifest')
    destination = candidate_directory(output_dir, existing=True)
    from classical_conditioning.external_artifacts import joaquim_root
    if not destination.is_relative_to(joaquim_root().resolve()):
        raise ValueError('Retained previews require an explicit JOAQUIM destination')
    files = []
    for ref in manifest['artifacts']:
        name = Path(ref['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Invalid preview artifact path')
        p = source.parent/name
        if not p.resolve().is_relative_to(source.parent) or p.is_symlink():
            raise ValueError('Preview artifact escapes bundle')
        if name.name == 'preview.svg':read_registry(p.read_bytes(),source.parent)
        if sha(p.read_bytes()) != ref['sha256']:
            raise ValueError('Preview artifact changed: '+str(p))
        files.append((p, name))
    files.append((source, Path('preview.json')))
    snapshot = Snapshot([p for p, _ in files])
    receipt = dict(scope=scope, scientific_selection_changed=False,
                   navigation_status='pending; register this bundle before task completion')
    with artifact_staging(destination, prefix='.retain-preview-') as staging:
        units = []
        for p, name in files:
            out = destination/name
            if out.exists(): raise FileExistsError(out)
            staged = staging/name; staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(p.read_bytes()); out.parent.mkdir(parents=True, exist_ok=True)
            units.append((staged, out))
        with state_lock(destination/'preview'):
            snapshot.check()
            def verify_saved():
                snapshot.check()
                for original,name in files:
                    if sha((destination/name).read_bytes())!=snapshot.hashes[str(original.resolve())]:
                        raise ValueError('Saved preview bytes changed')
            publish_transaction(tuple(units), staging, overwrite=False, verify_published=verify_saved)
    return receipt
