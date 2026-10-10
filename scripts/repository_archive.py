"""Audit repository content and transfer payloads only after JOAQUIM verification.

Commands: inventory, check, dry-run, transfer, prune, verify.
Transfer never prunes. Prune independently rechecks every source and destination.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import io
import zipfile
import json
import os
import posixpath
from urllib.parse import unquote
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from classical_conditioning.external_artifacts import DEFAULT_ARCHIVE, sha256

AUDIT = REPO / 'docs/maintenance/archive'
MANIFEST = AUDIT / 'transfer-inventory.json'
MEDIA = {'.png', '.svg', '.pdf', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.mov', '.mp3', '.wav'}
DATA = {'.parquet', '.pkl', '.pickle', '.npy', '.npz', '.h5', '.hdf5', '.arrow'}
# Named historical experiments, not every renderer. Supported/tested adapters stay.
ALTERNATIVE_SCRIPTS = {
    '.row1_v12_review_stage.py',
    *('scripts/' + n for n in (
        'render_3strace_plot_versions.py', 'render_delay_meanlog_check.py',
        'render_figure1_managua_review.py', 'render_figure2_managua_review.py',
        'render_figure2_stats_review.py', 'build_panel_e_f_binned_bars.py',
        'build_panel_e_individual_bout_means.py', 'build_panel_e_presentation_review.py',
        'build_panel_e_frozen_heatmap_signal_review.py', 'build_panel_e_frozen_v12_bins_review.py',
        'write_figure2_log_review_critique.py', 'assemble_delay_comparison_pdf.py',
    ))
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def safe_relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Unsafe manifest path: {name}')
    return p


def embedded_checks(path):
    """Verify embedded files, without rendering or modifying HTML."""
    if path.suffix != '.html':
        return []
    checks = []
    for script_id, body in re.findall(r'<script\b[^>]*id=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</script>', path.read_text(errors='strict'), re.S):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            if script_id == 'archive' and body.strip().startswith('UEsD'):
                zipped = base64.b64decode(body.strip(), validate=True)
                with zipfile.ZipFile(io.BytesIO(zipped)) as bundle:
                    if bundle.testzip() is not None:
                        raise ValueError(f'Embedded ZIP CRC mismatch: {path}')
                    manifest = json.loads(bundle.read('manifest.json'))
                    for item in [*manifest.get('files', []), *manifest.get('documentation', [])]:
                        member = item.get('archive_member')
                        if member:
                            data = bundle.read(member)
                            if hashlib.sha256(data).hexdigest() != item['sha256']:
                                raise ValueError(f'Embedded ZIP member hash mismatch: {path}:{member}')
                    checks.append(dict(script_id=script_id, zip_sha256=hashlib.sha256(zipped).hexdigest(), verified_members=len(bundle.namelist())))
            continue
        if not isinstance(obj, dict):
            continue
        for entry, item in obj.get('files', {}).items():
            if isinstance(item, dict) and 'base64' in item and 'sha256' in item:
                data = base64.b64decode(item['base64'], validate=True)
                digest = hashlib.sha256(data).hexdigest()
                if digest != item['sha256']:
                    raise ValueError(f'Embedded archive hash mismatch: {path}:{script_id}:{entry}')
                checks.append(dict(script_id=script_id, entry=entry, sha256=digest))
    return checks


def inventory(repo=REPO):
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0'))
    starting = json.loads((repo / 'docs/maintenance/archive/starting-inventory.json').read_text())
    start_hash = {r['path']: r.get('sha256') for r in starting['files']}
    text_sources = {}
    for folder in ('src', 'tests', 'scripts', 'docs', 'Plans', 'configs'):
        for p in (repo / folder).rglob('*'):
            if p.is_file() and p.suffix in {'.md', '.py', '.json', '.ps1'} and '__pycache__' not in p.parts:
                text_sources[str(p.relative_to(repo))] = p.read_text(errors='replace')
    reloc = json.loads((repo / 'docs/maintenance/archive/relocations.json').read_text())
    frozen_hashes = {e['sha256'] for e in reloc['extracted_frozen_records']}
    frozen_refs = defaultdict(set)
    historical_refs = defaultdict(set)
    def refs(obj, owner):
        if isinstance(obj, dict):
            if 'sha256' in obj:
                frozen_refs[obj['sha256']].add(owner)
                if isinstance(obj.get('path'), str): historical_refs[obj['sha256']].add(re.sub('/+', '/', obj['path'].replace('\\', '/')))
            for v in obj.values(): refs(v, owner)
        elif isinstance(obj, list):
            for v in obj: refs(v, owner)
    for p in (repo / 'records/frozen-analyses').rglob('*.json'):
        refs(json.loads(p.read_text()), str(p.relative_to(repo)))
    for p in (repo / 'configs/paper-figures/selections').glob('*.json'):
        refs(json.loads(p.read_text()), str(p.relative_to(repo)))
    # Review files are scientific evidence. Preserve them until verified external transfer.
    rows = []
    for p in sorted(repo.rglob('*')):
        rel = p.relative_to(repo)
        if any(x in rel.parts for x in ('.git', '.venv', '__pycache__', '.pytest_cache')) or not p.is_file():
            continue
        if str(rel).startswith('docs/maintenance/archive/'):
            continue  # Inventory records are control files, not recursive inputs.
        if p.is_symlink():
            raise ValueError(f'Repository content symlink needs explicit review: {rel}')
        key = str(rel); digest = sha256(p)
        consumers = sorted(name for name, content in text_sources.items()
                           if name != key and (key in content or (p.name in content and len(p.name) > 12)))
        frozen = sorted(frozen_refs.get(digest, set()))
        transfer = rel.parts[0] in {'reviews', 'outputs'} and key not in {'reviews/README.md', 'outputs/README.md'}
        candidate_code = key in ALTERNATIVE_SCRIPTS
        protected_code = digest in frozen_hashes or bool(frozen) or any(c.startswith(('tests/', 'src/')) for c in consumers)
        if candidate_code and not protected_code:
            transfer = True
        if rel.parts[0] == 'configs' and p.name.endswith('-review.json'):
            transfer = True
        if p.suffix.lower() in MEDIA | DATA:
            transfer = True
        role = 'supported code/documentation/configuration'
        reason = 'Retained active repository contract or implementation; consumer names are reference candidates, not inferred scientific approval'
        if rel.parts[0] == 'records': role = 'exact frozen definition/code'; reason = 'Immutable copy/extraction with SHA-256 evidence'
        if transfer:
            role = 'frozen payload' if frozen or 'freeze' in key else 'alternative/review evidence'
            reason = 'External scientific/visual payload; original stays until verified transfer'
            if candidate_code: role = 'superseded one-off candidate code'; reason = 'Named alternative; no supported source/test consumer or frozen dependency found'
        rows.append(dict(path=key, bytes=p.stat().st_size, sha256=digest,
                         git_status='tracked' if key in tracked else 'untracked',
                         role=role, consumers=consumers, freeze_dependencies=frozen,
                         action='archive_to_ssd' if transfer else 'retain',
                         state='pending_transfer' if transfer else 'retained', reason=reason,
                         starting_sha256=start_hash.get(key), historical_paths=sorted({'C:/Users/joaquim/Documents/ClassicalConditioning/' + key, *historical_refs[digest]})))
    duplicates = defaultdict(list)
    for r in rows:
        if r['action'] == 'archive_to_ssd': duplicates[r['sha256']].append(r['path'])
    for r in rows:
        if r['action'] == 'archive_to_ssd': r['canonical_copy'] = min(duplicates[r['sha256']])
    return dict(schema_version=1, date='2026-10-10', archive_root=str(DEFAULT_ARCHIVE),
                files=rows, inventory_policy='Reference-based audit; hash-protected snapshots retained; pending originals never pruned by inventory',
                control_files='docs/maintenance/archive/* are retained control records and excluded from recursive inventory',
                environment_and_caches=starting['environment_and_caches'])


def pending(manifest):
    return [r for r in manifest['files'] if r['action'] == 'archive_to_ssd']


def source_path(repo, row):
    p = repo / safe_relative(row['path'])
    if p.is_symlink() or repo.resolve() not in p.resolve().parents:
        raise ValueError(f'Unsafe source: {p}')
    return p


def validate_sources(manifest, repo):
    for row in pending(manifest):
        p = source_path(repo, row)
        if not p.is_file() or p.stat().st_size != row['bytes'] or sha256(p) != row['sha256']:
            raise ValueError(f'Source changed/unavailable; refresh inventory before transfer: {p}')


def validate_volume(destination):
    volume = Path('/Volumes/JOAQUIM')
    if not os.path.ismount(volume):
        raise FileNotFoundError('JOAQUIM is disconnected; no destination was created and no source was removed')
    resolved = destination.resolve()
    if volume.resolve() not in resolved.parents:
        raise ValueError('Archive destination must be on mounted JOAQUIM')
    if sys.platform == 'darwin':
        import plistlib
        info = plistlib.loads(subprocess.check_output(['diskutil', 'info', '-plist', str(volume)]))
        if info.get('VolumeName') != 'JOAQUIM':
            raise ValueError('Mounted volume identity does not match JOAQUIM')
    return volume


def payload_path(root, row):
    p = root / 'payload' / safe_relative(row['path'])
    if p.is_symlink() or root.resolve() not in p.resolve().parents:
        raise ValueError(f'Unsafe destination: {p}')
    return p


def verify_payload(manifest, root):
    count = 0
    for row in pending(manifest):
        p = payload_path(root, row)
        if not p.is_file() or p.stat().st_size != row['bytes'] or sha256(p) != row['sha256']:
            raise ValueError(f'Archive hash/size mismatch or missing payload: {p}')
        embedded_checks(p)
        count += 1
    return count


def bundle_links(manifest, repo, root):
    """Check archive-relative links and identify historical/external references.

    Immutable HTML is not rewritten. Relocated repository links resolve through
    the accompanying map; missing historical/external inputs remain explicit.
    """
    reloc_path = repo/'docs/maintenance/archive/relocations.json'
    moves = json.loads(reloc_path.read_text()).get('repository_moves', {}) if reloc_path.exists() else {}
    report = []
    for row in pending(manifest):
        if Path(row['path']).suffix != '.html': continue
        text = payload_path(root, row).read_text(errors='replace')
        for raw in sorted(set(re.findall(r'(?:href|src)=["\']([^"\']+)["\']', text))):
            if raw.startswith(('data:', '#', 'javascript:')): continue
            ref = unquote(raw.split('#')[0])
            if ':' in ref or ref.startswith('//'):
                report.append(dict(source=row['path'], reference=raw, status='external/historical URI; original preserved'))
                continue
            logical = posixpath.normpath(posixpath.join(str(Path(row['path']).parent), ref))
            if logical.startswith('../') or logical.startswith('/'):
                report.append(dict(source=row['path'], reference=raw, status='outside bundle; historical reference preserved'))
                continue
            target = root/'payload'/logical
            moved = moves.get(logical, logical)
            status = 'archive-relative target verified' if target.is_file() else 'repository target resolved through relocation map' if (repo/moved).is_file() else 'unavailable historical target; not fabricated'
            report.append(dict(source=row['path'], reference=raw, logical_path=logical, relocated_path=moved, status=status))
    return report


def transfer(manifest, repo, destination, *, check_mount=True):
    if check_mount: validate_volume(destination)
    validate_sources(manifest, repo)
    if destination.exists():
        raise FileExistsError(f'Published archive already exists; verify it rather than overwrite: {destination}')
    stage = destination.with_name(destination.name + '.staging')
    control_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    marker = stage / 'manifest-identity.json'
    if stage.is_symlink():
        raise ValueError('Staging destination cannot be a symlink')
    if stage.exists():
        if not marker.is_file() or json.loads(marker.read_text())['sha256'] != control_hash:
            raise ValueError('Staging destination collision: inventory differs; do not overwrite')
    else:
        required = sum(r['bytes'] for r in pending(manifest))
        parent = destination.parent
        # No mkdir before mount validation; free-space test on an existing ancestor.
        ancestor = parent
        while not ancestor.exists(): ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < required + 64 * 1024 * 1024:
            raise OSError('Insufficient archive space')
        stage.mkdir(parents=True)
        write_json(marker, dict(sha256=control_hash))
    existing_copy_report = []
    for row in pending(manifest):
        target = payload_path(stage, row)
        if target.exists():
            if target.is_file() and sha256(target) == row['sha256']: continue
            raise ValueError(f'Staging file collision: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.copying')
        original = source_path(repo, row)
        prior = Path('/Volumes/JOAQUIM/ClassicalConditioning-Mac-20261009/ClassicalConditioning') / safe_relative(row['path'])
        reused = check_mount and prior.is_file() and sha256(prior) == row['sha256']
        copy_source = prior if reused else original
        existing_copy_report.append(dict(path=row['path'], prior_ssd_copy=str(prior), reused_verified_copy=bool(reused)))
        shutil.copyfile(copy_source, temp)
        if sha256(temp) != row['sha256']: raise ValueError(f'Copy hash mismatch: {temp}')
        os.replace(temp, target)
    count = verify_payload(manifest, stage)
    validate_sources(manifest, repo)
    write_json(stage / 'link-report.json', bundle_links(manifest, repo, stage))
    write_json(stage / 'existing-copy-inspection.json', existing_copy_report)
    # Preserve all repository and historical identities alongside the payload.
    relocations = repo/'docs/maintenance/archive/relocations.json'
    if relocations.is_file(): shutil.copyfile(relocations, stage/'relocations.json')
    write_json(stage / 'transfer-inventory.json', manifest)
    write_json(stage / 'verification.json', dict(verified_files=count, manifest_sha256=control_hash, payload_hashes_verified=True))
    if destination.exists():
        raise FileExistsError('Destination appeared during transfer; verified staging retained')
    os.rename(stage, destination)
    return count


def prune(manifest, repo, destination, *, check_mount=True):
    if check_mount: validate_volume(destination)
    archived = json.loads((destination / 'transfer-inventory.json').read_text())
    if archived != manifest: raise ValueError('Published archive inventory differs; refusing source removal')
    verify_payload(manifest, destination)
    # Check ALL remaining sources before deleting ANY; supports an interrupted prune.
    for row in pending(manifest):
        p = source_path(repo, row)
        if p.exists() and sha256(p) != row['sha256']:
            raise ValueError(f'Source changed after transfer; refusing removal: {p}')
    removed_bytes = 0
    removed = []
    for row in pending(manifest):
        p = source_path(repo, row)
        if p.exists():
            if sha256(p) != row['sha256']:
                raise ValueError(f'Source changed during prune; remaining source preserved: {p}')
            removed_bytes += p.stat().st_size
            p.unlink(); removed.append(row['path'])
    return dict(removed_paths=removed, removed_bytes=removed_bytes, verified_archive=str(destination))


def check_repository(manifest, repo):
    approved = {r['path']: r['sha256'] for r in pending(manifest)}
    issues = []
    relocations = repo/'docs/maintenance/archive/relocations.json'
    if relocations.is_file():
        for entry in json.loads(relocations.read_text()).get('extracted_frozen_records', []):
            record = repo / safe_relative(entry['record'])
            if not record.is_file() or sha256(record) != entry['sha256']:
                issues.append(entry['record'] + ' (frozen record changed)')
    for p in repo.rglob('*'):
        if not p.is_file() or any(x in p.relative_to(repo).parts for x in ('.git', '.venv', '__pycache__')): continue
        key = str(p.relative_to(repo))
        generated = p.suffix.lower() in MEDIA | DATA or (p.suffix=='.html' and re.search(r'<(?:img|svg)\b|data:image/',p.read_text(errors='replace')))
        if generated and (key not in approved or sha256(p) != approved[key]): issues.append(key)
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory','check','dry-run','transfer','verify','prune'])
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--manifest', type=Path, default=MANIFEST,
                        help='Independent transfer checkpoint; preserves earlier published inventories')
    args = parser.parse_args()
    if args.command == 'inventory':
        obj = inventory(); write_json(args.manifest,obj)
        retained = [r['path'] for r in obj['files'] if r['action']=='retain' and Path(r['path']).suffix in {'.py','.ps1','.cjs'}]
        write_json(AUDIT/'retained-code.json',dict(paths=retained,policy='Supported/tested consumers and exact frozen code retained'))
        print(json.dumps(dict(files=len(obj['files']),pending_files=len(pending(obj)),pending_bytes=sum(r['bytes'] for r in pending(obj)))))
        return
    obj = json.loads(args.manifest.read_text()); dest = args.archive_root or Path(obj['archive_root'])
    if args.command == 'check':
        issues = check_repository(obj,REPO);print(json.dumps(dict(unaccounted_generated_files=issues)));sys.exit(bool(issues))
    elif args.command == 'dry-run':
        validate_sources(obj,REPO)
        print(json.dumps(dict(destination=str(dest),mounted=os.path.ismount('/Volumes/JOAQUIM'),pending_files=len(pending(obj)),pending_bytes=sum(r['bytes'] for r in pending(obj)),action='No writes or removals performed')))
    elif args.command == 'transfer': print(json.dumps(dict(verified_files=transfer(obj,REPO,dest))))
    elif args.command == 'verify':
        validate_volume(dest);print(json.dumps(dict(verified_files=verify_payload(obj,dest))))
    else:
        result=prune(obj,REPO,dest)
        result['total_payload_bytes'] = sum(r['bytes'] for r in pending(obj))
        result['all_sources_pruned'] = all(not source_path(REPO, r).exists() for r in pending(obj))
        write_json(args.manifest.parent/'completed-transfer.json',result)
        if result['all_sources_pruned'] and args.manifest.resolve() == MANIFEST.resolve():
            index = REPO/'Plans/IMPLEMENTATION_STEP_INDEX.md'
            index.write_text(index.read_text().replace('Preparation implemented; SSD transfer pending', 'Complete; external payloads verified and pruned'))
        print(json.dumps(result))


if __name__ == '__main__': main()
