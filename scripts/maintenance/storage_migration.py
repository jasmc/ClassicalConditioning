"""Explicit, hash-guarded external storage migration; never selects science.

Inventory includes transport ZIP members and authenticated embedded HTML payloads.
The plan is an explicit per-file manifest. Copy, publish and prune are separate.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile
from contextlib import ExitStack
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def packed(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(packed(value))


def validate_mount(roots):
    for root in roots:
        root = Path(root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError(f'Unavailable or unsafe root: {root}')
        if str(root).startswith('/Volumes/'):
            mount = Path('/Volumes') / root.parts[2]
            if not mount.is_mount():
                raise ValueError(f'Storage is not mounted: {mount}')


def files(roots, excluded=()):
    result = []
    excluded = [Path(p) for p in excluded]
    for root in map(Path, roots):
        for base, directories, names in os.walk(root):
            directories[:] = sorted(d for d in directories
                                    if not any((Path(base) / d).is_relative_to(p) for p in excluded))
            for name in sorted(names):
                path = Path(base) / name
                if any(path.is_relative_to(p) for p in excluded):
                    continue
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError(f'Non-regular source requires review: {path}')
                result.append(path)
    return result


def zip_members(raw, container):
    rows = []
    with zipfile.ZipFile(raw) as archive:
        seen = set()
        for item in archive.infolist():
            if item.is_dir():
                continue
            if item.filename in seen:
                raise ValueError(f'Duplicate ZIP member: {container}:{item.filename}')
            seen.add(item.filename)
            h = hashlib.sha256()
            with archive.open(item) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(block)
            rows.append(dict(container=container, member=item.filename,
                             bytes=item.file_size, sha256=h.hexdigest()))
    return rows


def embedded_payloads(raw, container):
    """Inventory data-script payloads without changing or rendering the page."""
    rows = []
    text = raw.decode('utf-8', errors='replace')
    for identifier, body in re.findall(
            r'<script\b[^>]*id=["\']([^"\']+)["\'][^>]*>(.*?)</script>', text, re.S):
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            if body.strip().startswith('UEsD'):
                payload = base64.b64decode(body.strip(), validate=True)
                rows.extend(zip_members(io.BytesIO(payload), container + '#' + identifier))
            continue

        def walk(obj, name):
            if isinstance(obj, dict):
                if isinstance(obj.get('base64'), str) and obj.get('sha256'):
                    payload = base64.b64decode(obj['base64'], validate=True)
                    observed = hashlib.sha256(payload).hexdigest()
                    if observed != obj['sha256']:
                        raise ValueError(f'Embedded payload hash mismatch: {container}#{name}')
                    rows.append(dict(container=container, member=identifier + '/' + name,
                                     bytes=len(payload), sha256=observed))
                else:
                    for key, child in obj.items():
                        walk(child, name + '/' + key)
            elif isinstance(obj, list):
                for index, child in enumerate(obj):
                    walk(child, name + '/' + str(index))
        walk(value, '')
    return rows


def inventory(roots, excluded=()):
    validate_mount(roots)
    rows, embedded, notebooks = [], [], []
    for index, path in enumerate(files(roots, excluded)):
        before = path.stat()
        h = digest(path)
        row = dict(source=str(path), bytes=before.st_size, sha256=h,
                   allocated_bytes=before.st_blocks * 512)
        if path.suffix.lower() == '.zip':
            embedded.extend(zip_members(path, str(path)))
        elif path.suffix.lower() == '.html' and not path.name.startswith('._'):
            raw = path.read_bytes()
            embedded.extend(embedded_payloads(raw, str(path)))
            text = raw.decode('utf-8', errors='replace')
            if re.search(r'localStorage|indexedDB|review-notebook|notebook-config', text):
                notebooks.append(dict(source=str(path), sha256=h,
                                      status='browser-note transfer not verified'))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed during inventory: {path}')
        rows.append(row)
        if index and index % 2000 == 0:
            print(f'Inventoried {index} files', flush=True)
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                roots=list(map(str, roots)), excluded=list(map(str, excluded)),
                files=rows, embedded=embedded, notebooks=notebooks)


def contained(path, roots, *, parent_cache=None):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe path: {p}')
    if not any(p.is_relative_to(Path(root)) and p != Path(root) for root in roots):
        raise ValueError(f'Path outside declared roots: {p}')
    parent = p.parent
    if parent_cache is None:
        parent_safe = parent.resolve() == parent.absolute()
    else:
        if parent not in parent_cache:
            parent_cache[parent] = parent.resolve() == parent.absolute()
        parent_safe = parent_cache[parent]
    if not parent_safe or p.is_symlink():
        raise ValueError(f'Symlink or non-canonical path: {p}')
    return p


def validate_plan(plan):
    validate_mount(plan['roots'])
    if not plan.get('authorization', '').strip():
        raise ValueError('An explicit author instruction is required')
    sources, targets, parents = set(), {}, {}
    for row in plan['files']:
        source = contained(row['source'], plan['roots'], parent_cache=parents)
        if str(source) in sources:
            raise ValueError(f'Duplicate source: {source}')
        sources.add(str(source))
        if row['action'] not in {'move', 'copy', 'retain', 'delete'}:
            raise ValueError('Unknown action')
        if row['action'] == 'delete' and (not row.get('reason') or not row.get('recovery')):
            raise ValueError('Deletion needs a reason and verified recovery identities')
        if row['action'] in {'move', 'copy'}:
            target = contained(row['destination'], plan['roots'], parent_cache=parents)
            if target == source:
                raise ValueError('Move destination equals source')
            key = str(target).casefold()
            prior = targets.setdefault(key, (str(target), row['sha256'], row['bytes']))
            if prior != (str(target), row['sha256'], row['bytes']):
                raise ValueError(f'Case-insensitive destination collision: {target}')
    if sources.intersection(t[0] for t in targets.values()):
        raise ValueError('A migration destination is also a source')
    for row in plan.get('generated', []):
        target = contained(row['path'], plan['roots'], parent_cache=parents)
        key = str(target).casefold()
        identity = (str(target), row['sha256'], row['bytes'])
        if key in targets and targets[key] != identity:
            raise ValueError('Generated destination collision: ' + str(target))
        targets[key] = identity
    for target, _, _ in targets.values():
        if any(str(parent).casefold() in targets for parent in Path(target).parents):
            raise ValueError('A file destination is also a parent directory: ' + target)


def matches(path, row):
    path = Path(path)
    return path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256']


def equal_bytes(left, right):
    with Path(left).open('rb') as a, Path(right).open('rb') as b:
        while True:
            first, second = a.read(4 * 1024 * 1024), b.read(4 * 1024 * 1024)
            if first != second:
                return False
            if not first:
                return True


def check_sources(plan, *, allow_pruned=False):
    for index, row in enumerate(plan['files']):
        if row['action'] == 'retain' and row.get('mutable_navigation'):
            continue  # Existing navigation publisher and consumer evidence own it.
        if allow_pruned and row['action'] in {'move', 'delete'} and not Path(row['source']).exists():
            continue
        if not matches(row['source'], row):
            raise ValueError(f'Source changed or unavailable: {row["source"]}')
        if index and index % 4000 == 0:
            print(f'Authenticated {index} source files', flush=True)


def check_unexpected(plan):
    expected = {r['source'] for r in plan['files']}
    destinations = {r['destination'] for r in plan['files'] if r['action'] in {'move', 'copy'}}
    destinations.update(r['path'] for r in plan.get('generated', []))
    actual = {str(p) for p in files(plan['roots'], plan.get('excluded', []))}
    unexpected = actual - expected - destinations
    if unexpected:
        raise ValueError('Unexpected files; refresh or reconcile inventory: ' + str(sorted(unexpected)[:8]))


def check_recovery(row):
    for target in row.get('recovery', []):
        if not matches(target['path'], target):
            raise ValueError('Deletion recovery changed or unavailable: ' + target['path'])


def recover_zips(plan):
    """Recover every declared transport member before retiring its packaging."""
    validate_plan(plan)
    count = 0
    for transport in plan.get('zip_recovery', []):
        source = contained(transport['source'], plan['roots'])
        original = next(r for r in plan['files'] if r['source'] == str(source))
        if not matches(source, original):
            raise ValueError('Transport changed: ' + str(source))
        declared = {r['member']: r for r in transport['members']}
        with zipfile.ZipFile(source) as archive:
            actual = [r.filename for r in archive.infolist() if not r.is_dir()]
            if len(actual) != len(declared) or set(actual) != set(declared):
                raise ValueError('ZIP member set changed')
            for name in actual:
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('Unsafe ZIP member: ' + name)
                row = declared[name]
                target = contained(row['path'], plan['roots'])
                if target.exists():
                    if not matches(target, row):
                        raise ValueError('ZIP recovery collision: ' + str(target))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as output, archive.open(name) as member:
                    shutil.copyfileobj(member, output, 4 * 1024 * 1024)
                if not matches(target, row):
                    raise ValueError('Recovered ZIP member hash mismatch: ' + name)
                count += 1
    return dict(recovered_members=count)


def copy_generated(plan):
    validate_plan(plan)
    count = 0
    for row in plan.get('generated', []):
        if not row.get('source'):
            continue  # ZIP recovery has its own member-level source contract.
        target = contained(row['path'], plan['roots'])
        if target.exists():
            if not matches(target, row):
                raise ValueError('Generated artifact collision: ' + str(target))
            continue
        if not matches(row['source'], row):
            raise ValueError('Generated source changed: ' + row['source'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output, Path(row['source']).open('rb') as source:
            shutil.copyfileobj(source, output)
        if not matches(target, row):
            raise ValueError('Generated artifact copy mismatch')
        count += 1
    return dict(published_generated=count)


def copy(plan):
    validate_plan(plan)
    check_unexpected(plan)
    check_sources(plan)
    needed = sum(r['bytes'] for r in plan['files'] if r['action'] in {'move', 'copy'}
                 and not Path(r['destination']).exists())
    if shutil.disk_usage(plan['roots'][0]).free < needed + 64 * 1024 * 1024:
        raise OSError('Insufficient space for verified copies')
    count = 0
    for row in plan['files']:
        if row['action'] not in {'move', 'copy'}:
            continue
        target = Path(row['destination'])
        if target.exists():
            if not matches(target, row):
                raise ValueError(f'Destination collision; existing bytes preserved: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves concurrent changes. An interrupted partial
        # file is never silently overwritten; completed copies resume by hash.
        with target.open('xb') as output, Path(row['source']).open('rb') as source:
            shutil.copyfileobj(source, output, 4 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if not matches(target, row):
            raise ValueError('Copy verification failed: ' + str(target))
        count += 1
        if count % 1000 == 0:
            print(f'Copied {count} files', flush=True)
    check_sources(plan)
    return dict(copied_files=count)


def verify(plan, *, allow_pruned=False):
    validate_plan(plan)
    check_sources(plan, allow_pruned=allow_pruned)
    for row in plan['files']:
        if row['action'] in {'move', 'copy'} and not matches(row['destination'], row):
            raise ValueError('Destination changed/unavailable: ' + row['destination'])
        if row['action'] == 'delete':
            check_recovery(row)
    for row in plan.get('generated', []):
        if not matches(row['path'], row):
            raise ValueError('Generated/recovered artifact changed: ' + row['path'])
    return dict(counts=dict(Counter(r['action'] for r in plan['files'])),
                logical_bytes=sum(r['bytes'] for r in plan['files']))


def publish(plan, manifest, receipt, consumer_evidence):
    verify(plan)
    evidence = json.loads(Path(consumer_evidence).read_text())
    identity = digest(manifest)
    if evidence.get('manifest_sha256') != identity or evidence.get('issues') != []:
        raise ValueError('Consumer evidence is missing, stale, or has unresolved issues')
    for row in evidence.get('bindings', []):
        if not matches(row['path'], row):
            raise ValueError('Consumer/state changed before publication: ' + row['path'])
    if not evidence.get('bindings'):
        raise ValueError('Consumer evidence must bind actual authority files')
    check_unexpected(plan)
    save_new(receipt, dict(schema_version=1, status='published', manifest_sha256=identity,
                          consumer_evidence=str(consumer_evidence),
                          consumer_evidence_sha256=digest(consumer_evidence)))


def publish_controls(plan):
    """Publish storage indexes first and scoped pointers last under figure locks."""
    from classical_conditioning.figure_workspace import state_lock
    rows = sorted(plan['controls'], key=lambda r: r.get('role') == 'state')
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Each control/state destination must appear exactly once')
    previous, written = {}, []

    def check_guards():
        changed = {r['path'] for r in written}
        for guard in plan['guards']:
            if guard['path'] not in changed and not matches(guard['path'], guard):
                raise ValueError('Publication input advanced: ' + guard['path'])

    def replace(target, raw):
        fd, name = tempfile.mkstemp(prefix='.' + target.name + '.', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)

    with ExitStack() as locks:
        locks.enter_context(state_lock(Path(plan['lock'])))
        for row in sorted(rows, key=lambda r: r['path']):
            if row.get('role') == 'state':
                locks.enter_context(state_lock(Path(row['path'])))
        check_guards()
        for row in rows:
            target = Path(row['path'])
            observed = digest(target) if target.is_file() else None
            if observed != row.get('expected_sha256'):
                raise ValueError('Control/state advanced: ' + str(target))
            if not matches(row['source'], row):
                raise ValueError('Prepared control changed: ' + row['source'])
            previous[str(target)] = target.read_bytes() if target.exists() else None
        try:
            for row in rows:
                check_guards()
                replace(Path(row['path']), Path(row['source']).read_bytes())
                written.append(row)
                if not matches(row['path'], row):
                    raise ValueError('Published control hash mismatch')
            check_guards()
            for row in written:
                if not matches(row['path'], row):
                    raise ValueError('Published control advanced during commit: ' + row['path'])
        except BaseException:
            for row in reversed(written):
                target = Path(row['path'])
                if not matches(target, row):
                    continue  # Preserve a concurrent edit rather than overwrite it.
                raw = previous[str(target)]
                if raw is None: target.unlink()
                else: replace(target, raw)
            raise
    return dict(published_controls=[{k:r[k] for k in ['path','sha256','bytes']} for r in rows])


def prune(plan, manifest, receipt, *, limit=None):
    checkpoint = json.loads(Path(receipt).read_text())
    if checkpoint['manifest_sha256'] != digest(manifest) or checkpoint['status'] != 'published':
        raise ValueError('Migration is not authenticated and published')
    evidence = Path(checkpoint['consumer_evidence'])
    if digest(evidence) != checkpoint['consumer_evidence_sha256']:
        raise ValueError('Consumer evidence changed')
    for binding in json.loads(evidence.read_text())['bindings']:
        if not matches(binding['path'], binding):
            raise ValueError('Consumer changed after publication: ' + binding['path'])
    verify(plan, allow_pruned=True)
    check_unexpected(plan)
    for row in plan['files']:
        if row['action'] in {'move', 'delete'} and row.get('note_transfer_required') and not row.get('note_transfer_verified'):
            raise ValueError('Browser notes not verified: ' + row['source'])
    removed, logical, allocated, comparisons = [], 0, 0, 0
    for row in plan['files']:
        source = Path(row['source'])
        if row['action'] in {'retain', 'copy'} or not source.exists():
            continue
        if row.get('note_transfer_required') and not row.get('note_transfer_verified'):
            raise ValueError('Browser notes not verified: ' + str(source))
        if not matches(source, row):
            raise ValueError('Source changed during prune: ' + str(source))
        if row['action'] == 'move' and not matches(row['destination'], row):
            raise ValueError('Destination changed during prune: ' + row['destination'])
        if row['action'] == 'move':
            if not equal_bytes(source, row['destination']):
                raise ValueError('Direct byte comparison failed before removal: ' + str(source))
            comparisons += 1
        if row['action'] == 'delete':
            check_recovery(row)
        st = source.stat()
        source.unlink()
        removed.append(str(source)); logical += st.st_size; allocated += st.st_blocks * 512
        if limit and len(removed) >= limit:
            break
    return dict(removed=removed, logical_source_bytes_removed=logical,
                source_allocated_bytes_removed=allocated, direct_byte_comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'recover-zips', 'copy-generated', 'copy', 'verify', 'publish', 'prune'])
    parser.add_argument('--root', type=Path, action='append', default=[])
    parser.add_argument('--exclude', type=Path, action='append', default=[])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--consumer-evidence', type=Path)
    args = parser.parse_args()
    if args.command == 'inventory':
        result = inventory(args.root, args.exclude)
    else:
        plan = json.loads(args.manifest.read_text())
        if args.command == 'recover-zips': result = recover_zips(plan)
        elif args.command == 'copy-generated': result = copy_generated(plan)
        elif args.command == 'copy': result = copy(plan)
        elif args.command == 'verify': result = verify(plan, allow_pruned=True)
        elif args.command == 'publish':
            publish(plan, args.manifest, args.receipt, args.consumer_evidence)
            result = dict(status='published')
        else: result = prune(plan, args.manifest, args.receipt)
    save_new(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {'files', 'embedded', 'notebooks', 'removed'}}))


if __name__ == '__main__':
    main()
