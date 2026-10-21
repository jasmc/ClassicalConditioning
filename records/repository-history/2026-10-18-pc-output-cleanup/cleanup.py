"""Scoped, author-requested retirement of two obsolete PC output directories.

Prepare authenticates the SSD source artifacts and preserves historical records.
Apply checks the prepared inventories before deleting the exact named roots.
"""
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SSD = Path('J:/')
ROOTS = [Path('F:/ClassicalConditioning Outputs'), Path('F:/Results_refactored')]
ARCHIVE = SSD / 'ClassicalConditioning-Archive/pc-output-cleanup/2026-10-18'
spec = importlib.util.spec_from_file_location('archive_tool', REPO / 'scripts/repository_archive.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def save(name, obj):
    tool.write_json(HERE / name, obj)


def volume():
    label = subprocess.check_output(['powershell', '-NoProfile', '-Command',
        '(Get-Volume -DriveLetter J -ErrorAction Stop).FileSystemLabel'], text=True).strip()
    if label != 'JOAQUIM':
        raise RuntimeError('J: is not the verified JOAQUIM volume')


def inventory(root):
    if root.resolve() != root or str(root) not in [str(p) for p in ROOTS]:
        raise RuntimeError('Unexpected cleanup root')
    rows = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or p.is_junction():
            raise RuntimeError(f'Reparse point: {p}')
        if p.is_file():
            st = p.stat()
            rows.append(dict(path=p.relative_to(root).as_posix(), bytes=st.st_size,
                             mtime_ns=st.st_mtime_ns))
    return rows


def preserve(root, rows, label):
    dest = ARCHIVE / label
    manifest = dict(schema_version=1, date='2026-10-18', archive_root=str(dest),
                    source_root=str(root), files=[])
    for row in rows:
        manifest['files'].append(dict(path=row['path'], bytes=row['bytes'],
            sha256=tool.sha256(root / row['path']), action='archive_to_ssd'))
    save(label + '-transfer-inventory.json', manifest)
    # The supported tool's mount guard is Mac-specific. volume() above checks
    # the real Windows volume label; both destination and root are fixed here.
    tool.transfer(manifest, root, dest, check_mount=False)
    print(f'Preserved {len(rows)} files: {dest}', flush=True)
    return manifest


def prepare():
    volume()
    if (HERE / 'prepared.json').exists() or ARCHIVE.exists():
        raise RuntimeError('Existing transaction; do not overwrite')
    inventories = [inventory(p) for p in ROOTS]
    save('starting-inventory.json', dict(roots=[dict(root=str(p), files=r)
         for p, r in zip(ROOTS, inventories)]))
    # No textual consumers of this old processing root may remain in the repo.
    result = subprocess.run(['rg', '-l', '-i', '--hidden', '-g', '!.git/**',
        '-g', '!records/repository-history/2026-10-18-pc-output-cleanup/**',
        'results[_ ]refactored', '.'], cwd=REPO, capture_output=True, text=True)
    if result.returncode != 1:
        raise RuntimeError('Old processing-root consumers or search error: ' + result.stdout + result.stderr)
    old = ROOTS[1]
    retained = SSD / 'Digested Data/allDelay-full-v1'
    verified = []
    old_ids = sorted(p.name for p in (old / 'Processed data').iterdir()
                     if p.is_dir() and re.fullmatch(r'\d{8}_\d+', p.name))
    for index, fish in enumerate(old_ids, 1):
        name = fish + '_source_manifest.json'
        a = json.loads((old / 'Metadata' / name).read_text())
        b = json.loads((retained / 'Metadata' / name).read_text())
        for field in ('recording_id', 'recording_name', 'condition_id'):
            if a[field] != b[field]:
                raise RuntimeError(f'Identity mismatch: {fish} {field}')
        for role, src in a['sources'].items():
            other = b['sources'][role]
            if (src['sha256'], src['size_bytes']) != (other['sha256'], other['size_bytes']):
                raise RuntimeError(f'Raw-source provenance mismatch: {fish} {role}')
            raw = Path(other['path'].replace('/Volumes/JOAQUIM/', 'J:/'))
            if not raw.is_file() or raw.stat().st_size != other['size_bytes']:
                raise RuntimeError(f'Raw input missing or resized: {raw}')
        for role, artifact in a['artifacts'].items():
            other = b['artifacts'][role]
            if artifact['logical_sha256'] != other['logical_sha256']:
                raise RuntimeError(f'Intake artifact provenance mismatch: {fish} {role}')
            p = retained / 'Processed data' / fish / Path(other['path']).name
            if p.stat().st_size != other['size_bytes'] or tool.sha256(p) != other['sha256']:
                raise RuntimeError(f'Retained SSD artifact hash mismatch: {p}')
            verified.append(dict(path=str(p), bytes=other['size_bytes'], sha256=other['sha256']))
        required = ['candidate_temporal_outcomes-corrected-v3.parquet',
                    'candidate-trial-outcomes-corrected-v1.parquet',
                    'frame_activity_candidates-corrected-v1.parquet',
                    'frame_preprocessed_corrected-v1.parquet',
                    'movement_state_candidates-corrected-v2.parquet']
        for name in required:
            if not (retained / 'Processed data' / fish / name).is_file():
                raise RuntimeError(f'Replacement derived artifact missing: {fish} {name}')
        save('source-verification-progress.json', dict(completed_recordings=index,
            total_recordings=len(old_ids), verified_files=len(verified)))
        print(f'Verified retained source artifacts: {index}/{len(old_ids)} recordings', flush=True)
    save('retained-source-verification.json', dict(recordings=old_ids, files=verified,
        raw_sources='Manifest SHA-256/size identities matched; raw presence and size checked',
        intake_identity='Manifest logical SHA-256 matched; some Parquet byte hashes differ across Windows/Mac serialization',
        retained_intake='Actual SSD camera/protocol/tracking SHA-256 verified',
        derivatives='Replacement files present; old derivative bytes deliberately superseded'))
    mirrors = []
    for row in inventories[0]:
        p = ROOTS[0] / row['path']
        counterpart = SSD / 'ClassicalConditioning-Mac-20261009/pc-outputs' / Path(row['path']).relative_to('ORGER-JOAQUIM/outputs')
        digest = tool.sha256(p)
        matches = counterpart.is_file() and counterpart.stat().st_size == row['bytes'] and tool.sha256(counterpart) == digest
        mirrors.append(dict(source=str(p), counterpart=str(counterpart), sha256=digest, exact_match=matches))
    save('figure2-mirror-verification.json', mirrors)
    # Preserve every figure payload, including any byte version differing from
    # the old Mac handoff; no frozen original becomes unavailable after cleanup.
    preserve(ROOTS[0], inventories[0], 'figure2-pc-originals')
    provenance = [r for r in inventories[1] if Path(r['path']).suffix.lower() not in ('.parquet', '.incomplete')]
    preserve(ROOTS[1], provenance, 'old-delay-run-records-and-previews')
    for root, rows in zip(ROOTS, inventories):
        if inventory(root) != rows:
            raise RuntimeError(f'Source changed during preparation: {root}')
    save('prepared.json', dict(status='ready', source_roots=[str(p) for p in ROOTS],
        source_recordings_verified=len(old_ids), source_artifacts_verified=len(verified),
        figure2_exact_handoff_matches=sum(r['exact_match'] for r in mirrors),
        figure2_total=len(mirrors), archive=str(ARCHIVE),
        obsolete_derivatives='Unreferenced September delay-cohort-v1 legacy/candidate results, replaced by allDelay-full-v1',
        deletion_authorization='Author explicitly requested deletion of old, no-longer-useful contents of these two roots'))
    print('Prepared; no source files deleted.', flush=True)


def apply():
    volume()
    prepared = json.loads((HERE / 'prepared.json').read_text())
    if prepared['source_roots'] != [str(p) for p in ROOTS] or prepared['status'] != 'ready':
        raise RuntimeError('Unexpected prepared scope')
    snapshots = json.loads((HERE / 'starting-inventory.json').read_text())['roots']
    for root, snapshot in zip(ROOTS, snapshots):
        if inventory(root) != snapshot['files']:
            raise RuntimeError(f'Source changed; preserve and reconcile: {root}')
    manifests = []
    for label in ('figure2-pc-originals', 'old-delay-run-records-and-previews'):
        manifest = json.loads((HERE / (label + '-transfer-inventory.json')).read_text())
        tool.verify_payload(manifest, Path(manifest['archive_root']))
        manifests.append(manifest)
    for root, manifest in zip(ROOTS, manifests):
        report = tool.prune(manifest, root, Path(manifest['archive_root']), check_mount=False)
        save(root.name.replace(' ', '-') + '-archived-prune.json', report)
    # All remaining files are superseded derived Parquet outputs or abandoned
    # incomplete processing files. Raw inputs and retained SSD intake were
    # authenticated in prepare; unique historic run records are archived above.
    root = ROOTS[1]
    expected = {r['path']: r for r in snapshots[1]['files']}
    for row in inventory(root):
        if row != expected[row['path']] or Path(row['path']).suffix.lower() not in ('.parquet', '.incomplete'):
            raise RuntimeError('Unexpected remaining file; stop deletion')
    for root in ROOTS:
        if root.resolve() != root or str(root) not in prepared['source_roots']:
            raise RuntimeError('Deletion target escaped scope')
        shutil.rmtree(root)
    if any(p.exists() for p in ROOTS):
        raise RuntimeError('Cleanup incomplete')
    total = sum(r['bytes'] for s in snapshots for r in s['files'])
    save('completed.json', dict(status='complete', deleted_roots=[str(p) for p in ROOTS],
        deleted_files=sum(len(s['files']) for s in snapshots), deleted_bytes=total,
        preserved_archive=str(ARCHIVE), retained_project=str(SSD / 'Digested Data/allDelay-full-v1'),
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(json.dumps(dict(status='complete', freed_gib=round(total / 2**30, 3))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'apply'])
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else apply)()
