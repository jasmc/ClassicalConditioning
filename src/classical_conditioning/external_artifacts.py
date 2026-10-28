"""Resolve relocated evidence without rewriting immutable scientific manifests."""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path('/Volumes/JOAQUIM/ClassicalConditioning-Archive/repository-cleanup/2026-10-10')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_reference(value: str | Path, repo: Path = REPOSITORY) -> str:
    text = re.sub('/+', '/', str(value).replace('\\', '/'))
    for prefix in (str(repo).replace('\\', '/'), 'C:/Users/joaquim/Documents/ClassicalConditioning'):
        if text.casefold().startswith(prefix.casefold() + '/'):
            return text[len(prefix) + 1:]
    return text


@lru_cache(maxsize=8)
def _external_moves(index: str, mtime_ns: int, size: int):
    return json.loads(Path(index).read_text()).get('external_moves', {})


@lru_cache(maxsize=8)
def _external_directories(index: str, mtime_ns: int, size: int):
    return json.loads(Path(index).read_text()).get('external_directory_moves', {})


def relocated_external_directory(value: str | Path, *, repo: Path = REPOSITORY) -> Path:
    """Translate a recorded directory for supported readers/output defaults.

    File authentication still belongs to resolve_artifact and its caller. This
    does not create a missing destination or select scientific versions.
    """
    text = str(value).replace('\\', '/')
    if text.startswith('J:/'):
        text = '/Volumes/JOAQUIM/' + text[3:]
    elif text.startswith('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/'):
        text = '/Volumes/JOAQUIM/ClassicalConditioning-Mac-20261009/pc-outputs/' + text.split('/outputs/', 1)[1]
    path = Path(text)
    if '..' in path.parts:
        raise ValueError('External directory cannot contain parent traversal')
    index = repo / 'records/repository-archive/relocations.json'
    if not index.is_file():
        return path
    info = index.stat()
    moves = _external_directories(str(index), info.st_mtime_ns, info.st_size)
    for old in sorted(moves, key=len, reverse=True):
        if text == old or text.startswith(old + '/'):
            target = Path(moves[old])
            if not target.is_absolute() or '..' in target.parts:
                raise ValueError('Invalid external directory relocation')
            return target / text[len(old):].lstrip('/')
    return path


def relocated_external_path(value: str | Path, *, repo: Path = REPOSITORY,
                            expected_sha256: str | None = None) -> Path:
    """Resolve an exact storage move, authenticating each retained identity.

    This is storage compatibility only: no fallback, version choice or missing
    payload exemption. Published historical inventories keep their original bytes.
    """
    path = Path(value)
    index = repo / 'records/repository-archive/relocations.json'
    if not index.is_file():
        return path
    info = index.stat()
    moves = _external_moves(str(index), info.st_mtime_ns, info.st_size)
    seen = set()
    while str(path) in moves:
        if str(path) in seen:
            raise ValueError('External relocation cycle: ' + str(path))
        seen.add(str(path))
        move = moves[str(path)]
        target = Path(move['path'])
        if (not target.is_absolute() or '..' in target.parts
                or target.is_symlink()):
            raise ValueError('Invalid external relocation: ' + str(target))
        expected = expected_sha256 or move['sha256']
        if expected != move['sha256']:
            raise ValueError('External relocation version mismatch: ' + str(path))
        if not target.is_file():
            raise FileNotFoundError('Relocated artifact unavailable: ' + str(target))
        if target.stat().st_size != move['bytes'] or sha256(target) != expected:
            raise ValueError('Relocated artifact SHA-256/size mismatch: ' + str(target))
        path = target
    return path


def resolve_artifact(value: str | Path, *, expected_sha256: str | None = None,
                     repo: Path = REPOSITORY, archive_root: Path | None = None) -> Path:
    """Prefer relocated definitions; allow pending local payloads, then external bytes.

    A hash mismatch is never bypassed by silently selecting another version.
    """
    key = normalized_reference(value, repo)
    # Historical paths can identify several accepted byte versions. Explicit
    # hashes select the verified version without modifying immutable pointers.
    locations_path = repo / 'records/repository-archive/verified-artifact-locations.json'
    if expected_sha256 and locations_path.is_file():
        locations = json.loads(locations_path.read_text())
        for entry in locations.get('entries', []):
            if entry['sha256'] != expected_sha256 or key not in {
                normalized_reference(ref, repo) for ref in entry['references']
            }:
                continue
            relative = Path(entry['relative_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid verified artifact location')
            exact = relocated_external_path(Path(entry['archive_root']) / relative,
                                            repo=repo, expected_sha256=expected_sha256)
            if not exact.is_file():
                raise FileNotFoundError(f'Artifact unavailable: {value}, SHA-256 {expected_sha256}. Connect JOAQUIM; the verified version is at {exact}.')
            if sha256(exact) != expected_sha256:
                raise ValueError(f'Artifact SHA-256 mismatch: {exact}')
            return exact
    # Current study manuscripts have one external owner. A local navigation
    # stub must never be read as their content, even when the drive is absent.
    studies_path = repo / 'docs/paper/study-sources.json'
    if studies_path.is_file():
        studies = json.loads(studies_path.read_text())
        for entry in studies['entries']:
            if key != entry['reference']:
                continue
            root = Path(studies['root'])
            relative = Path(entry['path'])
            candidate = relocated_external_path(root / relative, repo=repo,
                                                  expected_sha256=expected_sha256 or entry['sha256'])
            if (relative.is_absolute() or '..' in relative.parts
                    or not candidate.resolve().is_relative_to(root.resolve())):
                raise ValueError('Invalid study source location')
            if not candidate.is_file():
                raise FileNotFoundError(f'Study source unavailable: {candidate}. Connect JOAQUIM; no local substitute was selected.')
            if sha256(candidate) != (expected_sha256 or entry['sha256']):
                raise ValueError(f'Study source SHA-256 mismatch: {candidate}')
            return candidate
    reloc_path = repo / 'records/repository-archive/relocations.json'
    reloc = json.loads(reloc_path.read_text()) if reloc_path.exists() else {}
    moves = reloc.get('repository_moves', {})
    moved, visited = key, set()
    while moved in moves:
        if moved in visited:
            raise ValueError(f'Repository relocation cycle: {key}')
        visited.add(moved)
        moved = moves[moved]
    candidate = Path(moved) if Path(moved).is_absolute() else repo / moved
    candidate = relocated_external_path(candidate, repo=repo, expected_sha256=expected_sha256)
    audit = repo / 'records/repository-archive'
    inventory_paths = [*sorted(audit.glob('*/transfer-inventory.json'), reverse=True),
                       audit / 'transfer-inventory.json']
    inventory, row = {}, None
    for inventory_path in inventory_paths:
        if not inventory_path.exists():
            continue
        checkpoint = json.loads(inventory_path.read_text())
        match = next((r for r in checkpoint.get('files', [])
                      if key in [r['path'], *r.get('historical_paths', [])]), None)
        if match is not None:
            inventory, row = checkpoint, match
            break
    if not candidate.is_file():
        external_prefixes = {
            'J:': Path('/Volumes/JOAQUIM'),
            'F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs': Path('/Volumes/JOAQUIM/ClassicalConditioning-Mac-20261009/pc-outputs'),
        }
        for prefix, root in external_prefixes.items():
            if key.startswith(prefix + '/'):
                candidate = relocated_external_path(root / key[len(prefix) + 1:], repo=repo,
                                                      expected_sha256=expected_sha256)
                break
    if not candidate.is_file() and row:
        root = archive_root or Path(os.environ.get('CLASSICAL_CONDITIONING_ARCHIVE_ROOT', inventory['archive_root']))
        candidate = relocated_external_path(root / 'payload' / row['path'], repo=repo,
                                              expected_sha256=expected_sha256 or row['sha256'])
    if not candidate.is_file():
        raise FileNotFoundError(f'Artifact unavailable: {value}. Connect JOAQUIM or supply its verified archive root; no substitute input was selected.')
    # Retained mutable guidance can change after an inventory checkpoint.
    # External payloads and immutable records still require exact bytes.
    immutable = moved.startswith(('records/frozen-analyses/', 'configs/paper-figures/selections/'))
    expected = expected_sha256 or (row['sha256'] if row and
                                  (row.get('action') == 'archive_to_ssd' or immutable) else None)
    if expected and sha256(candidate) != expected:
        for entry in reloc.get('extracted_frozen_records', []):
            snapshot = repo / entry['record']
            if entry['sha256'] == expected and snapshot.is_file() and sha256(snapshot) == expected:
                return snapshot  # Exact historical bytes, never a scientific substitute.
        # Consolidated guidance has a current canonical destination and an
        # authenticated original. Only an explicit hash requests that original;
        # ordinary navigation continues to use current guidance.
        migrations = sorted((repo / 'records/repository-history').glob('*/migration.json'))
        for migration in migrations if expected_sha256 else []:
            for entry in json.loads(migration.read_text()).get('dispositions', []):
                if (entry.get('sha256') != expected_sha256
                        or (key != entry['original_path'] and moved != entry['canonical_path'])
                        or not entry.get('preserved_original')):
                    continue
                snapshot = repo / entry['preserved_original']
                if not snapshot.resolve().is_relative_to(repo.resolve()):
                    raise ValueError('Invalid preserved original location')
                if not snapshot.is_file() or sha256(snapshot) != expected_sha256:
                    raise ValueError(f'Artifact SHA-256 mismatch: {snapshot}')
                return snapshot
        raise ValueError(f'Artifact SHA-256 mismatch: {candidate}')
    return candidate


def external_output(relative: str | Path) -> Path:
    """Choose an external destination without creating a mount or repo output tree."""
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError('Output name must be a relative path without parent traversal')
    configured = os.environ.get('CLASSICAL_CONDITIONING_ARTIFACT_ROOT')
    root = Path(configured).expanduser().resolve() if configured else Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current')
    if root == REPOSITORY or REPOSITORY in root.parents:
        raise ValueError('Permanent outputs must be outside the repository')
    if not configured and not os.path.ismount('/Volumes/JOAQUIM'):
        raise FileNotFoundError('JOAQUIM is not mounted. Connect it or set CLASSICAL_CONDITIONING_ARTIFACT_ROOT to an external output directory.')
    # An explicitly configured root must already exist: an unavailable volume is not fabricated.
    if configured and not root.is_dir():
        raise FileNotFoundError(f'Configured external artifact root is unavailable: {root}')
    if configured:
        return root / part
    migrated = relocated_external_directory(root / part)
    if migrated != root / part:
        return migrated
    if part.parts[0] in {'figures', 'shared'}:
        return root.parent / part
    if _external_directories(str(REPOSITORY / 'records/repository-archive/relocations.json'),
            (REPOSITORY / 'records/repository-archive/relocations.json').stat().st_mtime_ns,
            (REPOSITORY / 'records/repository-archive/relocations.json').stat().st_size):
        raise ValueError('Use an explicit figure/panel or shared output destination after storage consolidation')
    return root / part


def check_study_sources(repo: Path = REPOSITORY, *, external: bool = False) -> list[str]:
    """Validate discovery offline; authenticate canonical bytes only on request."""
    path = repo / 'docs/paper/study-sources.json'
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    issues, seen = [], set()
    root = Path(data['root'])
    if not (PurePosixPath(data['root']).is_absolute() or PureWindowsPath(data['root']).is_absolute()) or (root.is_absolute() and root.resolve().is_relative_to(repo.resolve())):
        issues.append('Study source root must be absolute and external')
    for entry in data['entries']:
        ref, relative = entry['reference'], Path(entry['path'])
        if ref in seen:
            issues.append('Duplicate study reference: ' + ref)
        seen.add(ref)
        if Path(ref).is_absolute() or '..' in Path(ref).parts:
            issues.append('Invalid study reference: ' + ref)
        if relative.is_absolute() or '..' in relative.parts or not (root / relative).resolve().is_relative_to(root.resolve()):
            issues.append('Invalid study source location: ' + str(relative))
        if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            issues.append('Invalid study source SHA-256: ' + ref)
        if external:
            try:
                resolve_artifact(ref, repo=repo)
            except (ValueError, FileNotFoundError) as error:
                issues.append(str(error))
    return issues
