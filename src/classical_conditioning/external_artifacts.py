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


def _windows_joaquim_roots() -> list[Path]:
    """Discover attached volumes by label, without assuming a drive letter."""
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    mask = kernel.GetLogicalDrives()
    roots = []
    for number in range(26):
        if not mask & (1 << number):
            continue
        root = f'{chr(65 + number)}:/'
        label = ctypes.create_unicode_buffer(261)
        if kernel.GetVolumeInformationW(ctypes.c_wchar_p(root), label, len(label),
                                        None, None, None, None, 0):
            if label.value == 'JOAQUIM':
                roots.append(Path(root))
    return roots


def _joaquim_roots() -> list[Path]:
    return _windows_joaquim_roots() if os.name == 'nt' else [Path('/Volumes/JOAQUIM')]


def joaquim_root() -> Path:
    """Use explicit configuration or one uniquely identified mounted volume."""
    configured = os.environ.get('CLASSICAL_CONDITIONING_JOAQUIM_ROOT')
    if configured:
        root = Path(configured).expanduser()
    else:
        roots = _joaquim_roots()
        roots = [p for p in roots if os.path.ismount(p)]
        if len(roots) > 1:
            raise ValueError('Ambiguous JOAQUIM volumes; configure CLASSICAL_CONDITIONING_JOAQUIM_ROOT')
        if not roots:
            raise FileNotFoundError('JOAQUIM volume unavailable; mount it or configure CLASSICAL_CONDITIONING_JOAQUIM_ROOT')
        root = roots[0]
    if not root.is_absolute() or not root.is_dir():
        raise FileNotFoundError(f'JOAQUIM volume unavailable: {root}')
    if root.resolve().is_relative_to(REPOSITORY.resolve()):
        raise ValueError('JOAQUIM volume must be outside the repository')
    return root


def _canonical_external(value: str | Path) -> str:
    text = str(value).replace('\\', '/')
    if text.upper().startswith('J:/'):
        return '/Volumes/JOAQUIM/' + text[3:]
    configured = os.environ.get('CLASSICAL_CONDITIONING_JOAQUIM_ROOT')
    if configured:
        prefix = str(Path(configured).expanduser()).replace('\\', '/').rstrip('/')
        if text.casefold() == prefix.casefold() or text.casefold().startswith(prefix.casefold() + '/'):
            return '/Volumes/JOAQUIM' + text[len(prefix):]
    elif os.name=='nt' and PureWindowsPath(text).drive and PureWindowsPath(text).drive.casefold()!=PureWindowsPath(REPOSITORY).drive.casefold():
        roots=_windows_joaquim_roots()
        for root in roots:
            prefix=str(root).replace('\\','/').rstrip('/')
            if text.casefold()==prefix.casefold() or text.casefold().startswith(prefix.casefold()+'/'):
                if len(roots)!=1:raise ValueError('Ambiguous JOAQUIM volumes; configure CLASSICAL_CONDITIONING_JOAQUIM_ROOT')
                return '/Volumes/JOAQUIM'+text[len(prefix):]
    return text


def _host_external(value: str | Path) -> Path:
    """Map the recorded volume to an existing host mount, without creating it."""
    text = _canonical_external(value)
    prefix = '/Volumes/JOAQUIM'
    if text == prefix or text.startswith(prefix + '/'):
        relative = PurePosixPath(text[len(prefix):].lstrip('/'))
        if '..' in relative.parts:
            raise ValueError('External path cannot contain parent traversal')
        root = joaquim_root()
        candidate = root.joinpath(*relative.parts)
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError('External path escapes the JOAQUIM volume')
        return candidate
    return Path(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_reference(value: str | Path, repo: Path = REPOSITORY) -> str:
    text = _canonical_external(re.sub('/+', '/', str(value).replace('\\', '/')))
    policy = repo / 'docs/project-maintenance/structure-policy.json'
    aliases = json.loads(policy.read_text(encoding='utf-8')).get('recorded_repository_roots', []) if policy.is_file() else []
    for prefix in (str(repo).replace('\\', '/'), 'C:/Users/joaquim/Documents/ClassicalConditioning', *aliases):
        prefix = prefix.replace('\\', '/').rstrip('/')
        if text.casefold().startswith(prefix.casefold() + '/'):
            relative = text[len(prefix) + 1:]
            if '..' in PurePosixPath(relative).parts:
                raise ValueError('Recorded repository path escapes its checkout alias')
            return relative
    return text


@lru_cache(maxsize=8)
def _external_moves(index: str, mtime_ns: int, size: int):
    return {str(k).replace('\\', '/'): v for k, v in
            json.loads(Path(index).read_text(encoding='utf-8')).get('external_moves', {}).items()}


@lru_cache(maxsize=8)
def _external_directories(index: str, mtime_ns: int, size: int):
    return {str(k).replace('\\', '/'): str(v).replace('\\', '/') for k, v in
            json.loads(Path(index).read_text(encoding='utf-8')).get('external_directory_moves', {}).items()}


def relocated_external_directory(value: str | Path, *, repo: Path = REPOSITORY) -> Path:
    """Translate a recorded directory for supported readers/output defaults.

    File authentication still belongs to resolve_artifact and its caller. This
    does not create a missing destination or select scientific versions.
    """
    text = _canonical_external(value)
    if text.startswith('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/'):
        text = '/Volumes/JOAQUIM/ClassicalConditioning-Mac-20261009/pc-outputs/' + text.split('/outputs/', 1)[1]
    if '..' in PurePosixPath(text).parts:
        raise ValueError('External directory cannot contain parent traversal')
    index = repo / 'records/repository-archive/relocations.json'
    if not index.is_file():
        return _host_external(text)
    info = index.stat()
    moves = _external_directories(str(index), info.st_mtime_ns, info.st_size)
    return _host_external(relocate_directory_reference(text, moves))


def relocate_directory_reference(text, moves):
    """Apply recorded directory moves without filesystem access or Path hooks."""
    seen, used = set(), set()
    while True:
        if text in seen:
            raise ValueError('External directory relocation cycle: ' + text)
        seen.add(text)
        old = next((p for p in sorted(moves, key=len, reverse=True)
                    if text == p or text.startswith(p + '/')), None)
        if old is None:
            return text
        target = PurePosixPath(moves[old])
        if not (target.is_absolute() or PureWindowsPath(str(target)).is_absolute()) or '..' in target.parts:
            raise ValueError('Invalid external directory relocation')
        updated = str(target / text[len(old):].lstrip('/'))
        if updated == text:
            return text
        if old in used:
            raise ValueError('External directory relocation cycle: ' + old)
        used.add(old)
        text = updated


def relocated_external_path(value: str | Path, *, repo: Path = REPOSITORY,
                            expected_sha256: str | None = None) -> Path:
    """Resolve an exact storage move, authenticating each retained identity.

    This is storage compatibility only: no fallback, version choice or missing
    payload exemption. Published historical inventories keep their original bytes.
    """
    text = _canonical_external(value)
    index = repo / 'records/repository-archive/relocations.json'
    if not index.is_file():
        return _host_external(text)
    info = index.stat()
    moves = _external_moves(str(index), info.st_mtime_ns, info.st_size)
    directories = _external_directories(str(index), info.st_mtime_ns, info.st_size)
    seen = set()
    identity, size = expected_sha256, None
    while True:
        if text in seen:
            raise ValueError('External relocation cycle: ' + text)
        seen.add(text)
        if text in moves:
            move = moves[text]
            logical = PurePosixPath(move['path'].replace('\\', '/'))
            if not (logical.is_absolute() or PureWindowsPath(str(logical)).is_absolute()) or '..' in logical.parts:
                raise ValueError('Invalid external relocation: ' + str(logical))
            if (identity is not None and identity != move['sha256']) or (size is not None and size != move['bytes']):
                raise ValueError('External relocation version mismatch: ' + text)
            identity, size = move['sha256'], move['bytes']
            text = str(logical)
        else:
            updated = relocate_directory_reference(text, directories)
            if updated == text:
                break
            text = updated
    target = _host_external(text)
    if size is not None:
        if not target.is_file():
            raise FileNotFoundError('Relocated artifact unavailable: ' + str(target))
        if target.is_symlink() or (size is not None and target.stat().st_size != size) or sha256(target) != identity:
            raise ValueError('Relocated artifact SHA-256/size mismatch: ' + str(target))
    return target


def resolve_artifact(value: str | Path, *, expected_sha256: str | None = None,
                     repo: Path = REPOSITORY, archive_root: Path | None = None,
                     allow_historical: bool = True) -> Path:
    """Prefer relocated definitions; allow pending local payloads, then external bytes.

    A hash mismatch is never bypassed by silently selecting another version.
    """
    key = normalized_reference(value, repo)
    if '..' in PurePosixPath(key.replace('\\', '/')).parts:
        raise ValueError('Artifact reference cannot contain parent traversal')
    # Historical paths can identify several accepted byte versions. Explicit
    # hashes select the verified version without modifying immutable pointers.
    locations_path = repo / 'records/repository-archive/verified-artifact-locations.json'
    if allow_historical and expected_sha256 and locations_path.is_file():
        locations = json.loads(locations_path.read_text(encoding="utf-8"))
        for entry in locations.get('entries', []):
            if entry['sha256'] != expected_sha256 or key not in {
                normalized_reference(ref, repo) for ref in entry['references']
            }:
                continue
            relative = Path(entry['relative_path'])
            if relative.is_absolute() or PureWindowsPath(entry['relative_path']).drive or '..' in PurePosixPath(entry['relative_path'].replace('\\', '/')).parts:
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
        studies = json.loads(studies_path.read_text(encoding="utf-8"))
        for entry in studies['entries']:
            if key != entry['reference']:
                continue
            root = _host_external(studies['root'])
            relative = Path(entry['path'])
            candidate = relocated_external_path(str(PurePosixPath(studies['root']) / relative.as_posix()), repo=repo,
                                                  expected_sha256=expected_sha256 or entry['sha256'])
            if (relative.is_absolute() or PureWindowsPath(entry['path']).drive
                    or '..' in PurePosixPath(entry['path'].replace('\\', '/')).parts
                    or not candidate.resolve().is_relative_to(root.resolve())):
                raise ValueError('Invalid study source location')
            if not candidate.is_file():
                raise FileNotFoundError(f'Study source unavailable: {candidate}. Connect JOAQUIM; no local substitute was selected.')
            if sha256(candidate) != (expected_sha256 or entry['sha256']):
                raise ValueError(f'Study source SHA-256 mismatch: {candidate}')
            return candidate
    external = _canonical_external(key)
    if external == '/Volumes/JOAQUIM' or external.startswith('/Volumes/JOAQUIM/'):
        candidate = relocated_external_path(external, repo=repo, expected_sha256=expected_sha256)
        if not candidate.is_file():
            raise FileNotFoundError('Artifact unavailable: ' + str(candidate))
        if expected_sha256 and sha256(candidate) != expected_sha256:
            raise ValueError('Artifact SHA-256 mismatch: ' + str(candidate))
        return candidate
    reloc_path = repo / 'records/repository-archive/relocations.json'
    reloc = json.loads(reloc_path.read_text(encoding="utf-8")) if reloc_path.exists() else {}
    moves = reloc.get('repository_moves', {})
    moved, visited = key, set()
    while moved in moves:
        if moved in visited:
            raise ValueError(f'Repository relocation cycle: {key}')
        visited.add(moved)
        moved = moves[moved]
    candidate = moved if (PurePosixPath(moved).is_absolute() or PureWindowsPath(moved).is_absolute()) else repo / moved
    candidate = relocated_external_path(candidate, repo=repo, expected_sha256=expected_sha256)
    audit = repo / 'records/repository-archive'
    inventory_paths = [*sorted(audit.glob('*/transfer-inventory.json'), reverse=True),
                       audit / 'transfer-inventory.json']
    inventory, row = {}, None
    matches = []
    for inventory_path in inventory_paths:
        if not inventory_path.exists():
            continue
        checkpoint = json.loads(inventory_path.read_text(encoding="utf-8"))
        match = next((r for r in checkpoint.get('files', [])
                      if key in [r['path'], *r.get('historical_paths', [])]), None)
        if match is not None:
            matches.append((checkpoint, match))
    eligible = [(checkpoint, match) for checkpoint, match in matches
                if expected_sha256 is None or match.get('sha256') == expected_sha256]
    if eligible:
        if not candidate.is_file() and len({match['sha256'] for _, match in eligible
                                          if match.get('action') == 'archive_to_ssd'}) > 1:
            raise ValueError('Ambiguous archived versions; supply expected_sha256: ' + key)
        inventory, row = eligible[0]
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
        if not allow_historical:
            raise ValueError(f'Active artifact SHA-256 mismatch: {candidate}')
        for entry in reloc.get('extracted_frozen_records', []):
            snapshot = repo / entry['record']
            if entry['sha256'] == expected and snapshot.is_file() and sha256(snapshot) == expected:
                return snapshot  # Exact historical bytes, never a scientific substitute.
        # Consolidated guidance has a current canonical destination and an
        # authenticated original. Only an explicit hash requests that original;
        # ordinary navigation continues to use current guidance.
        migrations = sorted((repo / 'records/repository-history').glob('*/migration.json'))
        for migration in migrations if expected_sha256 else []:
            for entry in json.loads(migration.read_text(encoding="utf-8")).get('dispositions', []):
                if (entry.get('sha256') != expected_sha256
                        or (key != normalized_reference(entry['original_path'], repo)
                            and moved != normalized_reference(entry['canonical_path'], repo))
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
    if not part.parts or part.is_absolute() or PureWindowsPath(relative).drive or '..' in part.parts:
        raise ValueError('Output name must be a relative path without parent traversal')
    configured = os.environ.get('CLASSICAL_CONDITIONING_ARTIFACT_ROOT')
    root = Path(configured).expanduser().resolve() if configured else _host_external('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current')
    if root == REPOSITORY or REPOSITORY in root.parents:
        raise ValueError('Permanent outputs must be outside the repository')
    # An explicitly configured root must already exist: an unavailable volume is not fabricated.
    if configured and not root.is_dir():
        raise FileNotFoundError(f'Configured external artifact root is unavailable: {root}')
    if configured:
        if not (root / part).resolve().is_relative_to(root):
            raise ValueError('Output path escapes its configured root')
        return root / part
    logical_root = PurePosixPath('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current')
    migrated = relocated_external_directory(str(logical_root / part.as_posix()))
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
    data = json.loads(path.read_text(encoding="utf-8"))
    issues, seen = [], set()
    root = Path(data['root'])
    if not (PurePosixPath(data['root']).is_absolute() or PureWindowsPath(data['root']).is_absolute()) or (root.is_absolute() and root.resolve().is_relative_to(repo.resolve())):
        issues.append('Study source root must be absolute and external')
    for entry in data['entries']:
        ref, relative = entry['reference'], Path(entry['path'])
        if ref in seen:
            issues.append('Duplicate study reference: ' + ref)
        seen.add(ref)
        if Path(ref).is_absolute() or PureWindowsPath(ref).drive or '..' in PurePosixPath(ref.replace('\\', '/')).parts:
            issues.append('Invalid study reference: ' + ref)
        if relative.is_absolute() or PureWindowsPath(entry['path']).drive or '..' in PurePosixPath(entry['path'].replace('\\', '/')).parts or not (root / relative).resolve().is_relative_to(root.resolve()):
            issues.append('Invalid study source location: ' + str(relative))
        if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            issues.append('Invalid study source SHA-256: ' + ref)
        if external:
            try:
                resolve_artifact(ref, repo=repo)
            except (ValueError, FileNotFoundError) as error:
                issues.append(str(error))
    return issues
