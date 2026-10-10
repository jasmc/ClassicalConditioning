"""Resolve relocated evidence without rewriting immutable scientific manifests."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

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


def resolve_artifact(value: str | Path, *, expected_sha256: str | None = None,
                     repo: Path = REPOSITORY, archive_root: Path | None = None) -> Path:
    """Prefer relocated definitions; allow pending local payloads, then external bytes.

    A hash mismatch is never bypassed by silently selecting another version.
    """
    key = normalized_reference(value, repo)
    reloc_path = repo / 'docs/maintenance/archive/relocations.json'
    reloc = json.loads(reloc_path.read_text()) if reloc_path.exists() else {}
    moved = reloc.get('repository_moves', {}).get(key, key)
    candidate = Path(moved) if Path(moved).is_absolute() else repo / moved
    audit = repo / 'docs/maintenance/archive'
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
                candidate = root / key[len(prefix) + 1:]
                break
    if not candidate.is_file() and row:
        root = archive_root or Path(os.environ.get('CLASSICAL_CONDITIONING_ARCHIVE_ROOT', inventory['archive_root']))
        candidate = root / 'payload' / row['path']
    if not candidate.is_file():
        raise FileNotFoundError(f'Artifact unavailable: {value}. Connect JOAQUIM or supply its verified archive root; no substitute input was selected.')
    expected = expected_sha256 or (row['sha256'] if row else None)
    if expected and sha256(candidate) != expected:
        for entry in reloc.get('extracted_frozen_records', []):
            snapshot = repo / entry['record']
            if entry['sha256'] == expected and snapshot.is_file() and sha256(snapshot) == expected:
                return snapshot  # Exact historical bytes, never a scientific substitute.
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
    return root / part
