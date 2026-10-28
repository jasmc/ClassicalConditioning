"""Read-only ownership, active-link, relocation and evidence checks.

Use --external to authenticate current external artifacts and archive payloads.
No rendering, transfer, pruning or scientific selection is performed.
"""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from urllib.parse import unquote

ROOT = _COMMAND_REPOSITORY
sys.path.insert(0, str(ROOT / 'src'))
from classical_conditioning.external_artifacts import resolve_artifact, sha256, check_study_sources

LINK = re.compile(r'(?<!!)\[[^\]]+\]\((<[^>]+>|[^)]+)\)')


def recorded_path(value: str, repo: Path, aliases=(), *, base: Path | None = None) -> Path:
    """Resolve explicit old checkout roots without rewriting recorded bytes.

    This is a local verification alias, not an external-payload substitution.
    Unknown absolute paths retain their external identity.
    """
    text = str(value).replace('\\', '/')
    for alias in aliases:
        root = alias.replace('\\', '/').rstrip('/')
        if text == root or text.startswith(root + '/'):
            relative = text[len(root):].lstrip('/')
            if '..' in Path(relative).parts:
                raise ValueError('Recorded repository path escapes its checkout alias')
            return (repo / relative).resolve()
    path = Path(value)
    return (path if path.is_absolute() else (base or repo) / path).resolve()


def markdown_anchors(path: Path) -> set[str]:
    body = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
    anchors = set(re.findall(r'<a\b[^>]*\bid=["\']([^"\']+)', body))
    counts: dict[str, int] = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', body, flags=re.M):
        heading = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading)
        slug = re.sub(r'[^\w -]', '', heading.casefold()).replace(' ', '-')
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(slug if not count else f'{slug}-{count}')
    return anchors


def check(repo: Path = ROOT, *, external: bool = False) -> dict:
    repo = repo.resolve()
    issues: list[str] = []
    unavailable: list[str] = []
    issues.extend(check_study_sources(repo, external=external))
    policy = json.loads((repo / 'docs/project-maintenance/structure-policy.json').read_text())
    aliases = policy.get('recorded_repository_roots', [])
    reading_catalogue = repo / 'docs/paper/figures/review-library.json'
    if reading_catalogue.is_file():
        spec = importlib.util.spec_from_file_location('review_library', repo / 'scripts/reviews/library.py')
        library = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(library)
        reading_report = library.check(json.loads(reading_catalogue.read_text()), repo, external=external)
        issues.extend('Review library: ' + issue for issue in reading_report['issues'])
    exceptions = set(policy['compatibility_files'])
    for path in (repo / 'docs').rglob('*'):
        if not path.is_file():
            continue
        key = path.relative_to(repo).as_posix()
        if key not in exceptions and not any(key.startswith(prefix) for prefix in policy['active_doc_prefixes']):
            issues.append('Misplaced documentation: ' + key)
        if key.startswith('docs/running-analysis/') and path.suffix == '.md':
            body = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
            headings = re.findall(r'^#{1,6}\s+(.+)$', body, flags=re.M)
            for heading in headings:
                if any(heading.casefold().startswith(prefix.casefold()) for prefix in policy.get('future_work_headings', [])):
                    issues.append('Future work/progress in operational guide: ' + key + ' -> ' + heading)
    for path in (repo / 'Plans').rglob('*'):
        if path.is_file() and path.name not in {'README.md', 'IMPLEMENTATION_STEP_INDEX.md'} and not re.fullmatch(r'\d{2}(?:\.\d+)?_[A-Z0-9_]+\.md', path.name):
            issues.append('Misplaced plan material: ' + path.name)
    for key in policy['forbidden_paths']:
        if (repo / key).exists():
            issues.append('Superseded layout remains: ' + key)
    archive = repo / 'records/repository-archive'
    moves = json.loads((archive / 'relocations.json').read_text())['repository_moves']
    for original in moves:
        key = original
        seen = set()
        while key in moves:
            if key in seen:
                issues.append('Relocation cycle: ' + original)
                break
            seen.add(key)
            key = moves[key]
        else:
            if not (PurePosixPath(key).is_absolute() or PureWindowsPath(key).is_absolute()) and not (repo / key).is_file():
                issues.append('Missing relocation destination: ' + original + ' -> ' + key)
    index = json.loads((repo / 'records/frozen-analyses/index.json').read_text())
    main_manifest = repo / policy['migration_manifest']
    manifests = [main_manifest, *[p for p in sorted((repo / 'records/repository-history').glob('*/migration.json')) if p != main_manifest]]
    # Frozen-index requirements cannot be relaxed by mutable-index policy.
    protected = [(entry, False) for entry in index['files']]
    for manifest in manifests:
        protected.extend((entry, True) for entry in json.loads(manifest.read_text()).get('protected_files', []))
    originals = {(entry['path'], entry['sha256']): entry['preserved_original']
                 for entry in policy.get('mutable_index_originals', [])}
    checked = set()
    for entry, migration_original in protected:
        key = entry['path']
        identity = (key, entry['sha256'], migration_original)
        if identity in checked:
            continue
        checked.add(identity)
        path = repo / key
        original = originals.get((key, entry['sha256'])) if migration_original else None
        if original:
            # Only named live recovery indexes can have evolving bytes. Their
            # migration originals remain checked independently and exactly.
            if key not in {'records/repository-archive/ARTIFACT_INDEX.md',
                           'records/repository-archive/verified-artifact-locations.json'}:
                issues.append('Invalid mutable index exception: ' + key)
                continue
            snapshot = repo / original
            if (not snapshot.resolve().is_relative_to(repo / 'records/repository-history')
                    or not path.is_file() or not snapshot.is_file()
                    or sha256(snapshot) != entry['sha256']):
                issues.append('Preserved index original changed: ' + key)
            continue
        if not path.is_file() or sha256(path) != entry['sha256']:
            issues.append('Preserved bytes changed: ' + key)
    # Current links must work directly. Historical evidence is byte-preserved
    # and uses the resolver; its old relative links are not current guidance.
    documents = [repo / 'README.md', repo / 'AGENTS.md']
    for folder in ('docs', 'Plans'):
        documents.extend((repo / folder).rglob('*.md'))
    documents.extend((repo / '.agents/skills').rglob('*.md'))
    for key in policy.get('active_navigation', []):
        if not (repo / key).is_file():
            issues.append('Missing active navigation: ' + key)
        documents.append(repo / key)
    catalogue = policy.get('script_catalogue')
    if catalogue:
        catalogue_path = repo / catalogue
        if not catalogue_path.is_file():
            issues.append('Missing script catalogue: ' + catalogue)
        else:
            body = catalogue_path.read_text()
            for path in (repo / 'scripts').rglob('*'):
                if path.is_file() and path.suffix in {'.py', '.js', '.cjs', '.ps1'}:
                    relative = path.relative_to(repo / 'scripts').as_posix()
                    if f']({relative})' not in body:
                        issues.append('Uncatalogued script: ' + relative)
    documents.extend((repo / 'records/frozen-analyses').rglob('README.md'))
    documents.extend([repo / 'records/frozen-analyses/selections.md', repo / 'records/README.md',
                      repo / 'configs/README.md', repo / 'scripts/README.md', repo / 'legacy/README.md'])
    anchor_cache: dict[Path, set[str]] = {}
    for path in documents:
        key = path.relative_to(repo).as_posix()
        if not path.is_file():
            continue
        body = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for match in LINK.finditer(body):
            target = match[1].strip('<>')
            if target.startswith(('http:', 'https:', 'app:', 'codex:', 'mailto:')):
                continue
            target, _, fragment = target.partition('#')
            target = unquote(target)
            if target.startswith('/Volumes/'):
                if external and not Path(target).exists():
                    unavailable.append(key + ': ' + target)
                continue
            destination = recorded_path(target, repo, aliases, base=path.parent) if target else path
            if not destination.exists():
                issues.append('Broken active link: ' + key + ' -> ' + target)
            elif fragment and destination.suffix == '.md':
                destination = destination.resolve()
                if destination not in anchor_cache:
                    anchor_cache[destination] = markdown_anchors(destination)
                if unquote(fragment) not in anchor_cache[destination]:
                    issues.append('Broken heading link: ' + key + ' -> ' + target + '#' + fragment)
    state_paths = sorted((repo / 'docs/paper/figures').rglob('current-state.json'))
    state_paths += [repo / key for key in policy.get('additional_figure_states', [])]
    collection = repo / 'docs/paper/figures/main-paper-current-state.json'
    if collection.is_file():
        state_paths.append(collection)
    registry_path = repo / 'configs/paper-figures/behavior-paper.json'
    registry = json.loads(registry_path.read_text()).get('panels', {}) if registry_path.is_file() else {}
    for panel_id, panel in registry.items():
        selection = panel.get('scoped_selection_record')
        if selection and ('/history/' in selection or not (repo / selection).is_file()):
            issues.append('Invalid current scientific selection: ' + panel_id + ' -> ' + selection)
    for state_path in state_paths:
        if not state_path.is_file():
            issues.append('Missing figure state: ' + str(state_path.relative_to(repo)))
            continue
        state = json.loads(state_path.read_text())
        def resolved(value):
            return recorded_path(value, repo, aliases)
        entries = list(state.get('scientific_selection', {}).get('records', []))
        selected_paths = {resolved(item['path']) for item in entries}
        # A panel-only/exploratory state must not be compared with every panel
        # of its parent figure. Only the whole-figure summary owns that check.
        whole_figure = re.fullmatch(r'figure-(\d+)', state_path.parent.name)
        if whole_figure:
            figure_number = int(whole_figure[1])
            for panel_id, panel in registry.items():
                selection = panel.get('scoped_selection_record')
                if re.match(rf'^fig-{figure_number}[A-Za-z]', panel_id) and selection:
                    if resolved(selection) not in selected_paths:
                        issues.append('Scientific registry/state disagreement: ' + panel_id)
                    elif panel.get('scientific_definition'):
                        source = repo / selection
                        if source.is_file():
                            definition = json.loads(source.read_text()).get('scientific_definition')
                            if definition is not None and definition != panel['scientific_definition']:
                                issues.append('Scientific definition/selection disagreement: ' + panel_id)
            for mapping in state.get('draft', {}).get('condition_order', []):
                panel_id = f"fig-{figure_number}{mapping['display_panel']}"
                recorded = registry.get(panel_id, {}).get('presentation_mapping')
                if recorded is not None and any(recorded.get(k) != mapping.get(k)
                                                for k in ('display_panel', 'source_panel', 'condition', 'fish_id')):
                    issues.append('Presentation mapping/state disagreement: ' + panel_id)
        entries += list(state.get('preserved_figures', {}).values())
        # Direct bound objects are current requirements. Do not recursively
        # demand that historical provenance inside evidence match today's code.
        entries += [value for value in state.values()
                    if isinstance(value, dict) and 'path' in value and 'sha256' in value]
        frozen_entries = [value for value in state.get('frozen_display', {}).values()
                          if isinstance(value, dict) and 'path' in value and 'sha256' in value]
        entries += frozen_entries
        entries += [value for value in state.get('draft', {}).values()
                    if isinstance(value, dict) and 'path' in value and 'sha256' in value]
        for figure in state.get('figures', []):
            if not (repo / figure['state']).is_file():
                issues.append('Missing collection figure state: ' + figure['state'])
        for item in entries:
            path = resolved(item['path'])
            if path.is_relative_to(repo):
                if not path.is_file() or sha256(path) != item['sha256']:
                    # A frozen display may retain an old legend at a mutable
                    # logical path. Only exact recorded recovery is permitted;
                    # active draft edits must still fail on the direct bytes.
                    try:
                        if not any(item is entry for entry in frozen_entries):
                            raise ValueError('Active draft requires direct bytes')
                        resolve_artifact(path, expected_sha256=item['sha256'], repo=repo)
                    except (ValueError, FileNotFoundError):
                        issues.append('Bound figure record changed: ' + str(path))
            elif external:
                if not path.is_file():
                    unavailable.append(str(path))
                elif sha256(path) != item['sha256']:
                    issues.append('Bound external artifact changed: ' + str(path))
    verified_transactions = []
    dispositions = []
    for binding in policy.get('archive_dispositions', []):
        path = repo / binding['path']
        if (not path.resolve().is_relative_to(archive) or not path.is_file()
                or sha256(path) != binding['sha256']):
            issues.append('Archive disposition changed: ' + binding['path'])
        else:
            dispositions.append(json.loads(path.read_text()))
    if external:
        if not Path('/Volumes/JOAQUIM').is_mount():
            unavailable.append('JOAQUIM is not mounted; external verification incomplete')
        else:
            spec = importlib.util.spec_from_file_location('repository_archive_check', repo / 'scripts/maintenance/archive.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for manifest_path in [archive / 'transfer-inventory.json', *sorted(archive.glob('*/transfer-inventory.json'))]:
                manifest = json.loads(manifest_path.read_text())
                try:
                    audit = module.audit_payload(manifest, Path(manifest['archive_root']), dispositions=dispositions)
                    verified_transactions.append({'manifest': str(manifest_path.relative_to(repo)), **audit})
                except FileNotFoundError as error:
                    unavailable.append(str(error))
                except (ValueError, OSError) as error:
                    issues.append(str(error))
            locations = json.loads((archive / 'verified-artifact-locations.json').read_text())['entries']
            for entry in locations:
                try:
                    resolve_artifact(entry['references'][0], expected_sha256=entry['sha256'], repo=repo)
                except FileNotFoundError as error:
                    unavailable.append(str(error))
                except ValueError as error:
                    issues.append(str(error))
    return {'issues': sorted(set(issues)), 'unavailable_external_evidence': sorted(set(unavailable)),
            'protected_files_checked': len(checked), 'current_states_checked': len(state_paths),
            'verified_transactions': verified_transactions,
            'external_verification': 'not-requested' if not external else 'incomplete' if unavailable else 'passed' if not issues else 'failed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true')
    args = parser.parse_args()
    result = check(external=args.external)
    print(json.dumps(result, indent=2))
    return int(bool(result['issues'] or (args.external and result['unavailable_external_evidence'])))


if __name__ == '__main__':
    raise SystemExit(main())
