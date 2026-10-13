"""Read-only ownership, active-link, relocation and evidence checks.

Use --external to authenticate current external artifacts and archive payloads.
No rendering, transfer, pruning or scientific selection is performed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from classical_conditioning.external_artifacts import resolve_artifact, sha256

LINK = re.compile(r'(?<!!)\[[^\]]+\]\((<[^>]+>|[^)]+)\)')


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
    issues: list[str] = []
    unavailable: list[str] = []
    policy = json.loads((repo / 'docs/project-maintenance/structure-policy.json').read_text())
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
    for path in (repo / 'Plans').iterdir():
        if path.is_file() and path.name not in {'README.md', 'IMPLEMENTATION_STEP_INDEX.md'} and not re.fullmatch(r'\d{2}_[A-Z0-9_]+\.md', path.name):
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
            if not Path(key).is_absolute() and not (repo / key).is_file():
                issues.append('Missing relocation destination: ' + original + ' -> ' + key)
    index = json.loads((repo / 'records/frozen-analyses/index.json').read_text())
    main_manifest = repo / policy['migration_manifest']
    manifests = [main_manifest, *[p for p in sorted((repo / 'records/repository-history').glob('*/migration.json')) if p != main_manifest]]
    protected = list(index['files'])
    for manifest in manifests:
        protected.extend(json.loads(manifest.read_text()).get('protected_files', []))
    checked = set()
    for entry in protected:
        key = entry['path']
        if key in checked:
            continue
        checked.add(key)
        path = repo / key
        if not path.is_file() or sha256(path) != entry['sha256']:
            issues.append('Preserved bytes changed: ' + key)
    # Current links must work directly. Historical evidence is byte-preserved
    # and uses the resolver; its old relative links are not current guidance.
    documents = [repo / 'README.md', repo / 'AGENTS.md']
    for folder in ('docs', 'Plans'):
        documents.extend((repo / folder).rglob('*.md'))
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
            destination = (Path(target) if Path(target).is_absolute() else path.parent / target) if target else path
            if not destination.exists():
                issues.append('Broken active link: ' + key + ' -> ' + target)
            elif fragment and destination.suffix == '.md':
                destination = destination.resolve()
                if destination not in anchor_cache:
                    anchor_cache[destination] = markdown_anchors(destination)
                if unquote(fragment) not in anchor_cache[destination]:
                    issues.append('Broken heading link: ' + key + ' -> ' + target + '#' + fragment)
    state_paths = list((repo / 'docs/paper/figures').glob('figure-*/current-state.json'))
    registry_path = repo / 'configs/paper-figures/behavior-paper.json'
    registry = json.loads(registry_path.read_text()).get('panels', {}) if registry_path.is_file() else {}
    for panel_id, panel in registry.items():
        selection = panel.get('scoped_selection_record')
        if selection and ('/history/' in selection or not (repo / selection).is_file()):
            issues.append('Invalid current scientific selection: ' + panel_id + ' -> ' + selection)
    for state_path in state_paths:
        state = json.loads(state_path.read_text())
        entries = list(state['scientific_selection']['records'])
        selected_paths = {Path(item['path']).resolve() for item in entries}
        figure_number = int(state_path.parent.name.removeprefix('figure-'))
        for panel_id, panel in registry.items():
            selection = panel.get('scoped_selection_record')
            if panel_id.startswith(f'fig-{figure_number}') and selection and (repo / selection).resolve() not in selected_paths:
                issues.append('Scientific registry/state disagreement: ' + panel_id)
        entries += list(state.get('preserved_figures', {}).values())
        for scope in ('frozen_display', 'draft'):
            entries += [state[scope][key] for key in ('working_svg', 'review', 'effective_presentation') if state.get(scope, {}).get(key)]
        for item in entries:
            path = Path(item['path'])
            if path.is_relative_to(repo):
                if not path.is_file() or sha256(path) != item['sha256']:
                    issues.append('Bound figure record changed: ' + str(path))
            elif external:
                if not path.is_file():
                    unavailable.append(str(path))
                elif sha256(path) != item['sha256']:
                    issues.append('Bound external artifact changed: ' + str(path))
    verified_transactions = []
    if external:
        if not Path('/Volumes/JOAQUIM').is_mount():
            unavailable.append('JOAQUIM is not mounted; external verification incomplete')
        else:
            spec = importlib.util.spec_from_file_location('repository_archive_check', repo / 'scripts/repository_archive.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for manifest_path in [archive / 'transfer-inventory.json', *sorted(archive.glob('*/transfer-inventory.json'))]:
                manifest = json.loads(manifest_path.read_text())
                try:
                    count = module.verify_payload(manifest, Path(manifest['archive_root']))
                    verified_transactions.append({'manifest': str(manifest_path.relative_to(repo)), 'files': count})
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
