"""Storage contract for the historical cross-metric comparison, not selection."""
from pathlib import Path
import hashlib
import json
import os
from classical_conditioning.external_artifacts import external_output, relocated_external_directory, REPOSITORY

REVIEW = 'outputs/pre-vs-late-train-early-test-review'
RECEIPT = 'metric-comparison-inputs.json'


def workspace(value=None):
    path = (relocated_external_directory(Path(value).expanduser()).resolve()
            if value is not None else external_output(REVIEW).resolve())
    if path == REPOSITORY or path.is_relative_to(REPOSITORY):
        raise ValueError('Metric review outputs must be outside the repository')
    if str(path).startswith('/Volumes/'):
        volume = Path(*path.parts[:3])
        if not os.path.ismount(volume):
            raise FileNotFoundError(f'Output volume unavailable: {volume}')
    if not path.parent.is_dir():
        raise FileNotFoundError(f'Output parent unavailable: {path.parent}')
    return path


def require_new_outputs(root, filenames):
    existing = [name for name in filenames if (root / name).exists()]
    if existing:
        raise FileExistsError('Preserve existing review outputs; choose a new directory: ' + ', '.join(existing))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_inputs(root):
    names = ['fish_window_summary.csv', 'fish_paired_change.csv']
    receipt = {'role': 'Historical descriptive comparison; no paper selection',
               'files': {name: file_sha(root / name) for name in names}}
    with (root / RECEIPT).open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')


def verified_input(root, expected_sha256=None):
    path = root / 'fish_paired_change.csv'
    expected = expected_sha256
    if expected is None:
        receipt = json.loads((root / RECEIPT).read_text(encoding='utf-8'))
        for name, value in receipt['files'].items():
            if Path(name).name != name or file_sha(root / name) != value:
                raise ValueError('Metric input receipt mismatch: ' + name)
        expected = receipt['files'][path.name]
    if file_sha(path) != expected:
        raise ValueError('Metric input SHA-256 mismatch: ' + str(path))
    return path
