"""Revalidate exact render measurements transported between Git line endings.

This does not generate browser evidence or relax artwork/geometry checks. The
saved SVG must match exactly, and every source identity must match either the
current bytes or their LF/CRLF-only counterpart. Original reports stay intact.
"""
import copy
from pathlib import Path

from classical_conditioning.figure_workspace import sha
from classical_conditioning.figures import layout_frames


def source_equivalence(path, expected):
    path = Path(path)
    raw = path.read_bytes()
    actual = sha(raw)
    if actual == expected:
        kind = 'exact'
    else:
        raw.decode('utf-8')
        lf = raw.replace(b'\r\n', b'\n')
        if expected not in {sha(lf), sha(lf.replace(b'\n', b'\r\n'))}:
            raise ValueError('Source changed beyond line endings: ' + str(path))
        kind = 'LF/CRLF only'
    return {'path': str(path.resolve()), 'recorded_sha256': expected,
            'local_sha256': actual, 'equivalence': kind}


def revalidate_layout(svg, saved, source_reference):
    """Return a current assessment and explicit transport receipt, never a render."""
    if saved.get('status') != 'passed' or saved.get('issues'):
        raise ValueError('Saved evidence must already pass')
    if saved.get('artifact_sha256') != sha(svg):
        raise ValueError('Saved evidence belongs to different SVG bytes')
    paths = {'policy_sha256': layout_frames.POLICY,
             'collector_sha256': layout_frames.COLLECTOR,
             'helper_sha256': Path(layout_frames.__file__)}
    equivalence = {key: source_equivalence(path, saved[key])
                   for key, path in paths.items()}
    browser = copy.deepcopy(saved['browser'])
    # The source report and its measured arrays are retained. Only the collector
    # identity is transported, under the independently verified byte proof.
    if browser.get('collector_sha256') != saved['collector_sha256']:
        raise ValueError('Saved browser/collector identities disagree')
    browser['collector_sha256'] = equivalence['collector_sha256']['local_sha256']
    result = layout_frames.assess(svg, browser)
    for key in ('artifact_sha256', 'spec_sha256', 'scope', 'final_size_mm'):
        if result.get(key) != saved.get(key):
            raise ValueError('Saved rendering identity changed: ' + key)
    if result['status'] != 'passed':
        raise ValueError('Transported measurements fail current checks: ' + str(result['issues']))
    result['measurement_provenance'] = {
        'source_evidence': source_reference,
        'source_equivalence': equivalence,
        'new_browser_measurements': False,
        'meaning': 'Exact saved SVG measurements revalidated after source line-ending transport',
    }
    return result
