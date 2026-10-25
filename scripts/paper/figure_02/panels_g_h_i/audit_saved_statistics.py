"""Audit explicit hash-bound saved G/H/I results and build an SSD review.

No model fitting, new resampling, figure edits, cohort selection or freezing.
"""
from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src')]

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classical_conditioning.analysis.review.saved_statistics_audit import audit_tables, annotation_labels, coefficient_rows, bootstrap_draws
from classical_conditioning.artifacts import artifact_staging, publish_transaction
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, sha, state_lock


def build(manifest_path, output, helper, helper_sha256):
    manifest_path, output, helper = map(lambda p: Path(p).resolve(), (manifest_path, output, helper))
    if output.is_relative_to(_COMMAND_REPOSITORY) or not output.parent.is_dir() or output.exists():
        raise ValueError('Use a new external output directory with an existing parent')
    if output.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():
        raise FileNotFoundError('JOAQUIM unavailable')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    snapshot = Snapshot([*command_sources(__file__), manifest_path, helper])
    snapshot.hashes[str(manifest_path)] = sha(raw)
    if snapshot.hashes[str(helper)] != helper_sha256:
        raise ValueError('Personal notebook helper identity differs')

    def bound(ref):
        path = Path(ref['path']).resolve()
        raw = path.read_bytes()
        if sha(raw) != ref['sha256'] or snapshot.hashes.setdefault(str(path), sha(raw)) != sha(raw):
            raise ValueError('Audit source identity differs: ' + str(path))
        return path

    ticket = json.loads(bound(manifest['starting_ticket']).read_bytes())
    snapshot.hashes.update(ticket['protected'])
    snapshot.check()
    state = json.loads(bound(manifest['selected_state']).read_bytes())
    if state['figure_id'] != 'fig2' or state['draft']['working_svg'] != manifest['selected_svg']:
        raise ValueError('Audit manifest differs from selected Figure 2')
    for ref in manifest['authority_sources']:
        bound(ref)
    svg = bound(manifest['selected_svg']).read_bytes()
    reports = {}
    for panel, specification in manifest['panels'].items():
        artifacts = {name: bound(ref) for name, ref in specification['artifacts'].items()}
        def csv(name): return pd.read_csv(artifacts[name])
        data = pd.read_parquet(artifacts[specification['model_input']])
        observed = pd.read_parquet(artifacts[specification['trajectory']])
        dtests, local, trials = [csv(name) for name in ['D-interaction-tests.csv', 'MR-local-block-tests.csv', 'phase-contrasts.csv']]
        diag = json.loads(artifacts['phase-diagnostics.json'].read_bytes())
        report = audit_tables(data, observed, dtests, local, trials, diag, csv('phase-coefficients.csv'),
                              pd.read_csv(artifacts['phase-fixed-covariance.csv'], index_col=0),
                              pd.read_parquet(artifacts['phase-residuals.parquet']))
        local_coefficients = {row.block_10_name: csv(row.block_10_name.replace(' ', '-') + '-local-coefficients.csv')
                              for row in local.itertuples() if np.isfinite(row.mean_p_raw)}
        report['saved_coefficient_rows_checked'] = coefficient_rows(dtests, local, csv('block-coefficients.csv'), local_coefficients)
        report['displayed_annotations'] = annotation_labels(svg, panel, dtests, local, trials)
        arrays = {}
        for condition, source in specification['bootstrap'].items():
            with np.load(artifacts[source['file']], allow_pickle=False) as saved:
                arrays[condition] = tuple(saved[source[k]] for k in ['fish_ids', 'indices', 'medians'])
        report['bootstrap'] = bootstrap_draws(data, observed, arrays)
        report['saved_diagnostics'] = {name: json.loads(path.read_bytes()) for name, path in artifacts.items()
                                       if name.endswith('-diagnostics.json') and name != 'all-fit-diagnostics.json'}
        report['response_window_s'] = specification['response_window_s']
        report['scientific_status'] = specification['scientific_status']
        report['source_gaps'] = specification['source_gaps']
        reports[panel] = report
    result = {'scope': 'Selected Figure 2 G/H/I; read-only saved-result arithmetic and XML association audit',
              'status': 'saved arithmetic and named annotations agree; scientific and provenance limits remain',
              'selected_state': manifest['selected_state'], 'selected_svg': manifest['selected_svg'], 'panels': reports,
              'no_models_refitted': True, 'no_new_resamples': True, 'no_selection_or_figure_changes': True,
              'coverage_limits': ['Does not re-extract raw tracking or approve cohort completeness.',
                  'XML labels and marker counts do not establish painted visibility or precise marker locations.',
                  'No independent biological validation, learning-onset/extinction claim, or production certification.']}

    sections = []
    def section(identity, title, paragraphs, headers=None, rows=None):
        sections.append((identity, title, paragraphs, headers, rows))
    section('orientation', 'What was checked', [
        'Start with the outcome distinction, compare the exact saved settings, then inspect the audit results and remaining gaps. This report checks the selected Figure 2 bytes; it selects no scientific alternative.',
        'The saved fish medians, contributing counts, Holm8/BH9/BH90 corrections, fixed-effect phase contrasts and intervals agree with their saved inputs. Named SVG statistical text and trial-marker counts agree with those results. No model was refitted and no new bootstrap indices were drawn.',
        'G retains its scoped accepted freeze. H and I remain unfrozen scientific drafts. Successful arithmetic and convergence are engineering evidence; model adequacy, biological interpretation and paper-wide eligibility remain separate.'])
    section('outcomes', 'What the curves and stars measure', [
        'The metric is tail-bend angular speed. In each window, take the median natural logarithm of finite positive valid detected-bout frames. Subtract the baseline log median from the response log median within fish/trial, then take the median across contributing fish. Baseline is [−15,0) s; response is [0,9), [0,13) and [0,20) s for G, H and I. Empty eligible windows are undefined, not zero activity.',
        'This describes movement intensity conditional on valid detected bouts, not movement probability or zero-inclusive activity. The pointwise 95% curve intervals use 5,000 saved whole-fish resamples within condition. The audit replayed those exact indices and checked their percentile endpoints; it did not create new samples.',
        'The statistical models instead fit the response-window log median with baseline log median as a covariate. Baseline subtraction in the displayed outcome and baseline adjustment in the model are different operations. Model contrasts target fixed-effect population means; descriptive medians and their bootstrap intervals do not supply the p-values behind the stars.',
        'D tests condition-by-block terms relative to Pre (Holm8). M tests local condition differences at centred trial (BH9). R tests local slope differences, raw and BH9. Trial stars test control-minus-experimental change relative to average Pre5–14 in the phase-aware model (BH90). Trial stars mark rejection at .05; the number of trial symbols does not encode a p-value tier.'])
    rows = []
    for panel, report in reports.items():
        for name in ['block-diagnostics.json', 'phase-diagnostics.json', 'Train-1-local-diagnostics.json']:
            diag = report['saved_diagnostics'][name]
            rows.append([panel, name.replace('-diagnostics.json', ''), diag['optimizer'], diag['used_random_effects'],
                         diag['observation_count'], str(diag['fallback_used']), diag['formula']])
    section('settings', 'Actual saved model settings', [
        'These entries come from run diagnostics, not current CLI defaults. The full audit JSON preserves every available model diagnostic, including local failures and sensitivity fits. The categorical block model uses condition-by-block fixed effects; the phase model uses condition-by-phase and schedule-defined Pre/Train/Test bases; local models use condition-by-centred-trial terms.',
        'Random intercepts/slopes describe fish departures from fixed population effects. The primary block and phase fits use correlated fish intercept/slope terms; local fits use fish intercepts. The source-bound H/I fitter explicitly fits maximum likelihood (reml=False). No explicit maximum-iteration setting is recorded; the historical library/runtime default is unresolved. Do not infer that value from the current environment.',
        'Positive trial-change contrast means a lower experimental-condition response relative to control and the Pre difference. D/local coefficient signs use experimental-minus-control; read each row definition before comparing directions.'],
        ['Panel', 'Model', 'Optimizer', 'Random structure', 'Rows', 'Fallback', 'Fixed formula'], rows)
    rows = []
    for panel, report in reports.items():
        counts = report['significant_counts']
        rows.append([panel, str(report['fish']), report['fitted_observations'], report['scheduled_observations'],
                     *[counts[k] for k in ['D_Holm8', 'M_BH9', 'R_raw', 'R_BH9', 'trial_BH90']],
                     f"{report['residual_excess_kurtosis']:.3f}", ', '.join(report['unavailable_local_blocks']) or 'none'])
    section('results', 'Verified results and diagnostics', [
        'Counts below reproduce saved statistical decisions, not numbers of learners or acquisition/extinction times. A non-rejection does not demonstrate absence or equivalence. Denominators distinguish scheduled fish/trials from finite fitted observations.',
        'All primary phase fits report convergence, full-rank fixed design, no singular covariance and no random-intercept fallback. Boundary warnings remain in the saved records. Residual excess kurtosis is large in all three panels; matching p-value arithmetic does not establish Gaussian-tail adequacy or calibrated simultaneous timing claims.'],
        ['Panel', 'Fish by condition', 'Fitted rows', 'Scheduled rows', 'D Holm8', 'M BH9', 'R raw', 'R BH9', 'Trial BH90', 'Tail excess kurtosis', 'Unavailable locals'], rows)
    gaps = [panel + ': ' + gap for panel, report in reports.items() for gap in report['source_gaps']]
    section('limits', 'Open evidence and decisions', [
        'Historical G data were recovered byte-for-byte from the authenticated archive; source aliases now resolve them without choosing another analysis. Exact H/I driver-source recovery is incomplete. Formula, coefficients, covariance, residuals and decisions are available and independently reconciled; a complete code/runtime reproduction is not established.',
        *gaps,
        'For I, the bootstrap draw file was snapshotted during this audit and replays correctly against the selected trajectory; no earlier bound producer hash was found for that draw file. Its current audit identity must not be presented as proof of its original producer lineage.',
        'Next scientific work belongs to the existing input/cohort and analysis plans: paper-wide denominator and technical membership, model robustness/calibration, scientifically meaningful contrasts, and any separately defined extinction question. Testing recovery and disappearing stars do not themselves validate extinction.',
        'Notebook notes are provisional. They cannot approve a cohort, select a method, freeze a figure or authorize changes. Browser notebook behavior, responsive layout and printing were not exercised in this run.'])
    source_rows = [[panel, len(specification['artifacts']), specification['scientific_status'], str(specification['response_window_s'])]
                   for panel, specification in manifest['panels'].items()]
    section('sources', 'Source identities and reproduction', [
        'audit.json records the numerical checks and exact diagnostics; guide-provenance.json binds every consumed file, implementation and protected state. The input manifest connects these files to the frozen G manifest and selected H/I completion/registry records. study.md is the editable explanatory source; study.html is its self-contained reading version.',
        'Reproduce with scripts/paper/figure_02/panels_g_h_i/audit_saved_statistics.py using the authenticated input manifest, personal notebook helper and a new external output directory. Existing manuscripts, reading files and notebook storage are never overwritten.'],
        ['Panel', 'Authenticated/snapshotted artifacts', 'Scientific status', 'Response seconds'], source_rows)
    manuscript = '# Figure 2 G/H/I — saved statistical audit\n\n'
    html_sections = []
    for identity, title, paragraphs, headers, rows in sections:
        manuscript += '## ' + title + '\n\n' + '\n\n'.join(paragraphs) + '\n\n'
        body = ''.join('<p>' + html.escape(p) + '</p>' for p in paragraphs)
        if headers:
            manuscript += '| ' + ' | '.join(headers) + ' |\n|' + '|'.join('---' for _ in headers) + '|\n'
            manuscript += '\n'.join('| ' + ' | '.join(str(v).replace('|', '\\|') for v in row) + ' |' for row in rows) + '\n\n'
            body += '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + html.escape(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'
        notebook = '<details class="notebook no-print" data-section="' + identity + '"><summary>Your review notebook</summary><label><input type="checkbox" data-field="understood">I can explain this</label><label>Review position<select data-field="position"><option value="unreviewed">Not reviewed</option><option value="questions">Questions remain</option><option value="tentative">Provisional preference</option><option value="ready">Ready for discussion</option></select></label>' + ''.join('<label>' + title + '<textarea data-field="' + field + '"></textarea></label>' for title, field in [('My understanding', 'understanding'), ('Open questions', 'questions'), ('Provisional decision and reason', 'decision')]) + '<span data-local-status></span></details>'
        html_sections.append('<section class="chapter" id="' + identity + '"><h2>' + html.escape(title) + '</h2>' + body + notebook + '</section>')
    config = {'guide_id': 'classical-conditioning-figure2-ghi-statistical-audit', 'guide_version': '20261025',
              'source_fingerprint': sha(raw + manuscript.encode()), 'sections': [{'id': s[0], 'title': s[1]} for s in sections]}
    controls = '<input id="guide-search" type="search" aria-label="Find a topic"><div id="search-results" aria-live="polite"></div><details id="chapter-menu" open><summary>Reading path</summary><nav>' + ''.join('<a href="#' + s[0] + '">' + s[1] + '</a> ' for s in sections) + '</nav></details><div aria-hidden="true"><span id="progress-fill"></span></div><button id="export-notes">Export notes</button><button id="import-notes">Import notes</button><button id="print-guide">Print</button><input id="import-file" type="file" accept=".json,application/json" hidden><p id="save-status" role="status"></p><p id="import-message" role="status"></p>'
    css = 'body{font:17px/1.6 system-ui;color:#18343c;background:#f6f5ef;margin:auto;max-width:1200px;padding:24px}section{background:white;padding:24px;margin:24px 0}a{color:#145e65}nav{display:flex;gap:12px;flex-wrap:wrap}.table-wrap{overflow-x:auto;max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #bbb;min-width:95px}textarea{display:block;width:95%;min-height:85px}label{display:block;margin:12px 0}button,input,select{font:inherit}button{margin:6px}.active,.search-hit{outline:2px solid #27726e}pre{white-space:pre-wrap;overflow-wrap:anywhere}dialog{max-width:750px}@media(max-width:650px){body,section{padding:12px}h1{font-size:1.6rem}}@media print{body{background:white;font-size:10pt}button,input,nav,.no-print,#chapter-menu,dialog{display:none!important}section{padding:0}.table-wrap{overflow:visible}table{font-size:6pt;width:100%;table-layout:fixed}td,th{min-width:0;padding:3px;overflow-wrap:anywhere}tr{break-inside:avoid}h2{break-after:avoid}}'
    dialog = '<dialog id="import-dialog"><h2>Import review notes</h2><p id="import-warning"></p><pre id="import-preview"></pre><p>Existing notes will be backed up before replacement.</p><button id="confirm-import">Replace notes with this import</button><button id="cancel-import">Cancel</button></dialog>'
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Figure 2 G/H/I statistical audit</title><style>' + css + '</style><header><h1>Figure 2 G/H/I statistical audit</h1><p>Saved-result reconciliation · no new fit or scientific selection</p>' + controls + '</header><main>' + ''.join(html_sections) + '</main>' + dialog + '<script type="application/json" id="guide-config">' + json.dumps(config).replace('<', '\\u003c') + '</script><script>' + helper.read_text() + '</script></html>'
    result_bytes = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    protected = [{'path': p, 'sha256': h} for p, h in ticket['protected'].items()]
    sources = [{'path': p, 'sha256': h} for p, h in snapshot.hashes.items() if p not in ticket['protected']]
    provenance = {'guide': {'path': str(output / 'study.html'), 'sha256': sha(page.encode())},
                  'manuscript': {'path': str(output / 'study.md'), 'sha256': sha(manuscript.encode())},
                  'audit': {'path': str(output / 'audit.json'), 'sha256': sha(result_bytes)},
                  'sources': sources, 'protected': protected,
                  'browser_behavior': 'Not exercised; exact personal notebook helper inlined, static checks required',
                  'scientific_authority_changed': False}
    with artifact_staging(output.parent, prefix='.ghi-audit-') as staging:
        bundle = staging / 'report'
        bundle.mkdir()
        for name, payload in [('audit.json', result_bytes), ('study.md', manuscript.encode()), ('study.html', page.encode()),
                              ('guide-provenance.json', (json.dumps(provenance, indent=2) + '\n').encode())]:
            (bundle / name).write_bytes(payload)
        expected = {p.name: sha(p.read_bytes()) for p in bundle.iterdir()}
        def verify():
            snapshot.check()
            for name, digest in expected.items():
                if sha((output / name).read_bytes()) != digest:
                    raise ValueError('Saved audit artifact differs: ' + name)
        with state_lock(manifest['selected_state']['path']):
            snapshot.check()
            publish_transaction([(bundle, output)], staging, overwrite=False, verify_published=verify)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--notebook-script', type=Path, required=True)
    parser.add_argument('--notebook-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output_dir, args.notebook_script, args.notebook_sha256)['guide']))


if __name__ == '__main__':
    main()
