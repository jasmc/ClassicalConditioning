"""Build an SSD-owned study of a completed BLUP comparison, without analysis."""
from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src')]

import argparse
import base64
import html
import json
import re
from pathlib import Path
import pandas as pd
from classical_conditioning.analysis.inference.panel_g_run import verify_stage
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, sha


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join('---' for _ in headers) + '|\n' + '\n'.join(
        '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows)


def number(value):
    return f'{value:.4f}' if pd.notna(value) else 'undefined'


def render_markdown(text):
    """Small source-preserving renderer for this guide's paragraphs/tables/code."""
    blocks = []
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        if block.startswith('```'):
            blocks.append('<details><summary>Implementation excerpt</summary><pre>' + html.escape('\n'.join(lines[1:-1])) + '</pre></details>')
        elif block.startswith('|'):
            rows = [[c.strip() for c in line.strip('|').split('|')] for line in lines]
            blocks.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in rows[0]) + '</tr></thead><tbody>' + ''.join(
                '<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in row) + '</tr>' for row in rows[2:]) + '</tbody></table></div>')
        elif block.startswith('- '):
            blocks.append('<ul>' + ''.join('<li>' + html.escape(line[2:]) + '</li>' for line in lines) + '</ul>')
        else:
            blocks.append('<p>' + html.escape(block).replace('\n', '<br>') + '</p>')
    return ''.join(blocks)


def build(analysis, output, helper, helper_sha256, manuscript_path=None, manuscript_sha256=None,
          guide_version='20261025-v01'):
    analysis, output, helper = analysis.resolve(), output.resolve(), helper.resolve()
    if output.is_relative_to(_COMMAND_REPOSITORY):
        raise ValueError('Study manuscripts and HTML belong outside the repository')
    if output.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():
        raise FileNotFoundError('JOAQUIM unavailable')
    if not output.parent.is_dir() or output.exists():
        raise FileExistsError('Use a new guide directory; preserve existing notes/manuscripts')
    helper_raw = helper.read_bytes()
    if sha(helper_raw) != helper_sha256:
        raise ValueError('Review notebook helper identity mismatch')
    manifest = analysis / 'artifact-manifest.json'
    manifest_sha = sha(manifest.read_bytes())
    verify_stage(analysis, manifest_sha)
    paths = [*command_sources(__file__), helper, manifest, *[analysis / p for p in
             ['result.json', 'fish-comparison.csv', 'saved-legacy-comparison.csv', 'analysis-config.json']]]
    protected = list((_COMMAND_REPOSITORY / 'docs/paper/figures').rglob('*state.json'))
    if (manuscript_path is None) != (manuscript_sha256 is None):
        raise ValueError('A revised manuscript requires its exact SHA-256')
    if manuscript_path is not None:
        manuscript_path = manuscript_path.resolve()
        paths.append(manuscript_path)
    snapshot = Snapshot([*paths, *protected])
    result = json.loads((analysis / 'result.json').read_text())
    fish = pd.read_csv(analysis / 'fish-comparison.csv')
    legacy = pd.read_csv(analysis / 'saved-legacy-comparison.csv')
    validation = result['validation']
    cond = result['conditions']
    scores = result['positive_model_evidence']
    sections = []
    def section(identity, title, body):
        sections.append(dict(id=identity, title=title, markdown=body.strip()))
    section('orientation', 'Read this first', f'''
This is the completed, exploratory corrected-BLUP comparison requested in the Figure 3 discussion. It is a separate study, not the selected main-paper Figure 3 and not a scientific freeze. The selected figures and cohort were preserved.

Start with the outcome definition, then read the validation table before inspecting individual estimates. Finally compare the legacy rules and record questions in the notebook. More positive flags or fewer exclusions do not identify the most accurate method.

The analysis uses {len(fish)} saved Delay/control fish: {cond.get('delay', 0)} Delay and {cond.get('control', 0)} Control. All input fish remain represented. Finite outcomes in both epochs make a contrast estimable; they do not prove enough precision for a biological classification.
''')
    section('outcome', 'Outcome, sign and scheduled trials', '''
The physical metric is the selected legacy tail-bend angular speed. Each trial's Figure 2 G outcome is median ln(positive valid detected-bout response vigor) minus median ln(positive valid detected-bout baseline vigor). This applies one logarithm, not a logarithm of a ratio of arithmetic means.

For Delay, the response is [0,9) s and baseline is [−15,0) s relative to CS onset. No-bout vigor is undefined, not zero. This outcome describes conditional movement intensity; it cannot establish movement probability or whole-session activity.

Acquisition compares Pre trials 5–14 with late-Train trials 40–64. The score is Pre minus late Train, so a positive value means lower conditional response vigor during late training. Test observations are not used to construct this score. Test was previously inspected, so it is not retrospectively called an untouched prospective holdout.

Recovery is reported separately as late-Test 85–94 minus late-Train 40–64. It is descriptive here, with a plug-in standard error, not a second requirement for being an acquisition responder. Scheduled gaps and the observed missingness pattern are retained.
''')
    model = result['primary_model']
    section('model', 'What the BLUP estimates', f'''
The model jointly estimates condition-by-epoch fixed means, correlated fish-specific random epoch means, different Control/Delay residual variances and within-epoch scheduled-gap AR1 residual correlation. Random epoch means reparameterize a random intercept and epoch slope; they are not an additional biological trajectory.

The fish estimate is the fixed contrast plus its random deviation. Reporting only the random deviation would answer how much the fish departs from its condition mean, rather than its total predicted change. The excess estimate subtracts the jointly estimated Control fixed change. Its plug-in error formula includes the shared-reference covariance.

Shrinkage pulls poorly measured fish toward their condition's fitted change. That can reduce noise under the model while concealing model misspecification; it does not create observations or guarantee individual truth. A BLUP is not a probability that a fish learned or an observed individual causal effect.

The fitted model converged, boundary={model['boundary']}, using {model['nobs']} finite epoch observations. Estimated residual AR1 correlation is {model['phi']:.4f}. An independent-residual fit is included as a sensitivity, not selected automatically.
''')
    rows = []
    for row in validation:
        rows.append([row['scenario'], f"{row['successful_refits']}/{row['attempted']}",
                     number(row['total_pointwise_coverage']), number(row['total_coverage_mcse']),
                     number(row['excess_pointwise_coverage']), number(row['family_lower_coverage']),
                     number(row['boundary_fraction'])])
    section('validation', 'Calibration and independent stress checks', f'''
Calibration used {result['calibration_successful']} successful refits out of {result['calibration_attempted']} requested datasets, with seed {result['config']['seed']}. Refitted parametric prediction errors supply fish-specific pointwise 95% intervals and a one-sided simultaneous lower bound for the 57-fish excess family. These have different coverage targets.

Each validation scenario uses a new random stream, disjoint from calibration. Matched draws follow the fitted Gaussian/AR1 model; heavy_tail uses standardized t(5) innovations; within_epoch_trend adds the implemented 0.15-unit drift across an epoch; null removes acquisition effects. All retain the observed trial-availability design. Refit failures and boundary solutions are recorded rather than hidden.

{table(['Scenario','Refits','Total pointwise','MCSE','Excess pointwise','Family lower','Boundary fraction'], rows)}

Pointwise coverage targets 0.95 across simulated fish; family-lower coverage targets 0.95 for all excess contrasts jointly. Monte Carlo error is based on independent simulation datasets, retaining within-dataset fish dependence. About 499 datasets cannot establish exact tail probabilities. Failure under a scenario limits the supported claim; fitting successfully is not validation.

The null excess-family-bound false-positive rate is {next(r['null_any_false_positive_rate'] for r in validation if r['scenario']=='null'):.4f}. This is rejection by the excess bound alone, before the additional positive-total requirement. It is simulated model-based evidence, not an empirical biological false-positive rate. These checks do not validate informative missingness, arbitrary nonlinear learning trajectories, raw preprocessing or other assays.
''')
    evidence_rows = [[r['condition_id'], int(r['sum']), int(r['count'])] for r in scores]
    section('evidence', 'Directional model evidence, not learner categories', f'''
A positive model-evidence flag requires both a positive total acquisition estimate and an excess-family lower bound above zero. It is a model-based directional criterion with no approved biologically meaningful margin. A failure to flag is uncertainty or insufficient evidence, not proof of nonlearning.

{table(['Condition','Positive model evidence','All input fish'], evidence_rows)}

No classifier has been selected. The family bound covers all 57 input fish, including controls, under the stated calibration model. Do not rename these counts learners/nonlearners, optimize them as an accuracy target or transfer them into Figures 1/2. Final representation, control calibration and permitted claims remain for joint scientific review.
''')
    fish_rows = [[r.fish_id, r.condition_id, f'{r.epoch0_n}/{r.epoch1_n}', number(r.raw_mean_contrast),
                  number(r.raw_median_contrast), number(r.acquisition),
                  f'[{number(r.total_lower)}, {number(r.total_upper)}]', number(r.excess_acquisition),
                  number(r.excess_family_lower), number(r.no_ar1_acquisition),
                  number(r.recovery_late_test_minus_late_train)] for r in fish.itertuples()]
    section('fish', 'All fish and matched comparisons', f'''
Every input fish appears below. Pre/Train counts are finite trial outcomes, not a primary-cohort exclusion rule. Compare the unpooled mean with the BLUP for a closer mean-target comparison. The unpooled median targets epoch medians, so a difference is not solely an effect of shrinkage.

The pointwise interval is the calibrated total-acquisition interval. The excess lower bound is simultaneous across fish; it is not interchangeable with that pointwise interval. Recovery is late Test minus late Train and is descriptive. No row has been dropped because its response looks unfavorable.

{table(['Fish','Condition','Pre/Train n','Raw mean','Raw median','BLUP total','Total 95% interval','Excess','Excess family lower','No-AR1 total','Recovery'], fish_rows)}
''')
    legacy_rows = []
    for (variant, condition), group in legacy.groupby(['variant', 'condition_id'], sort=True):
        decisions = group.decision.value_counts().to_dict()
        legacy_rows.append([variant, condition, len(group), json.dumps(decisions, sort_keys=True)])
    section('legacy', 'The four unchanged legacy comparators', f'''
These are the saved reapplications of nominal/new/improved/WIP rules to the same metric's pilot data. Their original decisions are copied exactly; none was rerun or silently repaired for this study.

{table(['Rule','Condition','Inventory rows','Saved decisions'], legacy_rows)}

The legacy rules use different transformations, epoch windows, eligibility and uncertainty, and require acquisition plus recovery. Some middle windows include early Test. Their labels therefore do not isolate an algorithmic comparison with a training-only G contrast, and their Test curves are not held-out confirmation. They are historical comparators, not ground-truth labels.

The corrected model separates total change, random deviation and the shared Control reference. Neither a higher legacy count nor a narrower BLUP interval proves better biological classification. Inspect disagreements and the data behind them before choosing an interpretation.
''')
    section('limits', 'What remains unvalidated', '\n\n'.join(result['limitations']) + '''

The no-AR1 estimates quantify one residual-model sensitivity; they do not test every covariance structure. Recovery's plug-in standard errors do not carry the acquisition bootstrap calibration. The complete inventory remains visible, but sparse fish can still have poor precision.

Population conditioning, individual expression, causal learning and discrete learner classes are different claims. Current results support inspection of continuous conditional-vigor expression. They do not by themselves approve paper-wide membership, a new scientific freeze, independent new-cohort validation or Figure 4 learner-stratified inference.
''')
    section('decision', 'Decisions to make together', '''
- Choose the scientific target first: continuous acquisition expression, excess relative to the control reference, acquisition-plus-recovery phenotype, or justified categories.
- Review calibration and stress failures before deciding which uncertainty statements are defensible. A threshold cannot repair failed coverage.
- Decide what adequate per-fish information and a meaningful effect mean, without using favorable responses to define primary cohort membership.
- Compare like estimands and inspect discordant/sparse examples. Preserve unclassified or inconclusive cases explicitly.
- After an explicit choice, record its exact scope and evidence in the existing decision register; then update the single-path main Figure 3. Figure 4 grouping must follow that choice and its validation limits.

Notebook entries below are provisional study notes. They do not change repository state, approve science, select a cohort or trigger publication.
''')
    manuscript = '# Corrected BLUP comparison — exploratory review\n\n' + '\n\n'.join('## ' + s['title'] + '\n\n' + s['markdown'] for s in sections) + '\n'
    if manuscript_path is not None:
        revised = manuscript_path.read_bytes()
        if sha(revised) != manuscript_sha256:
            raise ValueError('Revised manuscript identity mismatch')
        manuscript = revised.decode()
        chapters = re.split(r'^## ', manuscript, flags=re.M)[1:]
        if len(chapters) != len(sections):
            raise ValueError('Preserve the stable study section identities')
        for section, chapter in zip(sections, chapters):
            title, body = chapter.split('\n', 1)
            if title.strip() != section['title']:
                raise ValueError('Changed section meaning requires an explicit guide revision')
            section['markdown'] = body.strip()
    fingerprint = sha(manifest.read_bytes() + manuscript.encode() + helper_raw)
    config = dict(guide_id='classical-conditioning-figure3-corrected-blup', guide_version=guide_version,
                  source_fingerprint=fingerprint, sections=[{k: s[k] for k in ['id', 'title']} for s in sections])
    css = '''body{margin:0;color:#192d37;background:#f7f5ef;font:17px/1.65 system-ui}header,main{max-width:1120px;margin:auto;padding:24px}header{border-bottom:2px solid #236766}nav{display:flex;gap:12px;flex-wrap:wrap}a{color:#145e65}section{background:white;margin:24px 0;padding:24px;border-radius:8px}h1,h2{line-height:1.2}.table-wrap{max-width:100%;overflow-x:auto}table{border-collapse:collapse;font-size:14px;min-width:700px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#e8efed}textarea{width:95%;min-height:90px}label{display:block;margin:12px 0}button,input,select{font:inherit}button{margin:5px;padding:6px 10px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.active,.search-hit{outline:2px solid #27726e}dialog{max-width:750px}#progress-fill{display:block;height:4px;background:#27726e;width:0}@media(max-width:650px){header,main{padding:12px}section{padding:16px}h1{font-size:1.6rem}}@media print{body{background:white;font-size:10pt}header{padding:0}.no-print,nav,button,input,dialog,#chapter-menu{display:none!important}section{padding:0;margin:15pt 0;break-inside:auto}.table-wrap{overflow:visible}table{font-size:6pt;min-width:0;width:100%}td,th{padding:3px;overflow-wrap:anywhere}tr{break-inside:avoid}tr[hidden]{display:table-row}h2{break-after:avoid}a{color:black}}'''
    nav = ''.join('<a href="#' + s['id'] + '">' + html.escape(s['title']) + '</a>' for s in sections)
    chunks = []
    for s in sections:
        identity = s['id']
        notebook = '<details class="notebook no-print" data-section="' + identity + '"><summary>Your review notebook</summary><label><input type="checkbox" data-field="understood">I can explain this without AI</label><label>Review position<select data-field="position"><option value="unreviewed">Not reviewed</option><option value="questions">Questions remain</option><option value="tentative">Provisional preference</option><option value="ready">Ready for discussion</option></select></label><label>My understanding<textarea data-field="understanding"></textarea></label><label>Open questions<textarea data-field="questions"></textarea></label><label>Provisional decision and reason<textarea data-field="decision"></textarea></label><span data-local-status></span></details>'
        chunks.append('<section class="chapter" id="' + identity + '"><h2>' + html.escape(s['title']) + '</h2>' + render_markdown(s['markdown']) + notebook + '</section>')
    controls = '<input id="guide-search" type="search" aria-label="Find a topic"><div id="search-results" aria-live="polite"></div><details id="chapter-menu" open><summary>Reading path</summary><nav>' + nav + '</nav></details><div aria-hidden="true"><span id="progress-fill"></span></div><button id="export-notes">Export notes</button><button id="import-notes">Import notes</button><button id="print-guide">Print</button><input id="import-file" type="file" accept=".json,application/json" hidden><p id="save-status" role="status"></p><p id="import-message" role="status"></p>'
    dialog = '<dialog id="import-dialog"><h2>Import review notes</h2><p id="import-warning"></p><pre id="import-preview"></pre><p>Existing notes are backed up before replacement.</p><button id="confirm-import">Replace notes with this import</button><button id="cancel-import">Cancel</button></dialog>'
    download = base64.b64encode((analysis / 'fish-comparison.csv').read_bytes()).decode()
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Corrected BLUP comparison</title><style>' + css + '</style><header><h1>Corrected BLUP comparison</h1><p>Exploratory evidence for joint review · 57 Delay/control fish · no method selected</p>' + controls + '<a download="fish-comparison.csv" href="data:text/csv;base64,' + download + '">Download all fish values</a></header><main>' + ''.join(chunks) + '</main>' + dialog + '<script type="application/json" id="guide-config">' + json.dumps(config).replace('</', r'<\/') + '</script><script>' + helper_raw.decode() + '</script></html>'
    snapshot.check()
    output.mkdir()
    (output / 'study.md').write_text(manuscript)
    (output / 'study.html').write_text(page)
    provenance = {'guide': {'path': str(output / 'study.html'), 'sha256': sha(page.encode())},
                  'manuscript': {'path': str(output / 'study.md'), 'sha256': sha(manuscript.encode())},
                  'analysis_manifest': {'path': str(manifest), 'sha256': manifest_sha},
                  'sources': [{'path': p, 'sha256': digest} for p, digest in snapshot.hashes.items() if p not in map(str, protected)],
                  'protected': [{'path': str(p), 'sha256': snapshot.hashes[str(p.resolve())]} for p in protected],
                  'scientific_status': 'completed exploratory comparison; not method approval',
                  'runtime_resources': 'CSS, notebook script and download data embedded',
                  'browser_behavior': 'Not exercised here; retained notebook helper bytes, static integrity checked'}
    (output / 'guide-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    snapshot.check()
    print(json.dumps(provenance['guide']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--notebook-script', type=Path, required=True)
    parser.add_argument('--notebook-sha256', required=True)
    parser.add_argument('--manuscript', type=Path)
    parser.add_argument('--manuscript-sha256')
    parser.add_argument('--guide-version', default='20261025-v01')
    args = parser.parse_args()
    build(args.analysis_dir, args.output_dir, args.notebook_script, args.notebook_sha256,
          args.manuscript, args.manuscript_sha256, args.guide_version)


if __name__ == '__main__':
    main()
