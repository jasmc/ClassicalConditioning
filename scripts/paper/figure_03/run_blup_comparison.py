"""Complete the isolated Delay/control BLUP comparison; never select a method.

Consumes the authenticated starting ticket and saved 57-fish analysis inputs.
Writes only a new external run. Calibration and validation use disjoint random
streams; original missingness, scheduled gaps and cohort membership are retained.
"""
from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src')]

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from classical_conditioning.analysis.review import blup_comparison as b
from classical_conditioning.analysis.inference.panel_g_run import make_manifest, verify_stage, run_lock
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, sha


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def model_summary(model):
    return {key: model[key] for key in ['nll', 'theta', 'G', 'sigma', 'phi', 'beta',
            'covbeta', 'nobs', 'boundary', 'ar1', 'converged', 'iterations', 'optimizer_message']}


def run(ticket_path, output, calibration=999, validation=499, seed=20261016):
    if calibration < 199 or validation < 199:
        raise ValueError('This review requires at least 199 calibration and validation datasets')
    output = output.expanduser().resolve()
    if output.is_relative_to(_COMMAND_REPOSITORY):
        raise ValueError('Numerical and review outputs belong outside the repository')
    if output.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():
        raise FileNotFoundError('JOAQUIM unavailable')
    if not output.parent.is_dir() or output.exists():
        raise FileExistsError('Choose a new directory under an existing external parent')
    ticket_path = ticket_path.resolve()
    raw = ticket_path.read_bytes()
    ticket = json.loads(raw)
    source = ticket['source_inputs']
    paths = {Path(row['path']).name: Path(row['path']) for row in source}
    protected = list((_COMMAND_REPOSITORY / 'docs/paper/figures').rglob('*state.json'))
    snapshot = Snapshot([ticket_path, *command_sources(__file__), *protected])
    snapshot.hashes[str(ticket_path)] = sha(raw)
    for row in source:
        path = Path(row['path']).resolve()
        if sha(path.read_bytes()) != row['sha256']:
            raise ValueError('Saved input changed: ' + str(path))
        snapshot.hashes[str(path)] = row['sha256']
    snapshot.check()
    trials = pd.read_parquet(paths['trial-data.parquet'])
    inventory = pd.read_csv(paths['fish-inventory.csv'])
    legacy = pd.read_csv(paths['classification-manifest.csv'])
    if set(inventory.condition_id) != {'control', 'delay'}:
        raise ValueError('This comparison supports the recorded Control/Delay scope only')
    data = b.epoch_data(trials, inventory, epochs=((5, 14), (40, 64)))
    if len(data.fish) != len(inventory):
        raise ValueError('Inventory loss')
    config = dict(schema_version=1, scope='Delay/control exploratory BLUP comparison',
                  calibration=calibration, validation_per_scenario=validation, seed=seed,
                  acquisition_epochs=[[5, 14], [40, 64]], recovery_epochs=[[40, 64], [85, 94]],
                  metric='legacy_distal_angular_speed', outcome='response_logmedian - baseline_logmedian',
                  scenarios=['matched', 'heavy_tail', 'within_epoch_trend', 'null'],
                  scientific_selection_changed=False, final_method_selected=False,
                  trial_availability='Observed design held fixed; not a missingness-mechanism validation')
    output.mkdir()
    with run_lock(output), threadpool_limits(limits=1):
        write_json(output / 'analysis-config.json', config)
        write_json(output / 'starting-snapshot.json', snapshot.hashes)
        code = output / 'source-code'
        code.mkdir()
        # Retain focused implementation snapshots; dependency identities remain
        # bound in starting-snapshot without copying the entire application.
        for path in [Path(__file__), Path(b.__file__)]:
            (code / path.name).write_bytes(path.read_bytes())
        begin = time.monotonic()
        model = b.fit(data, multistart=True)
        write_json(output / 'primary-model.json', model_summary(model))
        def sample(stage, count, stream_seed, scenario):
            def progress(done, total):
                snapshot.check()
                print(json.dumps({'stage': stage, 'completed': done, 'requested': total,
                                  'elapsed_seconds': round(time.monotonic() - begin, 1)}), flush=True)
            draws = b.bootstrap(data, model, count, stream_seed, scenario, progress)
            np.savez_compressed(output / (stage + '.npz'), **{k: draws[k] for k in ['total_error', 'excess_error', 'boundary']})
            write_json(output / (stage + '-refits.json'), {k: v for k, v in draws.items() if k not in ['total_error', 'excess_error', 'boundary']})
            return draws
        calibration_draws = sample('calibration', calibration, seed, 'matched')
        quantiles = b.interval_quantiles(calibration_draws)
        write_json(output / 'calibration-quantiles.json', quantiles)
        scores = b.scored_fish(data, model, quantiles)
        no_ar = b.fit(data, ar1=False, multistart=True)
        write_json(output / 'independent-residual-model.json', model_summary(no_ar))
        no_ar_scores = b.scored_fish(data, no_ar)
        scores['no_ar1_acquisition'] = no_ar_scores.acquisition
        recovery_data = b.epoch_data(trials, inventory, epochs=((40, 64), (85, 94)))
        recovery_model = b.fit(recovery_data, multistart=True)
        write_json(output / 'recovery-model.json', model_summary(recovery_model))
        recovery = b.scored_fish(recovery_data, recovery_model)
        # The model contrast is epoch0 - epoch1. Recovery is its explicit
        # opposite: late-Test minus late-Train, not an acquisition requirement.
        scores['recovery_late_test_minus_late_train'] = -recovery.acquisition
        scores['recovery_plugin_se'] = recovery.plugin_se
        scores.to_csv(output / 'fish-comparison.csv', index=False)
        legacy.to_csv(output / 'saved-legacy-comparison.csv', index=False)
        validations = []
        for i, scenario in enumerate(config['scenarios']):
            draws = sample('validation-' + scenario, validation, seed + 101 + i, scenario)
            row = b.coverage(draws, quantiles)
            row.update(scenario=scenario, seed=seed + 101 + i,
                       independence='New random stream; no calibration datasets reused',
                       null_any_false_positive_rate=float((np.max(draws['excess_error'], axis=1) > quantiles['family_lower']).mean()) if scenario == 'null' else None)
            validations.append(row)
        write_json(output / 'validation.json', validations)
        summary = dict(config=config, fish=len(scores), conditions=inventory.condition_id.value_counts().to_dict(),
                       primary_model=model_summary(model), validation=validations,
                       positive_model_evidence=scores.groupby('condition_id').positive_model_evidence.agg(['sum', 'count']).reset_index().to_dict('records'),
                       calibration_successful=len(calibration_draws['boundary']), calibration_attempted=calibration,
                       calibration_boundary_fraction=float(calibration_draws['boundary'].mean()),
                       retention='All input fish retained; estimability does not establish adequate precision',
                       elapsed_seconds=time.monotonic() - begin,
                       limitations=['Conditional positive-bout vigor, not movement probability or biological learner truth',
                                    'Parametric calibration assumes the fitted model; stress checks are finite scenarios',
                                    'Observed missingness held fixed; non-random missingness not validated',
                                    'Legacy outcomes, epochs and acquisition-plus-recovery targets differ',
                                    'No approved meaningful-effect threshold or learner representation',
                                    'Delay/control pilot does not validate full Trace assays or a paper-wide cohort'])
        write_json(output / 'result.json', summary)
        snapshot.check()
        digest = make_manifest(output)
        verify_stage(output, digest)
        snapshot.check()
        print(json.dumps({'status': 'completed exploratory comparison', 'output': str(output),
                          'artifact_manifest_sha256': digest, 'fish': len(scores),
                          'scientific_selection_changed': False}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ticket', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=int, default=999)
    parser.add_argument('--validation', type=int, default=499)
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    run(args.source_ticket, args.output_dir, args.calibration, args.validation, args.seed)


if __name__ == '__main__':
    main()
