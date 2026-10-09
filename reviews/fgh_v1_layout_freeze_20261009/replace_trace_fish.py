"""Replace only G's fish, using the selected Version 1 processing unchanged."""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, shutil
import numpy as np
import pandas as pd
from dataclasses import asdict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NEW = REPO / 'reviews/fgh_v1_trace20230310_08_20261009'
DATA = NEW / 'frozen-version1'
sys.path.insert(0, str(REPO/'reviews/fgh_c_bout_vs_bins_20261008'))
import build_comparison as processing
import shared_colourbar as layout
from freeze_and_style import digest, snapshot

def build():
    snapshot()  # Verify the original freeze before copying unchanged F/H.
    NEW.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    spec = list(processing.FISH[1]); spec[2] = '20230310_08'; spec = tuple(spec)
    project = Path(spec[3]); proc = project/'Processed data'/spec[2]
    source_paths = [project/'Metadata'/f'{spec[2]}_{suffix}.json' for suffix in
                    ['source_manifest','corrected-preprocess_complete','candidate-corrected_complete']]
    source, angles, metric = [json.loads(p.read_text()) for p in source_paths]
    assert source['recording_id'] == spec[2] and source['condition_id'] == 'trace'
    assert angles['status'] == metric['status'] == 'complete'
    expected = {'camera': source['artifacts']['camera']['sha256'],
                'protocol': source['artifacts']['protocol']['sha256'],
                'angles': angles['frames_sha256'], 'coverage': metric['metrics_sha256']}
    paths = {'camera': proc/'camera.parquet', 'protocol': proc/'stimulus_events.parquet',
             'angles': proc/'frame_preprocessed_corrected.parquet', 'coverage': proc/spec[4]}
    inputs = [{'kind': key, 'path': str(p), 'sha256': digest(p)} for key,p in paths.items()]
    assert {r['kind']:r['sha256'] for r in inputs} == expected
    protocol = pd.read_parquet(paths['protocol'])
    assert protocol.Type.eq('Cycle').sum() == 94
    from classical_conditioning.analysis.figure4 import verify_expected_us
    measured_us, paired_trials = verify_expected_us(protocol, 'all3sTrace')
    assert paired_trials == 46 and abs(measured_us - 13) < .1
    metadata = {'panel':'G', 'fish':spec[2], 'inputs':inputs,
                'metadata_inputs':[{'path':str(p), 'sha256':digest(p)} for p in source_paths]}
    (NEW/'manifest.json').write_text(json.dumps({'panels':[metadata]}, indent=2))
    processing.PRIOR = NEW  # The unchanged reader validates this fish's own inputs.
    print('Reconstructing Version 1 for', spec[2], flush=True)
    frames, cadence, _ = processing.load_frames(spec)
    frames['C_sample'] = np.nan; frames['C_sample_unclipped'] = np.nan
    runs=[]; statistics=[]
    for trial,p in frames.groupby('trial', sort=True):
        baseline = p.time_s.ge(-15)&p.time_s.lt(0)
        values, uncapped, stats = processing.c_values(p.bout_median_log, baseline)
        frames.loc[p.index,'C_sample'] = values
        frames.loc[p.index,'C_sample_unclipped'] = uncapped
        p=p.copy(); p['C_sample']=values
        runs.append(processing.sample_runs(p,cadence))
        statistics.append({'panel':'G','trial':int(trial),'version':'BoutSamples',**stats})
    allruns=pd.concat(runs,ignore_index=True)
    assert allruns.sample_count.sum() == frames.C_sample.notna().sum()
    assert np.isfinite(allruns.C).all() and allruns.C.between(-1,1).all()
    assert set(frames.trial.unique()) == set(range(5,95))
    frames.to_parquet(DATA/'PanelG_sample_data.parquet',index=False)
    allruns.to_csv(DATA/'PanelG_display_sample_runs.csv',index=False)
    prior=json.loads((HERE/'frozen-version1/freeze.json').read_text())
    for p in 'FH':
        src=HERE/'frozen-version1'/f'Panel{p}_display_sample_runs.csv'
        shutil.copy2(src,DATA/src.name)
        assert digest(src) == digest(DATA/src.name)
    oldstats=pd.read_csv(HERE/'frozen-version1/sample_baseline_statistics.csv')
    merged=pd.concat([oldstats[oldstats.panel.isin(['F','H'])],pd.DataFrame(statistics)],ignore_index=True)
    merged.sort_values(['panel','trial']).to_csv(DATA/'sample_baseline_statistics.csv',index=False)
    shutil.copy2(HERE/'frozen-version1/selected_processing_code.py',DATA/'selected_processing_code.py')
    for p in source_paths:
        shutil.copy2(p,DATA/p.name)
    record={k:v for k,v in prior.items() if k not in ['snapshot_files','original_sample_tables','undefined_trial']}
    record.update({'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'approval_date_local':'2026-10-09',
        'approval_evidence':'use a different fish for 3sTrace. use "20230310_08" instead.',
        'supersedes':str(HERE/'frozen-version1/freeze.json'),
        'supersedes_sha256':digest(HERE/'frozen-version1/freeze.json'),
        'fish':{'F':'20221115_07','G':'20230310_08','H':'20221115_09'},
        'replacement_inputs':metadata, 'cadence':asdict(cadence),
        'undefined_trials':[{'panel':'G','trial':s['trial']} for s in statistics if not s['defined']]+[{'panel':'H','trial':16}],
        'original_sample_tables':[r for r in prior['original_sample_tables'] if r['panel']!='G']+
            [{'panel':'G','path':str(DATA/'PanelG_sample_data.parquet'),'sha256':digest(DATA/'PanelG_sample_data.parquet')}],
        'snapshot_files':[{'path':str(p),'sha256':digest(p)} for p in DATA.iterdir() if p.is_file() and p.name!='freeze.json']})
    (DATA/'freeze.json').write_text(json.dumps(record,indent=2)+'\n')
    return spec

def main():
    if '--render-only' in sys.argv:
        spec=list(processing.FISH[1]);spec[2]='20230310_08';spec=tuple(spec)
    else:
        spec=build()
    def verified_snapshot():
        record=json.loads((DATA/'freeze.json').read_text())
        for r in record['snapshot_files']:
            assert digest(Path(r['path'])) == r['sha256']
        return record
    layout.HERE=NEW;layout.OUT=NEW/'shared-colourbar'
    layout.FISH=(processing.FISH[0],spec,processing.FISH[2])
    layout.snapshot=verified_snapshot
    layout.main()
    config_path=REPO/'configs/paper-figures/figure1-fgh-version1-freeze-20261009.json'
    config=json.loads(config_path.read_text())
    config['previous_freeze_manifest']=str(HERE/'frozen-version1/freeze.json')
    config['freeze_manifest']=str(DATA/'freeze.json')
    config['freeze_manifest_sha256']=digest(DATA/'freeze.json')
    config['fish']={'F':'20221115_07','G':'20230310_08','H':'20221115_09'}
    config['layout_validation']=str(layout.OUT/'validation.json')
    config['layout_validation_sha256']=digest(layout.OUT/'validation.json')
    config['bundle']=config['current_layout']['pdf']
    config['bundle_sha256']=digest(Path(config['bundle']))
    config_path.write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':
    main()
