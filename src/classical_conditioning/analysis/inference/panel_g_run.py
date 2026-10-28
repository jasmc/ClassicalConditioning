"""Immutable stage directories and an atomic, locked independent-run pointer."""
from __future__ import annotations
import argparse
import errno
from contextlib import contextmanager
from datetime import datetime, timezone
import os
if os.name == 'nt':
    import msvcrt
else:
    import fcntl
import hashlib
import json
import importlib.metadata
from pathlib import Path
import shutil
import sys
import uuid
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .panel_g_review import ReviewConfig, clean_json, save_json, prepare_input


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x') as f:
        json.dump(clean_json(value),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


@contextmanager
def run_lock(root):
    with (root/'.run.lock').open('a+b') as f:
        if os.name == 'nt':
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                raise ValueError('Another process owns this independent run') from error
        else:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Another process owns this independent run') from error
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def make_manifest(directory):
    files=[{'path':str(p.relative_to(directory)),'sha256':sha(p)}
           for p in sorted(directory.rglob('*')) if p.is_file()
           and p.name!='artifact-manifest.json' and not p.name.startswith('._')]
    save_json(directory/'artifact-manifest.json',{'schema_version':2,'artifacts':files,
        'scientific_status':'exploratory','original_selection_changed':False})
    return sha(directory/'artifact-manifest.json')


def verify_stage(directory,expected_manifest):
    if sha(directory/'artifact-manifest.json')!=expected_manifest:
        raise ValueError('Stage manifest changed')
    entries=json.loads((directory/'artifact-manifest.json').read_text())['artifacts']
    expected={e['path'] for e in entries}
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()
            and p.name!='artifact-manifest.json' and not p.name.startswith('._')}
    if actual!=expected:raise ValueError('Stage file inventory changed')
    for e in entries:
        if sha(directory/e['path'])!=e['sha256']:raise ValueError('Stage artifact changed: '+e['path'])


def stage_transaction(root,state,stage,source_files,callback,verify_originals):
    """Publish evidence before the pointer; prior stage artifacts are immutable."""
    before=sha(root/'analysis-state.json') if (root/'analysis-state.json').exists() else None
    if state.get('latest'):
        previous=root/state['latest']['directory']
        verify_stage(previous,state['latest']['manifest_sha256'])
    else:previous=None
    destination=root/'stages'/(stage+'-'+uuid.uuid4().hex[:12])
    destination.parent.mkdir(exist_ok=True)
    if previous:shutil.copytree(previous,destination,ignore=shutil.ignore_patterns('._*'))
    else:destination.mkdir()
    starting=[{'path':str(p),'sha256':sha(p)} for p in source_files]
    snapshots=destination/'source-snapshots'/stage;snapshots.mkdir(parents=True,exist_ok=True)
    for i,p in enumerate(source_files):shutil.copyfile(p,snapshots/f'{i:02d}-{p.name}')
    save_json(destination/f'{stage}-sources.json',starting)
    if stage=='calibrate':
        # Resume only identical code, settings and input bytes from an
        # unpublished candidate. Published predecessors remain unchanged.
        for candidate in sorted((root/'stages').glob('calibrate-*'),reverse=True):
            if candidate==destination or not (candidate/'stage-failure.json').exists():continue
            progress=candidate/'calibration-progress.json';sources=candidate/'calibrate-sources.json'
            if (progress.exists() and sources.exists() and json.loads(sources.read_text())==starting
                    and sha(candidate/'model-input.parquet')==sha(destination/'model-input.parquet')):
                shutil.copyfile(progress,destination/'calibration-progress.json');break
    save_json(destination/'runtime-environment.json',{'python':sys.version,
        'packages':{n:importlib.metadata.version(n) for n in
            ['numpy','pandas','scipy','statsmodels','patsy','matplotlib','pyarrow','threadpoolctl']}})
    try:
        callback(destination)
        verify_originals()
        for e in starting:
            if sha(Path(e['path']))!=e['sha256']:raise ValueError('Source changed while running: '+e['path'])
        if previous:verify_stage(previous,state['latest']['manifest_sha256'])
        current=sha(root/'analysis-state.json') if (root/'analysis-state.json').exists() else None
        if current!=before:raise ValueError('Run pointer changed while computing')
        save_json(destination/f'{stage}-complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
            'stage':stage,'original_G_hashes_unchanged':True,'sources':starting})
        manifest=make_manifest(destination)
        verify_stage(destination,manifest)
        updated={**state,'latest':{'directory':str(destination.relative_to(root)),
            'manifest_sha256':manifest},'completed':[*state.get('completed',[]),stage]}
        atomic_json(root/'analysis-state.json',updated)
        return updated
    except BaseException as e:
        # Failed candidate stays inspectable; it never becomes the selected run.
        save_json(destination/'stage-failure.json',{'stage':stage,'error':repr(e),
            'published':False,'previous_pointer_preserved':True})
        raise


def import_inputs(parent,out,runner):
    parent=parent.resolve()
    known={e['path']:e['sha256'] for e in json.loads((parent/'artifact-manifest.json').read_text())['artifacts']}
    names=['scheduled-window-outcomes.parquet','model-input.parquet','fish-technical-audit.csv',
           'fish-block-coverage.csv','recording-inventory.json','input-lineage.json',
           'input-verification.json','protected-originals.json','field-definitions.json']
    imported=[]
    for name in names:
        p=parent/name
        if sha(p)!=known.get(str(p)):raise ValueError('Changed parent input: '+name)
        shutil.copyfile(p,out/name)
        if sha(out/name)!=known[str(p)]:raise ValueError('Parent input changed while copying: '+name)
        imported.append({'path':str(p),'sha256':known[str(p)]})
    lineage=json.loads((out/'input-lineage.json').read_text())
    for i,e in enumerate(lineage):
        if sha(Path(e['path']))!=e['sha256']:raise ValueError('Changed bound input: '+e['path'])
        if (i+1)%50==0:print(f'Input authentication {i+1}/{len(lineage)}',flush=True)
    data=prepare_input(pd.read_parquet(out/'scheduled-window-outcomes.parquet'))
    saved=pd.read_parquet(out/'model-input.parquet')
    pd.testing.assert_frame_equal(data,saved)
    inventory=json.loads((runner.FREEZE/'sources/selected-source-inventory.json').read_text())
    frozen=next(e for e in inventory if e['path'].endswith('\\fish-logmedian-differences.parquet'))
    replica=next(e for e in lineage if Path(e['path']).name=='fish-logmedian-differences.parquet')
    if replica['sha256']!=frozen['sha256']:raise ValueError('Frozen numerical reference changed')
    ref=pd.read_parquet(replica['path'])
    joined=data[['condition_id','fish_id','trial_number','z']].merge(ref,
        on=['condition_id','fish_id','trial_number'],how='outer',validate='one_to_one',indicator=True)
    if not joined._merge.eq('both').all():raise ValueError('Frozen eligibility mismatch')
    np.testing.assert_allclose(joined.z,joined.ratio,rtol=0,atol=1e-12)
    save_json(out/'input-import.json',{'parent':str(parent),'imported':imported,
        'all_bound_inputs_verified':len(lineage),'frozen_G_numeric_match':True,
        'maximum_difference':float(abs(joined.z-joined.ratio).max()),
        'original_reconstruction_sources':'parent run prepare-runner snapshot; imported as authenticated data'})
    for e in imported:
        if sha(Path(e['path']))!=e['sha256']:raise ValueError('Parent input changed during preparation')


def main(runner):
    from classical_conditioning.external_artifacts import relocated_external_directory, joaquim_root
    parser=argparse.ArgumentParser(description='Guarded independent G analysis; preserves every published stage')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--prepare-from',type=Path)
    parser.add_argument('--stage',default='all',choices=['all','prepare','legacy','population','stabilize',
        'sensitivities','individual','timing-review','hyperparameters','cohorts','calibrate','robust-calibrate','validation-review','report','verify'])
    parser.add_argument('--bootstrap',type=int,default=999)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261014)
    parser.add_argument('--calibration-repetitions',type=int,default=10)
    parser.add_argument('--calibration-bootstrap',type=int,default=99)
    parser.add_argument('--robust-calibration-repetitions',type=int,default=100)
    parser.add_argument('--cohort-bootstrap',type=int,default=199)
    parser.add_argument('--attempt',default='initial')
    parser.add_argument('--rebuild-review',action='store_true')
    args=parser.parse_args()
    runner.MOUNT=joaquim_root()
    runner.ANALYSIS_ROOT=relocated_external_directory(runner.ANALYSIS_ROOT).resolve()
    runner.FROZEN_EXTERNAL=relocated_external_directory(runner.FROZEN_EXTERNAL).resolve()
    out=relocated_external_directory(args.output_dir).resolve()
    if (args.workers<1 or args.calibration_repetitions<1 or args.calibration_bootstrap<19
            or args.robust_calibration_repetitions<1 or args.cohort_bootstrap<19):
        raise ValueError('Invalid worker/resampling/simulation settings')
    if not runner.MOUNT.is_dir() or not out.is_relative_to(runner.ANALYSIS_ROOT) or out==runner.ANALYSIS_ROOT:
        raise ValueError('Use the dedicated external independent-G namespace')
    config=ReviewConfig(bootstrap=args.bootstrap,seed=args.seed)
    settings=clean_json({'review':config.__dict__,'calibration_repetitions':args.calibration_repetitions,
        'calibration_bootstrap':args.calibration_bootstrap,'cohort_bootstrap':args.cohort_bootstrap,
        'robust_calibration_repetitions':args.robust_calibration_repetitions})
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run-config.json').exists() and not (out/'analysis-state.json').exists():
        raise ValueError('Preserve old-layout runs; choose a new destination')
    with run_lock(out),threadpool_limits(limits=1):
        if (out/'analysis-state.json').exists():
            state=json.loads((out/'analysis-state.json').read_text())
            if state['settings']!=settings:raise ValueError('Run settings changed; use a new run')
        else:
            protected=runner.protected_files()
            if (out/'start-ticket.json').exists():
                for e in json.loads((out/'start-ticket.json').read_text())['protected']:
                    runner.authenticate(Path(e['path']),e['sha256'])
            save_json(out/'protected-originals.json',protected)
            state={'schema_version':2,'settings':settings,'completed':[],
                'scientific_status':'exploratory; no primary target/threshold/cohort approval'}
        def protected_check():
            for e in json.loads((out/'protected-originals.json').read_text()):
                runner.authenticate(Path(e['path']),e['sha256'])
        protected_check()
        if state.get('latest'):verify_stage(out/state['latest']['directory'],state['latest']['manifest_sha256'])
        for completed in state['completed']:
            if not (out/state['latest']['directory']/(completed+'-complete.json')).is_file():
                raise ValueError('Completion pointer lacks authenticated stage evidence: '+completed)
        if args.stage=='verify':print('Verified independent stage and protected G');return
        from .panel_g_legacy import evaluate_legacy,run_cohort_sensitivities
        from .panel_g_marginal import run_marginal_calibration,run_marginal_candidates,summarize_marginal_calibration
        from .panel_g_hyperparameters import run_hyperparameters
        order=['prepare','legacy','population','stabilize','sensitivities','individual','timing-review',
               'hyperparameters','cohorts','calibrate','robust-calibrate','validation-review','report']
        stages=order if args.stage=='all' else [args.stage]
        src=runner.ROOT/'src/classical_conditioning/analysis/inference'
        source_files=[Path(runner.__file__),*sorted(src.glob('panel_g_*.py')),src/'learning_onset.py']
        for stage in stages:
            if stage in state['completed'] and not (stage=='report' and args.rebuild_review):
                print('Verified completed stage: '+stage,flush=True);continue
            if stage!='prepare' and 'prepare' not in state['completed']:raise ValueError('Prepare must complete first')
            print('Starting guarded stage: '+stage,flush=True)
            def callback(d):
                if stage=='prepare':
                    if args.prepare_from:
                        if (out/'start-ticket.json').exists():
                            ticket=json.loads((out/'start-ticket.json').read_text())
                            if str(args.prepare_from.resolve())!=ticket['parent_run'] or sha(args.prepare_from/'artifact-manifest.json')!=ticket['parent_manifest']['sha256']:
                                raise ValueError('Parent run differs from starting ticket')
                        import_inputs(args.prepare_from,d,runner)
                    else:
                        shutil.copyfile(out/'protected-originals.json',d/'protected-originals.json');runner.prepare(d)
                elif stage=='legacy':evaluate_legacy(d,runner.ROOT)
                elif stage=='population':runner.population(d,config,args.workers)
                elif stage=='stabilize':runner.stabilize(d,config,args.workers)
                elif stage=='sensitivities':runner.sensitivities(d,config,args.workers)
                elif stage=='individual':runner.individual(d,config)
                elif stage=='timing-review':runner.timing_review(d,config)
                elif stage=='hyperparameters':run_hyperparameters(d,config)
                elif stage=='cohorts':run_cohort_sensitivities(d,config,args.workers,args.cohort_bootstrap);run_marginal_candidates(d,config)
                elif stage=='calibrate':runner.calibrate(d,config,args.workers,args.calibration_repetitions,args.calibration_bootstrap)
                elif stage=='robust-calibrate':run_marginal_calibration(d,config,args.workers,args.robust_calibration_repetitions)
                elif stage=='validation-review':summarize_marginal_calibration(d,args.robust_calibration_repetitions)
                elif stage=='report':runner.report(d,config)
            state=stage_transaction(out,state,stage,source_files,callback,protected_check)
            print('Published '+stage+': '+str(out/state['latest']['directory']),flush=True)
