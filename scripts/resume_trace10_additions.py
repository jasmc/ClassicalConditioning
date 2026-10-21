"""Resume the sealed 10sTrace inventory, verifying and preserving old stages.

Each fish has its own worker process. An existing completed stage is verified,
not rebuilt. Any partially published stage fails closed. Old unpublished staging
is preserved. All new stage calls use overwrite=False and immutable D: sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from process_trace10_additions import PROJECT, TRACE3, sha, save_new

REPO = Path(__file__).resolve().parents[1]
META = PROJECT / "Metadata"
ORIGINAL = META / "trace10-additions-20261018-run.json"
STAGES = ("intake", "corrected-preprocess", "candidate-metrics", "movement-state",
          "temporal-profiles", "trial-outcomes")


def now():
    return datetime.now(timezone.utc).isoformat()


def state(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def stage_paths(rid):
    d, q = PROJECT/"Processed data"/rid, PROJECT/"Quality checks"/rid
    return {
        "intake": [d/"camera.parquet",d/"tracking.parquet",d/"stimulus_events.parquet",
                   q/"acquisition_summary.json",q/"acquisition_report.html",
                   *[q/"figures"/s for s in ("camera_timing.png","tracking_overview.png","protocol_timeline.png")],
                   META/f"{rid}_source_manifest.json"],
        "corrected-preprocess": [d/"frame_preprocessed_corrected.parquet",q/"corrected_preprocessing_summary.json",META/f"{rid}_corrected-preprocess_complete.json"],
        "candidate-metrics": [d/"frame_activity_candidates-corrected.parquet",q/"candidate-corrected_activity_summary.json",META/f"{rid}_candidate-corrected_complete.json"],
        "movement-state": [d/"movement_state_candidates-corrected.parquet",q/"movement-candidate-corrected_summary.json",META/f"{rid}_movement-candidate-corrected_complete.json"],
        "temporal-profiles": [d/"candidate_temporal_outcomes-corrected.parquet",q/"candidate-corrected_temporal_outcomes_summary.json",META/f"{rid}_candidate-temporal-outcomes-corrected_complete.json"],
        "trial-outcomes": [d/"candidate-trial-outcomes-corrected.parquet",d/"candidate-trial-outcomes-corrected_coverage.parquet",q/"candidate-trial-outcomes-corrected_summary.json",META/f"{rid}_candidate-trial-outcomes-corrected_complete.json"],
    }


def dispositions(exists):
    """Require a contiguous prefix of wholly published stages."""
    result, missing = {}, False
    for name in STAGES:
        flags = exists[name]
        if any(flags) and not all(flags):
            raise ValueError("Partially published stage must be preserved for reconciliation: "+name)
        present = all(flags)
        if missing and present:
            raise ValueError("Published downstream stage has missing upstream stage: "+name)
        if not present: missing = True
        result[name] = "verify-existing" if present else "build-missing"
    return result


def code_manifest():
    files = sorted((REPO/"src/classical_conditioning").rglob("*.py"))
    files += [Path(__file__),REPO/"scripts/process_trace10_additions.py"]
    return {str(p.resolve()):sha(p) for p in files}


def make_plan(run_id):
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    if Path(original["project"]).resolve()!=PROJECT.resolve():raise ValueError("Project identity mismatch")
    for p,h in original['preserved_handoff_hashes'].items():
        if sha(p)!=h:raise ValueError("Original handoff changed: "+p)
    rows=[]
    for row in original["rows"]:
        if row['issues']:raise ValueError("Original fish has unresolved source issue")
        if sha(row['sources']['protocol'])!=row['protocol']['sha256']:raise ValueError("Protocol changed: "+row['recording_id'])
        record={**row,"source_states":{k:state(Path(v)) for k,v in row['sources'].items()}}
        paths=stage_paths(row['recording_id'])
        try:record['stage_actions']=dispositions({k:[p.is_file() for p in ps] for k,ps in paths.items()})
        except ValueError as e:record['recovery_issue']=str(e)
        other=TRACE3/"Metadata"/f"{row['recording_id']}_source_manifest.json"
        if other.exists():
            m=json.loads(other.read_text())
            if m['recording_id']!=row['recording_id']:raise ValueError("Cross-assay manifest identity mismatch")
            if any(Path(m['sources'][k]['path']).resolve()!=Path(v).resolve() for k,v in row['sources'].items()):
                record['recovery_issue']='Cross-assay source collision requires review'
            record['shared_3s_manifest']={'path':str(other),'sha256':sha(other)}
        rows.append(record)
    baseline={str(p.resolve()):state(p) for p in PROJECT.rglob('*') if p.is_file()
              and p.name not in {run_id+'.stdout.log',run_id+'.stderr.log'}}
    return {'run_id':run_id,'created_utc':now(),'project':str(PROJECT),
            'original_plan':str(ORIGINAL),'original_plan_sha256':sha(ORIGINAL),
            'code_sha256':code_manifest(),'baseline':baseline,
            'preserved_handoff_hashes':original['preserved_handoff_hashes'],'rows':rows,
            'policy':'verify existing stages; build missing stages overwrite=False; one process per fish; preserve all old logs and staging'}


def protect_existing(exclude=()):
    excluded={Path(p).resolve() for p in exclude}
    old={str(p.resolve()).casefold() for p in PROJECT.rglob('*') if p.is_file() and p.resolve() not in excluded}
    def guard(event,args):
        candidates=[]
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            mode=args[1] or '';flags=args[2] if len(args)>2 else 0
            if any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):candidates=[args[0]]
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime'}:candidates=[args[0]]
        elif event=='os.rename':candidates=args[:2]
        for v in candidates:
            if isinstance(v,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(v)).resolve()
                if p.drive.lower()=='d:' or str(p).casefold() in old or any(x.casefold()=='results (paper)' for x in p.parts):
                    raise PermissionError(f'Preservation guard blocked {event}: {p}')
    sys.addaudithook(guard)


def verify_single(paths,recipe,rid,hash_key):
    artifact,summary_path,marker_path=paths
    summary=json.loads(summary_path.read_text());marker=json.loads(marker_path.read_text())
    h=sha(artifact)
    if (marker.get('status')!='complete' or marker.get('recipe')!=recipe or summary.get('recipe')!=recipe
        or marker.get('recording_id')!=rid or summary.get('recording_id')!=rid
        or marker.get(hash_key)!=h or marker.get('summary_sha256')!=sha(summary_path)
        or summary.get('artifact',{}).get('sha256')!=h
        or Path(summary['artifact']['path']).resolve()!=artifact.resolve()):
        raise ValueError('Completed stage hash/identity mismatch: '+recipe)
    return summary,h


def verify_fish(row,through='trial-outcomes',raw=False):
    from classical_conditioning.artifacts import load_and_verify_source_manifest
    from classical_conditioning.preprocessing.candidate_metrics_from_corrected_frames import verify_corrected_preprocess_source
    from classical_conditioning.analysis.trial_outcomes import verify_candidate_trial_outcomes
    rid=row['recording_id'];paths=stage_paths(rid)
    name,artifacts,_=load_and_verify_source_manifest(PROJECT,rid)
    m=json.loads(paths['intake'][-1].read_text())
    if name!=row['recording_name']:raise ValueError('Source alias changed')
    for k,p in row['sources'].items():
        s=m['sources'][k]
        if Path(s['path']).resolve()!=Path(p).resolve() or s['size_bytes']!=Path(p).stat().st_size:
            raise ValueError('Raw source path/size mismatch')
        if raw and s['sha256']!=sha(p):raise ValueError('Raw source hash mismatch: '+k)
    if m['sources']['protocol']['sha256']!=row['protocol']['sha256']:raise ValueError('Protocol lineage mismatch')
    if through=='intake':return
    corrected=verify_corrected_preprocess_source(PROJECT,rid)
    if any(corrected['summary']['input_artifacts'][k]['sha256']!=a['sha256'] for k,a in artifacts.items()):
        raise ValueError('Stale corrected intake lineage')
    if through=='corrected-preprocess':return
    metric,mh=verify_single(paths['candidate-metrics'],'tail-candidate-corrected',rid,'metrics_sha256')
    if metric['input_artifacts']['corrected_preprocess']['sha256']!=corrected['frames_sha256']:
        raise ValueError('Stale candidate corrected-frame lineage')
    if any(metric['input_artifacts'][k]['sha256']!=a['sha256'] for k,a in artifacts.items()):
        raise ValueError('Stale candidate intake lineage')
    if through=='candidate-metrics':return
    movement,wh=verify_single(paths['movement-state'],'movement-candidate-corrected',rid,'movement_sha256')
    if movement['inputs']['candidate_metrics']['sha256']!=mh:raise ValueError('Stale movement metrics')
    if through=='movement-state':return
    temporal,_=verify_single(paths['temporal-profiles'],'candidate-temporal-outcomes-corrected',rid,'profiles_sha256')
    if (temporal['experiment']!='all10sTrace' or temporal['inputs']['candidate_frames']['sha256']!=mh
        or temporal['inputs']['movement_state']['sha256']!=wh
        or temporal['inputs']['protocol']['sha256']!=artifacts['protocol']['sha256']):
        raise ValueError('Stale temporal lineage/experiment')
    if through=='temporal-profiles':return
    v=verify_candidate_trial_outcomes(PROJECT,rid,metric_recipe='tail-candidate-corrected')
    if (v.summary['experiment']!='all10sTrace' or v.summary['identity']['condition_id']!=row['condition']
        or v.summary['config']['response_window_s']!=[0.,20.]):raise ValueError('Wrong outcome condition/window')


def worker(plan,row,result_path):
    from classical_conditioning.intake import RecordingSources,intake_recording
    from classical_conditioning.preprocessing.corrected_frame_preprocessing import build_corrected_preprocessing
    from classical_conditioning.preprocessing.candidate_metrics_from_corrected_frames import build_candidate_activity_metrics_from_corrected
    from classical_conditioning.analysis.movement_state import build_candidate_movement_state
    from classical_conditioning.analysis.temporal_profiles import build_candidate_temporal_profiles
    from classical_conditioning.analysis.trial_outcomes import build_candidate_trial_outcomes
    protect_existing()
    rid=row['recording_id'];result={'recording_id':rid,'condition':row['condition'],'stages':[],
        'built':[],'reused':[],'incomplete_protocol':not row['protocol']['standard_event_counts']}
    try:
        if row.get('recovery_issue'):raise ValueError(row['recovery_issue'])
        for k,p in row['sources'].items():
            if state(Path(p))!=row['source_states'][k]:raise ValueError('Raw state changed: '+k)
        actions=dispositions({k:[p.is_file() for p in ps] for k,ps in stage_paths(rid).items()})
        if actions!=row['stage_actions']:raise ValueError('Published stage state changed after planning')
        existing=[s for s in STAGES if actions[s]=='verify-existing']
        if existing:
            print(f'VERIFY {rid} through {existing[-1]}',flush=True)
            verify_fish(row,through=existing[-1],raw=True)
            result['reused']=existing;result['stages']=existing.copy()
        parts={k:Path(v) for k,v in row['sources'].items()}
        sources=RecordingSources(row['recording_name'],rid,**parts)
        calls={
            'intake':lambda:intake_recording(parts['protocol'].parent,PROJECT,recording_id=rid,sources=sources,overwrite=False),
            'corrected-preprocess':lambda:build_corrected_preprocessing(PROJECT,rid,overwrite=False),
            'candidate-metrics':lambda:build_candidate_activity_metrics_from_corrected(PROJECT,rid,overwrite=False),
            'movement-state':lambda:build_candidate_movement_state(PROJECT,rid,metric_recipe='tail-candidate-corrected',overwrite=False),
            'temporal-profiles':lambda:build_candidate_temporal_profiles(PROJECT,rid,metric_recipe='tail-candidate-corrected',experiment_name='all10sTrace',overwrite=False),
            'trial-outcomes':lambda:build_candidate_trial_outcomes(PROJECT,rid,metric_recipe='tail-candidate-corrected',experiment_name='all10sTrace',overwrite=False)}
        for stage in STAGES:
            if stage in existing:continue
            print(f'BUILD {rid} {stage}',flush=True);calls[stage]()
            result['built'].append(stage);result['stages'].append(stage)
            print(f'STAGE {rid} {stage}',flush=True)
        verify_fish(row)
        result.update(status='completed',lineage_verified=True)
    except Exception as e:
        traceback.print_exc();result.update(status='failed',error=repr(e))
    result['finished_utc']=now();save_new(result_path,result)
    return 0 if result['status']=='completed' else 1


def execute(plan,plan_path,digest,prefix):
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p]
    kernel.CreateMutexW.restype=ctypes.c_void_p
    mutex=kernel.CreateMutexW(None,True,'Local\\ClassicalConditioningTrace10Additions')
    if not mutex or ctypes.get_last_error()==183:raise RuntimeError('Another 10sTrace recovery worker is active')
    protect_existing(exclude=[Path(str(prefix)+s) for s in ('.stdout.log','.stderr.log')])
    results=[];started=time.time();events=Path(str(prefix)+'-events.jsonl')
    for index,row in enumerate(plan['rows'],1):
        if code_manifest()!=plan['code_sha256']:raise ValueError('Processing code changed after recovery plan')
        rid=row['recording_id'];out=Path(str(prefix)+'-'+rid+'-result.json')
        print(f'START {rid} [{index}/{len(plan["rows"])}]',flush=True)
        args=[sys.executable,str(Path(__file__).resolve()),'--plan',str(plan_path),'--plan-sha256',digest,
              '--worker',rid]
        child=subprocess.run(args,cwd=REPO)
        result=json.loads(out.read_text()) if out.exists() else {
            'recording_id':rid,'status':'failed','error':f'worker stopped without result, exit={child.returncode}'}
        results.append(result)
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
        print(f'DONE {rid} {result["status"]} elapsed={time.time()-started:.1f}s',flush=True)
    changed=[p for p,s in plan['baseline'].items() if not Path(p).exists() or state(Path(p))!=s]
    handoffs=[p for p,h in plan['preserved_handoff_hashes'].items() if sha(p)!=h]
    report={'run_id':plan['run_id'],'finished_utc':now(),'results':results,
            'counts':dict(Counter(r['status'] for r in results)),
            'existing_files_changed':changed,'preserved_handoffs_changed':handoffs,
            'elapsed_seconds':time.time()-started,'plan_sha256':digest}
    report['status']='completed-and-verified' if not changed and not handoffs and all(r['status']=='completed' for r in results) else 'incomplete-or-review-required'
    save_new(Path(str(prefix)+'-completion.json'),report)
    with Path(str(prefix)+'-completion.md').open('x',encoding='utf-8') as f:
        f.write(f"# 10sTrace recovery completion\n\nStatus: {report['status']}; counts: {report['counts']}.\n\n")
        for r in results:f.write(f"- {r['recording_id']}: {r['status']}; reused {r.get('reused',[])}; built {r.get('built',[])}; incomplete protocol {r.get('incomplete_protocol')}; error {r.get('error','none')}.\n")
    print(json.dumps({'status':report['status'],'counts':report['counts']},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-only',action='store_true');ap.add_argument('--run-id',default='trace10-additions-20261019-recovery')
    ap.add_argument('--plan',type=Path);ap.add_argument('--plan-sha256');ap.add_argument('--worker')
    args=ap.parse_args()
    if args.plan_only:
        if not re.fullmatch(r'[a-z0-9-]+',args.run_id):raise ValueError('Invalid run ID')
        p=META/(args.run_id+'-plan.json');plan=make_plan(args.run_id);save_new(p,plan)
        print(json.dumps({'plan':str(p),'sha256':sha(p),'fish':len(plan['rows']),
                          'issues':{r['recording_id']:r['recovery_issue'] for r in plan['rows'] if r.get('recovery_issue')}}),flush=True);return
    if not args.plan or sha(args.plan)!=args.plan_sha256:raise ValueError('Exact sealed plan hash required')
    plan=json.loads(args.plan.read_text());prefix=META/plan['run_id']
    if Path(plan['project']).resolve()!=PROJECT.resolve() or sha(ORIGINAL)!=plan['original_plan_sha256']:raise ValueError('Original plan/project changed')
    if code_manifest()!=plan['code_sha256']:raise ValueError('Code changed after plan')
    if args.worker:
        row=next(r for r in plan['rows'] if r['recording_id']==args.worker)
        raise SystemExit(worker(plan,row,Path(str(prefix)+'-'+args.worker+'-result.json')))
    if any(Path(str(prefix)+s).exists() for s in ('-events.jsonl','-completion.json')):raise FileExistsError('Preserve existing recovery run')
    execute(plan,args.plan,args.plan_sha256,prefix)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
