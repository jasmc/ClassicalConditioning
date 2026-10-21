"""Authenticated fixed-10sTrace caches and separate-panel statistics.

Source drives are read-only; saved records are exclusive and resumable.
Scientific roster selection/publication is a separate, explicit operation.
"""
from pathlib import Path
import argparse, gc, json, os, subprocess, sys, traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
from classical_conditioning.figures.pooled_log_heatmap import analyze_recording,pool_equal_fish
from classical_conditioning.analysis.trial_outcomes import verify_candidate_trial_outcomes
from review_figure2_block_log_median import calculate_trials,read_windows,COL
import build_figure2_trace3_panels as reference

PROJECT=Path(r'F:\Digested Data\all10sTrace-full-v1')
META=PROJECT/'Metadata'
RUN=META/'trace10-additions-20261019-recovery-plan.json'
COMPLETION=META/'trace10-additions-20261019-recovery-completion.json'
OUT=Path('J:/ClassicalConditioning Outputs/current/reviews/figure-02-10strace-20261020')
STATE=REPO/'docs/paper/figures/figure-02/current-state.json'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
def save(p,data):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def check(refs):
    for r in refs:
        if sha256_file(Path(r['path']))!=r['sha256']:raise ValueError('Changed input '+r['path'])
def guard():
    reference.OUT=OUT
    reference.guard_calculation_writes()

def seal():
    if (OUT/'plan.json').exists():raise FileExistsError('Existing sealed run; resume it')
    if not OUT.parent.is_dir():raise FileNotFoundError('JOAQUIM required')
    OUT.mkdir(exist_ok=True)
    state=load(STATE);run=load(RUN);done=load(COMPLETION)
    assert done['status']=='completed-and-verified' and len(done['results'])==63
    assert not done['existing_files_changed'] and not done['preserved_handoffs_changed']
    assert sha256_file(RUN)==done['plan_sha256']
    assert all(r['lineage_verified'] and len(r['stages'])==6 for r in done['results'])
    refs=[state['draft'][k] for k in ['working_svg','review','legend','semantic_registry']]
    refs+=list(state['draft']['matched_trace3_scientific_draft'][k] for k in ['data','statistics'])
    check(refs)
    deps=[Path(__file__),REPO/'src/classical_conditioning/figures/pooled_log_heatmap.py',
          REPO/'scripts/review_figure2_block_log_median.py',REPO/'scripts/analyze_figure2_trace3_matched.py',
          REPO/'scripts/render_figure2_delay_logmedian.py',REPO/'scripts/render_figure2_delay_phase_lmm.py',
          REPO/'scripts/render_figure2_delay_legacy_metric_lme.py',REPO/'src/classical_conditioning/analysis/bout_block_statistics.py']
    rows=[]
    for r in run['rows']:
        rid=r['recording_id'];s=PROJECT/'Quality checks'/rid/'candidate-trial-outcomes-corrected_summary.json'
        summary=load(s)
        assert summary['experiment']=='all10sTrace' and summary['identity']['condition_id']==r['condition']
        assert summary['config']['baseline_window_s']==[-15.,0.] and summary['config']['response_window_s']==[0.,20.]
        rows.append({'recording_id':rid,'condition':r['condition'],'summary':ref(s),
             'shared_control_identity':r['shared_control_identity'],'protocol_audit':r['protocol'],
             'incomplete_protocol':not r['protocol']['standard_event_counts']})
    assert pd.Series([r['condition'] for r in rows]).value_counts().to_dict()=={'control':32,'trace':31}
    ticket=Snapshot([STATE,REPO/'docs/paper/figures/figure-01/current-state.json',*deps])
    ticket.hashes.update({r['path']:r['sha256'] for r in refs})
    save(OUT/'starting-ticket.json',ticket.hashes)
    save(OUT/'plan.json',{'cohort':rows,'processing':[ref(RUN),ref(COMPLETION)],'references':refs,
         'science_code':[ref(p) for p in deps],'source_drives_read_only':['D:','F:'],
         'roster_status':'all 63 inventoried; technical gates and common analysis roster not yet selected',
         'methods':{'C':'selected direct 1s original-frame positive bout log medians; baseline-bin median; equal fish mean',
                    'F':'measured-frame median log difference [0,20) minus [-15,0); min3/5, separate Holm18',
                    'I':'selected G/H models and whole-fish bootstrap5000 seed10; panel-separate families'}})
    print('SEALED 63 completed acquisitions',flush=True)

def preflight():
    plan=load(OUT/'plan.json');check(plan['processing']+plan['science_code'])
    rows=[]
    for r in plan['cohort']:
        rid=r['recording_id'];p=PROJECT/'Processed data'/rid
        camera=pd.read_parquet(p/'camera.parquet');cadence=estimate_camera_cadence(camera)
        protocol=pd.read_parquet(p/'stimulus_events.parquet');n=int(protocol.Type.eq('Cycle').sum())
        rows.append({'recording_id':rid,'condition':r['condition'],'shared_control_identity':r['shared_control_identity'],
            'cs_count':n,'has_frame_loss_evidence':bool(cadence.has_frame_loss_evidence),
            'cadence':__import__('dataclasses').asdict(cadence),'incomplete_protocol':r['incomplete_protocol'],
            'heatmap_gate_pass':not cadence.has_frame_loss_evidence and n==94})
        print('PREFLIGHT',rid,rows[-1]['heatmap_gate_pass'],flush=True)
        del camera;gc.collect()
    save(OUT/'preflight.json',{'rows':rows,'issues':[r for r in rows if not r['heatmap_gate_pass']]})

def fish(rid):
    guard();plan=load(OUT/'plan.json');check(plan['science_code'])
    row=next(r for r in plan['cohort'] if r['recording_id']==rid)
    cache=OUT/'fish';cache.mkdir(exist_ok=True);rp=cache/(rid+'.json')
    if rp.exists():check(load(rp)['outputs']);print('CACHED',rid,flush=True);return
    if list(cache.glob(rid+'-*.parquet')):raise FileExistsError('Preserve uncommitted partial '+rid)
    verified=verify_candidate_trial_outcomes(PROJECT,rid,metric_recipe='tail-candidate-corrected')
    assert verified.summary['experiment']=='all10sTrace' and verified.summary['config']['response_window_s']==[0.,20.]
    check([row['summary']])
    proc=PROJECT/'Processed data'/rid
    cycles=pd.read_parquet(proc/'stimulus_events.parquet')
    cycles=cycles.loc[cycles.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    intervals=[(float(t)-15000,float(t)+20000) for t in cycles.iloc[4:].Beg]
    mp=proc/'frame_activity_candidates-corrected.parquet';wp=proc/'movement_state_candidates-corrected.parquet'
    metrics=read_windows(mp,['FrameID','AbsoluteTime',COL],intervals)
    movement=read_windows(wp,['FrameID','AbsoluteTime','valid','moving'],intervals)
    trials=calculate_trials(metrics,movement,cycles,range(5,min(len(cycles),94)+1),20)
    trials['fish_id']=rid;trials['condition_id']=row['condition']
    sources=[ref(p) for p in [mp,wp,proc/'stimulus_events.parquet']]
    for a,key in zip(sources,['candidate_frames','movement_state','protocol']):assert a['sha256']==verified.summary['inputs'][key]
    trials.to_parquet(cache/(rid+'-windows.parquet'),index=False)
    record={'recording_id':rid,'condition':row['condition'],'observed_cs':len(cycles),
            'incomplete_protocol':row['incomplete_protocol'],'window_sources':sources,
            'outputs':[ref(cache/(rid+'-windows.parquet'))]}
    del metrics,movement;gc.collect()
    gate=next(r for r in load(OUT/'preflight.json')['rows'] if r['recording_id']==rid)
    if not gate['heatmap_gate_pass']:
        record.update(heatmap='review-required',heatmap_error='preflight timing/coverage gate; no schedule padding or silent exclusion')
    else:
        try:
            matrix,bins,info=analyze_recording(PROJECT,rid,'','all10sTrace',row['condition'])
            bins.to_parquet(cache/(rid+'-bins.parquet'),index=False)
            record.update(heatmap='complete',heatmap_provenance=info)
            record['outputs'].append(ref(cache/(rid+'-bins.parquet')))
        except Exception as e:
            traceback.print_exc();record.update(heatmap='review-required',heatmap_error=repr(e))
    save(rp,record);print('DONE',rid,record['heatmap'],flush=True)

def calculate():
    guard();plan=load(OUT/'plan.json');check(plan['science_code'])
    if not (OUT/'preflight.json').exists():preflight()
    for i,row in enumerate(plan['cohort'],1):
        print('START',row['recording_id'],i,'/63',flush=True)
        subprocess.run([sys.executable,str(Path(__file__)),'--fish',row['recording_id']],cwd=REPO,check=True)
    if (OUT/'calculation.json').exists():check(load(OUT/'calculation.json')['outputs']);return
    records=[load(OUT/'fish'/(r['recording_id']+'.json')) for r in plan['cohort']]
    for r in records:check(r['outputs'])
    windows=pd.concat([pd.read_parquet(OUT/'fish'/(r['recording_id']+'-windows.parquet')) for r in records],ignore_index=True)
    windows.to_parquet(OUT/'windows-observed.parquet',index=False)
    save(OUT/'calculation.json',{'status':'calculated; roster review before publication',
         'issues':[r for r in records if r['heatmap']!='complete'],
         'fish_records':[ref(OUT/'fish'/(r['recording_id']+'.json')) for r in records],
         'outputs':[ref(OUT/'windows-observed.parquet')],'plan':ref(OUT/'plan.json')})
    print('CALCULATION COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seal',action='store_true');ap.add_argument('--preflight',action='store_true');ap.add_argument('--calculate',action='store_true');ap.add_argument('--fish');args=ap.parse_args()
    if args.seal:seal()
    elif args.preflight:guard();preflight()
    elif args.fish:fish(args.fish)
    elif args.calculate:calculate()
    else:ap.error('Choose operation')
