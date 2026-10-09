"""Authenticate recorded F inputs and regenerate unavailable J previews.

No raw/frame files are copied. Each model retains its original definition.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import pandas as pd

ROOT=Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly')
REVIEW=ROOT/'row3-trial-ratio-review'
LOCAL=Path('outputs/baseline-window-review/figure2-pre15/figure-2G_delay-control_tail_length_weighted_angular_l1.figure.json')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''): h.update(block)
    return h.hexdigest()

def dump(path,data):
    path.write_text(json.dumps(data,indent=2,default=str)+'\n',encoding='utf-8')

def prepare():
    local=json.loads(LOCAL.read_text())
    assert local['cohort_hash']=='9ef4b9297c939e0d34a99a4d606d0f9c8c2a42a0a7b9aee20d947823ef66f2d5'
    verified=[]
    for item in local['input_artifacts']:
        original=Path(item['path'])
        if 'Digested Data' not in original.parts:
            continue  # Renderer code is not an input-data copy.
        path=Path('F:/')/original.relative_to(original.anchor)
        actual=sha(path)
        assert actual==item['sha256'],str(path)
        verified.append({'path':str(path),'sha256':actual,'recorded_path':str(original)})
    manifest=next(x for x in verified if x['path'].endswith('cohort-manifest-v1.parquet'))
    cohort=pd.read_parquet(manifest['path'])
    assert len(cohort)==57 and cohort.primary_included.all()
    assert cohort.condition_id.value_counts().to_dict()=={'delay':29,'control':28}
    outcomes=[x for x in verified if Path(x['path']).name=='candidate-trial-outcomes-corrected-v1.parquet']
    assert len(outcomes)==57
    trials=pd.concat([pd.read_parquet(x['path']) for x in outcomes],ignore_index=True)
    trials=trials.loc[trials.metric_id.eq('legacy_distal_angular_speed')&trials.alignment.eq('CS')&trials.trial_number.between(5,94)]
    assert len(trials)==5130 and set(trials.fish_id)==set(cohort.fish_id)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    source=ROOT/'sources'/(stamp+'-authenticated-f-recovery')
    source.mkdir(parents=True,exist_ok=False)
    REVIEW.mkdir(parents=True,exist_ok=True)
    # This table exists only to compare the rejected all-frame definition;
    # it is never a source for the corrected bout-only curves.
    oldfish=trials[['fish_id','condition_id','trial_number']].copy()
    oldfish['Fish median response / baseline']=trials.response_total_activity/trials.baseline_total_activity
    oldpath=source/'regenerated-all-frame-reference.parquet'
    oldfish.to_parquet(oldpath,index=False)
    lineage={'status':'regenerated from hash-matching original F inputs; not recovered J export bytes',
             'authorization':'User: Use F: if the inputs match (2026-10-09)',
             'recorded_local_sidecar':{'path':str(LOCAL.resolve()),'sha256':sha(LOCAL)},
             'cohort_hash':local['cohort_hash'],'verified_input_count':len(verified),'inputs':verified}
    dump(source/'storage-recovery.json',lineage)
    panel=source/'authenticated-legacy-source.figure.json'
    dump(panel,{'analysis_identity':{'metric_id':'legacy_distal_angular_speed','cohort_hash':local['cohort_hash'],
        'recovery':'definitions reconstructed on authenticated original F inputs; J exports inaccessible'},
        'inputs':outcomes+[manifest], 'code_dependencies':[], 'outputs':[],
        'panel_data':{'path':str(oldpath),'sha256':sha(oldpath)},'storage_recovery':lineage})
    print(f'F authentication passed: {len(verified)} recorded input hashes, 57 fish, 5130 scheduled trials',flush=True)
    return panel

def call(module,args):
    sys.argv=[module.__file__]+args
    module.main()

def main():
    import render_figure2_delay_legacy_metric_lme as mean
    if '--resume' in sys.argv:
        state=json.loads((REVIEW/'f-recovery-current.json').read_text())
        panel=Path(state['source_panel']); meanout=Path(state['mean'])
        side=json.loads((meanout/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').read_text())
        for item in side['inputs']+side['outputs']: assert sha(item['path'])==item['sha256']
    else:
        panel=prepare()
        meanout=REVIEW/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-delay-boutonly-lme')
        state={'assembly_root':str(ROOT),'source_panel':str(panel),'mean':str(meanout)}
        dump(REVIEW/'f-recovery-current.json',state)
        call(mean,['--assembly-root',str(ROOT),'--source-panel',str(panel),'--output-dir',str(meanout)])
        side=json.loads((meanout/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').read_text())
    snapshot=meanout/'analysis-script.py'
    if not snapshot.exists():
        assert sha(mean.__file__)==side['code'][0]['sha256']
        snapshot.write_bytes(Path(mean.__file__).read_bytes())
    import render_figure2_delay_review_display as display
    for scale in ['ratio','log-ratio']:
        if scale in state: continue
        before=set(REVIEW.iterdir())
        call(display,['--assembly-root',str(ROOT),'--analysis-dir',str(meanout),'--scale',scale])
        state[scale]=str(next(iter(set(REVIEW.iterdir())-before)))
        dump(REVIEW/'f-recovery-current.json',state)
    import render_figure2_delay_phase_lmm as phase
    if 'phase' not in state:
        before=set(REVIEW.iterdir())
        call(phase,['--review-root',str(REVIEW),'--analysis-dir',str(meanout)])
        phaseout=next(iter(set(REVIEW.iterdir())-before))
        state['phase']=str(phaseout); dump(REVIEW/'f-recovery-current.json',state)
    else: phaseout=Path(state['phase'])
    import render_figure2_delay_logmedian as median
    before=set(REVIEW.iterdir())
    call(median,['--review-root',str(REVIEW),'--analysis-dir',str(meanout),'--phase-dir',str(phaseout),'--source-panel',str(panel)])
    state['logmedian']=str(next(iter(set(REVIEW.iterdir())-before)))
    state['completed']=True
    dump(REVIEW/'f-recovery-current.json',state)
    print('ALL_F_VERSIONS_COMPLETE='+json.dumps(state),flush=True)

if __name__=='__main__': main()
