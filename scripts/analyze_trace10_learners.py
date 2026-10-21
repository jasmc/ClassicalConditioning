"""Transparent acquisition ranking and audit for the approved fixed10sTrace cohort."""
from pathlib import Path
import argparse,json,os,sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from analyze_figure2_trace10 import OUT as DATA,PROJECT,load,ref,save,check
from classical_conditioning.figure_workspace import Snapshot

OUT=Path('J:/ClassicalConditioning Outputs/current/reviews/figure-01-trace10-learner-selection-20261020')
STATE=REPO/'docs/paper/figures/figure-01/trace10-learner-selection/current-state.json'
METHOD=REPO/'records/figure-reviews/figure-01/trace10-learner-method-before-ranking-20261020.md'
CURRENT='20230307_04'
EPOCHS=[(5,14),(40,64),(25,39),(50,64),(15,24),(65,69),(90,94)]
PRIMARY_BLOCKS=[(5,9),(10,14),*( (a,a+4) for a in range(40,65,5))]
VALUE='legacy_log_median_difference'

def protect():
    root=OUT.resolve();old={str(p.resolve()).casefold() for p in OUT.rglob('*') if p.is_file()}
    def guard(event,args):
        paths=[]
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            mode,flags=args[1] or '',args[2] if len(args)>2 else 0
            if any(c in mode for c in 'wax+') or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):paths=[args[0]]
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime'}:paths=[args[0]]
        elif event=='os.rename':paths=list(args[:2])
        for value in paths:
            if not isinstance(value,(str,bytes,os.PathLike)):continue
            p=Path(os.fsdecode(value)).resolve()
            if p.drive.lower() in ['d:','f:']:raise PermissionError('Read-only source '+str(p))
            if p.drive.lower()=='j:' and p!=root and root not in p.parents:raise PermissionError('Outside learner scope '+str(p))
            if str(p).casefold() in old:raise PermissionError('Preserve existing learner artifact '+str(p))
    sys.addaudithook(guard)

def median(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)];return float(np.median(a)) if len(a) else np.nan
def epoch(g,a,b,col=VALUE):return g.loc[g.trial_number.between(a,b),col]
def eligible(g,a,b,minimum):return all(epoch(g,start,min(start+4,b)).notna().sum()>=minimum for start in range(a,b+1,5))
def score(g,drop=()):
    x=g.loc[~g.trial_number.isin(drop)];return median(epoch(x,5,14))-median(epoch(x,40,64))

def calculate():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Preserve existing learner run')
    if not OUT.parent.is_dir():raise FileNotFoundError('Authenticated JOAQUIM required')
    OUT.mkdir(exist_ok=True);protect()
    roster=load(DATA/'common-roster.json');assert roster['status']=='author-approved'
    calc=load(DATA/'calculation.json');check(calc['outputs']+calc['fish_records'])
    plan=load(DATA/'plan.json');selected={r['recording_id']:r for r in roster['selected']};excluded={r['recording_id']:r['reason'] for r in roster['excluded']}
    windows=pd.read_parquet(DATA/'windows-observed.parquet');assert windows.fish_id.nunique()==63
    states=[REPO/'docs/paper/figures/figure-01/current-state.json',REPO/'docs/paper/figures/figure-02/current-state.json',REPO/'docs/paper/figures/figure-01/10strace-example/current-state.json',STATE]
    ticket=Snapshot([*states,Path(__file__),METHOD,DATA/'common-roster.json',DATA/'calculation.json',DATA/'windows-observed.parquet'])
    save(OUT/'starting-ticket.json',ticket.hashes)
    save(OUT/'plan.json',{'roster':ref(DATA/'common-roster.json'),'full_inventory':ref(DATA/'plan.json'),'cached_values':ref(DATA/'windows-observed.parquet'),'pre_ranking_method':ref(METHOD),'code':ref(Path(__file__)),'comparator':CURRENT,'source_drives_read_only':['D:','F:'],'figure_replacement_or_freeze_authorized':False,'cohort':roster['counts'],'classifier_status':'provisional operational expression; no validated biological categories'})
    rows=[];deletions=[];protocol_audits=[]
    for row in plan['cohort']:
        rid=row['recording_id'];g=windows.loc[windows.fish_id.eq(rid)].sort_values('trial_number');rec=load(DATA/'fish'/(rid+'.json'));check(rec['outputs'])
        r={'fish_id':rid,'condition':row['condition'],'era':rid[:4],'selected_common_roster':rid in selected,'exclusion_reason':excluded.get(rid,''),'shared_control_identity':row['shared_control_identity'],'incomplete_protocol':row['incomplete_protocol'],'observed_window_trials':len(g),'finite_outcome_trials':int(g[VALUE].notna().sum()),'primary_score':score(g)}
        for a,b in EPOCHS:
            key=f'{a}_{b}';r['median_'+key]=median(epoch(g,a,b));r['n_'+key]=int(epoch(g,a,b).notna().sum())
            phase=g.loc[g.trial_number.between(a,b)];den=phase.response_measured_frames.sum()
            r['positive_moving_frame_fraction_'+key]=float(phase.response_positive_moving_samples.sum()/den) if den else np.nan
            r['baseline_log_median_'+key]=median(phase.baseline_log_median)
            r['response_log_median_'+key]=median(phase.response_log_median)
            r['median_measured_response_frames_'+key]=median(phase.response_measured_frames)
            for minimum in [2,3,4]:r[f'eligible_{key}_min{minimum}']=eligible(g,a,b,minimum)
        for a,b in PRIMARY_BLOCKS:r[f'block_{a}_{b}_n']=int(epoch(g,a,b).notna().sum())
        for minimum in [2,3,4]:r[f'primary_eligible_min{minimum}']=r[f'eligible_5_14_min{minimum}'] and r[f'eligible_40_64_min{minimum}']
        r['baseline_drift_late_minus_pre']=r['baseline_log_median_40_64']-r['baseline_log_median_5_14']
        for a,b in [(25,39),(50,64),(15,24)]:r[f'score_{a}_{b}']=r['median_5_14']-r[f'median_{a}_{b}']
        for a,b in PRIMARY_BLOCKS:
            value=score(g,range(a,b+1));deletions.append({'fish_id':rid,'scenario':f'drop_block_{a}_{b}','kind':'block','score':value})
        for t in [*range(5,15),*range(40,65)]:deletions.append({'fish_id':rid,'scenario':'drop_trial_'+str(t),'kind':'trial','score':score(g,[t])})
        r['catch_outcome_median']=median(g.loc[g.trial_number.isin([25,39,53,59]),VALUE]);r['catch_n']=int(g.loc[g.trial_number.isin([25,39,53,59]),VALUE].notna().sum())
        r['catch_late53_59_median']=median(g.loc[g.trial_number.isin([53,59]),VALUE])
        r['test_recovery_lateTest_minus_lateTrain']=r['median_90_94']-r['median_40_64']
        if rec['heatmap']=='complete':
            info=rec['heatmap_provenance'];r.update(heatmap_finite_cells=info['finite_cells'],heatmap_defined_trials=info['defined_trials'],expected_us_s=info['expected_us_s'],paired_training_trials=info['paired_training_trials'],frame_id_gaps=info['cadence']['missing_frame_ids'],buffer_capacity_exceeded=info['cadence']['buffer_capacity_exceeded'])
        else:r.update(heatmap_finite_cells=np.nan,heatmap_defined_trials=np.nan,expected_us_s=np.nan,paired_training_trials=np.nan,frame_id_gaps=0,buffer_capacity_exceeded=rid=='20230315_04')
        # Fresh recorded-event check, independent of the cache's expected-US summary.
        pp=PROJECT/'Processed data'/rid/'stimulus_events.parquet';events=pd.read_parquet(pp);cs=events.loc[events.Type.eq('Cycle')].sort_values('Beg');us=events.loc[events.Type.eq('Reinforcer'),'Beg'].to_numpy(float);pairs=[];catches=[]
        if row['condition']=='trace':
            for trial,onset in enumerate(cs.Beg.iloc[14:64],15):
                delta=(us-float(onset))/1000;near=delta[np.abs(delta-20)<=.1]
                if len(near)==1:pairs.append({'trial':trial,'latency_s':float(near[0])})
                elif len(near)==0:catches.append(trial)
                else:raise ValueError('Ambiguous US timing '+rid)
            r.update(actual_us_min_s=min(x['latency_s'] for x in pairs),actual_us_max_s=max(x['latency_s'] for x in pairs),actual_paired_training=len(pairs),unpaired_training_cs=','.join(map(str,catches)))
            if rid in selected:assert len(pairs)==46 and catches==[25,39,53,59]
        protocol_audits.append({'fish_id':rid,'source':ref(pp),'cs_count':len(cs),'pairs':pairs,'unpaired_training_cs':catches})
        # Supported trial coverage distinguishes missing/invalid tracking from immobility.
        cp=PROJECT/'Processed data'/rid/'candidate-trial-outcomes-corrected_coverage.parquet';coverage=pd.read_parquet(cp)
        cov=coverage.loc[coverage.metric_id.eq('legacy_distal_angular_speed')]
        for a,b in [(5,14),(40,64)]:
            part=cov.loc[cov.trial_number.between(a,b)]
            r[f'valid_response_fraction_{a}_{b}']=median(part.response_valid_fraction)
        r['tracking_coverage_source']=str(cp);r['tracking_coverage_sha256']=ref(cp)['sha256']
        rows.append(r)
    table=pd.DataFrame(rows);deletion=pd.DataFrame(deletions)
    for minimum in [2,3,4]:
        mask=table.selected_common_roster&table.condition.eq('trace')&table[f'primary_eligible_min{minimum}']
        table.loc[mask,f'rank_min{minimum}']=table.loc[mask,'primary_score'].rank(ascending=False,method='min')
    primary=table.selected_common_roster&table.primary_eligible_min3
    controls=table.loc[primary&table.condition.eq('control')]
    for a,b in [(25,39),(50,64),(15,24)]:
        mask=table.selected_common_roster&table.condition.eq('trace')&table.eligible_5_14_min3&table[f'eligible_{a}_{b}_min3']
        table.loc[mask,f'rank_{a}_{b}']=table.loc[mask,f'score_{a}_{b}'].rank(ascending=False,method='min')
    for idx,row in table.iterrows():
        reference=controls.loc[controls.fish_id.ne(row.fish_id)] if row.condition=='control' else controls
        table.loc[idx,'control_reference_n']=len(reference)
        table.loc[idx,'descriptive_control_upper_tail']=(1+int((reference.primary_score>=row.primary_score).sum()))/(1+len(reference)) if np.isfinite(row.primary_score) else np.nan
        era=reference.loc[reference.era.eq(row.era)];table.loc[idx,'era_control_reference_n']=len(era)
        table.loc[idx,'era_control_upper_tail']=(1+int((era.primary_score>=row.primary_score).sum()))/(1+len(era)) if len(era) and np.isfinite(row.primary_score) else np.nan
        d=deletion.loc[deletion.fish_id.eq(row.fish_id)]
        for kind in ['block','trial']:
            x=d.loc[d.kind.eq(kind),'score'];table.loc[idx,kind+'_deletion_score_min']=x.min();table.loc[idx,kind+'_deletion_score_max']=x.max();table.loc[idx,kind+'_deletion_positive_fraction']=float(x.gt(0).mean())
    eligible_ids=table.loc[primary&table.condition.eq('trace'),'fish_id']
    deletion['rank']=np.nan;dm=deletion.fish_id.isin(eligible_ids)
    deletion.loc[dm,'rank']=deletion.loc[dm].groupby('scenario').score.rank(ascending=False,method='min')
    for kind in ['block','trial']:
        extrema=deletion.loc[dm&deletion.kind.eq(kind)].groupby('fish_id')['rank'].agg(['min','max'])
        table[kind+'_deletion_rank_best']=table.fish_id.map(extrema['min']);table[kind+'_deletion_rank_worst']=table.fish_id.map(extrema['max'])
    table['provisional_strong_suppression_expression']=primary&table.condition.eq('trace')&table.primary_score.gt(0)&table.median_40_64.lt(0)&table.descriptive_control_upper_tail.le(.05)&table.block_deletion_positive_fraction.eq(1)
    table['classification_status']=np.where(~table.selected_common_roster,'technical exclusion; available score retained',np.where(~table.primary_eligible_min3,'insufficient distributed feature coverage',np.where(table.provisional_strong_suppression_expression,'provisional strong suppression expression','eligible continuous score; criterion not met; not proven nonlearning')))
    table.to_csv(OUT/'full-score-coverage-exclusion-table.csv',index=False);deletion.to_csv(OUT/'deletion-and-rank-sensitivity.csv',index=False)
    controls[['fish_id','era','primary_score','control_reference_n','descriptive_control_upper_tail','era_control_reference_n','era_control_upper_tail']].to_csv(OUT/'control-reference-leave-one-out.csv',index=False) if 'control_reference_n' in controls else table.loc[primary&table.condition.eq('control'),['fish_id','era','primary_score','control_reference_n','descriptive_control_upper_tail','era_control_reference_n','era_control_upper_tail']].to_csv(OUT/'control-reference-leave-one-out.csv',index=False)
    top=table.loc[primary&table.condition.eq('trace')].sort_values('primary_score',ascending=False).head(5).fish_id.tolist();shortlist=list(top)
    for col in ['rank_25_39','rank_50_64']:
        shortlist.extend(table.loc[table[col].le(3),'fish_id'].tolist())
    shortlist=list(dict.fromkeys([*shortlist,CURRENT]));save(OUT/'shortlist.json',{'fish':shortlist,'primary_top5':top,'comparator':CURRENT,'selection_rule':'pre-ranking bounded shortlist; no preferred candidate selected yet'})
    save(OUT/'protocol-audit.json',{'rows':protocol_audits})
    save(OUT/'calculation.json',{'status':'scores and audits computed; visual candidate recommendation pending','all_inventory_rows':len(table),'common_roster':roster['counts'],'eligible_features':table.loc[primary].condition.value_counts().to_dict(),'provisional_flagged':table.loc[table.provisional_strong_suppression_expression,'fish_id'].tolist(),'control_tail_resolution':1/(len(controls)+1),'method':ref(METHOD),'plan':ref(OUT/'plan.json'),'shortlist':ref(OUT/'shortlist.json'),'outputs':[ref(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['calculation.json']],'limitations':['no independently validated binary learner category','positive moving-frame intensity is distinct from movement probability','control-tail ranks are descriptive, exchangeability not established','deletion ranges are sensitivity, not confidence intervals','Test has been observed; no prospective holdout claim']})
    ticket.check()
    fields=['fish_id','primary_score','n_5_14','n_40_64','median_5_14','median_40_64','rank_min3','rank_25_39','rank_50_64','descriptive_control_upper_tail','provisional_strong_suppression_expression']
    print(table.loc[table.condition.eq('trace')&table.selected_common_roster].sort_values('rank_min3')[fields].to_string(index=False));print('SHORTLIST',shortlist)

if __name__=='__main__':calculate()
