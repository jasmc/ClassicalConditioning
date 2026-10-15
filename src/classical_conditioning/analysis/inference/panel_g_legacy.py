"""Source-pinned historical rule projections and explicit sensitivity cohorts.

Historical trial-segmented pickle inputs are not available here. Corrected
detector/event projections are named as such; they are never exact old cohorts.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from .panel_g_review import (PHASE_WINDOWS,fit_with_retry,design_for,estimate,
    bootstrap_worker,simultaneous_band,timing_bounds,save_json)

REVISIONS=('2f63ef48361d6e80d1b8e1393894d7e5a3dedf55','bf46bf7b02baaf6d8138d9772f255881f86c3c78')
FILES=('1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py',
       '5_NormalizedVigorPlotting_LogMedian.py','experiment_configuration.py',
       'general_configuration.py','analysis_utils.py')
QC_BLOCKS={'Late Pre-Train':range(10,15),'Early Test':range(65,70),'Late Test':range(90,95)}
RECIPES={
 'all_selected_G':(),
 'legacy_any_block_6':('any_block_6',),
 'legacy_baseline_3_projection':('baseline_3',),
 'legacy_CR_3_projection':('cr_3',),
 'legacy_baseline_and_CR_projection':('baseline_3','cr_3'),
 'legacy_train_US_projection':('train_us',),
 'legacy_combined_with_viability_proxy':('last_us_proxy','train_us','retrain_us','baseline_3','cr_3'),
 'all_required_blocks_6_alternative':('all_blocks_6',),
}


def historical_sources(root,out):
    destination=out/'legacy-sources';destination.mkdir(exist_ok=True)
    catalog=[]
    for revision in REVISIONS:
        for name in FILES:
            raw=subprocess.run(['git','show',revision+':'+name],cwd=root,check=True,capture_output=True).stdout
            text=raw.decode()
            required={FILES[0]:['RUN_DISCARD = False','def check_viability','def check_train'],
                FILES[1]:['minimum_trials_per_fish_per_block = 6',
                          'counts[counts >= minimum_trials_per_fish_per_block]'],
                FILES[2]:['us_window_qc: int = 5','min_number_trials_with_bouts_per_block: int = 3']}
            if any(token not in text for token in required.get(name,[])):
                raise ValueError('Historical rule source differs from reviewed recipe: '+revision+':'+name)
            path=destination/(revision[:7]+'-'+name);path.write_bytes(raw)
            catalog.append({'revision':revision,'path_at_revision':name,
                'sha256':hashlib.sha256(raw).hexdigest(),'snapshot':str(path)})
    save_json(out/'legacy-source-catalog.json',{'sources':catalog,
        'experiment':'allDelay','preprocessing_discard_default':False,'normalized_discard_default':False,
        'normalized_minimum':6,'normalized_logic':'ANY block, then retain other eligible rows',
        'US_window_s':[0,5],'baseline_window_s':[-15,0],'CR_window_s':[0,9],
        'boundaries':'inclusive','US_train_ordinals':[18,63],
        'QC_blocks':{k:list(v) for k,v in QC_BLOCKS.items()},'minimum_bout_trials_per_QC_block':3,
        'exact_historical_inputs_available':False,'primary_membership_changed':False,
        'recipes_registered_before_effect_estimation':RECIPES})


def bout_in_window(times,event,left,right):
    a=np.searchsorted(times,event+int(left*1000),side='left')
    b=np.searchsorted(times,event+int(right*1000),side='right')
    return b>a


def coverage_rules(eligible_trials,baseline_counts,cr_counts):
    counts=[sum(a<=t<=a+9 for t in eligible_trials) for a in range(5,95,10)]
    return {'any_block_6':any(n>=6 for n in counts),'all_blocks_6':all(n>=6 for n in counts),
            'baseline_3':all(n>=3 for n in baseline_counts),
            'cr_3':all(n>=3 for n in cr_counts)}


def evaluate_legacy(out,root):
    historical_sources(root,out)
    windows=pd.read_parquet(out/'scheduled-window-outcomes.parquet')
    data=pd.read_parquet(out/'model-input.parquet')
    lineage=json.loads((out/'input-lineage.json').read_text())
    sources={(Path(e['path']).parent.name,Path(e['path']).name):e for e in lineage}
    rows,details=[] ,[]
    for index,((condition,fish),group) in enumerate(windows.groupby(['condition_id','fish_id'],observed=True)):
        entries=[sources[(fish,name)] for name in ['movement_state_candidates-corrected-v2.parquet','stimulus_events.parquet']]
        for e in entries:
            if hashlib.sha256(Path(e['path']).read_bytes()).hexdigest()!=e['sha256']:
                raise ValueError('Changed source: '+e['path'])
        movement=pd.read_parquet(entries[0]['path'],columns=['AbsoluteTime','moving','valid'])
        protocol=pd.read_parquet(entries[1]['path']).sort_values('Beg',kind='stable')
        times=movement.loc[movement.moving & movement.valid,'AbsoluteTime'].to_numpy(np.int64)
        us=protocol.loc[protocol.Type.astype(str).eq('Reinforcer')].reset_index(drop=True)
        train=[]
        for number,event in enumerate(us.itertuples(),1):
            present=bout_in_window(times,int(event.Beg),0,5)
            if 18<=number<=63:train.append(present)
            details.append({'condition_id':str(condition),'fish_id':fish,'rule':'US_bout_projection',
                'US_ordinal':number,'train_US':18<=number<=63,'bout_present':present,
                'raw_duration_ms':int(event.End-event.Beg)})
        # The legacy rule tested a frame-annotated end in a trial-segmented
        # pickle. Raw duration below is a separate proxy, never an exact end.
        last_proxy=False
        if len(us):
            last=us.iloc[-1]
            last_proxy=(last.End-last.Beg>=400 and bout_in_window(times,int(last.Beg),0,5))
        baseline_counts=[];cr_counts=[]
        for block,trials in QC_BLOCKS.items():
            subset=group.loc[group.trial_number.isin(trials)]
            bas=sum(bout_in_window(times,int(r.event_start_absolute_time_ms),-15,0) for r in subset.itertuples())
            cr=sum(bout_in_window(times,int(r.event_start_absolute_time_ms),0,9) for r in subset.itertuples())
            baseline_counts.append(bas);cr_counts.append(cr)
            details.append({'condition_id':str(condition),'fish_id':fish,'rule':'CS_QC_projection',
                'block':block,'observed_trials':len(subset),'baseline_bout_trials':bas,'CR_bout_trials':cr})
        trialnums=data.loc[(data.condition_id==condition)&(data.fish_id==fish),'trial_number'].to_numpy()
        row={'condition_id':str(condition),'fish_id':fish,'eligible_outcomes':len(trialnums),
            **coverage_rules(trialnums,baseline_counts,cr_counts),
            'train_us':bool(train) and all(train),'retrain_us':True,
            'retrain_reason':'historical allDelay has no Re-Train group; bypass',
            'last_us_proxy':bool(last_proxy),'last_us_exact_status':'not_evaluable',
            'last_us_exact_reason':'historical annotated US-end/segmented pickle unavailable',
            'projection':'authenticated frozen detector and event times; not historical Bout/700fps pickle'}
        rows.append(row)
        print(f'Historical rule projection {index+1}/57: {fish}',flush=True)
    matrix=pd.DataFrame(rows)
    matrix.to_csv(out/'legacy-discard-rule-matrix.csv',index=False)
    pd.DataFrame(details).to_csv(out/'legacy-discard-rule-details.csv',index=False)
    membership=[];flow=[];overlap=[]
    for recipe,rules in RECIPES.items():
        kept=np.ones(len(matrix),bool)
        for rule in rules:
            before=kept.copy();kept &= matrix[rule].to_numpy(bool)
            for condition in ['control','delay']:
                is_c=matrix.condition_id.eq(condition).to_numpy()
                flow.append({'recipe':recipe,'rule':rule,'condition_id':condition,
                    'before':int((before&is_c).sum()),'removed_here':int((before&~kept&is_c).sum()),
                    'remaining':int((kept&is_c).sum())})
        for r,keep in zip(matrix.itertuples(),kept):
            membership.append({'recipe':recipe,'condition_id':r.condition_id,'fish_id':r.fish_id,
                'included':bool(keep),'failed_rules':';'.join(rule for rule in rules if not getattr(r,rule)),
                'role':'author-authorized sensitivity only; selected G and primary cohort unchanged'})
    rule_names=sorted({r for v in RECIPES.values() for r in v})
    for a in rule_names:
        for b in rule_names:
            overlap.append({'rule_a':a,'rule_b':b,'fail_both':int((~matrix[a]&~matrix[b]).sum())})
    pd.DataFrame(membership).to_csv(out/'legacy-sensitivity-membership.csv',index=False)
    pd.DataFrame(flow).to_csv(out/'legacy-cumulative-flow.csv',index=False)
    pd.DataFrame(overlap).to_csv(out/'legacy-failure-overlap.csv',index=False)
    save_json(out/'legacy-assessment-status.json',{'status':'source_pinned_projections_complete',
        'fish':len(matrix),'recipes':list(RECIPES),'exact_historical_cohort_reproduced':False,
        'unevaluable_rule':'historical frame-annotated last-US end; original segmented Bout data',
        'raw_US_end_proxy_is_explicit_alternative':True,'primary_membership_changed':False})


def run_cohort_sensitivities(out,config,workers=3,nboot=199):
    data=pd.read_parquet(out/'model-input.parquet')
    membership=pd.read_csv(out/'legacy-sensitivity-membership.csv')
    summaries=[];timings=[]
    seeds=np.random.SeedSequence(config.seed+844).generate_state(nboot)
    for recipe,roster in membership.groupby('recipe',sort=False):
        keys=roster.loc[roster.included,['condition_id','fish_id']]
        selected=data.merge(keys,on=['condition_id','fish_id'],validate='many_to_one')
        counts=selected.groupby('condition_id',observed=True).fish_id.nunique().to_dict()
        directory=out/'cohort-sensitivities'/recipe;directory.mkdir(parents=True,exist_ok=True)
        for target in ['difference','adjusted_response']:
            if min(counts.get('control',0),counts.get('delay',0))<6:
                save_json(directory/(target+'-result.json'),{'status':'insufficient_fish','counts':counts});continue
            fitted,diagnostic=fit_with_retry(selected,target)
            save_json(directory/(target+'-fit.json'),diagnostic)
            if fitted is None:
                save_json(directory/(target+'-result.json'),{'status':'fit_failed','counts':counts});continue
            baseline=float(selected.baseline_logmedian.mean())
            features,cm,_=design_for(fitted,target,baseline);values,se=estimate(fitted,cm)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                draws=list(pool.map(bootstrap_worker,[(selected,target,int(s),baseline) for s in seeds]))
            lo,hi,critical,maxima=simultaneous_band(values,se,draws)
            success=sum(d['success'] for d in draws);accepted=success>=np.ceil(.95*nboot)
            if not accepted:lo[:]=np.nan;hi[:]=np.nan
            trial_counts=selected.groupby(['condition_id','trial_number'],observed=True).fish_id.nunique().unstack(0).reindex(range(5,95)).fillna(0)
            adequate=trial_counts.min(axis=1).to_numpy()>=config.min_local_fish
            pd.DataFrame({'trial_number':range(5,95),'estimate':values[:90],
                'simultaneous_lower':lo[:90],'simultaneous_upper':hi[:90],
                'coverage_adequate':adequate}).to_csv(directory/(target+'-trajectory.csv'),index=False)
            for i,name in enumerate(PHASE_WINDOWS):
                summaries.append({'recipe':recipe,'target':target,'contrast':name,'control_fish':counts['control'],
                    'delay_fish':counts['delay'],'eligible_rows':len(selected),'estimate':values[90+i],
                    'simultaneous_lower':lo[90+i],'simultaneous_upper':hi[90+i],
                    'bootstrap_successful':success,'bootstrap_requested':nboot,'paper_approved':False})
            if accepted:
                for f in config.threshold_fractions:
                    row=timing_bounds(range(5,95),values[:90],lo[:90],hi[:90],-np.log1p(-f),config.persistence,adequate)
                    timings.append({'recipe':recipe,'target':target,'threshold_fraction':f,**row})
            save_json(directory/(target+'-result.json'),{'status':'exploratory' if accepted else 'band_failed',
                'bootstrap_requested':nboot,'bootstrap_successful':success,'critical':critical,
                'counts':counts,'paper_approved':False,'cohort_selection':'registered rules, not chosen by significance'})
            print(f'Cohort sensitivity {recipe} / {target}: {counts}, {success}/{nboot} refits',flush=True)
    pd.DataFrame(summaries).to_csv(out/'legacy-cohort-phase-results.csv',index=False)
    pd.DataFrame(timings).replace([np.inf,-np.inf],np.nan).to_csv(out/'legacy-cohort-timing-results.csv',index=False)
