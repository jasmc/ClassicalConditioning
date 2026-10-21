"""Separate Holm18 panel families, reusing the selected raw sign/BM recipes."""
from pathlib import Path
import argparse,json,sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from classical_conditioning.analysis.bout_block_statistics import exact_sign,independent_test,VALUE,PAIRS
from analyze_figure2_trace10 import OUT,STATE,load,ref,save,check,guard,pool_equal_fish
import analyze_figure2_trace3_matched as matched

def saved_table(path,data):
    if path.exists():
        old=pd.read_csv(path) if path.suffix=='.csv' else pd.read_parquet(path)
        pd.testing.assert_frame_equal(old,data,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-14)
        return
    if path.suffix=='.csv':data.to_csv(path,index=False)
    else:data.to_parquet(path,index=False)

def blocks(windows):
    parts=[]
    for order,(label,a,b) in enumerate([('PT',10,14),('ET',65,69),('LT',90,94)]):
        g=windows.loc[windows.trial_number.between(a,b)].groupby(['condition_id','fish_id']).legacy_log_median_difference
        t=g.agg(**{VALUE:'median','eligible_log_trials':'count'}).reset_index()
        t['Selected block order']=order;t['Selected block']=label;t['minimum_trials']=3
        t['Eligible']=t.eligible_log_trials.ge(3);parts.append(t)
    return pd.concat(parts,ignore_index=True)

def separate_tests(panel,data):
    assert not data.duplicated(['condition_id','fish_id','Selected block order']).any()
    conditioned='delay' if panel=='D' else 'trace';rows=[];changes=[]
    w=data.loc[data.Eligible].pivot(index=['condition_id','fish_id'],columns='Selected block order',values=VALUE).reindex(columns=[0,1,2])
    for block in range(3):
        groups={}
        for c in [conditioned,'control']:
            x=w.xs(c)[block].dropna().to_numpy();groups[c]=x
            rows.append(dict(panel=panel,family='zero',condition=c,left=block,right=block,test='exact two-sided sign versus zero',n_pairs=len(x),**exact_sign(x)))
        t,c=groups[conditioned],groups['control']
        rows.append(dict(panel=panel,family='between',condition=conditioned+' vs control',left=block,right=block,test='two-sided Brunner-Munzel; t calibration',n_conditioned=len(t),n_control=len(c),**independent_test(t,c)))
    for lo,hi in PAIRS:
        groups={}
        for c in [conditioned,'control']:
            paired=w.xs(c)[[lo,hi]].dropna();x=(paired[hi]-paired[lo]).to_numpy();groups[c]=x
            rows.append(dict(panel=panel,family='within',condition=c,left=lo,right=hi,test='exact two-sided sign on paired fish changes',n_pairs=len(x),**exact_sign(x)))
            changes.extend(dict(panel=panel,condition=c,fish_id=rid,left=lo,right=hi,change=float(v)) for rid,v in zip(paired.index,x))
        t,c=groups[conditioned],groups['control']
        rows.append(dict(panel=panel,family='change',condition=conditioned+' vs control',left=lo,right=hi,test='two-sided Brunner-Munzel on paired fish changes; t calibration',n_conditioned=len(t),n_control=len(c),median_change_conditioned=float(np.median(t)),median_change_control=float(np.median(c)),**independent_test(t,c)))
    t=pd.DataFrame(rows);assert len(t)==18 and np.isfinite(t.p_raw).all()
    t['p_holm18']=multipletests(t.p_raw,method='holm')[1]
    t['stars']=t.p_holm18.map(lambda p:'****' if p<.0001 else '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns')
    t['family_scope']='18 tests within '+panel;t['statistical_unit']='fish'
    return t,pd.DataFrame(changes)

def de():
    guard();state=load(STATE);refs=state['draft']['matched_trace3_scientific_draft'];check([refs['data'],refs['statistics']])
    source=Path(refs['data']['path']).parent;completion=load(Path(refs['statistics']['path']))
    wp=source/'windows.parquet';wr=next(r for r in completion['outputs'] if Path(r['path'])==wp);check([wr])
    assert ref(matched.D_SOURCE)['sha256']=='65324bfb9cb50e5b690c99fd018a73ace4fb4ffc1428345b64dddd54a8e56c84'
    windows=pd.read_parquet(wp);assert windows.fish_id.nunique()==179
    d=pd.read_parquet(matched.D_SOURCE);e=blocks(windows)
    count='eligible_log_trials' if 'eligible_log_trials' in d else 'eligible_trials'
    if count in d:assert (d.Eligible==d[count].ge(3)).all()
    # Frozen D values and eligibility remain untouched; only its family is corrected.
    directory=OUT/'DE-corrections';directory.mkdir(exist_ok=True)
    for p,data in [('D',d),('E',e)]:
        target=directory/(p+'-statistics-Holm18.csv')
        if target.exists():raise FileExistsError('Preserve existing '+str(target))
        tests,changes=separate_tests(p,data);tests.to_csv(target,index=False)
        data.to_parquet(directory/(p+'-fish-blocks.parquet'),index=False)
        changes.to_parquet(directory/(p+'-paired-changes.parquet'),index=False)
    old=pd.read_parquet(source/'E-fish-blocks.parquet')
    change=e.merge(old[['fish_id','condition_id','Selected block order','Eligible']],on=['fish_id','condition_id','Selected block order'],suffixes=('_new','_old'))
    change.to_csv(directory/'E-eligibility-changes.csv',index=False)
    save(directory/'calculation.json',{'status':'computed; artwork not yet corrected','author_direction':'each panel separate stats; minimum3 eligible trials per five-trial block',
        'D_values_unchanged':True,'source_D':ref(matched.D_SOURCE),'source_E_trials':wr,
        'E_old_eligible_blocks':int(change.Eligible_old.sum()),'E_new_eligible_blocks':int(change.Eligible_new.sum()),
        'inputs':[ref(Path(__file__)),refs['data'],refs['statistics']],
        'outputs':[ref(p) for p in directory.iterdir() if p.is_file()]})
    print('D/E separate Holm18 calculations complete',flush=True)

def prepare_selected():
    guard();roster=load(OUT/'common-roster.json');assert roster['status']=='author-approved'
    records=load(OUT/'calculation.json');check(records['outputs']+records['fish_records'])
    selected=roster['selected'];ids=[r['recording_id'] for r in selected]
    windows=pd.read_parquet(OUT/'windows-observed.parquet');windows=windows.loc[windows.fish_id.isin(ids)]
    saved_table(OUT/'windows.parquet',windows.reset_index(drop=True))
    f=blocks(windows);tests,changes=separate_tests('F',f)
    saved_table(OUT/'F-fish-blocks.parquet',f);saved_table(OUT/'F-statistics-Holm18.csv',tests);saved_table(OUT/'F-paired-changes.parquet',changes)
    bins=pd.concat([pd.read_parquet(OUT/'fish'/(rid+'-bins.parquet')) for rid in ids],ignore_index=True)
    pool_equal_fish(bins,{r['recording_id']:r['condition'] for r in selected}).to_parquet(OUT/'C-pooled-bins.parquet',index=False)
    from render_figure2_delay_legacy_metric_lme import bootstrap_trajectories
    x=windows.rename(columns={'legacy_log_median_difference':'ratio'}).copy();x.condition_id=x.condition_id.replace({'trace':'delay'})
    curves,draws=bootstrap_trajectories(x,n_boot=5000,seed=10);curves.condition_id=curves.condition_id.replace({'delay':'trace'})
    curves.to_parquet(OUT/'I-trajectory.parquet',index=False)
    np.savez_compressed(OUT/'I-bootstrap-draws.npz',**{f'{c}_{k}':v for c,d in draws.items() for k,v in d.items()})
    directory=OUT/'I-models';directory.mkdir(exist_ok=True)
    windows.to_parquet(directory/'windows.parquet',index=False)
    save(directory/'plan.json',{'selected':selected,'excluded_recordings':roster['excluded'],
        'inputs':[ref(OUT/'common-roster.json'),ref(directory/'windows.parquet'),*load(OUT/'plan.json')['science_code']]})
    print('Selected C/F/I numerical summaries prepared',flush=True)

def models():
    guard();directory=OUT/'I-models';plan=load(directory/'plan.json')
    counts=pd.Series([r['condition'] for r in plan['selected']]).value_counts().to_dict()
    matched.OUT=directory;original_save=matched.save
    def save_with_scope(path,data):
        if Path(path).name=='completion.json':
            data['cohort_counts']=counts;data['excluded_recordings']=plan['excluded_recordings']
            data['display_panel']='I';data['internal_filenames']='H/D/M/R retain reference implementation names; assays are not pooled'
            data['implementation']=ref(REPO/'scripts/analyze_figure2_trace3_matched.py')
        original_save(path,data)
    matched.save=save_with_scope;matched.run_models()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['de','prepare-selected','models']);a=p.parse_args()
    {'de':de,'prepare-selected':prepare_selected,'models':models}[a.operation]()
