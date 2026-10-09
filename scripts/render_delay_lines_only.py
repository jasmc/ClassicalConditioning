"""Compare four observed Delay outcome definitions without bands or inference."""
from pathlib import Path
import json
import hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    state=json.loads((ROOT/'f-recovery-current.json').read_text())
    mean,median=Path(state['mean']),Path(state['logmedian'])
    fishpath=mean/'fish-ratios.parquet'
    side=json.loads((mean/'Fig2_PanelG_delay_legacy_LME_bootstrap5000.figure.json').read_text())
    record=next(r for r in side['outputs'] if Path(r['path'])==fishpath)
    assert sha(fishpath)==record['sha256']
    windows=median/'window-logmedians-all-trials.parquet'
    extraction=json.loads((median/'extraction-complete.json').read_text())
    assert sha(windows)==extraction['table_sha256']
    fish=pd.read_parquet(fishpath)
    rows=pd.read_parquet(windows).merge(fish[['fish_id','trial_number']],on=['fish_id','trial_number'],validate='one_to_one')
    assert len(rows)==len(fish)==4811 and rows.fish_id.nunique()==57
    data=fish.rename(columns={'ratio':'mean_ratio'}).merge(rows[['fish_id','trial_number','baseline_window_median','response_window_median','baseline_logmedian','response_logmedian']],on=['fish_id','trial_number'],validate='one_to_one')
    data['log_mean_ratio']=np.log(data.mean_ratio)
    data['median_ratio']=data.response_window_median/data.baseline_window_median
    data['B_trial']=data.response_logmedian-data.baseline_logmedian
    summary=data.groupby(['condition_id','trial_number'],observed=True)[['mean_ratio','log_mean_ratio','median_ratio','B_trial']].median().reset_index()
    # Verify the newly displayed lines reproduce the saved observed curves.
    for directory,column in [(mean,'mean_ratio'),(median,'B_trial')]:
        old=pd.read_parquet(directory/'bootstrap-summary.parquet')
        check=summary.merge(old[['condition_id','trial_number','median']],on=['condition_id','trial_number'],validate='one_to_one')
        np.testing.assert_allclose(check[column],check['median'],atol=1e-14,rtol=0)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/(stamp+'-delay-lines-only'); out.mkdir()
    summary.to_csv(out/'condition-median-lines.csv',index=False)
    data.to_parquet(out/'fish-trial-values.parquet',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':11})
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,layout='constrained')
    specs=[('A','mean_ratio','Arithmetic window means','mean(response) / mean(baseline)',1),
        ('B','log_mean_ratio','Log of arithmetic-mean ratio','ln[mean(response) / mean(baseline)]',0),
        ('C','median_ratio','Raw window medians','median(response) / median(baseline)',1),
        ('D','B_trial','Historical LogMedian difference = handoff B','median(ln response) - median(ln baseline)',0)]
    ratios=np.concatenate([summary.mean_ratio,summary.median_ratio,np.exp(summary.log_mean_ratio),np.exp(summary.B_trial)])
    lower=np.floor(ratios.min()*20)/20; upper=np.ceil(ratios.max()*20)/20
    for ax,(letter,column,title,formula,reference) in zip(axes.ravel(),specs):
        for condition,color in [('control','#27aae1'),('delay','#f00098')]:
            d=summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
            ax.plot(d.trial_number,d[column],color=color,lw=1.5,label=condition.capitalize()).pop().set_gid(f'delay-{letter}-median-{condition}')
        ax.axhline(reference,color='.4',lw=.8,zorder=0)
        for boundary in [14.5,64.5]: ax.axvline(boundary,color='.6',ls=':',lw=.8,zorder=0)
        for trial,label in [(9.5,'Pre'),(39.5,'Training'),(79.5,'Test')]:
            ax.text(trial,.96,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9,color='.5')
        ax.set(xlim=(4,95),ylim=(lower,upper) if reference==1 else (np.log(lower),np.log(upper)),
            title=letter+' | '+title+'\n'+formula,ylabel='Median across fish')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',frameon=False,fontsize=10)
    for ax in axes[-1]: ax.set_xlabel('Global CS trial')
    fig.suptitle('Delay | four observed data definitions\nSame 57 fish, 4,811 usable fish-trials; baseline [-15,0) s, response [0,9) s',fontsize=15)
    fig.supxlabel('Valid bout frames only. A/B/C retain zero vigor; D logs positive vigor only. No smoothing.',fontsize=10)
    for ext in ['png','svg']: fig.savefig(out/('Delay_four_versions_lines_only.'+ext),dpi=180)
    plt.close(fig)
    record={'status':'descriptive comparison; no panel freeze or inference',
        'definitions':specs,'colors':{'blue':'control','magenta':'delay'},
        'bootstrap_or_model_results_used':False,'bands':False,'stats_marks':False,
        'source_inputs':[{'path':str(p),'sha256':sha(p)} for p in [fishpath,windows]],
        'scheduled_trials':[5,94],'usable_fish_trials':4811,'fish':57,
        'zero_policy':'means/raw medians retain zero; LogMedian excludes nonpositive bout frames before log',
        'renderer':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'outputs':[{'path':str(p),'sha256':sha(p)} for p in out.iterdir()]}
    (out/'lines-only.figure.json').write_text(json.dumps(record,indent=2)+'\n')
    print('OUTPUT_DIRECTORY='+str(out))

if __name__=='__main__': main()
