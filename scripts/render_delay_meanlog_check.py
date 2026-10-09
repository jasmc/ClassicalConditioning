"""Fresh mean(log positive bout frames) comparison, without bands or inference."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import render_figure2_delay_logmedian as extractor

ROOT=Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')

def main():
    state=json.loads((ROOT/'f-recovery-current.json').read_text())
    source=Path(state['logmedian'])
    oldpath=source/'window-logmedians-all-trials.parquet'
    saved=json.loads((source/'extraction-complete.json').read_text())
    assert extractor.sha(oldpath)==saved['table_sha256']
    old=pd.read_parquet(oldpath)
    out=ROOT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-delay-meanlog-check')
    out.mkdir()
    print('OUTPUT_DIRECTORY='+str(out),flush=True)
    extractor.PANEL_SOURCE=Path(state['source_panel'])
    rows,lineage=extractor.extract(out)
    check=rows.merge(old,on=['fish_id','trial_number'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==len(rows)==len(old)==5130
    for column in ['baseline_logmedian','response_logmedian','baseline_positive_frames','response_positive_frames']:
        np.testing.assert_allclose(check[column+'_new'],check[column+'_old'],rtol=0,atol=1e-14,equal_nan=True)
    rows['B_logmedian']=rows.response_logmedian-rows.baseline_logmedian
    rows['B_meanlog']=rows.response_logmean-rows.baseline_logmean
    np.testing.assert_array_equal(np.isfinite(rows.B_logmedian),np.isfinite(rows.B_meanlog))
    assert np.isfinite(rows.B_meanlog).sum()==4811
    rows.to_parquet(out/'fish-trial-window-summaries.parquet',index=False)
    summary=rows.groupby(['condition_id','trial_number'],observed=True)[['B_logmedian','B_meanlog']].median().reset_index()
    counts=rows.groupby(['condition_id','trial_number'],observed=True).B_meanlog.count().reset_index(name='contributing_fish')
    summary=summary.merge(counts,on=['condition_id','trial_number'],validate='one_to_one')
    summary.to_csv(out/'condition-median-lines.csv',index=False)
    # The condition aggregation remains median across fish in both versions.
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(11.5,9),sharex=True,layout='constrained')
    values=summary[['B_logmedian','B_meanlog']].to_numpy()
    lo=np.floor(np.nanmin(values)*20)/20; hi=np.ceil(np.nanmax(values)*20)/20
    colors={'control':'#27aae1','delay':'#f00098'}
    for ax,column,title in [(axes[0],'B_logmedian','Historical LogMedian: median(ln response) - median(ln baseline)'),
        (axes[1],'B_meanlog','New mean-log version: mean(ln response) - mean(ln baseline)')]:
        for condition,color in colors.items():
            d=summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
            ax.plot(d.trial_number,d[column],color=color,lw=1.5,label=condition.capitalize())
        ax.set_title(title)
    for condition,color in colors.items():
        d=summary.loc[summary.condition_id.eq(condition)].sort_values('trial_number')
        axes[2].plot(d.trial_number,d.B_logmedian,color=color,lw=1.4,ls='--',label=condition.capitalize()+' | LogMedian')
        axes[2].plot(d.trial_number,d.B_meanlog,color=color,lw=1.5,label=condition.capitalize()+' | mean-log')
    axes[2].set_title('Direct comparison: solid = mean-log; dashed = LogMedian')
    for ax in axes:
        ax.axhline(0,color='.4',lw=.8,zorder=0)
        for boundary in [14.5,64.5]: ax.axvline(boundary,color='.6',ls=':',lw=.8,zorder=0)
        for trial,label in [(9.5,'Pre'),(39.5,'Training'),(79.5,'Test')]:
            ax.text(trial,.96,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9,color='.5')
        ax.set(xlim=(4,95),ylim=(lo,hi),ylabel='Median across fish\n(log-vigor change)')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',frameon=False,fontsize=9,ncol=2 if ax is axes[2] else 1)
    axes[-1].set_xlabel('Global CS trial')
    fig.suptitle('Delay | change only the within-window summary\nSame positive bout frames, 57 fish and 4,811 usable fish-trials',fontsize=14)
    fig.supxlabel('Baseline [-15,0) s; response [0,9) s. Log frames first; no pseudocount, smoothing, bands or statistical marks.',fontsize=9)
    for ext in ['png','svg']: fig.savefig(out/('Delay_meanlog_vs_LogMedian.'+ext),dpi=180)
    plt.close(fig)
    verification={'same_window_logmedians_and_sample_counts':True,'same_trial_eligibility':True,
        'scheduled_rows':len(rows),'usable_rows':int(np.isfinite(rows.B_meanlog).sum()),
        'cohort_fish':rows.fish_id.nunique(),'population_aggregation':'median across fish for both outcomes',
        'meanlog_formula':'mean(ln positive response bout frames) minus mean(ln positive baseline bout frames)',
        'meanlog_is_not':'ln arithmetic response mean minus ln arithmetic baseline mean',
        'source_inputs':lineage,'old_trial_data':{'path':str(oldpath),'sha256':extractor.sha(oldpath)},
        'code':[{'path':str(p.resolve()),'sha256':extractor.sha(p)} for p in [Path(__file__),Path(extractor.__file__)]],
        'status':'descriptive candidate; no model fit, statistical transfer, panel freeze or prior export replacement'}
    verification['outputs']=[{'path':str(p),'sha256':extractor.sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
    extractor.dump(out/'comparison.figure.json',verification)
    print('COMPLETED_DIRECTORY='+str(out),flush=True)

if __name__=='__main__': main()
