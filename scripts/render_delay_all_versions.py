"""Additional matched-input comparisons and PDF page manifest (no model refits)."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from render_figure2_delay_legacy_metric_lme import bootstrap_trajectories,stars

REVIEW=Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')
COLORS={'control':'#27aae1','delay':'#f00098'}

def observed(ax,summary,title,ylabel,reference=1,spread='ci'):
    for cond,color in COLORS.items():
        d=summary.loc[summary.condition_id.eq(cond)].sort_values('trial_number')
        ax.plot(d.trial_number,d['median'],color=color,lw=1.3,label=cond.capitalize())
        low,high=('ci_lower','ci_upper') if spread=='ci' else ('q25','q75')
        ax.fill_between(d.trial_number,d[low],d[high],color=color,alpha=.22,lw=0)
    ax.axhline(reference,color='.4',lw=.7)
    for b in [14.5,64.5]: ax.axvline(b,color='.5',lw=.8,ls=':')
    ax.set(xlim=(4,95),ylabel=ylabel,title=title)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=9)

def main():
    state=json.loads((REVIEW/'f-recovery-current.json').read_text())
    assert state['completed']
    mean,median,phase=(Path(state[k]) for k in ['mean','logmedian','phase'])
    out=REVIEW/'all-versions-comparison'
    out.mkdir(exist_ok=True)
    mean_summary=pd.read_parquet(mean/'bootstrap-summary.parquet')
    rows=pd.read_parquet(median/'window-logmedians-all-trials.parquet')
    base=pd.read_parquet(mean/'model-input.parquet')
    matched=rows.merge(base[['fish_id','trial_number']],on=['fish_id','trial_number'],validate='one_to_one')
    assert len(matched)==4811
    literal=matched[['fish_id','condition_id','trial_number']].copy()
    literal['ratio']=matched.response_window_median/matched.baseline_window_median
    assert np.isfinite(literal.ratio).all() and (literal.ratio>0).all()
    literal.to_parquet(out/'literal-window-median-ratios.parquet',index=False)
    literal_summary,literal_draws=bootstrap_trajectories(literal,n_boot=5000,seed=10)
    literal_summary.to_parquet(out/'literal-window-median-bootstrap.parquet',index=False)
    historical=matched[['fish_id','condition_id','trial_number']].copy()
    historical['ratio']=np.exp(matched.response_logmedian-matched.baseline_logmedian)
    historical_summary,historical_draws=bootstrap_trajectories(historical,n_boot=5000,seed=10)
    historical_summary.to_parquet(out/'exp-logmedian-bootstrap.parquet',index=False)
    for condition in COLORS:
        original=np.load(mean/(condition+'-bootstrap-draws.npz'))
        np.testing.assert_array_equal(original['indices'],literal_draws[condition]['indices'])
        np.testing.assert_array_equal(original['indices'],historical_draws[condition]['indices'])
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    bounds=np.concatenate([s[['ci_lower','ci_upper']].to_numpy().ravel() for s in [mean_summary,literal_summary,historical_summary]])
    bounds=bounds[np.isfinite(bounds)]
    margin=.05*(bounds.max()-bounds.min())
    for ax,summary,title in zip(axes,[mean_summary,literal_summary,historical_summary],
        ['Arithmetic window means: mean(response) / mean(baseline)',
         'Literal window medians: median(response) / median(baseline)',
         'Historical LogMedian summary: exp[median(ln response) − median(ln baseline)]']):
        observed(ax,summary,title,'Condition median ratio\n95% fish-bootstrap CI')
        ax.set_ylim(bounds.min()-margin,bounds.max()+margin)
    axes[-1].set_xlabel('Global CS trial')
    fig.suptitle('Same fish, initial bout mask and windows; identical 5,000 fish draws, seed 10\nMean/median retain zero vigor; LogMedian logs positive vigor only',fontsize=12)
    fig.savefig(out/'matched-window-summaries.png',dpi=190); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    observed(axes[0],mean_summary,'Uncertainty of the condition median: 95% bootstrap confidence interval','Median ratio [95% CI]')
    observed(axes[1],mean_summary,'Between-fish spread: 25th–75th percentiles (IQR)','Median ratio [fish IQR]',spread='iqr')
    axes[-1].set_xlabel('Global CS trial')
    fig.suptitle('Same blue/magenta median lines; two different meanings of a band',fontsize=13)
    fig.savefig(out/'ci-versus-iqr.png',dpi=190); plt.close(fig)
    # Phase trial marks with the original mean-derived block D/M/R tests.
    dtests=pd.read_csv(mean/'D-interaction-tests.csv'); local=pd.read_csv(mean/'MR-local-block-tests.csv')
    trial=pd.read_csv(phase/'phase-contrasts.csv')
    fig=plt.figure(figsize=(11,7),layout='constrained')
    grid=fig.add_gridspec(2,1,height_ratios=[1.7,3])
    marks=fig.add_subplot(grid[0]); ax=fig.add_subplot(grid[1],sharex=marks)
    marks.set(ylim=(-.5,4.7),yticks=[4,3,2,1,0],yticklabels=['D: interaction · Holm8','M: block difference · BH9','R: slope · raw','R: slope · BH9','Trial: phase change · BH90'])
    marks.tick_params(length=0,labelbottom=False,labelsize=9)
    marks.spines[['top','right','left','bottom']].set_visible(False)
    for _,r in dtests.iterrows():
        if r.p_holm<.05: marks.text(r.center,4,'D'+stars(r.p_holm),ha='center',color='#a97900')
    for _,r in local.iterrows():
        for col,y,label,color in [('mean_p_fdr',3,'M','.4'),('slope_p_raw',2,'R','#a43c24'),('slope_p_fdr',1,'R','#a43c24')]:
            if r[col]<.05: marks.text(r.center,y,label+stars(r[col]),ha='center',color=color)
    sig=trial.loc[trial.change_p_fdr<.05]
    marks.scatter(sig.trial_number,np.zeros(len(sig)),color='black',marker='*',s=26,lw=0)
    for i,name in enumerate(['Pre','Tr1','Tr2','Tr3','Tr4','Tr5','Te1','Te2','Te3']):
        marks.text(9.5+i*10,4.55,name,ha='center',fontsize=9,color='.4')
    observed(ax,mean_summary,'Observed medians stay visible; black stars use a separate phase-aware LMM','Response / baseline bout intensity\ncondition median [95% fish-bootstrap CI]')
    ax.set_xlabel('Global CS trial')
    fig.suptitle('Arithmetic window means | phase-aware trial tests\nD/M/R retain their block definitions; no onset claim',fontsize=13)
    fig.savefig(out/'mean-phase-statistics.png',dpi=190); plt.close(fig)
    # Record result comparisons and validations for the single-file report.
    meanresult=json.loads((mean/'result-summary.json').read_text())
    assert (meanresult['D_significant'],meanresult['M_significant'],meanresult['R_raw_significant'],meanresult['R_FDR_significant'],meanresult['trial_FDR_significant'])==(6,4,2,1,67)
    medresult=json.loads((median/'result-summary.json').read_text())
    # Shorter axis wording for the compiled review; underlying results stay intact.
    import render_figure2_delay_logmedian as renderer
    renderer.PHASE_SOURCE=phase
    median_render=out/'logmedian-display'
    median_render.mkdir(exist_ok=True)
    tables={name:pd.read_csv(median/(name+'-contrasts.csv')) for name in ['global','phase','phase-powell','phase-intercept']}
    renderer.render(pd.read_parquet(median/'bootstrap-summary.parquet'),tables,
        pd.read_csv(median/'D-interaction-tests.csv'),pd.read_csv(median/'MR-local-block-tests.csv'),median_render)
    report={'matched_rows':len(matched),'matched_fish':matched.fish_id.nunique(),
        'literal_median_ratio_vs_exp_logmedian_max_difference':float(abs(literal.ratio-historical.ratio).max()),
        'identical_bootstrap_draw_indices':True,'mean':meanresult,'logmedian':medresult,
        'phase':json.loads((phase/'comparison-summary.json').read_text()),
        'mean_model_adequacy':json.loads((phase/'model-adequacy.json').read_text()),
        'paths':state}
    (out/'comparison-report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    png='Fig2_PanelG_delay_legacy_LME_bootstrap5000.png'
    pages=[
      {'title':'Same inputs, different window summaries','image':str(out/'matched-window-summaries.png'),'caption':'Literal medians provide the requested typical-intensity version. Means and raw medians retain zero vigor. Historical LogMedian excludes nonpositive values before logging; it can also differ with even counts because median interpolation and logarithms do not commute. All three use the same initial bout mask, 4,811 matched fish-trials, 57 fish and resampling indices. The literal-median panel is descriptive; its own LMM has not been fitted.'},
      {'title':'Bootstrap CI versus IQR','image':str(out/'ci-versus-iqr.png'),'caption':'Blue = control, magenta = Delay. Both lines are medians across fish. Upper bands estimate uncertainty about each condition median; lower bands show the middle 50% of fish. Bootstrap is the resampling method used to obtain the confidence interval, not a third type of spread.'},
      {'title':'Arithmetic means: ratio display, global trial model','image':str(Path(state['ratio'])/png),'caption':'Each fish contributes mean response bout vigor / mean baseline bout vigor. The line is the condition median. Bands are pointwise 95% whole-fish bootstrap CIs. D/M/R and black stars come from log-response LMMs; they do not directly test these median curves. The global spline yields 67 trial rejections, including all ten Pre trials; this is a warning against interpreting stars as learning onset.'},
      {'title':'Arithmetic means: natural-log ratio display','image':str(Path(state['log-ratio'])/png),'caption':'Calculate each fish’s arithmetic-mean ratio, take ln(ratio), then take the median across fish. The reference is zero. The model tests and stars are identical to the ratio display. Changing the axis/display transformation does not turn arithmetic window means into historical LogMedian.'},
      {'title':'Arithmetic means: observed medians and fitted contrasts','image':str(phase/'Fig2_Delay_observed_and_fitted_contrasts.png'),'caption':'Top: observed ratio medians, blue control and magenta Delay, with fish-bootstrap CIs. Bottom: gray global-spline and purple phase-aware fitted contrasts, with pointwise model CIs. Positive contrasts mean a larger response reduction in Delay relative to control, compared with average Pre. Models adjust ln(response) for ln(baseline) and fish random intercepts/slopes.'},
      {'title':'Arithmetic means: phase-aware statistics marks','image':str(out/'mean-phase-statistics.png'),'caption':'The phase-aware model gives Pre, training and test separate shapes and intercepts. Its trial rejections cover trials 19–70, with none in Pre. D/M/R use the same block models as the earlier mean version. The phase sensitivity reduces curve borrowing across boundaries; it does not establish onset or remove the heavy residual tails.'},
      {'title':'Historical LogMedian summary: new fitted statistics','image':str(median_render/'Fig2_Delay_LogMedian_stats.png'),'caption':'Per fish: median(ln positive response bout vigor) minus median(ln positive baseline bout vigor); then median across fish. Reference zero. Bands: 95% whole-fish bootstrap CIs. D/M/R are freshly fitted for this outcome; black stars use its phase-aware model. Test3 M/R are unavailable because that local fit is singular. No historical second log(x+1), rolling median or downsampling is added.'},
      {'title':'LogMedian and mean outcomes: fitted contrasts separately','image':str(median_render/'Fig2_Delay_LogMedian_observed_fitted.png'),'caption':'Top: observed historical-summary LogMedian differences, blue control and magenta Delay. Bottom: gray arithmetic-mean phase contrast and purple LogMedian phase contrast. These are different outcome summaries fitted with the same phase-model form, not fits through the displayed medians. The shaded contrast bands are model-based intervals, distinct from descriptive bootstrap bands.'}
    ]
    (out/'pdf-pages.json').write_text(json.dumps(pages,indent=2)+'\n')
    print('COMPARISON_DIRECTORY='+str(out),flush=True)

if __name__=='__main__': main()
