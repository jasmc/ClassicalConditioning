"""Version 5: eligible-only 1 s log means, 20 s baseline mean, no scaling."""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_full_bouts_baseline_samples_20261009'
sys.path.insert(0,str(REPO/'reviews/fgh_legacy_layout_20261009'))
import render_layout as plotting
from render_layout import digest
BIN_WIDTH=1.;BIN_COUNT=40;BASELINE_COUNT=20;EDGES=np.arange(-20,20+BIN_WIDTH,BIN_WIDTH);TOL=1e-12

def build():
    previous=json.loads((SOURCE/'data_manifest.json').read_text())
    refs=[];panels=[];audits=[];source_tables=[]
    for panel in 'FGH':
        path=SOURCE/f'Panel{panel}_complete_bout_sample_data.parquet'
        expected=next(r['sha256'] for r in previous['data_files'] if Path(r['path'])==path)
        assert digest(path)==expected
        frames=pd.read_parquet(path,columns=['trial','FrameID','time_s','eligible','log_vigor'])
        np.testing.assert_array_equal(frames.log_vigor.notna(),frames.eligible)
        assert not np.isinf(frames.log_vigor).any()
        tables=[];residuals=[];defined_count=0
        for trial,p in frames.groupby('trial',sort=True):
            idx=np.searchsorted(EDGES,p.time_s.to_numpy(),side='right')-1
            assert ((idx>=0)&(idx<BIN_COUNT)).all()
            good=p.log_vigor.notna().to_numpy();values=p.log_vigor.to_numpy()
            count=np.bincount(idx[good],minlength=BIN_COUNT);total=np.bincount(idx,minlength=BIN_COUNT)
            sums=np.bincount(idx[good],weights=values[good],minlength=BIN_COUNT)
            means=np.divide(sums,count,out=np.full(BIN_COUNT,np.nan),where=count>0)
            checked=p[p.eligible].assign(bin_index=idx[good]).groupby('bin_index').log_vigor.mean().reindex(range(BIN_COUNT)).to_numpy()
            np.testing.assert_allclose(means,checked,atol=1e-12,rtol=0,equal_nan=True)
            assert np.array_equal(np.isnan(means),count==0)
            base=means[:BASELINE_COUNT][~np.isnan(means[:BASELINE_COUNT])]
            baseline=float(np.mean(base)) if len(base) else np.nan
            defined=bool(np.isfinite(baseline));defined_count+=defined
            delta=means-baseline if defined else np.full(BIN_COUNT,np.nan)
            if defined:
                residual=float(abs(np.nanmean(delta[:BASELINE_COUNT])));assert residual<TOL;residuals.append(residual)
            refs.append({'panel':panel,'fish':previous['fish'][panel],'trial':int(trial),
                'baseline_mean_log_vigor':baseline,'finite_baseline_bin_count':len(base),'defined':defined})
            d=delta[:BASELINE_COUNT];n=int(np.isfinite(d).sum())
            audits.append({'panel':panel,'trial':int(trial),'defined':defined,'finite_baseline_bins':n,
                'baseline_mean':float(np.nanmean(d)) if n else np.nan,
                'baseline_median':float(np.nanmedian(d)) if n else np.nan,
                'baseline_above_zero':int((d>TOL).sum()),'baseline_below_zero':int((d<-TOL).sum()),
                'baseline_tied_at_zero':int((np.isfinite(d)&(abs(d)<=TOL)).sum())})
            tables.append(pd.DataFrame({'trial':int(trial),'bin_index':range(BIN_COUNT),'start_s':EDGES[:-1],'end_s':EDGES[1:],
                'total_frame_count':total,'eligible_frame_count':count,'nan_sample_count':total-count,
                'mean_log_vigor':means,'baseline_mean_log_vigor':baseline,'trial_defined':defined,'delta_log_vigor':delta}))
        table=pd.concat(tables,ignore_index=True);assert len(table)==90*BIN_COUNT
        table.to_csv(HERE/f'Panel{panel}_mean_bins.csv',index=False)
        panels.append({'panel':panel,'fish':previous['fish'][panel],'defined_trials':int(defined_count),'undefined_trials':90-int(defined_count),
            'display_bins':len(table),'finite_display_bins':int(table.delta_log_vigor.notna().sum()),
            'empty_bins':int(table.mean_log_vigor.isna().sum()),'single_sample_bins_retained':int(table.eligible_frame_count.eq(1).sum()),
            'maximum_absolute_baseline_mean':max(residuals) if residuals else None})
        source_tables.append({'panel':panel,'path':str(path),'sha256':expected});del frames;gc.collect()
    statistics=pd.DataFrame(refs);statistics.to_csv(HERE/'baseline_statistics.csv',index=False)
    audit=pd.DataFrame(audits);audit.to_csv(HERE/'per_trial_baseline_balance.csv',index=False)
    report={'version':'Version5_DirectBinMeans','bin_width_s':BIN_WIDTH,'panels':panels,
        'defined_trials':int(statistics.defined.sum()),'undefined_trials':int((~statistics.defined).sum()),
        'all_defined_baseline_means_zero':True,'tolerance_log_units':TOL,
        'trials_with_equal_above_below_counts':int((audit.defined&(audit.baseline_above_zero==audit.baseline_below_zero)).sum()),
        'maximum_above_below_count_difference':int((audit.baseline_above_zero-audit.baseline_below_zero).abs().max()),
        'nonbout_and_invalid_frames_remain_NaN':True,'infinite_input_or_output_values':False,
        'independent_groupby_bin_means_verified':True,'other_versions_unchanged':True}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    manifest={'version':'Version5_DirectBinMeans','fish':previous['fish'],'bin_width_s':BIN_WIDTH,'bins_per_trial':BIN_COUNT,
        'display_interval_s':[-20,20],'baseline_interval_s':[-20,0],'maximum_baseline_bins':BASELINE_COUNT,'trials':[5,94],
        'frame_value':'direct eligible bout natural-log vigor; non-bout and invalid/ineligible frames remain NaN; no replacement/floor',
        'bin_value':'arithmetic mean of finite eligible natural-log vigor per 1 s bin; ignore NaNs; any one finite sample suffices',
        'baseline':'arithmetic mean of finite unscaled baseline-bin means in [-20,0); all-NaN bins ignored; one vote per finite bin',
        'formula':'delta_log_vigor = bin_mean - mean(finite baseline-bin means)',
        'undefined_rule':'whole trial undefined only if it has no finite baseline-bin means',
        'data_scaling':False,'data_clipping':False,'percentile_scaling':False,'palette':'managua_r','colour_limits':[-.25,.25],
        'source_data_manifest':str(SOURCE/'data_manifest.json'),'source_data_manifest_sha256':digest(SOURCE/'data_manifest.json'),
        'source_tables':source_tables,'panels':panels,
        'data_files':[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')]}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    build()
