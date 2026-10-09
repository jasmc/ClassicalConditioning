"""Centre direct half-second means on their own finite baseline-bin distribution."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_full_bouts_baseline_samples_20261009'
sys.path[:0]=[str(REPO/'reviews/fgh_v1_trace20230310_08_20261009'),str(REPO/'reviews/fgh_c_bout_vs_bins_20261008')]
import build_new_versions as plotting
from build_comparison import digest

def main():
    previous=json.loads((SOURCE/'data_manifest.json').read_text())
    references=[];checks=[]
    for panel in 'FGH':
        source=SOURCE/f'Panel{panel}_direct_bins.csv'
        expected=next(r['sha256'] for r in previous['data_files'] if Path(r['path'])==source)
        assert digest(source)==expected
        bins=pd.read_csv(source)
        before=bins[['trial','bin_index','start_s','end_s','eligible_frame_count','mean_log_vigor']].copy()
        for trial,p in bins.groupby('trial',sort=True):
            baseline=p[p.bin_index.between(10,39)].mean_log_vigor.dropna().to_numpy()
            if len(baseline):lo,m,hi=np.quantile(baseline,[.1,.5,.9],method='linear');width=(hi-lo)/2
            else:lo=m=hi=width=np.nan
            values=np.clip((p.mean_log_vigor.to_numpy()-m)/width,-1,1) if width>0 else np.full(len(p),np.nan)
            bins.loc[p.index,'C']=values
            bins.loc[p.index,'baseline_reference_median']=m
            bins.loc[p.index,'baseline_reference_P10']=lo
            bins.loc[p.index,'baseline_reference_P90']=hi
            fish=previous['fish'][panel]
            references.append({'panel':panel,'fish':fish,'trial':int(trial),'baseline_finite_bin_count':len(baseline),
                'p10':float(lo),'p50':float(m),'p90':float(hi),'C_scale':float(width),'defined':bool(width>0)})
            if width>0:
                residual=float(abs(np.nanmedian(values[p.bin_index.between(10,39)])))
                assert residual<1e-11
                checks.append({'panel':panel,'trial':int(trial),'baseline_median_residual':residual})
        pd.testing.assert_frame_equal(before,bins[before.columns],check_exact=True)
        bins.to_csv(HERE/f'Panel{panel}_direct_bins.csv',index=False)
    statistics=pd.DataFrame(references);statistics.to_csv(HERE/'baseline_statistics.csv',index=False)
    manifest={'status':'Version2 baseline-bin population approved by user',
        'approval':'Use baseline-bin means; guarantee displayed median zero (recommended)',
        'source_data_manifest':str(SOURCE/'data_manifest.json'),'source_data_manifest_sha256':digest(SOURCE/'data_manifest.json'),
        'fish':previous['fish'],'C':'clip((bin_mean-P50)/((P90-P10)/2),-1,1)',
        'display':'direct eligible framewise natural-log means in 0.5 s cells; no bout-median substitution',
        'baseline':'finite unscaled baseline-bin means in [-15,0); one vote per finite bin; P10/P50/P90 NumPy linear',
        'sample_versions':'unchanged complete-bout sample C/D; retain unbinned timepoint baseline',
        'palette':'managua_r','limits':[-1,1],
        'data_files':[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')]}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'fish':previous['fish'],'direct_cell_means_and_eligible_counts_unchanged':True,
        'defined_trials':len(checks),'maximum_absolute_displayed_baseline_median':max(r['baseline_median_residual'] for r in checks),
        'undefined_trials':statistics[~statistics.defined][['panel','trial']].to_dict('records'),'checks':checks}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    plotting.OUT=HERE;plotting.render('C_DirectBins')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
