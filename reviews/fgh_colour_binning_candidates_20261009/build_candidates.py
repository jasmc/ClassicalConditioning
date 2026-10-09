"""Render independent colour experiments and audit V5 exported numeric data."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
import render_candidates as plotting
from render_candidates import digest
V4=REPO/'reviews/fgh_version4_quartersecond_means_20261009'
V5=REPO/'reviews/fgh_version5_onesecond_means_20261009'

def audit():
    results=[]
    for panel in 'FGH':
        a=pd.read_csv(V4/f'Panel{panel}_mean_bins.csv')
        b=pd.read_csv(V5/f'Panel{panel}_mean_bins.csv')
        assert len(b)==3600 and np.isfinite(b.baseline_mean_log_vigor).all()
        assert (b.end_s-b.start_s).eq(1).all()
        assert np.array_equal(np.isnan(b.mean_log_vigor),b.eligible_frame_count.eq(0))
        assert np.array_equal(np.isnan(b.delta_log_vigor),np.isnan(b.mean_log_vigor))
        np.testing.assert_allclose(b.delta_log_vigor,b.mean_log_vigor-b.baseline_mean_log_vigor,atol=2e-15,rtol=0,equal_nan=True)
        for trial,q in b.groupby('trial'):
            base=q[q.start_s.lt(0)].mean_log_vigor.dropna().mean()
            np.testing.assert_allclose(q.baseline_mean_log_vigor,base,atol=2e-15,rtol=0)
        # Independent readback using exact quarter-bin eligible sample weights.
        a['new_bin']=a.bin_index//4
        a['sum_log']=a.mean_log_vigor.fillna(0)*a.eligible_frame_count
        g=a.groupby(['trial','new_bin']).agg(count=('eligible_frame_count','sum'),total=('total_frame_count','sum'),sum_log=('sum_log','sum'),unweighted=('mean_log_vigor','mean'))
        weighted=g.sum_log.div(g['count'].replace(0,np.nan)).to_numpy()
        np.testing.assert_allclose(b.mean_log_vigor,weighted,atol=2e-14,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(b.eligible_frame_count,g['count'])
        np.testing.assert_array_equal(b.total_frame_count,g.total)
        values=b.delta_log_vigor.to_numpy();finite=values[np.isfinite(values)]
        results.append({'panel':panel,'bin_width_s':1,'rows':len(b),'finite_bins':len(finite),
            'maximum_difference_from_unweighted_old_bin_means':float(np.nanmax(abs(b.mean_log_vigor.to_numpy()-g.unweighted.to_numpy()))),
            'weighted_quarter_bin_reconstruction_verified':True,
            'minimum_delta':float(finite.min()),'maximum_delta':float(finite.max()),
            'fraction_finite_saturated_at_fixed_limits':float(np.mean(abs(finite)>.25)),
            'max_abs_baseline_mean':float(b[b.start_s.lt(0)].groupby('trial').delta_log_vigor.mean().abs().max())})
    report={'exported_csv_readback_verified':True,'panels':results,
        'build_script':str(V5/'build_version5.py'),'build_script_sha256':digest(V5/'build_version5.py')}
    (HERE/'version5_exported_data_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    audit()
    plotting.render('Version5_DirectBinMeans',V5,'delta_log_vigor',True)
    plotting.render('Version4_Contrast',V4,'delta_log_vigor',True,contrast=True)
    plotting.render('Version5_Contrast',V5,'delta_log_vigor',True,contrast=True)
