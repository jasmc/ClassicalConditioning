"""Author-approved trial baseline quartiles: median-centred, four colours."""
from pathlib import Path
import json,sys,csv,io
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
import render_candidates as plotting
from render_candidates import digest
SOURCE=REPO/'reviews/fgh_version4_quartersecond_means_20261009'
OUT=REPO/'reviews/fgh_version6_trial_quartiles_20261009'

def build():
    OUT.mkdir(exist_ok=True)
    original=json.loads((SOURCE/'data_manifest.json').read_text())
    files={Path(r['path']):r['sha256'] for r in original['data_files']}
    records=[];checks=[]
    for panel in 'FGH':
        path=SOURCE/f'Panel{panel}_mean_bins.csv';assert digest(path)==files[path]
        cells=pd.read_csv(path);pieces=[]
        for trial,q in cells.groupby('trial',sort=True):
            q=q.copy();base=q[q.start_s.lt(0)].mean_log_vigor.dropna().to_numpy()
            p25,p50,p75=np.quantile(base,[.25,.5,.75],method='linear')
            boundaries=np.array([p25-p50,0.,p75-p50])
            q['baseline_P25_log_vigor']=p25;q['baseline_P50_log_vigor']=p50;q['baseline_P75_log_vigor']=p75
            q['median_centred_log_vigor']=q.mean_log_vigor-p50
            good=q.median_centred_log_vigor.notna()
            q['quartile_class']=np.nan
            q.loc[good,'quartile_class']=np.searchsorted(boundaries,q.loc[good,'median_centred_log_vigor'],side='right')+1
            # A common ordinal display score; not a continuous amplitude transform.
            q['quartile_display_score']=q.quartile_class.map({1:-1.,2:-1/3,3:1/3,4:1.})
            centered_base=q[q.start_s.lt(0)].median_centred_log_vigor.dropna()
            residual=abs(float(centered_base.median()));assert residual<2e-15
            counts=[int(q.quartile_class.eq(c).sum()) for c in range(1,5)]
            records.append({'panel':panel,'fish':original['fish'][panel],'trial':int(trial),
                'finite_baseline_bins':len(base),'P25_log_vigor':p25,'P50_log_vigor':p50,'P75_log_vigor':p75,
                'P25_median_centred':boundaries[0],'P50_median_centred':0.,'P75_median_centred':boundaries[2],
                'baseline_median_residual':residual,'collapsed_boundaries':bool(np.any(np.diff(boundaries)==0)),
                **{f'display_class_{i+1}_count':n for i,n in enumerate(counts)}})
            # Independently check raw-value intervals; ties belong to upper class.
            expected=np.ones(good.sum(),dtype=int)
            raw=q.loc[good,'mean_log_vigor'].to_numpy()
            for t in [p25,p50,p75]:expected+=raw>=t
            np.testing.assert_array_equal(q.loc[good,'quartile_class'],expected)
            pieces.append(q)
        out=pd.concat(pieces,ignore_index=True)
        # Preserve original V4 columns losslessly in numeric readback.
        pd.testing.assert_frame_equal(out[cells.columns],cells,check_exact=True)
        # Keep every original CSV field verbatim; append new columns separately.
        original_rows=list(csv.reader(io.StringIO(path.read_text())))
        extras=list(csv.reader(io.StringIO(out.drop(columns=cells.columns).to_csv(index=False))))
        assert len(original_rows)==len(extras)
        with (OUT/f'Panel{panel}_mean_bins.csv').open('w',newline='') as handle:
            csv.writer(handle).writerows(a+b for a,b in zip(original_rows,extras))
        read=pd.read_csv(OUT/f'Panel{panel}_mean_bins.csv')
        pd.testing.assert_frame_equal(read[cells.columns],cells,check_exact=True)
        np.testing.assert_array_equal(read.quartile_class.isna(),cells.mean_log_vigor.isna())
        assert set(read.quartile_class.dropna().unique())<=set(range(1,5))
        checks.append({'panel':panel,'rows':len(read),'finite_classified_bins':int(read.quartile_class.notna().sum()),
            'original_v4_columns_preserved':True,'class_assignments_and_nan_masks_verified':True})
    stats=pd.DataFrame(records);stats.to_csv(OUT/'per_trial_quartile_thresholds.csv',index=False)
    positions=[0.,.25,.75,1.];colours=[to_hex(plotting.CMAP(x)) for x in positions]
    display={'thresholds':[1.5,2.5,3.5],'thresholds_coordinate':'quartile class index for rendering only',
        'colours':colours,'managua_r_sample_positions':positions,'reference':'each trial finite pre-CS 0.25 s mean bins in [-20,0)',
        'quantile_method':'linear','scientific_boundaries':'per-trial P25-P50, 0, P75-P50 in log units',
        'threshold_table':str(OUT/'per_trial_quartile_thresholds.csv'),
        'threshold_table_sha256':digest(OUT/'per_trial_quartile_thresholds.csv'),
        'tie_policy':'lower-inclusive, upper-exclusive; exact threshold ties enter upper class',
        'class_intervals':['x < P25','P25 <= x < P50','P50 <= x < P75','x >= P75'],
        'display_scores':[-1.,-1/3,1/3,1.],'missing_colour':'#000000'}
    report={'version':'Version6_TrialQuartiles','defined_trials':len(stats),'bin_width_s':.25,'panels':checks,
        'max_abs_baseline_median':float(stats.baseline_median_residual.max()),
        'collapsed_quartile_trials':int(stats.collapsed_boundaries.sum()),'median_is_zero_verified':True,
        'original_numeric_v4_columns_preserved':True,'build_script_sha256':digest(Path(__file__))}
    manifest={'version':'Version6_TrialQuartiles','fish':original['fish'],'bin_width_s':.25,
        'bins_per_trial':160,'baseline_interval_s':[-20,0],'display_interval_s':[-20,20],
        'source_data_manifest':str(SOURCE/'data_manifest.json'),'source_data_manifest_sha256':digest(SOURCE/'data_manifest.json'),
        'source_tables':[{'path':str(SOURCE/f'Panel{p}_mean_bins.csv'),'sha256':digest(SOURCE/f'Panel{p}_mean_bins.csv')} for p in 'FGH'],
        'bin_value':original['bin_value'],'formula':'median_centred_log_vigor = mean_log_vigor - baseline P50',
        'colour_mapping':display,'data_clipping':False,'continuous_amplitude_scaling':False,
        'common_display_range':'ordinal quartile scores [-1,-1/3,+1/3,+1]; four classes, not physical amplitude',
        'approval_evidence':{'bin_width':'0.25 s, matching current Version 4',
            'quartiles':'separate quartiles within each trial. each trial data will need to be scaled to be in the same range and scale',
            'central_boundary':'P50 and 0 must be the same thing. confirm how to do that',
            'reference_and_median_subtraction':'Before CS: use [-20,0) s to define each trial\u2019s median and quartiles'},
        'data_files':[{'path':str(p),'sha256':digest(p)} for p in OUT.glob('*.csv')]}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    plotting.render('Version6_TrialQuartiles',OUT,'quartile_class',True,discrete=display)
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
