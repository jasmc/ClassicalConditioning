"""Author-approved five baseline bands with a narrow dark managua_r P45-P55 centre."""
from pathlib import Path
import json,sys,csv,io
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
import render_candidates as plotting
from render_candidates import digest
SOURCE=REPO/'reviews/fgh_version5_onesecond_means_20261009'
OUT=REPO/'reviews/fgh_version6_onesecond_centralband_20261009'

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
            p25,p45,p50,p55,p75=np.quantile(base,[.25,.45,.5,.55,.75],method='linear')
            boundaries=np.array([p25-p50,p45-p50,p55-p50,p75-p50])
            assert boundaries[1] <= 0 < boundaries[2]
            assert np.searchsorted(boundaries,0.,side='right')+1==3
            q['baseline_P45_log_vigor']=p45;q['baseline_P55_log_vigor']=p55;
            q['baseline_P25_log_vigor']=p25;q['baseline_P50_log_vigor']=p50;q['baseline_P75_log_vigor']=p75
            q['median_centred_log_vigor']=q.mean_log_vigor-p50
            good=q.median_centred_log_vigor.notna()
            q['band_class']=np.nan
            q.loc[good,'band_class']=np.searchsorted(boundaries,q.loc[good,'median_centred_log_vigor'],side='right')+1
            # A common ordinal display score; not a continuous amplitude transform.
            q['band_display_score']=q.band_class.map({1:-1.,2:-.5,3:0.,4:.5,5:1.})
            centered_base=q[q.start_s.lt(0)].median_centred_log_vigor.dropna()
            residual=abs(float(centered_base.median()));assert residual<2e-15
            counts=[int(q.band_class.eq(c).sum()) for c in range(1,6)]
            records.append({'panel':panel,'fish':original['fish'][panel],'trial':int(trial),
                'finite_baseline_bins':len(base),'P25_log_vigor':p25,'P45_log_vigor':p45,'P50_log_vigor':p50,'P55_log_vigor':p55,'P75_log_vigor':p75,
                'P25_median_centred':boundaries[0],'P45_median_centred':boundaries[1],'P50_median_centred':0.,'P55_median_centred':boundaries[2],'P75_median_centred':boundaries[3],
                'baseline_median_residual':residual,'collapsed_boundaries':bool(np.any(np.diff(boundaries)==0)),
                **{f'display_class_{i+1}_count':n for i,n in enumerate(counts)}})
            # Independently check raw-value intervals; ties belong to upper class.
            expected=np.ones(good.sum(),dtype=int)
            raw=q.loc[good,'mean_log_vigor'].to_numpy()
            for t in [p25,p45,p55,p75]:expected+=raw>=t
            np.testing.assert_array_equal(q.loc[good,'band_class'],expected)
            pieces.append(q)
        out=pd.concat(pieces,ignore_index=True)
        # Preserve original V5 columns losslessly in numeric readback.
        pd.testing.assert_frame_equal(out[cells.columns],cells,check_exact=True)
        # Keep every original CSV field verbatim; append new columns separately.
        original_rows=list(csv.reader(io.StringIO(path.read_text())))
        extras=list(csv.reader(io.StringIO(out.drop(columns=cells.columns).to_csv(index=False))))
        assert len(original_rows)==len(extras)
        with (OUT/f'Panel{panel}_mean_bins.csv').open('w',newline='') as handle:
            csv.writer(handle).writerows(a+b for a,b in zip(original_rows,extras))
        read=pd.read_csv(OUT/f'Panel{panel}_mean_bins.csv')
        pd.testing.assert_frame_equal(read[cells.columns],cells,check_exact=True)
        np.testing.assert_array_equal(read.band_class.isna(),cells.mean_log_vigor.isna())
        assert set(read.band_class.dropna().unique())<=set(range(1,6))
        checks.append({'panel':panel,'rows':len(read),'finite_classified_bins':int(read.band_class.notna().sum()),
            'original_v5_columns_preserved':True,'class_assignments_and_nan_masks_verified':True})
    stats=pd.DataFrame(records);stats.to_csv(OUT/'per_trial_quartile_thresholds.csv',index=False)
    positions=[0.,.25,.5,.75,1.];colours=[to_hex(plotting.CMAP(x)) for x in positions]
    assert colours[2] != '#000000'
    display={'thresholds':[1.5,2.5,3.5,4.5],'band_labels':['Low','Below','Centre','Above','High'],'colourbar_label':'Baseline bands (P50 = 0)','thresholds_coordinate':'quartile class index for rendering only',
        'colours':colours,'managua_r_sample_positions':positions,'reference':'each trial finite pre-CS 1 s mean bins in [-20,0)',
        'quantile_method':'linear','scientific_boundaries':'per-trial P25-P50, P45-P50, P55-P50, P75-P50 in log units','central_band':'P45 <= x < P55, contains median-centred zero','central_colour':colours[2]+', actual dark managua_r midpoint; missing values remain pure black',
        'threshold_table':str(OUT/'per_trial_quartile_thresholds.csv'),
        'threshold_table_sha256':digest(OUT/'per_trial_quartile_thresholds.csv'),
        'tie_policy':'lower-inclusive, upper-exclusive; exact threshold ties enter upper class',
        'class_intervals':['x < P25','P25 <= x < P45','P45 <= x < P55','P55 <= x < P75','x >= P75'],
        'display_scores':[-1.,-.5,0.,.5,1.],'missing_colour':'#000000'}
    report={'version':'Version6_OneSecondCentralBand','defined_trials':len(stats),'bin_width_s':1.0,'panels':checks,
        'max_abs_baseline_median':float(stats.baseline_median_residual.max()),
        'collapsed_band_trials':int(stats.collapsed_boundaries.sum()),'median_is_zero_verified':True,'zero_in_central_band_all_trials':True,'central_baseline_percentile_interval':[45,55],
        'original_numeric_v5_columns_preserved':True,'build_script_sha256':digest(Path(__file__))}
    manifest={'version':'Version6_OneSecondCentralBand','fish':original['fish'],'bin_width_s':1.0,
        'bins_per_trial':40,'baseline_interval_s':[-20,0],'display_interval_s':[-20,20],
        'source_data_manifest':str(SOURCE/'data_manifest.json'),'source_data_manifest_sha256':digest(SOURCE/'data_manifest.json'),
        'source_tables':[{'path':str(SOURCE/f'Panel{p}_mean_bins.csv'),'sha256':digest(SOURCE/f'Panel{p}_mean_bins.csv')} for p in 'FGH'],
        'bin_value':original['bin_value'],'formula':'median_centred_log_vigor = mean_log_vigor - baseline P50',
        'colour_mapping':display,'data_clipping':False,'continuous_amplitude_scaling':False,
        'common_display_range':'ordinal band scores [-1,-0.5,0,+0.5,+1]; five classes, not physical amplitude',
        'approval_evidence':{'centre_colour':'the centre has to be dark as in managua_r. always follow managua. but center cannot be total dark','central_band':'Middle 10% of baseline values: P45-P55, giving five bands','bin_width':'apply also a 1 s binning in all versions 6; discard the 4 color version 6',
            'quartiles':'separate quartiles within each trial. each trial data will need to be scaled to be in the same range and scale',
            'central_boundary':'P50 and 0 must be the same thing. confirm how to do that',
            'reference_and_median_subtraction':'Before CS: use [-20,0) s to define each trial\u2019s median and quartiles'},
        'data_files':[{'path':str(p),'sha256':digest(p)} for p in OUT.glob('*.csv')]}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'numeric_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    plotting.render('Version6_OneSecondCentralBand',OUT,'band_class',True,discrete=display)
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
