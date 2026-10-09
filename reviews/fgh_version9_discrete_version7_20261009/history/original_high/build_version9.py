"""V9: original five V6 colours, fixed bands on unchanged scaled V7 values."""
from pathlib import Path
import copy,json,sys,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_version7_unpooled_recipe_20261009'
PALETTE=REPO/'reviews/fgh_version6_onesecond_centralband_20261009/data_manifest.json'
BOUNDARIES=np.array([-.5,-.1,.1,.5])
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    source=json.loads((SOURCE/'data_manifest.json').read_text())
    hashes={Path(r['path']).name:r['sha256'] for r in source['data_files']}
    preserved={str(p):digest(p) for p in SOURCE.iterdir() if p.is_file()}
    palette=copy.deepcopy(json.loads(PALETTE.read_text())['colour_mapping'])
    palette.update(thresholds=[1.5,2.5,3.5,4.5],reference='fixed boundaries on Version 7 scaled values',
        scientific_boundaries=BOUNDARIES.tolist(),central_band='-0.1 <= scaled value < +0.1',
        colourbar_label='Scaled baseline bands',class_intervals=['z < -0.5','-0.5 <= z < -0.1','-0.1 <= z < 0.1','0.1 <= z < 0.5','z >= 0.5'])
    # Remove inherited V6 percentile table references: V9 uses fixed boundaries.
    for key in ['threshold_table','threshold_table_sha256','quantile_method']:palette.pop(key,None)
    checks=[];inputs=[]
    for panel in 'FGH':
        path=SOURCE/f'Panel{panel}_mean_bins.csv'; assert digest(path)==hashes[path.name]
        table=pd.read_csv(path); original=table.copy(deep=True)
        z=table.scaled_baseline_vigor.to_numpy(); finite=np.isfinite(z)
        bands=np.full(len(table),np.nan); bands[finite]=np.searchsorted(BOUNDARIES,z[finite],side='right')+1
        table['band_class']=bands
        table['band_display_score']=np.where(np.isfinite(bands),(bands-3)/2,np.nan)
        pd.testing.assert_frame_equal(table[original.columns],original)
        assert np.array_equal(np.isnan(bands),np.isnan(z))
        assert np.all(bands[finite & np.isclose(z,0,atol=1e-12,rtol=0)]==3)
        assert np.all(bands[finite & (z<-.5)]==1) and np.all(bands[finite & (z>=.5)]==5)
        for trial,t in table[table.scaled_trial_defined].groupby('trial'):
            b=t.loc[t.start_s.ge(-15)&t.start_s.lt(0),'scaled_baseline_vigor'].dropna()
            assert abs(b.median())<1e-12
            assert np.searchsorted(BOUNDARIES,float(b.median()),side='right')+1==3
        table.to_csv(HERE/path.name,index=False)
        readback=pd.read_csv(HERE/path.name)
        np.testing.assert_allclose(readback.scaled_baseline_vigor,original.scaled_baseline_vigor,atol=1e-12,rtol=0,equal_nan=True)
        checks.append({'panel':panel,'defined_trials':int(table.groupby('trial').scaled_trial_defined.first().sum()),
                       'class_counts':{str(c):int(table.band_class.eq(c).sum()) for c in range(1,6)},
                       'missing_cells':int(table.band_class.isna().sum())})
        inputs.append({'path':str(path),'sha256':digest(path)})
    (HERE/'scaling_parameters.csv').write_bytes((SOURCE/'scaling_parameters.csv').read_bytes())
    (HERE/'baseline_statistics.csv').write_bytes((SOURCE/'baseline_statistics.csv').read_bytes())
    description='Start from unchanged Version 7 scaled 0.5 s bout-summary values and its defined-trial mask. Assign fixed bands at -0.5, -0.1, +0.1 and +0.5. Exact threshold ties enter the upper band. Zero lies in the dark central band [-0.1,+0.1). Apply original Version 6 managua_r colours; this uses fixed scaled boundaries rather than Version 6 percentile boundaries.'
    manifest=copy.deepcopy(source)
    manifest.update(version='Version9_DiscreteVersion7',revision='fixed five colour bands on Version 7 values',display_value_column='band_class',
        colour_mapping=palette,colour_limits=[1,5],colourbar_label='Scaled baseline bands',banding_description=description,
        colour_description='Five original V6 colours; fixed scaled boundaries -0.5/-0.1/+0.1/+0.5',
        source_version7_manifest=str(SOURCE/'data_manifest.json'),source_version7_manifest_sha256=digest(SOURCE/'data_manifest.json'),
        source_version7_numeric_files=inputs,source_palette_manifest=str(PALETTE),source_palette_manifest_sha256=digest(PALETTE),
        data_files=[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')])
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    report={'fixed_scaled_boundaries':BOUNDARIES.tolist(),'colours':palette['colours'],
        'defined_trials':sum(p['defined_trials'] for p in checks),'undefined_trials':13,
        'version7_physical_and_scaled_values_unchanged':True,'same_missing_and_undefined_mask_as_version7':True,
        'zero_in_central_band_for_all_defined_trials':True,'upper_band_threshold_ties':True,'panels':checks}
    np.testing.assert_array_equal(np.searchsorted(BOUNDARIES,BOUNDARIES,side='right')+1,[2,3,4,5])
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2))
    code=(REPO/'reviews/fgh_colour_binning_candidates_20261009/render_candidates.py').read_text().replace("('Version4_','Version5_','Version6_')","('Version4_','Version5_','Version6_','Version9_')")
    (HERE/'render_version9.py').write_text(code); sys.path.insert(0,str(HERE))
    import render_version9
    render_version9.render('Version9_DiscreteVersion7',HERE,'band_class',True,discrete=palette)
    assert all(digest(Path(p))==h for p,h in preserved.items())
    (HERE/'version7_preservation.json').write_text(json.dumps(preserved,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':build()
