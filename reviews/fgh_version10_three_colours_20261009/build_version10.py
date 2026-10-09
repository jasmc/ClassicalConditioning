"""V10 merges V9 outer band pairs; underlying physical/scaled values retained."""
from pathlib import Path
import copy,json,hashlib,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_version9_discrete_version7_20261009'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    source=json.loads((SOURCE/'data_manifest.json').read_text())
    hashes={Path(r['path']).name:r['sha256'] for r in source['data_files']}
    protected={str(p):digest(p) for p in SOURCE.iterdir() if p.is_file()}
    mapping=copy.deepcopy(source['colour_mapping'])
    mapping.update(colours=[mapping['colours'][i] for i in [0,2,4]],
        managua_r_sample_positions=[mapping['managua_r_sample_positions'][i] for i in [0,2,4]],
        thresholds=[1.5,2.5],scientific_boundaries=[-.1,.1],band_labels=['Low','Centre','High'],
        merged_band_meanings=['Low + Below','Centre','Above + High'],
        colourbar_label='Scaled baseline bands',class_intervals=['z < -0.1','-0.1 <= z < +0.1','z >= +0.1'],display_scores=[-1,0,1])
    inputs=[];checks=[]
    for panel in 'FGH':
        path=SOURCE/f'Panel{panel}_mean_bins.csv';assert digest(path)==hashes[path.name]
        table=pd.read_csv(path);old=table.copy(deep=True)
        table['version9_band_class']=table.band_class
        table['version9_band_display_score']=table.band_display_score
        table['band_class']=table.band_class.map({1:1,2:1,3:2,4:3,5:3})
        table['band_display_score']=table.band_class-2
        untouched=[c for c in old.columns if c not in ['band_class','band_display_score']]
        pd.testing.assert_frame_equal(table[untouched],old[untouched])
        z=table.scaled_baseline_vigor.to_numpy();finite=np.isfinite(z)
        np.testing.assert_array_equal(table.band_class.to_numpy()[finite],np.searchsorted([-.1,.1],z[finite],side='right')+1)
        assert np.array_equal(table.band_class.isna(),old.band_class.isna())
        assert np.all(table.loc[np.isclose(z,0,atol=1e-12,rtol=0),'band_class']==2)
        table.to_csv(HERE/path.name,index=False)
        checks.append({'panel':panel,'defined_trials':int(table.groupby('trial').scaled_trial_defined.first().sum()),
            'class_counts':{str(c):int(table.band_class.eq(c).sum()) for c in [1,2,3]},'missing_cells':int(table.band_class.isna().sum())})
        inputs.append({'path':str(path),'sha256':digest(path)})
    for name in ['scaling_parameters.csv','baseline_statistics.csv']:(HERE/name).write_bytes((SOURCE/name).read_bytes())
    description='Use unchanged Version 9 physical and Version 7 scaled values. Merge Low with Below (old classes 1/2), retain Centre (old class 3), and merge Above with High (old classes 4/5). Fixed boundaries are -0.1 and +0.1; exact ties enter the upper band. Cyan #81e7ff represents Low/Below, dark purple #582948 represents Centre, and softer amber #e09f57 represents Above/High. The same 13 sparse Control trials remain undefined.'
    manifest=copy.deepcopy(source)
    manifest.update(version='Version10_ThreeColours',revision='three colours by merging Version 9 outer pairs',
        colour_mapping=mapping,colour_limits=[1,3],banding_description=description,
        colour_description='Three V9 colours; fixed boundaries -0.1/+0.1; dark centre and softer amber',
        source_version9_manifest=str(SOURCE/'data_manifest.json'),source_version9_manifest_sha256=digest(SOURCE/'data_manifest.json'),
        source_version9_numeric_files=inputs,data_files=[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')])
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    report={'defined_trials':sum(p['defined_trials'] for p in checks),'undefined_trials':13,
        'fixed_scaled_boundaries':[-.1,.1],'colours':mapping['colours'],'physical_and_scaled_values_unchanged':True,
        'three_classes_equal_merged_version9_classes':True,'missing_mask_unchanged':True,'zero_in_dark_centre':True,'panels':checks}
    (HERE/'numeric_verification.json').write_text(json.dumps(report,indent=2))
    code=(SOURCE/'render_version9.py').read_text().replace('Version9_','Version10_')
    (HERE/'render_version10.py').write_text(code);sys.path.insert(0,str(HERE))
    import render_version10
    render_version10.render('Version10_ThreeColours',HERE,'band_class',True,discrete=mapping)
    assert all(digest(Path(p))==h for p,h in protected.items())
    (HERE/'version9_preservation.json').write_text(json.dumps(protected,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':build()
