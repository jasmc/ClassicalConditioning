"""Read back both versions and check their explicitly different baseline units."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageChops
from build_comparison import c_values,digest

HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
stats=pd.read_csv(OUT/'C_trial_baseline_statistics.csv')
records=[]
for panel in 'FGH':
    frames=pd.read_parquet(OUT/f'Panel{panel}_sample_data.parquet')
    bins=pd.read_parquet(OUT/f'Panel{panel}_direct_halfsecond_bins.parquet')
    runs=pd.read_csv(OUT/f'Panel{panel}_display_sample_runs.csv')
    assert len(bins)==7200
    for trial,p in frames.groupby('trial'):
        independently=p.loc[p.eligible].groupby('bout_id').log_vigor.transform('median').reindex(p.index)
        np.testing.assert_allclose(independently,p.bout_median_log,rtol=0,atol=1e-12,equal_nan=True)
        base=p.time_s.ge(-15)&p.time_s.lt(0)
        expected,_,meta=c_values(p.bout_median_log,base)
        np.testing.assert_allclose(expected,p.C_sample,rtol=0,atol=1e-12,equal_nan=True)
        r=runs[runs.trial.eq(trial)]
        assert r.sample_count.sum()==p.C_sample.notna().sum()
        b=bins[bins.trial.eq(trial)].sort_values('bin_index')
        expected_bin=p.assign(k=np.floor((p.time_s+20)/.5).astype(int)).groupby('k').log_vigor.mean().reindex(range(80)).to_numpy()
        np.testing.assert_allclose(expected_bin,b.direct_log_bin,rtol=0,atol=1e-12,equal_nan=True)
        basem=np.zeros(80,bool);basem[10:40]=True
        bc,_,bm=c_values(b.direct_log_bin,basem)
        np.testing.assert_allclose(bc,b.C_bin,rtol=0,atol=1e-12,equal_nan=True)
        for version,m in [('BoutSamples',meta),('DirectBins',bm)]:
            saved=stats[stats.panel.eq(panel)&stats.trial.eq(trial)&stats.version.eq(version)].iloc[0]
            assert saved.baseline_count==m['baseline_count']
            np.testing.assert_allclose([saved.p10,saved.p50,saved.p90,saved.C_scale],
                [m['p10'],m['p50'],m['p90'],m['C_scale']],rtol=0,atol=1e-12)
        records.append({'panel':panel,'trial':int(trial),'eligible_sample_count':int(p.eligible.sum()),
            'finite_sample_C_count':int(p.C_sample.notna().sum()),'finite_bin_C_count':int(b.C_bin.notna().sum()),
            'sample_baseline_median':float(np.nanmedian(expected[base])) if meta['defined'] else None,
            'bin_baseline_median':float(np.nanmedian(bc[basem])) if bm['defined'] else None})
pages=[('C_'+v+'_Panel'+panel) for v in ['BoutSamples','DirectBins'] for panel in 'FGH']
for i,name in enumerate(pages):
    original=Image.open(OUT/'pdf_checks'/f'{name}.png').convert('RGB')
    bundled=Image.open(OUT/'pdf_checks'/f'bundle-{i+1}.png').convert('RGB')
    assert ImageChops.difference(original,bundled).getbbox() is None
report={'readback_trials_per_version':270,'direct_bins_read_back':21600,
    'maximum_sample_baseline_median_residual':max(abs(r['sample_baseline_median']) for r in records if r['sample_baseline_median'] is not None),
    'maximum_bin_baseline_median_residual':max(abs(r['bin_baseline_median']) for r in records if r['bin_baseline_median'] is not None),
    'pdf_pages':6,'pdf_visual_review':'All six rendered panel pages inspected; bundle pages pixel-identical to inspected originals.',
    'records':records,'files':[{'path':str(p),'sha256':digest(p)} for p in [Path(__file__),HERE/'README.md',OUT/'C_two_versions_FGH.pdf']]}
(OUT/'readback_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['records','files']},indent=2))
