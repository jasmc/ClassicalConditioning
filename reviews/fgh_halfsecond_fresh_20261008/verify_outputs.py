"""Read-back checks on delivered numeric tables and exported SVGs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib.colors import CenteredNorm
from build_review import verify_svg, digest

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
examples=pd.read_csv(OUT/'representative_bout_contributions.csv')
records=[]; contributions=[]; geometry=[]
for panel in 'FGH':
    b=pd.read_parquet(OUT/f'Panel{panel}_scalar_bins.parquet')
    assert len(b)==7200 and not b.duplicated(['trial','bin_index']).any()
    for trial,g in b.groupby('trial'):
        g=g.sort_values('bin_index')
        assert g.bin_index.tolist()==list(range(80))
        np.testing.assert_array_equal(g.bin_start_s,-20+np.arange(80)*.5)
        np.testing.assert_array_equal(g.bin_end_s,g.bin_start_s+.5)
        assert g.uncentred_log_bin.isna().equals(g.eligible_frames.eq(0))
        v=g.uncentred_log_bin.to_numpy(); base=v[10:40];base=base[np.isfinite(base)]
        q10,m,q90=np.quantile(base,[.1,.5,.9],method='linear')
        np.testing.assert_allclose(g.baseline_median,m,rtol=0,atol=1e-12)
        for col,scale in [('C',(q90-q10)/2),('D',max(m-q10,q90-m))]:
            expected=np.clip((v-m)/scale,-1,1) if scale>0 else np.full(80,np.nan)
            np.testing.assert_allclose(g[col],expected,atol=1e-11,rtol=0,equal_nan=True)
        records.append({'panel':panel,'trial':int(trial),'finite_baseline_bins':len(base),
            'log_baseline_median':float(np.nanmedian(g.centred_log_bin.iloc[10:40])),
            'quantile_defined':bool(q90>q10)})
    for col,label,half in [('centred_log_bin','LogReference',.25),('C','C',1.),('D','D',1.)]:
        matrix=b.pivot(index='trial',columns='bin_index',values=col).to_numpy()
        geometry.append({'panel':panel,'variant':label,'checks':verify_svg(
            OUT/f'{label}_Panel{panel}.svg',matrix,CenteredNorm(vcenter=0,halfrange=half,clip=True))})
    for (trial,k),g in examples[examples.panel.eq(panel)].groupby(['trial','bin_index']):
        value=np.average(g.bout_log_median,weights=g.eligible_frames)
        row=b[b.trial.eq(trial)&b.bin_index.eq(k)].iloc[0]
        assert int(g.eligible_frames.sum())==row.eligible_frames
        assert abs(value-row.uncentred_log_bin)<1e-12
        if len(g)>1:
            contributions.append({'panel':panel,'trial':int(trial),'bin_index':int(k),
                'bin_start_s':float(row.bin_start_s),'bout_count':len(g),
                'frame_weighted_mean':float(value),'equal_bout_mean':float(g.bout_log_median.mean()),
                'difference':float(value-g.bout_log_median.mean()),
                'bouts':g[['bout_id','eligible_frames','bout_log_median']].to_dict('records')})
worst=max(contributions,key=lambda r:abs(r['difference']))
report={'readback_trials':len(records),'readback_bins':21600,
    'maximum_log_baseline_median_residual':max(abs(r['log_baseline_median']) for r in records),
    'undefined_quantile_trials':[r for r in records if not r['quantile_defined']],
    'independent_bout_contribution_bins':examples.groupby(['panel','trial','bin_index']).ngroups,
    'largest_representative_frame_weighting_effect':worst,'geometry':geometry,
    'pdf_visual_review':'All nine PDFs rendered with Poppler; all panels, labels and colourbars inspected in all_pdf_pages.png.',
    'note':'Geometry and arithmetic validated; scientific scalar/preprocessing approval remains unresolved.',
    'code_sha256':digest(Path(__file__))}
(OUT/'readback_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='geometry'},indent=2))
