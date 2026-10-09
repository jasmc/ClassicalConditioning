"""Resume the fully authenticated frame extraction after a gated local failure."""
from pathlib import Path
import json
import sys
import pandas as pd
import render_figure2_delay_logmedian as median

def main():
    root=Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')
    state=json.loads((root/'f-recovery-current.json').read_text())
    previous=root/'20261009T153341564032Z-delay-logmedian'
    data=pd.read_parquet(previous/'window-logmedians-all-trials.parquet')
    assert len(data)==5130 and data.fish_id.nunique()==57 and not data.duplicated(['fish_id','trial_number']).any()
    panel=Path(state['source_panel'])
    expected={x['path']:x['sha256'] for x in json.loads(panel.read_text())['storage_recovery']['inputs']}
    lineage=json.loads((previous/'frame-source-manifest.json').read_text())
    assert len(lineage)==114
    for item in lineage: assert expected[item['path']]==item['sha256']
    diag=json.loads((previous/'Test-3-local-diagnostics.json').read_text())
    assert diag['singular'] and diag['diagnostic_status']=='failed'
    for name in ['block','global','phase','phase-powell','phase-intercept']:
        assert json.loads((previous/(name+'-diagnostics.json')).read_text())['diagnostic_status']=='ok'
    median.dump(previous/'extraction-complete.json',{'source_panel_sha256':median.sha(panel),
        'table_sha256':median.sha(previous/'window-logmedians-all-trials.parquet'),
        'manifest_sha256':median.sha(previous/'frame-source-manifest.json'),
        'record_created_after_interrupted_fit':'Extraction completed 57/57 before singular Test3 local fit; no raw data copied'})
    before=set(root.iterdir())
    sys.argv=[median.__file__,'--review-root',str(root),'--analysis-dir',state['mean'],
        '--phase-dir',state['phase'],'--source-panel',str(panel),'--extraction-dir',str(previous)]
    median.main()
    state['logmedian']=str(next(iter(set(root.iterdir())-before)))
    state['completed']=True
    median.dump(root/'f-recovery-current.json',state)
    print('COMPLETE='+json.dumps(state),flush=True)

if __name__=='__main__': main()
