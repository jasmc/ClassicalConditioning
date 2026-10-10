from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from classical_conditioning.analysis.figure4 import verify_expected_us
source=json.loads((HERE/'frozen-version1/20230310_08_source_manifest.json').read_text())
assert source['recording_id']=='20230310_08' and source['condition_id']=='trace'
protocol=pd.read_parquet('F:/Digested Data/all3sTrace-full-v1/Processed data/20230310_08/stimulus_events.parquet')
us,n=verify_expected_us(protocol,'all3sTrace')
assert n==46 and abs(us-13)<.1
frames=pd.read_parquet(HERE/'frozen-version1/PanelG_sample_data.parquet',columns=['trial','time_s','C_sample'])
baseline=frames[frames.time_s.ge(-15)&frames.time_s.lt(0)]
medians=baseline.groupby('trial').C_sample.median()
assert medians.notna().sum()==90
residual=float(medians.abs().max())
assert residual<1e-11
record={'fish':'20230310_08','condition':'trace','training_paired_trials':n,
        'measured_US_onset_s':us,'defined_trial_baselines':int(medians.notna().sum()),
        'maximum_absolute_baseline_C_median':residual,
        'numeric_recipe':'selected Version 1 unchanged; no half-second binning'}
(HERE/'replacement_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
