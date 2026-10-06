"""Additional route/protocol and finite-window numerical audit, without publishing data."""
import contextlib,io,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'legacy/helpers')); sys.path.insert(0,str(ROOT))
from experiment_configuration import ExperimentType,get_experiment_config
from tests.test_legacy_preprocessing_parity import oracle
from general_configuration import config
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.ingestion.readers import read_protocol
OUT=Path(sys.argv[1]); result={'routes':[], 'window_comparisons':[]}
for route in ExperimentType:
    try:
        with contextlib.redirect_stdout(io.StringIO()): cfg=get_experiment_config(route.value)
    except (ValueError, NotImplementedError): continue
    row={'route':route.value,'raw_path':str(cfg.path_home),'placeholder_path':str(cfg.path_home)=='.','condition_samples':[]}
    if str(cfg.path_home)!='.':
        paths=sorted(cfg.path_home.glob('*stim control.txt'))
        row['protocol_file_count']=len(paths)
        for condition in cfg.cond_types:
            token=cfg.cond_dict[condition]['name in original path'].lower()
            candidates=[p for p in paths if ('_'+token+'_') in p.name.lower()]
            if not candidates: row['condition_samples'].append({'condition':condition,'status':'no matching raw protocol found'}); continue
            p=candidates[0]
            try:
                frame=read_protocol(p).frame
                row['condition_samples'].append({'condition':condition,'path':str(p),'sha256':sha256_file(p),'event_counts':{str(k):int(v) for k,v in frame.Type.value_counts().items()},'status':'parsed'})
            except Exception as e: row['condition_samples'].append({'condition':condition,'path':str(p),'status':'parse_error','error':str(e)})
    result['routes'].append(row)
for project,rid in [(Path('J:/Digested Data/allDelay-full-v1'),'20221115_07'),(Path('J:/Digested Data/allDelay-full-v1'),'20221115_09'),(Path('F:/Digested Data/all3sTrace-full-v1'),'20230227_03'),(Path('F:/Digested Data/all3sTrace-full-v1'),'20230306_01')]:
    p=project/'Processed data'/rid/'tracking.parquet'
    a=next(pq.ParquetFile(p).iter_batches(batch_size=50000,columns=[f'angle{i}' for i in range(16)])).to_pandas().to_numpy()
    a=np.cumsum(a*180/np.pi,axis=1)
    d=pd.DataFrame(a,columns=[f'Angle of point {i} (deg)' for i in range(16)]); d.insert(0,config.time_trial_frame_label,np.arange(len(d)))
    original=oracle('filter_data')(d.copy(),3,10)
    old=d.copy(); cols=d.columns[1:]; old.loc[:,cols]=d.loc[:,cols].rolling(10,center=True).mean(); old=old.dropna(); old[cols]=old[cols].astype('float32')
    import analysis_utils
    fixed=analysis_utils.filter_data(d.copy(),3,10)
    speed=lambda f: f[config.tail_angle_label].diff().abs().to_numpy()*.7
    result['window_comparisons'].append({'recording_id':rid,'scope':'first 50000 tracking rows, uniform-grid numerical filter isolation; not full preprocessing reproduction','original_vs_repaired_max_abs_deg':float(np.max(np.abs(original[cols].to_numpy()-fixed[cols].to_numpy()))),'omitted_spatial_filter_max_distal_difference_deg':float(np.max(np.abs(original[config.tail_angle_label]-old[config.tail_angle_label]))),'omitted_spatial_filter_median_speed_difference_deg_per_ms':float(np.nanmedian(np.abs(speed(original)-speed(old))))})
(OUT/'route_and_filter_comparisons.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

