"""Read-only preprocessing evidence; writes only a new audit directory."""
import argparse, ast, contextlib, io, json, sys, subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'legacy/helpers'))
from experiment_configuration import ExperimentType,get_experiment_config
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    result={'scope':'read-only source audit; no cohort reprocessing or analysis rerun','routes':[], 'recordings':[], 'unreadable_manifests':[]}
    for route in ExperimentType:
        try:
            with contextlib.redirect_stdout(io.StringIO()): cfg=get_experiment_config(route.value)
            result['routes'].append({'route':route.value,'configured':True,'conditions':cfg.cond_types,'cs_duration_s':cfg.cs_duration,'minimum_cs':cfg.min_number_cs_trials,'minimum_us':cfg.min_number_us_trials,'raw_path':str(cfg.path_home),'raw_path_exists':cfg.path_home.exists(),'shared_preprocessing':True})
        except Exception as e: result['routes'].append({'route':route.value,'configured':False,'error':str(e)})
    projects=[Path('J:/Digested Data/allDelay-full-v1'),Path('F:/Digested Data/all3sTrace-full-v1')]
    for project in projects:
        manifests=sorted((project/'Metadata').glob('*_source_manifest.json'))
        selected={}
        for path in manifests:
            try: m=json.loads(path.read_text(encoding='utf-8-sig'))
            except Exception as e:
                result['unreadable_manifests'].append({'path':str(path),'error':str(e)}); continue
            key=m.get('condition_id','unknown')
            if key not in selected or m['recording_id']=='20221115_07': selected[key]=(path,m)
        for condition,(mp,m) in selected.items():
            rid=m['recording_id']; folder=project/'Processed data'/rid
            r={'project':str(project),'condition':condition,'recording_id':rid,'manifest':str(mp),'manifest_sha256':sha256_file(mp),'sources':{}}
            for kind,filename in [('camera','camera.parquet'),('tracking','tracking.parquet'),('protocol','stimulus_events.parquet')]:
                p=folder/filename; rec=m['artifacts'][kind]; h=sha256_file(p)
                r['sources'][kind]={'path':str(p),'manifest_path':rec['path'],'relocated':str(p).replace('/','\\').lower()!=rec['path'].lower(),'sha256':h,'hash_matches':h==rec['sha256'],'rows':pq.ParquetFile(p).metadata.num_rows}
                if h!=rec['sha256']: raise ValueError(f'Hash mismatch: {p}')
            c=pd.read_parquet(folder/'camera.parquet'); ids=c.FrameID.to_numpy(); dt=np.diff(c.ElapsedTime)
            r['camera']={'rows':len(c),'first_id':int(ids[0]),'last_id':int(ids[-1]),'missing_ids':int(np.maximum(np.diff(ids)-1,0).sum()),'duplicate_or_reversed_ids':int((np.diff(ids)<=0).sum()),'arrival_gaps_gt10ms':int((dt>10).sum()),'arrival_intervals_lt0_5ms':int((dt<.5).sum()),'startup_rows_legacy':13999}
            try:
                cadence=estimate_camera_cadence(c); r['cadence']=asdict(cadence)|{'fps':cadence.framerate,'assumption':'constant acquisition cadence anchored to stable arrival; hardware exposure timestamps unavailable'}
            except ValueError as e: r['cadence_error']=str(e)
            t=pq.ParquetFile(folder/'tracking.parquet'); matched=missing=nonfinite=wraps=0; previous=None
            for batch in t.iter_batches(columns=['FrameID']+[f'angle{i}' for i in range(16)]):
                d=batch.to_pandas(); f=d.FrameID.to_numpy(); ix=np.searchsorted(ids,f); ok=ix<len(ids); ok[ok]=ids[ix[ok]]==f[ok]; matched+=int(ok.sum()); missing+=int((~ok).sum())
                a=d.iloc[:,1:].to_numpy(); nonfinite+=int((~np.isfinite(a)).sum()); cumul=a.sum(axis=1)
                delta=np.diff(np.r_[previous,cumul]) if previous is not None else np.diff(cumul)
                wraps+=int((np.abs(delta)>np.pi).sum()); previous=cumul[-1]
            r['tracking']={'camera_matched_rows':matched,'camera_unmatched_rows':missing,'nonfinite_angle_cells':nonfinite,'summed_angle_changes_gt_pi':wraps}
            p=pd.read_parquet(folder/'stimulus_events.parquet'); r['protocol']={'counts':{str(k):int(v) for k,v in p.Type.value_counts().items()},'nonpositive_durations':int((p.End<=p.Beg).sum()),'events_start_outside_arrival_span':int(((p.Beg<c.AbsoluteTime.iloc[0])|(p.Beg>c.AbsoluteTime.iloc[-1])).sum())}
            result['recordings'].append(r)
            (out/'audit.json').write_text(json.dumps(result,indent=2))
            print(f'{condition} {rid}: {len(c)} camera rows; {r["camera"]["arrival_gaps_gt10ms"]} arrival gaps >10ms',flush=True)
    result['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result['code_hashes']={str(p.relative_to(ROOT)):sha256_file(p) for p in [ROOT/'legacy/modules/my_functions.py',ROOT/'legacy/helpers/analysis_utils.py',ROOT/'src/classical_conditioning/preprocessing/acquisition_timing.py',Path(__file__).resolve()]}
    (out/'audit.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
