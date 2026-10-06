"""Execute original cadence arithmetic with plotting replaced by no-ops."""
import sys,json,contextlib,io
from pathlib import Path
import pandas as pd
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tests.test_legacy_preprocessing_parity import oracle
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence
class NoPlot:
    def __getattr__(self,name): return lambda *a,**k:None
class Plot(NoPlot):
    def subplots(self,*a,**k): return NoPlot(),[NoPlot() for _ in range(5)]
f=oracle('framerate_and_reference_frame'); f.__globals__['plt']=Plot();f.__globals__['gen_var'].max_interval_between_frames=.005;f.__globals__['gen_var'].buffer_size=700
rows=[]
for project,rid in [(Path('J:/Digested Data/allDelay-full-v1'),'20221115_07'),(Path('J:/Digested Data/allDelay-full-v1'),'20221115_09'),(Path('F:/Digested Data/all3sTrace-full-v1'),'20230227_03'),(Path('F:/Digested Data/all3sTrace-full-v1'),'20230306_01')]:
    c=pd.read_parquet(project/'Processed data'/rid/'camera.parquet')
    for discard in (0,13999):
        d=c.iloc[discard:].reset_index(drop=True)
        with contextlib.redirect_stdout(io.StringIO()): rate,ref,loss=f(d.copy(),rid,None)
        repaired=estimate_camera_cadence(d)
        rows.append({'recording_id':rid,'discard_rows':discard,'original_fps':rate,'original_reference':int(ref),'original_loss_flag':bool(loss),'repaired_fps':repaired.framerate,'repaired_reference':repaired.reference_frame_id,'repaired_loss_flag':repaired.has_frame_loss_evidence,'fps_difference':repaired.framerate-rate})
(Path(sys.argv[1])/'original_cadence_comparisons.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
