"""Strict-pre-US sensitivity and separate actual-US audit for the bounded shortlist."""
from pathlib import Path
import gc,sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from analyze_trace10_learners import OUT,DATA,PROJECT,load,ref,save,check,protect,score,median
from review_figure2_block_log_median import read_windows,calculate_trials,COL

def main():
    protect();short=load(OUT/'shortlist.json')['fish'];audits=[];us_rows=[]
    for rid in short:
        record=load(DATA/'fish'/(rid+'.json'));check(record['window_sources'])
        proc=PROJECT/'Processed data'/rid;protocol=pd.read_parquet(proc/'stimulus_events.parquet')
        cycles=protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
        intervals=[(float(t)-15000,float(t)+21500) for t in cycles.iloc[4:].Beg]
        metrics=read_windows(proc/'frame_activity_candidates-corrected.parquet',['FrameID','AbsoluteTime',COL],intervals)
        movement=read_windows(proc/'movement_state_candidates-corrected.parquet',['FrameID','AbsoluteTime','valid','moving'],intervals)
        strict=calculate_trials(metrics,movement,cycles,range(5,95),19.9)
        strict['fish_id']=rid;strict.to_parquet(OUT/(rid+'-strict-pre-US-19p9s.parquet'),index=False)
        original=pd.read_parquet(DATA/'fish'/(rid+'-windows.parquet'))
        us=protocol.loc[protocol.Type.eq('Reinforcer'),'Beg'].to_numpy(float)
        absolute=metrics.AbsoluteTime.to_numpy(float);raw=metrics[COL].to_numpy(float)
        valid=movement.valid.to_numpy(bool)&movement.moving.to_numpy(bool)&np.isfinite(raw)&(raw>0)
        logs=np.full(len(raw),np.nan);logs[valid]=np.log(raw[valid])
        latencies=[]
        for trial in range(15,65):
            onset=float(cycles.iloc[trial-1].Beg);lat=(us-onset)/1000;near=us[np.abs(lat-20)<=.1]
            if len(near)!=1:continue
            actual=float(near[0]);latencies.append((actual-onset)/1000)
            masks=[(absolute>=actual-1000)&(absolute<actual),(absolute>=actual)&(absolute<actual+1000)]
            x=[logs[m][np.isfinite(logs[m])] for m in masks]
            us_rows.append({'fish_id':rid,'trial_number':trial,'actual_us_latency_s':(actual-onset)/1000,'preUS_1s_log_median':median(x[0]),'postUS_1s_log_median':median(x[1]),'postUS_minus_preUS':median(x[1])-median(x[0]),'preUS_positive_moving_frames':len(x[0]),'postUS_positive_moving_frames':len(x[1])})
        audits.append({'fish_id':rid,'primary_20s_score':score(original),'strict_pre_US_19p9s_score':score(strict),'score_difference':score(strict)-score(original),'latency_min_s':min(latencies),'latency_max_s':max(latencies),'strict_window_end_s':19.9,'minimum_margin_before_actual_US_s':min(latencies)-19.9,'paired_trials':len(latencies),'strict_pre_US_values':ref(OUT/(rid+'-strict-pre-US-19p9s.parquet')),'window_sources':record['window_sources']})
        print('AUDITED',rid,flush=True);del metrics,movement,logs,raw,absolute;gc.collect()
    pd.DataFrame(us_rows).to_csv(OUT/'actual-US-pre-post-1s-audit.csv',index=False)
    save(OUT/'strict-pre-US-sensitivity.json',{'status':'computed; secondary sensitivity only; primary outcome unchanged','rationale':'Nominal [0,20) can include at most tens of milliseconds after slightly early actualUS. [0,19.9) ends before every observed shortlisted pairedUS and checks that candidate contrast is not postUS driven.','actual_US_audit_definition':'Median log positive valid moving-frame vigor during [actualUS,actualUS+1s) minus [actualUS-1s,actualUS); counts retained; never used for score/labels/ranking.','rows':audits,'actual_US_values':ref(OUT/'actual-US-pre-post-1s-audit.csv')})

if __name__=='__main__':main()
