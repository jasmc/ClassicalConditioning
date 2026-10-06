"""Compare arrival-clock and legacy presumed-cadence processing for Figure 1."""
import json
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_figure1_vigor_alignment import ROOT, PROJECT, OUT as PRIOR, TIME, digest
from build_figure1_legacy_vigor_heatmaps import read_windows
from classical_conditioning.preprocessing.acquisition_timing import estimate_camera_cadence, presumed_acquisition_times
from classical_conditioning.analysis.movement_state import (
    smooth_contiguous_median, rolling_extreme_envelope, detect_legacy_envelope_bouts,
    MovementCalibrationConfig, _odd_window_samples)
from classical_conditioning.analysis.temporal_profiles import _signed_bout_log_vigor

OUT=ROOT/'audit-legacy-cadence-20261006'


def main():
    manifest=json.loads((PROJECT/'Metadata/20221115_07_source_manifest.json').read_text())
    proc=PROJECT/'Processed data/20221115_07'
    camera_path=proc/'camera.parquet'
    assert digest(camera_path)==manifest['artifacts']['camera']['sha256']
    evidence=json.loads((PRIOR/'audit.json').read_text())
    for path,sha in evidence['verified_hashes'].items(): assert digest(path)==sha
    camera=pd.read_parquet(camera_path)
    cadence=estimate_camera_cadence(camera)
    elapsed,absolute=presumed_acquisition_times(camera,cadence)
    offset=absolute-camera.AbsoluteTime.to_numpy()
    cycles=pd.read_parquet(proc/'stimulus_events.parquet')
    cycles=cycles.loc[cycles.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
    previous=pd.read_parquet(PRIOR/'joined_frames.parquet')
    saved=pd.read_parquet(ROOT/'heatmaps/Fig1_PanelF_Delay_legacy-vigor_v2.parquet')
    config=MovementCalibrationConfig()
    median_window=_odd_window_samples(config.smoothing_window_ms,cadence.interval_ms)
    max_window=_odd_window_samples(config.envelope_max_window_ms,cadence.interval_ms)
    min_window=_odd_window_samples(config.envelope_min_window_ms,cadence.interval_ms)
    parts=[]; summaries=[]; binparts=[]
    for trial in (9,17,63,66,93):
        onset=int(cycles.iloc[trial-1].Beg)
        c=read_windows(proc/'frame_preprocessed_corrected-v1.parquet',
            ['FrameID','AbsoluteTime','DeltaTimeMs','FrameStep','timestamp_valid','frame_valid',
             *[f'angle{i}' for i in range(16)]],[(onset-22000,onset+22001)])
        index=np.searchsorted(camera.FrameID.to_numpy(),c.FrameID)
        np.testing.assert_array_equal(camera.FrameID.to_numpy()[index],c.FrameID)
        assumed_absolute=absolute[index]
        presumed_delta=np.r_[np.nan,np.diff(elapsed[index])]
        seconds=(assumed_absolute-onset)/1000
        cumul=c[[f'angle{i}' for i in range(16)]].to_numpy().sum(axis=1)
        delta=np.r_[np.nan,np.diff(cumul)]
        derivative_valid=(c.FrameStep.to_numpy()==1)&c.timestamp_valid.to_numpy()&c.frame_valid.to_numpy()&np.r_[False,c.frame_valid.to_numpy()[:-1]]&np.isfinite(presumed_delta)&(presumed_delta>0)&(presumed_delta<=10)
        raw=np.abs(np.arctan2(np.sin(delta),np.cos(delta)))/presumed_delta
        raw[~derivative_valid]=np.nan
        smoothed=smooth_contiguous_median(raw,c.FrameStep.to_numpy(),window_samples=median_window)
        envelope=rolling_extreme_envelope(smoothed,c.FrameStep.to_numpy(),max_window_samples=max_window,min_window_samples=min_window)
        valid=derivative_valid&np.isfinite(envelope)
        moving,boutids=detect_legacy_envelope_bouts(envelope,raw,presumed_delta,c.FrameStep.to_numpy(),valid,
            envelope_threshold=config.envelope_threshold_rad_per_ms,amplitude_threshold=config.bout_amplitude_threshold_rad_per_ms,
            minimum_bout_duration_ms=config.minimum_bout_duration_ms,maximum_interbout_gap_ms=config.maximum_interbout_gap_ms)
        keep=(seconds>=-20)&(seconds<20)
        c=c.loc[keep].copy(); seconds=seconds[keep]; raw=raw[keep]; valid=valid[keep]; moving=moving[keep]; boutids=boutids[keep]
        c['ArrivalAbsoluteTime']=c.AbsoluteTime
        c['AbsoluteTime']=assumed_absolute[keep]; c[TIME]=seconds
        c['ArrivalDeltaTimeMs']=c.DeltaTimeMs; c['DeltaTimeMs']=presumed_delta[keep]
        c['Trial number']=trial; c['Vigor']=raw; c['valid']=valid; c['moving']=moving; c['bout_id']=boutids
        c['eligible']=valid&moving&(boutids>0)&np.isfinite(raw)&(raw>0)
        c['raw_on_bout_frames']=c.Vigor.where(c.eligible)
        baseline=float(np.median(np.log(c.loc[c.eligible & c[TIME].ge(-15)&c[TIME].lt(0),'Vigor'])))
        c['centred_log']=np.log(c.raw_on_bout_frames)-baseline
        mapping=c.loc[c.eligible].groupby('bout_id').centred_log.median()
        c['bout_median']=c.bout_id.map(mapping).where(c.eligible)
        c['bin_index']=np.floor((seconds+20)/.5).astype(int)
        values=_signed_bout_log_vigor(raw,seconds,c.bin_index.to_numpy(),valid,moving,boutids,
                                      bin_count=80,baseline_start_s=-15,baseline_end_s=0)
        rebuilt=c.groupby('bin_index').bout_median.mean().reindex(range(80)).to_numpy()
        np.testing.assert_allclose(rebuilt,values,rtol=0,atol=1e-12,equal_nan=True)
        np.testing.assert_array_equal(np.isfinite(c.raw_on_bout_frames),np.isfinite(c.bout_median))
        parts.append(c)
        old=previous.loc[previous['Trial number'].eq(trial)]
        oldbins=saved.loc[saved['Trial number'].eq(trial)].sort_values('Time bin center (s)')['Signed log vigor'].to_numpy()
        binparts.append(pd.DataFrame({'Trial number':trial,'Time bin center (s)':np.arange(-19.75,20,.5),
                                     'Signed log vigor':values,'Arrival-clock signed log vigor':oldbins}))
        summaries.append(dict(trial=trial,frames=len(c),presumed_raw_invalid=int((~np.isfinite(raw)).sum()),
            arrival_raw_invalid=int((~np.isfinite(old.Vigor)).sum()),arrival_valid_fraction=float(old.valid.mean()),
            presumed_valid_fraction=float(valid.mean()),arrival_eligible_frames=int(old.eligible.sum()),
            presumed_eligible_frames=int(c.eligible.sum()),arrival_missing_bins=int(np.isnan(oldbins).sum()),
            presumed_missing_bins=int(np.isnan(values).sum()),arrival_max_raw=float(old.Vigor.max()),
            presumed_max_raw=float(c.Vigor.max()),median_clock_correction_ms=float((c.AbsoluteTime-c.ArrivalAbsoluteTime).median())))
    OUT.mkdir(parents=True,exist_ok=True)
    frames=pd.concat(parts,ignore_index=True); bins=pd.concat(binparts,ignore_index=True)
    frames.to_parquet(OUT/'presumed_cadence_frames.parquet',index=False)
    bins.to_parquet(OUT/'presumed_cadence_bins.parquet',index=False)
    plt.rcParams['svg.fonttype']='none'
    fig,axes=plt.subplots(5,3,figsize=(14,9),sharex=True,sharey='col',layout='constrained')
    low=min(-.25,float(frames.bout_median.min())); high=max(.25,float(frames.bout_median.max()))
    for row,trial in enumerate((9,17,63,66,93)):
        p=frames.loc[frames['Trial number'].eq(trial)]; b=bins.loc[bins['Trial number'].eq(trial)]
        axes[row,0].plot(p[TIME],p.raw_on_bout_frames,color='black',lw=.5)
        axes[row,1].plot(p[TIME],p.bout_median,color='#c85a17',lw=.8)
        for t,v in zip(b['Time bin center (s)'],b['Signed log vigor']):
            if np.isfinite(v): axes[row,2].bar(t-.25,v,width=.5,align='edge',color='#c85a17',alpha=.7)
            else: axes[row,2].axvspan(t-.25,t+.25,color='#dddddd',lw=0)
        for col in range(3):
            axes[row,col].axvline(0,color='#0d7f3c',lw=.8); axes[row,col].axvline(10,color='#0d7f3c',ls='--',lw=.8)
            axes[row,col].set_xlim(-20,20); axes[row,col].tick_params(labelsize=8)
            if col: axes[row,col].set_ylim(low-.05,high+.05); axes[row,col].axhline(0,color='grey',lw=.4)
            if row==4: axes[row,col].set_xlabel('Seconds relative to CS onset')
        axes[row,0].set_ylabel(f'Trial {trial}\nRaw vigor (rad/ms)',fontsize=9)
    for col,title in enumerate(('Raw on detected bout frames','Signed bout medians · same frames','0.5 s means · NaNs ignored')): axes[0,col].set_title(title,fontsize=11)
    fig.suptitle(f'Legacy presumed acquisition cadence · {cadence.framerate:.6f} FPS\nSeparate timing review; baseline [-15,0) s',fontsize=13)
    fig.savefig(OUT/'presumed_cadence_review.svg');fig.savefig(OUT/'presumed_cadence_review.png',dpi=150);plt.close(fig)
    result=dict(cadence=asdict(cadence),fps=cadence.framerate,detector_samples=[median_window,max_window,min_window],
                clock_difference_ms_quantiles=pd.Series(offset[cadence.reference_position:]).quantile([0,.5,.99,1]).to_dict(),
                checks=summaries,original_artifacts_preserved=True,
                time_source='Constant cadence anchored at stable legacy reference; presumed capture clock',
                limitations='Assumes capture cadence constant and exported IDs correspond to capture sequence; absolute reference retains arrival latency.',
                frames_sha256=digest(OUT/'presumed_cadence_frames.parquet'),bins_sha256=digest(OUT/'presumed_cadence_bins.parquet'))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
