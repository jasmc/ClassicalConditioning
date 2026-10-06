"""Audit >10 ms intervals back to immutable camera/tracking text logs."""
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_figure1_vigor_alignment import ROOT, PROJECT, OUT as AUDIT, TIME, digest

OUT=ROOT/'audit-camera-gaps-and-bin-mapping-20261006'


def raw_excerpt(path, selected_ids):
    """Hash the complete raw file and preserve exact selected text lines."""
    h=hashlib.sha256(); lines=[]
    with path.open('rb') as source:
        header=source.readline(); h.update(header)
        for line in source:
            h.update(line)
            try: frame=int(line.split(None,1)[0])
            except (ValueError, IndexError): continue
            if frame in selected_ids: lines.append(line.decode().rstrip())
    return h.hexdigest(),header.decode().strip(),lines


def main():
    manifest=json.loads((PROJECT/'Metadata/20221115_07_source_manifest.json').read_text())
    proc=PROJECT/'Processed data/20221115_07'
    camera_path=proc/'camera.parquet'; tracking_path=proc/'tracking.parquet'
    assert digest(camera_path)==manifest['artifacts']['camera']['sha256']
    assert digest(tracking_path)==manifest['artifacts']['tracking']['sha256']
    camera=pd.read_parquet(camera_path)
    delta=np.r_[np.nan,np.diff(camera.ElapsedTime)]
    steps=np.r_[0,np.diff(camera.FrameID)]
    absolute_delta=np.r_[np.nan,np.diff(camera.AbsoluteTime)]
    gaps=np.flatnonzero(delta>10)
    frames=pd.read_parquet(AUDIT/'joined_frames.parquet')
    local=frames.loc[frames.DeltaTimeMs.gt(10)].copy()
    indices=np.searchsorted(camera.FrameID.to_numpy(),local.FrameID)
    np.testing.assert_array_equal(camera.FrameID.to_numpy()[indices],local.FrameID)
    np.testing.assert_allclose(camera.ElapsedTime.to_numpy()[indices],local.ElapsedTime,atol=0,rtol=0)
    np.testing.assert_allclose(delta[indices],local.DeltaTimeMs,atol=0,rtol=0)
    assert np.all(steps[indices]==1)
    # Exact raw records around the missing-bin example and three largest gaps.
    chosen=[8619056]+camera.FrameID.to_numpy()[gaps[np.argsort(delta[gaps])[-3:]]].tolist()
    wanted={int(i) for center in chosen for i in range(int(center)-5,int(center)+12)}
    OUT.mkdir(parents=True,exist_ok=True)
    sources={}
    for kind in ('camera','tracking'):
        source=manifest['sources'][kind]; path=__import__('pathlib').Path(source['path'])
        sha,header,lines=raw_excerpt(path,wanted)
        assert sha==source['sha256']
        (OUT/f'raw_{kind}_excerpts.txt').write_text(header+'\n'+'\n'.join(lines)+'\n')
        sources[kind]={'path':str(path),'sha256':sha,'selected_lines':len(lines)}
        if kind=='camera':
            parsed=pd.read_csv(OUT/'raw_camera_excerpts.txt',sep=r'\s+')
            stored=camera.loc[camera.FrameID.isin(parsed.FrameID)].reset_index(drop=True)
            np.testing.assert_array_equal(parsed.FrameID,stored.FrameID)
            np.testing.assert_allclose(parsed.ElapsedTime,stored.ElapsedTime,atol=1e-8,rtol=0)
            np.testing.assert_array_equal(parsed.AbsoluteTime,stored.AbsoluteTime)
    excerpt=camera.loc[camera.FrameID.isin(wanted)].copy()
    at=np.searchsorted(camera.FrameID.to_numpy(),excerpt.FrameID)
    excerpt['DeltaTimeMs']=delta[at]; excerpt['AbsoluteDeltaMs']=absolute_delta[at]
    excerpt.to_csv(OUT/'camera_interval_examples.csv',index=False)
    local[['Trial number','FrameID','AbsoluteTime','ElapsedTime','DeltaTimeMs',TIME]].to_csv(OUT/'selected_trial_long_intervals.csv',index=False)
    # Quantify short intervals after spikes without assuming what the camera clock represents.
    safe=gaps[gaps+10<len(delta)]
    following=np.stack([delta[safe+i] for i in range(1,11)],axis=1)
    gap_summary=dict(camera_rows=len(camera),long_intervals=len(gaps),
        frame_id_gaps=int((steps[1:]!=1).sum()),long_interval_frame_id_gaps=int((steps[gaps]!=1).sum()),
        delta_quantiles_ms=pd.Series(delta[gaps]).quantile([0,.5,.9,.99,1]).to_dict(),
        selected_trial_long_intervals=len(local),
        selected_trial_counts=local.groupby('Trial number').size().to_dict(),
        following_10_contain_interval_below_05ms_fraction=float((following<.5).any(axis=1).mean()),
        next_interval_median_ms=float(np.median(following[:,0])),
        long_interval_spacing_seconds_quantiles=pd.Series(np.diff(camera.ElapsedTime.to_numpy()[gaps])/1000).quantile([.1,.5,.9]).to_dict(),
        example=excerpt.loc[excerpt.FrameID.between(8619053,8619065)].to_dict('records'))
    # Find a bin where framewise logs and the actual bout-summary mean differ.
    rows=[]
    for (trial,k),q in frames.loc[frames.eligible].groupby(['Trial number','bin_index']):
        rows.append(dict(trial=int(trial),bin_index=int(k),left_s=-20+k*.5,
                         frame_log_mean=float(q.centred_log.mean()),bout_bin=float(q.bout_median.mean()),
                         n=len(q),bouts=q.bout_id.nunique()))
    bins=pd.DataFrame(rows)
    bins['difference']=abs(bins.frame_log_mean-bins.bout_bin)
    ex=bins.loc[bins.bouts.ge(2)].sort_values('difference',ascending=False).iloc[0]
    trial=int(ex.trial); k=int(ex.bin_index)
    q=frames.loc[frames['Trial number'].eq(trial)&frames.bin_index.eq(k)].copy()
    table=q.loc[q.eligible].groupby('bout_id').agg(eligible_frames=('bout_median','size'),
        bout_median=('bout_median','first'),frame_log_mean=('centred_log','mean'))
    table['contribution_sum']=table.eligible_frames*table.bout_median
    table.to_csv(OUT/'bin_contributions.csv')
    bins.to_csv(OUT/'frame_log_vs_bout_bins.csv',index=False)
    q.to_csv(OUT/'bin_example_frames.csv',index=False)
    plt.rcParams['svg.fonttype']='none'
    fig,ax=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
    ax[0].scatter(q.loc[q.eligible,TIME],q.loc[q.eligible,'centred_log'],s=7,color='#aaaaaa',label='Individual centred logs')
    ax[0].plot(q[TIME],q.bout_median,color='#c85a17',lw=2,label='Bout medians used for binning')
    ax[0].legend(fontsize=9); ax[0].set_ylabel('Signed log vigor')
    ax[1].bar(ex.left_s,ex.bout_bin,width=.5,align='edge',color='#c85a17')
    ax[1].set_ylabel('0.5 s mean'); ax[1].set_xlabel('Seconds relative to CS onset')
    for a in ax: a.set_xlim(ex.left_s,ex.left_s+.5); a.axhline(0,color='black',lw=.5)
    fig.suptitle(f'Trial {trial}, bin [{ex.left_s:g},{ex.left_s+.5:g}) s: {int(ex.n)} valid frames, {int(ex.bouts)} bouts')
    fig.savefig(OUT/'column2_to_column3.svg'); fig.savefig(OUT/'column2_to_column3.png',dpi=150); plt.close(fig)
    report=dict(raw_sources_verified=sources,gap_summary=gap_summary,bin_example=ex.to_dict(),
        bin_contributions=table.reset_index().to_dict('records'),
        interpretation='Long intervals exist in original camera timestamps with consecutive FrameIDs; no evidence that angle rows were removed. Cause of camera timestamp delays cannot be determined from these logs alone.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
