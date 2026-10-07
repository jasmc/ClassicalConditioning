"""Provisional single-fish heatmap comparison; never changes assembly or freezes.

Run with .venv-trace/Scripts/python.exe. Acquisition and detector operations
are reused verbatim from the hashed v5 builder, with a read-only trial callback.
"""
from pathlib import Path
import sys, json, hashlib, gc, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT, FISH, digest
OUT = ROOT/'heatmap-reference-review-20261007'
V5 = ROOT/'cadence-review-v5-20261007'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'svg.fonttype':'none','font.size':9})
records, allbins, summaries, provenance = [], [], [], []
captured = []

def capture(trial, seconds, raw, valid, moving, bouts):
    eligible = valid & moving & (bouts>0) & np.isfinite(raw) & (raw>0)
    base = (seconds>=-15)&(seconds<0)
    captured.append({'trial':trial, 'seconds':seconds[eligible].copy(),
        'raw':raw[eligible].copy(), 'bouts':bouts[eligible].copy(),
        'baseline_total_frames':int(base.sum()),
        'baseline_detector_valid':int((base&valid).sum())})

source = REPO/'scripts/build_figure1_cadence_review_v5.py'
code = source.read_text(encoding='utf-8')
needle = "        if panel=='F' and trial in TRIALS:"
assert code.count(needle)==1
code = code.replace(needle, '        capture(trial,seconds,raw[s],valid[s],moving[s],boutids[s])\n'+needle)
namespace = {'__file__':str(source), '__name__':'review_v5_readonly', 'capture':capture}
exec(compile(code,str(source),'exec'),namespace)

def binmean(indices, values):
    ok=np.isfinite(values)
    n=np.bincount(indices[ok],minlength=80)
    sums=np.bincount(indices[ok],weights=values[ok],minlength=80)
    return np.divide(sums,n,out=np.full(80,np.nan),where=n>0),n

for spec in FISH:
    panel,name,fish,*_=spec
    print('Verifying and comparing',panel,name,fish,flush=True)
    saved=V5/f'Fig1_Panel{panel}_{name.replace(" ","")}_presumed-cadence_v5.svg'
    meta=json.loads(saved.with_suffix('.svg.json').read_text())
    assert digest(saved)==meta['svg_sha256']
    assert digest(Path(meta['panel_data']))==meta['panel_data_sha256']
    manifest=json.loads((V5/'build_manifest.json').read_text())
    assert manifest['panels'][panel]['input_artifacts']==meta['input_artifacts']
    captured.clear()
    rebuilt,traces,events,prov=namespace['fish_data'](spec)
    old=pd.read_parquet(meta['panel_data'])
    np.testing.assert_allclose(rebuilt['Signed log vigor'],old['Signed log vigor'],atol=1e-12,rtol=0,equal_nan=True)
    baseline_logs=[np.log(p['raw'][(p['seconds']>=-15)&(p['seconds']<0)]) for p in captured]
    pool=np.concatenate(baseline_logs)
    reference=float(np.median(pool))
    fishrows=[]
    for p,logs in zip(captured,baseline_logs):
        trial=p['trial']; t=p['seconds']; raw=p['raw']; bouts=p['bouts']
        logged=np.log(raw); baseline=float(np.median(logs)) if len(logs) else np.nan
        frame=pd.DataFrame({'bout':bouts,'log':logged,'raw':raw})
        boutlog=frame.groupby('bout')['log'].transform('median').to_numpy()
        indices=np.floor((t+20)/.5).astype(int)
        uncentred,n=binmean(indices,boutlog)
        trialcentred=uncentred-baseline; fixed=uncentred-reference
        current=old.loc[old['Trial number'].eq(trial),'Signed log vigor'].to_numpy()
        np.testing.assert_allclose(current,trialcentred,atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(np.isfinite(uncentred),np.isfinite(fixed))
        if len(logs): np.testing.assert_array_equal(np.isfinite(current),np.isfinite(fixed))
        low,high=np.quantile(logs,[.1,.9]) if len(logs) else (np.nan,np.nan)
        # Diagnostic only: on the SAME bout-median signal, isolate division/clipping.
        width=high-low
        percentile=np.clip((boutlog-low)/width,0,1) if width>0 else np.full(len(t),np.nan)
        pbin,_=binmean(indices,percentile)
        row={'panel':panel,'fish':fish,'trial':trial,'baseline_log_median':baseline,
             'fish_log_reference':reference,'trial_minus_fish':baseline-reference,
             'baseline_moving_frames':len(logs),'baseline_moving_seconds':len(logs)/prov['fps'],
             'baseline_bouts':len(np.unique(bouts[(t>=-15)&(t<0)])),
             'baseline_total_frames':p['baseline_total_frames'],
             'baseline_detector_valid':p['baseline_detector_valid'],
             'baseline_log_p10':low,'baseline_log_p90':high,'p90_p10_width':width,
             'baseline_log_iqr':float(np.subtract(*np.quantile(logs,[.75,.25]))) if len(logs) else np.nan,
             'eligible_frames':len(t),'supported_bins':int((n>0).sum()),
             'p10_p90_frames_clipped_fraction':float(np.mean((boutlog<low)|(boutlog>high))) if width>0 else np.nan}
        # Sensitivity to one baseline bout: descriptive, not independent-frame CI.
        base=(t>=-15)&(t<0); baseids=np.unique(bouts[base])
        leave=[np.median(logged[base&(bouts!=b)]) for b in baseids if (base&(bouts!=b)).any()]
        row['max_leave_one_baseline_bout_shift']=float(np.max(np.abs(np.array(leave)-baseline))) if leave else np.nan
        for start,stop,label in [(-15,0,'pre'),(0,9,'cs_before_delay_us'),(0,10,'cs'),(10,20,'post')]:
            mask=(np.arange(-19.75,20,.5)>=start)&(np.arange(-19.75,20,.5)<stop)
            for key,vals in [('uncentred',uncentred),('trial',trialcentred),('fixed',fixed)]:
                a=vals[mask]; row[label+'_'+key]=float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        records.append(row); fishrows.append(row)
        allbins.extend({'panel':panel,'fish':fish,'trial':trial,'time_s':float(tm),'contributing_frames':int(nn),
            'uncentred_log':float(u),'trial_centred':float(c),'fish_centred':float(f),
            'diagnostic_p10_p90_clipped':float(q)}
            for tm,nn,u,c,f,q in zip(np.arange(-19.75,20,.5),n,uncentred,trialcentred,fixed,pbin))
    r=pd.DataFrame(fishrows)
    b=pd.DataFrame([x for x in allbins if x['panel']==panel])
    summary={'panel':panel,'fish':fish,'fixed_log_reference':reference,'fixed_reference_rad_per_ms':float(np.exp(reference)),
        'baseline_log_range':[float(r.baseline_log_median.min()),float(r.baseline_log_median.max())],
        'baseline_median_amplitude_ratio_max_min':float(np.exp(r.baseline_log_median.max()-r.baseline_log_median.min())),
        'baseline_frames_min_median_max':[float(x) for x in r.baseline_moving_frames.quantile([0,.5,1])],
        'baseline_bouts_min_median_max':[float(x) for x in r.baseline_bouts.quantile([0,.5,1])],
        'missing_baseline_trials':r.loc[r.baseline_moving_frames.eq(0),'trial'].tolist(),
        'less_than_3_baseline_bouts_trials':r.loc[r.baseline_bouts.lt(3),'trial'].tolist(),
        'largest_offset_trial':int(r.loc[r.trial_minus_fish.abs().idxmax(),'trial']),
        'largest_leave_one_bout_shift':float(r.max_leave_one_baseline_bout_shift.max())}
    for col in ['trial_centred','fish_centred']:
        finite=b[col].dropna()
        summary[col+'_saturation_025']=float((finite.abs()>.25).mean())
        summary[col+'_saturation_075']=float((finite.abs()>.75).mean())
    summary['phase_means']={}
    for phase,lo,hi in [('Pre-Train',5,14),('Train',15,64),('Test',65,94)]:
        summary['phase_means'][phase]=r.loc[r.trial.between(lo,hi),[
            'baseline_log_median','cs_uncentred','cs_trial','cs_fixed']].mean().to_dict()
    summaries.append(summary);provenance.append(prov)
    del rebuilt,traces,events;gc.collect()

bins=pd.DataFrame(allbins);stats=pd.DataFrame(records)
bins.to_parquet(OUT/'candidate_bins.parquet',index=False)
stats.to_csv(OUT/'baseline_trial_audit.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=True))

def heatmap_gallery():
    fig,axes=plt.subplots(3,5,figsize=(18,11),layout='constrained')
    rawlo,rawhi=np.quantile(bins.uncentred_log.dropna(),[.01,.99])
    configs=[('trial_centred','Trial median centred\ncurrent display ±0.25','managua_r',-.25,.25),
        ('trial_centred','Same stored values\nwider display ±0.75','managua_r',-.75,.75),
        ('fish_centred','Fixed fish baseline\ndisplay ±0.75','managua_r',-.75,.75),
        ('uncentred_log','Uncentred ln(vigor)\nshared 1–99% display','viridis',rawlo,rawhi),
        ('fish_centred','Same fixed values\ndiverging palette ±0.75','RdBu_r',-.75,.75)]
    for i,spec in enumerate(FISH):
        panel,name,fish,*_=spec; part=bins.loc[bins.panel.eq(panel)]
        for j,(col,title,palette,lo,hi) in enumerate(configs):
            a=axes[i,j];cmap=plt.get_cmap(palette).copy();cmap.set_bad('black')
            matrix=part.pivot(index='trial',columns='time_s',values=col).to_numpy()
            im=a.imshow(matrix,aspect='auto',interpolation='nearest',extent=(-20,20,94.5,4.5),cmap=cmap,vmin=lo,vmax=hi)
            for t in [0,10]: a.axvline(t,color='#00a65a',lw=.8)
            for y in [14.5,64.5]:a.axhline(y,color='white',lw=.6)
            if spec[-1] is not None:a.plot([spec[-1]]*2,[14.5,64.5],color='#aa40bc',ls='--',lw=.8)
            a.set_title(f'{panel} · {name} {fish}\n{title}',fontsize=9)
            a.set_xlabel('Seconds from measured CS onset');a.set_ylabel('Trial')
            fig.colorbar(im,ax=a,shrink=.75,extend='both')
    fig.suptitle('Identical reconstructed acquisition clock, metric, bout support and 0.5 s bins\nBlack = no eligible contribution; colour limits clip display only. Provisional, not selected.',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'candidate_gallery.{ext}',dpi=160)
    plt.close(fig)
heatmap_gallery()

fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
for i,s in enumerate(summaries):
    r=stats.loc[stats.panel.eq(s['panel'])];a=axes[i,0]
    a.plot(r.trial,r.baseline_log_median,'o-',ms=2,lw=.6);a.axhline(s['fixed_log_reference'],color='black',ls='--')
    a.set_title(f"{s['panel']} baseline ln(vigor) and fixed reference")
    a=axes[i,1];a.plot(r.trial,r.baseline_bouts,label='baseline bouts');a.set_title('Available baseline bouts');a.legend()
    a=axes[i,2]
    for field,label in [('cs_uncentred','Uncentred'),('cs_trial','Trial centred'),('cs_fixed','Fish centred')]:a.plot(r.trial,r[field],label=label,lw=.9)
    a.set_title('CS [0,10) mean supported bins');a.legend(fontsize=7)
    for a in axes[i]:
        a.set_xlabel('Trial');a.axvline(14.5,color='gray',lw=.5);a.axvline(64.5,color='gray',lw=.5)
fig.savefig(OUT/'baseline_diagnostics.png',dpi=160);plt.close(fig)

# Numerical representatives include requested Delay trials and each fish's most shifted/sparsest baseline.
chosen=[]
for s in summaries:
    r=stats.loc[stats.panel.eq(s['panel'])]
    selected=set([9,17,63,66,93,s['largest_offset_trial'],int(r.loc[r.baseline_bouts.idxmin(),'trial'])])
    chosen.append(r.loc[r.trial.isin(selected)])
pd.concat(chosen).to_csv(OUT/'representative_trials.csv',index=False)

# Saved historical alternatives: a file inventory and visual contact sheets preserve their identities.
roots=[REPO/'outputs',ROOT,ROOT.parent/'baseline-window-review',ROOT.parent/'figure1-cd-baseline-review',ROOT.parent/'figure1-examples']
inventory=[];pictures=[];seen=set()
for root in roots:
    for p in root.rglob('*'):
        if OUT in p.parents or not p.is_file() or p in seen:continue
        if not ('fig1' in str(p).lower() or 'figure1' in str(p).lower() or 'figure-1' in str(p).lower()):continue
        if p.suffix.lower() not in ['.png','.svg','.json','.parquet','.pdf']:continue
        seen.add(p);inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
        if p.suffix.lower()=='.png' and 'preflight' not in str(p):pictures.append(p)
pd.DataFrame(inventory).to_csv(OUT/'saved_alternative_inventory.csv',index=False)
for page in range((len(pictures)+11)//12):
    canvas=Image.new('RGB',(1500,1200),'#eeeeee');d=ImageDraw.Draw(canvas)
    for k,p in enumerate(pictures[page*12:(page+1)*12]):
        x=(k%3)*500;y=(k//3)*300
        with Image.open(p) as im:thumb=ImageOps.contain(im.convert('RGB'),(490,255))
        canvas.paste(thumb,(x+(490-thumb.width)//2,y))
        label=str(p.relative_to(REPO)) if REPO in p.parents else str(p.relative_to(ROOT.parent))
        d.text((x+5,y+258),label[:68]+'\n'+label[68:136],fill='black')
    canvas.save(OUT/f'saved_gallery_{page+1:02}.png')

manifest={'selection_status':'provisional; explicit selection required before freezing','baseline_s':[-15,0],
    'builder':str(Path(__file__)),'builder_sha256':digest(Path(__file__)),
    'v5_builder':str(source),'v5_builder_sha256':digest(source),
    'v5_reproduction':'all 21600 saved bins reproduced within 1e-12; NaNs identical',
    'callback_instrumentation':'read-only arrays after v5 detector; original builder and artifacts unchanged',
    'reference':'median pooled eligible frame logs from SAME [-15,0) windows across trials 5–94, separately per fish; frame-duration weighted',
    'signal':'median ln(raw vigor) per bout in [-20,20), repeated on identical eligible frames; finite-only bin means',
    'uncentred_units':'ln(vigor / (1 rad/ms)); numerical raw unit retained',
    'missing_baseline_policy':'trial centred all NaN; fixed/uncentred retain eligible bouts with no trial baseline',
    'diagnostic_p10_p90':'isolates range division on current bout medians; NOT historical log-bout-mean recreation; not recommended candidate',
    'fish_provenance':provenance}
manifest['outputs']=[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()]
(OUT/'review_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(summaries,indent=2),flush=True)
print('Review output:',OUT,flush=True)
