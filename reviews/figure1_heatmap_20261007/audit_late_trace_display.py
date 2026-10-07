"""Audit yellow saturation and missingness without changing F/G/H panels."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT,digest
SRC=ROOT/'fgh-trial-baseline-bin-centred-20261007'
OUT=ROOT/'fgh-late-trace-diagnostic-20261007'
OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((SRC/'build_manifest.json').read_text())
rec=next(x for x in manifest['panels'] if x['panel']=='G')
path=Path(rec['panel_data']);assert digest(path)==rec['panel_data_sha256']
b=pd.read_parquet(path)
upstream=ROOT/'fgh-rebuild-panel-e-contract-20261007'
upmanifest=json.loads((upstream/'build_manifest.json').read_text())
fp=upstream/'PanelG_representative_frames.parquet'
expected=next(x['sha256'] for x in upmanifest['outputs'] if Path(x['path'])==fp)
assert digest(fp)==expected
f=pd.read_parquet(fp)
rows=[]
for phase,lo,hi in [('Pre-Train',5,14),('Train',15,64),('Test',65,94),('Late Test',85,94)]:
    q=b[b.trial.between(lo,hi)]
    for region,l,h in [('full',-20,20),('baseline',-15,0),('CS',0,10),('post',10,20)]:
        p=q[q.bin_center_s.ge(l)&q.bin_center_s.lt(h)];v=p.signed_bout_log_bin.dropna()
        rows.append({'phase':phase,'region':region,'bins':len(p),'NaN_bins':int(p.signed_bout_log_bin.isna().sum()),
            'median':float(v.median()),'mean':float(v.mean()),'q10':float(v.quantile(.1)),
            'q25':float(v.quantile(.25)),'q75':float(v.quantile(.75)),'q90':float(v.quantile(.9)),
            'yellow_saturation_025':float((v>.25).mean()),'blue_saturation_025':float((v<-.25).mean()),
            'yellow_saturation_075':float((v>.75).mean()),'blue_saturation_075':float((v<-.75).mean()),
            'within_005':float((v.abs()<=.05).mean()),'median_eligible_fraction':float(p.eligible_fraction.median())})
stats=pd.DataFrame(rows);stats.to_csv(OUT/'phase_region_statistics.csv',index=False)
trials=b.groupby('trial').agg(NaN_bins=('signed_bout_log_bin',lambda x:int(x.isna().sum())),
    median_eligible_fraction=('eligible_fraction','median'),minimum_eligible_frames=('eligible_frames','min'))
trials.to_csv(OUT/'trial_missingness.csv')
plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,4,figsize=(17,7),layout='constrained')
matrix=b.pivot(index='trial',columns='bin_center_s',values='signed_bout_log_bin').to_numpy()
coverage=b.pivot(index='trial',columns='bin_center_s',values='eligible_fraction').to_numpy()
configs=[(matrix,'Same values · managua_r ±0.25','managua_r',-.25,.25),
    (matrix,'Same values · managua_r ±0.75','managua_r',-.75,.75),
    (coverage,'Eligible frame fraction per bin','viridis',0,1),
    (np.isnan(matrix).astype(float),'Missing bins: black=NaN, white=finite','Greys',0,1)]
for a,(values,title,palette,low,high) in zip(axs,configs):
    cmap=plt.get_cmap(palette).copy();cmap.set_bad('black')
    im=a.imshow(values,aspect='auto',interpolation='nearest',extent=(-20,20,94.5,4.5),cmap=cmap,vmin=low,vmax=high)
    a.set_title(title);a.set_xlabel('Seconds from CS');a.set_ylabel('Trial')
    for t in [-15,0]:a.axvline(t,color='#00a65a',lw=.8)
    for y in [14.5,64.5]:a.axhline(y,color='#888888',lw=.8)
    if palette!='Greys':fig.colorbar(im,ax=a,shrink=.7,extend='both' if palette=='managua_r' else 'neither')
fig.suptitle('G diagnostic only: colour spread and missingness are separate\nBaseline is between the green guides. No stored values or approved panels changed.')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'G_colour_coverage_missingness.{ext}',dpi=180)
plt.close(fig)

# Trial 93: verify actual eligible-frame support against binned coverage, and show skew.
p=f[f.trial.eq(93)];q=b[b.trial.eq(93)].sort_values('bin_center_s')
np.testing.assert_array_equal(p.groupby('bin_index').eligible.sum().reindex(range(80),fill_value=0).to_numpy(),q.eligible_frames)
shift=float(q.display_centre_offset_from_previous.iloc[0])
fig,axs=plt.subplots(4,1,figsize=(13,8),layout='constrained',gridspec_kw={'height_ratios':[1.5,1.5,.6,1]})
axs[0].plot(p.time_s,p.raw_on_eligible,color='black',lw=.5);axs[0].set_ylabel('Raw eligible\nrad/ms')
axs[1].plot(p.time_s,p.bout_median-shift,color='#777777',lw=.8)
axs[1].set_ylabel('Repeated bout log\nnew reference');axs[1].axhline(0,color='black',lw=.5)
cmap=plt.get_cmap('managua_r').copy();cmap.set_bad('black')
axs[2].imshow(q.signed_bout_log_bin.to_numpy()[None,:],extent=(-20,20,0,1),aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25)
axs[2].set_yticks([]);axs[2].set_ylabel('G row 93')
axs[3].bar(q.bin_center_s,q.eligible_fraction,width=.5,color='#777777');axs[3].set_ylim(0,1);axs[3].set_ylabel('Eligible fraction')
for a in axs:
    a.set_xlim(-20,20);a.axvspan(-15,0,color='#0d7f3c',alpha=.08);a.axvline(0,color='green',lw=.7)
axs[-1].set_xlabel('Seconds from CS');fig.suptitle('G trial 93 · 10 NaN bins / 80; 11,596 eligible frames / 28,102\nEvidence only; one bout median repeats across its support, while a bin requires any finite contribution')
fig.savefig(OUT/'G_trial93_support.png',dpi=180);plt.close(fig)

summary={'phase_region_stats':rows,'total_NaN_bins':int(b.signed_bout_log_bin.isna().sum()),
    'test_NaN_bins':int(b[b.trial.ge(65)].signed_bout_log_bin.isna().sum()),
    'zero_NaN_test_trials':trials.loc[(trials.index>=65)&trials.NaN_bins.eq(0)].index.tolist(),
    'trial93_NaN_bins':int(q.signed_bout_log_bin.isna().sum()),
    'baseline_definition':'median finite displayed baseline bins, not arithmetic mean or width normalization',
    'bin_missingness_definition':'NaN only when zero finite eligible bout-frame contributions',
    'source':str(path),'source_sha256':digest(path),'frames':str(fp),'frames_sha256':digest(fp),
    'source_files_unchanged':digest(path)==rec['panel_data_sha256'],
    'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()]}
(OUT/'audit.json').write_text(json.dumps(summary,indent=2))
print(stats[stats.phase.eq('Test')].to_string(index=False));print('NaNs',summary['total_NaN_bins'],'Test',summary['test_NaN_bins'],'zero-NaN test trials',summary['zero_NaN_test_trials']);print(OUT)
