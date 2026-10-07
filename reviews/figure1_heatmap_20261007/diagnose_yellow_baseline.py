"""Explain the exact option-C G Test colours using its scalar bin values."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from baseline_colour_mapping import baseline_colour_norm

ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
SRC=ROOT/'fgh-all-options-baseline-colour-centred-20261007'
OUT=ROOT/'fgh-baseline-yellow-explanation-20261007'
OUT.mkdir(exist_ok=True)
p=SRC/'all_options_bins.parquet'
manifest=json.loads((SRC/'colour_review_manifest.json').read_text())
record=next(r for r in manifest['outputs'] if Path(r['path'])==p)
source_hash=hashlib.sha256(p.read_bytes()).hexdigest()
assert source_hash==record['sha256']
b=pd.read_parquet(p)
g=b[b.panel.eq('G')&b.trial.between(65,94)].copy()
base=g[g.bin_center_s.ge(-15)&g.bin_center_s.lt(0)]
rows=[]
for trial,group in base.groupby('trial'):
    values=group.C.dropna().to_numpy()
    assert abs(np.median(values))<1e-12
    rows.append({'trial':int(trial),'below_zero':int((values<-1e-12).sum()),
                 'above_zero':int((values>1e-12).sum()),'at_zero':int((np.abs(values)<=1e-12).sum()),
                 'NaN_baseline_bins':int(group.C.isna().sum()),
                 'blue_endpoint_bins':int((values<=-1).sum()),
                 'yellow_endpoint_bins':int((values>=1).sum()),'median':float(np.median(values))})
pd.DataFrame(rows).to_csv(OUT/'G_test_baseline_counts_by_trial.csv',index=False)
base[['trial','bin_center_s','uncentred_bout_log_bin','colour_centred_log_bin','C_unclipped','C']].to_csv(OUT/'G_test_baseline_scalar_bins.csv',index=False)
norm=baseline_colour_norm(1)
cmap=plt.get_cmap('managua_r').copy();cmap.set_bad('black')
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(15,4.9),layout='constrained')
matrix=g.pivot(index='trial',columns='bin_center_s',values='C').to_numpy()
im=axes[0].pcolormesh(np.arange(-20,20.5,.5),np.arange(64.5,95),
                       np.ma.masked_invalid(matrix),cmap=cmap,norm=norm,shading='flat')
axes[0].set_ylim(94.5,64.5)
axes[0].add_patch(Rectangle((-15,64.5),15,30,fill=False,edgecolor='#19ad67',linewidth=2))
axes[0].set_title('Exact G Test rows, option C\nBaseline [−15,0) outlined in green')
axes[0].set_xlabel('Seconds from CS');axes[0].set_ylabel('Trial')
values=base.C.dropna().to_numpy()
counts,edges=np.histogram(values,bins=np.linspace(-1,1,31))
centres=(edges[:-1]+edges[1:])/2
axes[1].bar(centres,counts,width=np.diff(edges),color=cmap(norm(centres)),edgecolor='white',linewidth=.25)
axes[1].axvline(0,color='#582948',ls='--',lw=1)
axes[1].set_title('856 finite baseline scalar bins\n423 below zero · 423 above zero · 10 at zero')
axes[1].set_xlabel('Displayed bin value (clipped)');axes[1].set_ylabel('Number of baseline bins')
axes[1].text(.04,.93,'73 at blue endpoint\n192 at yellow endpoint',transform=axes[1].transAxes,va='top')
r=base[base.trial.eq(93)].sort_values('C').dropna(subset=['C'])
rv=r.C.to_numpy();rank=np.arange(1,len(rv)+1)
axes[2].scatter(rank,rv,c=cmap(norm(rv)),edgecolor='#444444',s=65,linewidth=.45)
axes[2].axhline(0,color='#582948',ls='--',lw=1)
axes[2].set_ylim(-1.08,1.08)
axes[2].set_title('Trial 93: its 27 baseline bins, sorted\n13 negative · median 0 · 13 positive')
axes[2].set_xlabel('Baseline bin rank');axes[2].set_ylabel('Displayed bin value (clipped)')
axes[2].text(.05,.08,'0 at blue endpoint\n7 at yellow endpoint',transform=axes[2].transAxes)
fig.suptitle('Median centring balances ranks; deviations above and below the median can have different magnitudes',fontsize=13)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'G_test_baseline_colour_explanation.{ext}',dpi=180)
plt.close(fig)
report={'source':str(p),'source_sha256':source_hash,'panel':'G','option':'C','trials':[65,94],
        'baseline_s':[-15,0],'finite_baseline_bins':len(values),'missing_baseline_bins':int(base.C.isna().sum()),
        'below_zero':int((values<-1e-12).sum()),'above_zero':int((values>1e-12).sum()),
        'at_zero':int((np.abs(values)<=1e-12).sum()),
        'blue_endpoint_bins':int((values<=-1).sum()),'yellow_endpoint_bins':int((values>=1).sum()),
        'median_negative_magnitude':float(np.median(np.abs(values[values<-1e-12]))),
        'median_positive_magnitude':float(np.median(values[values>1e-12])),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'diagnostic_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
