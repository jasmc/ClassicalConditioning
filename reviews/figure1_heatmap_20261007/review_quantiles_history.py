"""F/G/H quantile review with historical snapshots; never freezes a choice."""
from pathlib import Path
import sys,json,subprocess,ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image,ImageOps,ImageDraw
from quantile_candidates import trial_quantile_variants
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT,FISH,digest
SRC=ROOT/'fgh-trial-baseline-bin-centred-20261007'
OUT=ROOT/'fgh-quantile-history-review-20261007'
OUT.mkdir(parents=True,exist_ok=True)
history=Path(__file__).parent/'history';history.mkdir(exist_ok=True)
snapshots=[]
for commit,date in [('2f63ef4','2026-02-14'),('bf46bf7','2026-03-24')]:
    for name in ['1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py','2_ExampleFishPlotting.py']:
        rev=subprocess.check_output(['git','-c','core.fsmonitor=false','rev-parse',commit],cwd=REPO,text=True).strip()
        content=subprocess.check_output(['git','-c','core.fsmonitor=false','show',f'{commit}:{name}'],cwd=REPO)
        path=history/f'{date}_{name}';path.write_bytes(content)
        snapshots.append({'commit':rev,'date':date,'git_path':name,'snapshot':str(path),'sha256':digest(path)})
# Execute only the pure historical colour-limit helper to reproduce its defect.
march=history/'2026-03-24_2_ExampleFishPlotting.py';tree=ast.parse(march.read_text(encoding='utf-8-sig'))
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_compute_blockwise_symmetric_vlims')
ns={'pd':pd,'np':np,'Sequence':list}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(march),'exec'),ns)
legacy_limits=ns[node.name](pd.DataFrame([[0.,.5,1.]],index=[5]),[[5]])
assert legacy_limits==(0.,0.)
manifest=json.loads((SRC/'build_manifest.json').read_text());parts=[]
for spec in FISH:
    panel=spec[0];rec=next(x for x in manifest['panels'] if x['panel']==panel)
    p=Path(rec['panel_data']);assert digest(p)==rec['panel_data_sha256']
    b=pd.read_parquet(p);b['panel']=panel;parts.append(b)
allbins=pd.concat(parts,ignore_index=True)
base=allbins.loc[allbins.bin_center_s.ge(-15)&allbins.bin_center_s.lt(0),'signed_bout_log_bin']
base=base[np.isfinite(base)];qlow,qhigh=np.quantile(base,[.1,.9]);limit=max(abs(qlow),abs(qhigh))
assert limit>0
quantile_rows=[];summaries=[]
for i,b in enumerate(parts):
    b['legacy_formula_on_log_bins']=np.nan;b['median_anchored_quantile']=np.nan
    for trial,g in b.groupby('trial'):
        mask=(g.bin_center_s.ge(-15)&g.bin_center_s.lt(0)).to_numpy()
        linear,anchored,meta=trial_quantile_variants(g.signed_bout_log_bin,mask)
        b.loc[g.index,'legacy_formula_on_log_bins']=linear;b.loc[g.index,'median_anchored_quantile']=anchored
        quantile_rows.append({'panel':b.panel.iloc[0],'trial':int(trial),**meta})
    parts[i]=b
    for col in ['signed_bout_log_bin','legacy_formula_on_log_bins','median_anchored_quantile']:
        summaries.append({'panel':b.panel.iloc[0],'candidate':col,'NaN_bins':int(b[col].isna().sum())})
allbins=pd.concat(parts,ignore_index=True);allbins.to_parquet(OUT/'candidate_bins.parquet',index=False)
q=pd.DataFrame(quantile_rows);q.to_csv(OUT/'per_trial_quantile_audit.csv',index=False)
baseline_rows=allbins[allbins.bin_center_s.ge(-15)&allbins.bin_center_s.lt(0)]
trial_medians=baseline_rows.groupby(['panel','trial']).median_anchored_quantile.median()
assert np.allclose(trial_medians.dropna(),0,atol=1e-12)
invalid_trials=set(trial_medians[trial_medians.isna()].index)
valid_rows=np.array([(p,t) not in invalid_trials for p,t in zip(allbins.panel,allbins.trial)])
assert np.array_equal(allbins.loc[valid_rows,'signed_bout_log_bin'].isna(),allbins.loc[valid_rows,'median_anchored_quantile'].isna())
configs=[('A','signed_bout_log_bin',-.25,.25,'Current reference · ±0.25','Centred log-bin value'),
    ('B','signed_bout_log_bin',-limit,limit,f'Shared baseline P10/P90\nsymmetric display ±{limit:.3f}','Centred log-bin value'),
    ('C','legacy_formula_on_log_bins',0,1,'Trial P10/P90 → 0–1\nlegacy formula on log bins','Trial quantile-scaled log-bin value'),
    ('D','median_anchored_quantile',-1,1,'Trial median = 0\nsymmetric baseline P10/P90 range','Trial quantile-scaled log-bin value')]
plt.rcParams.update({'font.size':8,'svg.fonttype':'none'})
fig,axs=plt.subplots(3,4,figsize=(17,12),layout='constrained')
for i,spec in enumerate(FISH):
    panel,name,fish,*_=spec;b=parts[i]
    for j,(letter,col,lo,hi,title,label) in enumerate(configs):
        a=axs[i,j];cmap=plt.get_cmap('managua_r').copy();cmap.set_bad('black')
        vals=b.pivot(index='trial',columns='bin_center_s',values=col).to_numpy()
        im=a.imshow(vals,aspect='auto',interpolation='nearest',extent=(-20,20,94.5,4.5),cmap=cmap,vmin=lo,vmax=hi)
        a.set_title(f'{letter} · {panel} {name} {fish}\n{title}',fontsize=9)
        for t in [0,10]:a.axvline(t,color='#0d7f3c',lw=.7)
        for y in [14.5,64.5]:a.axhline(y,color='white',lw=.6)
        if spec[-1] is not None:a.plot([spec[-1]]*2,[14.5,64.5],color='#78358c',ls=':',lw=.8)
        a.set_xlabel('Seconds from CS');a.set_ylabel('Trial')
        fig.colorbar(im,ax=a,shrink=.72,label=label,extend='both' if letter in ['A','B'] else 'neither')
fig.suptitle('F/G/H review · identical clock, metric, bout support, bins and [−15,0) reference windows\nA/B change colour clipping only; C/D divide by trial-dependent ranges and clip stored transformed values. managua_r throughout.',fontsize=12)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'quantile_comparison.{ext}',dpi=180)
plt.close(fig)

for letter,col,lo,hi,title,label in configs[1:]:
    thumbs=[]
    for i,spec in enumerate(FISH):
        panel,name,fish,*_=spec;b=parts[i]
        fig=plt.figure(figsize=(6.1,5.5));grid=fig.add_gridspec(3,1,height_ratios=[10,50,30],left=.17,right=.78,bottom=.16,top=.80,hspace=.10)
        mat=b.pivot(index='trial',columns='bin_center_s',values=col)
        for k,(phase,first,last) in enumerate([('Pre-Train',5,14),('Train',15,64),('Test',65,94)]):
            a=fig.add_subplot(grid[k]);a.set_facecolor('black');cmap=plt.get_cmap('managua_r').copy();cmap.set_bad('black')
            a.pcolormesh(np.arange(-20,20.5,.5),np.arange(first-.5,last+1.5,1),np.ma.masked_invalid(mat.reindex(range(first,last+1)).to_numpy()),cmap=cmap,norm=Normalize(lo,hi,clip=True),shading='flat',rasterized=False)
            a.set_ylim(last+.5,first-.5);a.set_yticks([first,last]);a.set_ylabel(phase);a.set_xlim(-20,20);a.set_xticks([-20,-10,0,10,20])
            if k<2:a.tick_params(labelbottom=False)
            else:a.set_xlabel('Time from measured CS onset (s)')
            for t in [0,10]:a.axvline(t,color='#0d7f3c',lw=.8,ls='--' if t==10 else '-')
            if phase=='Train' and spec[-1] is not None:a.axvline(spec[-1],color='#78358c',ls=':',lw=.8)
        fig.text(.04,.955,panel,fontsize=19,weight='bold');fig.text(.17,.955,f'{name} · {fish}',fontsize=13,weight='bold')
        fig.text(.17,.88,title,fontsize=9)
        cmap=plt.get_cmap('managua_r').copy()
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo,hi,clip=True),cmap=cmap),cax=fig.add_axes([.825,.16,.023,.64]),extend='both' if letter=='B' else 'neither');cb.set_label(label)
        fig.text(.17,.055,'0.5 s bins · baseline [−15,0) · black = NaN',fontsize=8)
        fig.text(.17,.026,'Colour clipping only; input values preserved' if letter=='B' else 'Per-trial range division and clipping; input retained separately',fontsize=8)
        stem=OUT/f'{letter}_Panel{panel}_{name.replace(" ","")}'
        for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=220)
        plt.close(fig)
        with Image.open(stem.with_suffix('.png')) as im:thumbs.append(ImageOps.contain(im.convert('RGB'),(900,820)))
    gallery=Image.new('RGB',(2700,820),'white')
    for i,im in enumerate(thumbs):gallery.paste(im,(900*i,0))
    gallery.save(OUT/f'{letter}_F-G-H.png')

metrics=[]
for panel,b in zip(['F','G','H'],parts):
    for phase,first,last in [('Pre-Train',5,14),('Train',15,64),('Test',65,94)]:
        p=b[b.trial.between(first,last)&b.bin_center_s.ge(-15)&b.bin_center_s.lt(0)]
        v=p.signed_bout_log_bin.dropna()
        metrics.append({'panel':panel,'phase':phase,'baseline_bins':len(p),'finite_baseline_bins':len(v),
            'A_baseline_saturation':float((v.abs()>.25).mean()),
            'B_baseline_saturation':float((v.abs()>limit).mean()),
            'C_baseline_median':float(p.legacy_formula_on_log_bins.median()),
            'D_baseline_median':float(p.median_anchored_quantile.median()),
            'D_baseline_saturation':float((p.median_anchored_quantile.dropna().abs()>=1).mean())})
pd.DataFrame(metrics).to_csv(OUT/'candidate_phase_statistics.csv',index=False)
stats=q.groupby('panel').agg(width_min=('width','min'),width_median=('width','median'),width_max=('width','max'),linear_baseline_median_min=('baseline_median_linear','min'),linear_baseline_median_max=('baseline_median_linear','max'))
stats.to_csv(OUT/'trial_range_summary.csv')
report={'selection_status':'review only; no candidate frozen','baseline_s':[-15,0],'palette':'managua_r',
    'pooled_baseline_quantiles':[float(qlow),float(qhigh)],'common_symmetric_display_range':[-float(limit),float(limit)],
    'pooled_reference_scope':'all finite displayed baseline bins, trials 5–94, all three example fish; equal bin weight; display only',
    'candidate_B':'no stored-value change; symmetric limit is max(abs(pooled baseline P10),abs(pooled baseline P90))',
    'candidate_C':'historical affine P10/P90 formula applied to CURRENT LOG-BOUT BINS with approved windows; not historical raw-frame pipeline reproduction; not median-centred',
    'candidate_D':'(value-P50)/max(P50-P10,P90-P50) per trial; stored output clipped to [-1,1]; preserves sample baseline median zero; explicit different units',
    'candidate_D_validation':{'valid_trials':int(trial_medians.notna().sum()),'undefined_trials':int(trial_medians.isna().sum()),'maximum_absolute_baseline_median':float(trial_medians.abs().max()),'same_missingness_as_input_for_valid_trials':True},
    'sparse_baseline_trials':q.loc[q.n.lt(10),['panel','trial','n']].to_dict('records'),
    'historical_limit_helper_reproduction':{'input':[[0,.5,1]],'output':list(legacy_limits),'defect':'degenerate colour interval for nonnegative clipped values'},
    'historical_snapshots':snapshots,'candidate_missingness':summaries,'phase_stats':metrics,
    'degenerate_quantile_trials':q[q.status.ne('ok')][['panel','trial','status']].to_dict('records'),
    'degenerate_anchored_trials':q.loc[q['anchored_status'].notna(),['panel','trial','anchored_status']].to_dict('records') if 'anchored_status' in q else [],
    'code':[{'path':str(p),'sha256':digest(p)} for p in [Path(__file__),Path(__file__).with_name('quantile_candidates.py')]],
    'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file() and p.name!='review_manifest.json']}
(OUT/'review_manifest.json').write_text(json.dumps(report,indent=2))
print('Shared quantile colour limits',report['common_symmetric_display_range']);print(stats.to_string());print(pd.DataFrame(metrics).query('panel == "G"').to_string(index=False));print(OUT)
