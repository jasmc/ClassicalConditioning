"""New review variants from the hash-bound frame tables; no detector changes."""
from pathlib import Path
from types import SimpleNamespace
import sys, json, gc
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO/'reviews/fgh_D_samples_and_direct_bins_20261009'
sys.path[:0] = [str(REPO/'reviews/fgh_c_bout_vs_bins_20261008'), str(REPO/'reviews/fgh_v1_layout_freeze_20261009')]
from build_comparison import sample_runs, digest
from freeze_and_style import CMAP, NORM, COLOURS
from shared_colourbar import verify, SIZE, LEFTS, WIDTH
FISH = [('F','Delay','20221115_07',9), ('G','3 s Trace','20230310_08',13), ('H','Control','20221115_09',None)]
EDGES = np.arange(-20,20.5,.5)

def scale(x, base, mode):
    x=np.asarray(x,float);finite=np.asarray(base,bool)&np.isfinite(x)
    if finite.any():
        p10,m,p90=np.quantile(x[finite],[.1,.5,.9],method='linear')
        width=(p90-p10)/2 if mode=='C' else max(m-p10,p90-m)
    else:p10=m=p90=width=np.nan
    raw=(x-m)/width if width>0 else np.full_like(x,np.nan)
    result=np.clip(raw,-1,1)
    if width>0:
        assert abs(np.nanmedian(result[finite]))<1e-11
        assert np.array_equal(np.isnan(result),np.isnan(x))
    return result,{'baseline_count':int(finite.sum()),'p10':float(p10),'p50':float(m),
        'p90':float(p90),'scale':float(width),'defined':bool(width>0)}

def build_data():
    OUT.mkdir(exist_ok=True)
    freeze_path=HERE/'frozen-version1/freeze.json'
    freeze=json.loads(freeze_path.read_text())
    prior=json.loads((REPO/'reviews/fgh_c_bout_vs_bins_20261008/outputs/manifest.json').read_text())
    tables={p['panel']:p for p in freeze['original_sample_tables']}
    statistics=[];counts=[]
    for letter,_,fish,_ in FISH:
        item=tables[letter];assert digest(Path(item['path']))==item['sha256']
        frames=pd.read_parquet(item['path'],columns=['trial','FrameID','time_s','eligible','bout_id','log_vigor','bout_median_log'])
        assert frames.trial.nunique()==90
        assert np.array_equal(frames.log_vigor.notna(),frames.eligible)
        assert np.array_equal(frames.bout_median_log.notna(),frames.eligible)
        interval=freeze['cadence']['interval_ms'] if letter=='G' else next(p['interval_ms'] for p in prior['source_inputs'] if p['panel']==letter)
        cadence=SimpleNamespace(interval_ms=interval)
        runs=[];bins=[]
        for trial,p in frames.groupby('trial',sort=True):
            baseline=p.time_s.ge(-15)&p.time_s.lt(0)
            d,stats=scale(p.bout_median_log,baseline,'D')
            p=p.copy();p['C_sample']=d # sample_runs' generic value interface; CSV names it D.
            runs.append(sample_runs(p,cadence).rename(columns={'C':'D'}))
            statistics.append({'panel':letter,'fish':fish,'trial':int(trial),'variant':'D_BoutSamples',**stats})
            indices=np.searchsorted(EDGES,p.time_s.to_numpy(),side='right')-1
            good=p.log_vigor.notna().to_numpy()
            assert ((indices>=0)&(indices<80)).all()
            count=np.bincount(indices[good],minlength=80)
            sums=np.bincount(indices[good],weights=p.log_vigor.to_numpy()[good],minlength=80)
            b=np.divide(sums,count,out=np.full(80,np.nan),where=count>0)
            # Independent groupby checks the finite means against the exported scalars.
            direct=p.loc[good].assign(bin_index=indices[good]).groupby('bin_index').log_vigor.mean().reindex(range(80)).to_numpy()
            np.testing.assert_allclose(b,direct,atol=1e-12,rtol=0,equal_nan=True)
            bb=np.zeros(80,bool);bb[10:40]=True
            values={}
            for mode in ['C','D']:
                values[mode],stats=scale(b,bb,mode)
                statistics.append({'panel':letter,'fish':fish,'trial':int(trial),'variant':mode+'_DirectBins',**stats})
            bins.append(pd.DataFrame({'trial':int(trial),'bin_index':range(80),'start_s':EDGES[:-1],
                'end_s':EDGES[1:],'eligible_frame_count':count,'mean_log_vigor':b,**values}))
        run=pd.concat(runs,ignore_index=True);cells=pd.concat(bins,ignore_index=True)
        assert len(cells)==7200
        run.to_csv(OUT/f'Panel{letter}_D_sample_runs.csv',index=False)
        cells.to_csv(OUT/f'Panel{letter}_direct_bins.csv',index=False)
        counts.append({'panel':letter,'fish':fish,'D_sample_runs':len(run),
                       'direct_bins':len(cells),'empty_direct_bins':int(cells.mean_log_vigor.isna().sum())})
        del frames;gc.collect()
        print('Calculated',letter,fish,flush=True)
    pd.DataFrame(statistics).to_csv(OUT/'baseline_statistics.csv',index=False)
    manifest={'status':'new review variants; original C freeze retained',
        'source_freeze':str(freeze_path),'source_freeze_sha256':digest(freeze_path),
        'source_tables':list(tables.values()),'counts':counts,
        'D':'clip((x-P50)/max(P50-P10,P90-P50),-1,1)',
        'C':'clip((x-P50)/((P90-P10)/2),-1,1)',
        'sample_variant':'bout medians repeated on eligible samples; sample baseline; D; no binning',
        'bin_variant':'mean of eligible framewise natural-log vigor; NO bout-median substitution; 0.5 s bins; finite baseline bins vote once',
        'baseline_interval':'[-15,0)','display_interval':'[-20,20)','quantiles':'NumPy linear',
        'palette':'managua_r','limits':[-1,1],'empty_values':'black, with no gap filling',
        'data_files':[{'path':str(p),'sha256':digest(p)} for p in OUT.glob('*.csv')]}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def render(kind):
    data=json.loads((OUT/'data_manifest.json').read_text())
    for item in data['data_files']:assert digest(Path(item['path']))==item['sha256']
    mode,variant=kind.split('_',1)
    fig=plt.figure(figsize=SIZE);tables=[]
    for i,(letter,name,fish,us) in enumerate(FISH):
        if variant=='BoutSamples':
            runs=pd.read_csv(OUT/f'Panel{letter}_{mode}_sample_runs.csv').rename(columns={mode:'value'})
            runs=runs.rename(columns={'value':'C'})
        else:
            cells=pd.read_csv(OUT/f'Panel{letter}_direct_bins.csv')
            runs=cells[cells[mode].notna()][['trial','start_s','end_s',mode]].rename(columns={mode:'value'})
            runs=runs.rename(columns={'value':'C'})
            assert np.allclose(runs.end_s-runs.start_s,.5)
        tables.append((letter,runs))
        ax=fig.add_axes([LEFTS[i],.13,WIDTH,.75]);ax.set_facecolor('black')
        patches=[Rectangle((r.start_s,r.trial-.5),r.end_s-r.start_s,1) for r in runs.itertuples()]
        coll=PatchCollection(patches,facecolors=CMAP(NORM(runs.C.to_numpy())),edgecolors='none',antialiaseds=False)
        coll.set_gid(f'frozen_samples_{letter}');ax.add_collection(coll)
        ax.set(xlim=(-20,20),ylim=(94.5,4.5),xticks=[-20,0,20],yticks=list(range(10,91,10)))
        ax.tick_params(direction='out',length=2.5,width=.6,labelsize=7,top=False,right=False,pad=2)
        ax.set_xlabel('Time from CS onset (s)',fontsize=8,labelpad=4)
        for spine in ax.spines.values():spine.set_linewidth(.6);spine.set_color('#333333')
        for t in [0,10]:ax.axvline(t,color='#168241',linewidth=.65,alpha=.8)
        if us is not None:ax.plot([us]*2,[14.5,64.5],color='#964bad',linewidth=.6,linestyle=':')
        for y in [14.5,64.5]:ax.axhline(y,color='white',linewidth=.7)
        if i==2:
            for text,y in [('Pre',9.5),('Train',39.5),('Test',79.5)]:
                ax.text(1.045,y,text,transform=ax.get_yaxis_transform(),rotation=90,va='center',ha='left',fontsize=7,color='#444444')
        fig.text(LEFTS[i]-.035,.947,letter,fontsize=14,weight='bold',va='center')
        fig.text(LEFTS[i]+WIDTH/2,.947,name,fontsize=10,ha='center',va='center',color=COLOURS[letter])
        fig.text(LEFTS[i]+WIDTH/2,.909,fish,fontsize=7,ha='center',va='center',color='#555555')
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=fig.add_axes([.875,.21,.012,.59]),ticks=[-1,-.5,0,.5,1])
    bar.set_label('Vigor relative to baseline',fontsize=8,labelpad=4)
    bar.ax.tick_params(direction='out',length=2,width=.5,labelsize=7,pad=2)
    bar.outline.set_linewidth(.6);bar.solids.set_rasterized(False);bar.solids.set_edgecolor('face')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[t.get_window_extent(renderer) for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text()]
    assert all(b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1 for b in boxes)
    stem=OUT/f'FGH_{kind}_shared_colourbar'
    for ext in ['svg','png','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    reports=verify(stem.with_suffix('.svg'),tables)
    for report in reports:report['display_rectangles']=report.pop('sample_runs')
    record={'variant':kind,'scale':data[mode],'data_manifest_sha256':digest(OUT/'data_manifest.json'),
        'panels':reports,'shared_colourbar_count':1,'palette':'managua_r','limits':[-1,1],
        'outputs':[{'path':str(stem.with_suffix('.'+ext)),'sha256':digest(stem.with_suffix('.'+ext))} for ext in ['svg','png','pdf']]}
    (OUT/f'{kind}_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Rendered',kind,flush=True)

if __name__=='__main__':
    if '--stage-one' in sys.argv:build_data();render('D_BoutSamples')
    else:render(sys.argv[1])
