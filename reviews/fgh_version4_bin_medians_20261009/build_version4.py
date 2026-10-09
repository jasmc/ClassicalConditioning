"""Direct log-vigor bin medians, scalar-per-bin baseline, no data scaling."""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_full_bouts_baseline_samples_20261009'
sys.path.insert(0,str(REPO/'reviews/fgh_v1_layout_freeze_20261009'))
from freeze_and_style import CMAP,COLOURS,digest
import shared_colourbar as geometry
FISH=[('F','Delay','20221115_07',9),('G','3 s Trace','20230310_08',13),('H','Control','20221115_09',None)]
EDGES=np.arange(-20,20.5,.5)
COLOUR_NORM=Normalize(vmin=-.25,vmax=.25,clip=True)
STEM=HERE/'FGH_V4_direct_bin_medians_log025'

def build():
    source_manifest=json.loads((SOURCE/'data_manifest.json').read_text())
    references=[];panels=[];source_inputs=[]
    for letter,_,fish,_ in FISH:
        path=SOURCE/f'Panel{letter}_complete_bout_sample_data.parquet'
        expected=next(r['sha256'] for r in source_manifest['data_files'] if Path(r['path'])==path)
        assert digest(path)==expected
        frames=pd.read_parquet(path,columns=['trial','FrameID','time_s','eligible','log_vigor'])
        assert np.array_equal(frames.log_vigor.notna(),frames.eligible)
        assert not np.isinf(frames.log_vigor).any()
        cells=[];residuals=[]
        for trial,p in frames.groupby('trial',sort=True):
            index=np.searchsorted(EDGES,p.time_s.to_numpy(),side='right')-1
            assert ((index>=0)&(index<80)).all()
            good=p.log_vigor.notna().to_numpy()
            count=np.bincount(index[good],minlength=80)
            total=np.bincount(index,minlength=80)
            medians=p.assign(bin_index=index).groupby('bin_index').log_vigor.median().reindex(range(80)).to_numpy()
            assert np.array_equal(np.isnan(medians),count==0)
            # Independent np.nanmedian readback: each non-NaN contributes, no coverage gate.
            direct=np.array([np.nanmedian(p.log_vigor.to_numpy()[index==i]) if count[i] else np.nan for i in range(80)])
            np.testing.assert_array_equal(medians,direct)
            finite_base=medians[10:40][~np.isnan(medians[10:40])]
            reference=float(np.median(finite_base)) if len(finite_base) else np.nan
            delta=medians-reference
            if len(finite_base):
                residual=float(abs(np.nanmedian(delta[10:40])));assert residual<1e-11;residuals.append(residual)
            cells.append(pd.DataFrame({'trial':int(trial),'bin_index':range(80),'start_s':EDGES[:-1],'end_s':EDGES[1:],
                'total_frame_count':total,'eligible_frame_count':count,'nan_sample_count':total-count,
                'median_log_vigor':medians,'baseline_median_log_vigor':reference,'delta_log_vigor':delta}))
            references.append({'panel':letter,'fish':fish,'trial':int(trial),'finite_baseline_bin_count':len(finite_base),
                'baseline_median_log_vigor':reference,'defined':bool(len(finite_base))})
        table=pd.concat(cells,ignore_index=True);assert len(table)==7200
        table.to_csv(HERE/f'Panel{letter}_median_bins.csv',index=False)
        low=table[table.eligible_frame_count.eq(1)]
        assert low.median_log_vigor.notna().all()
        panels.append({'panel':letter,'fish':fish,'bins':len(table),'finite_bins':int(table.delta_log_vigor.notna().sum()),
            'bins_with_one_eligible_sample':len(low),'empty_bins':int(table.median_log_vigor.isna().sum()),
            'maximum_absolute_baseline_median':max(residuals) if residuals else None,
            'values_outside_colour_limits':int(table.delta_log_vigor.abs().gt(.25).sum()),
            'maximum_absolute_unclipped_value':float(table.delta_log_vigor.abs().max())})
        source_inputs.append({'panel':letter,'path':str(path),'sha256':expected})
        del frames;gc.collect()
    statistics=pd.DataFrame(references);statistics.to_csv(HERE/'baseline_statistics.csv',index=False)
    manifest={'version':'Version4_DirectBinMedians','status':'user-requested additional version',
        'fish':source_manifest['fish'],'confirmed_choices':['direct eligible framewise log vigor','subtract baseline median, no scaling','existing eligible moving mask'],
        'source_tables':source_inputs,'display_interval_s':[-20,20],'trials':[5,94],'bin_width_s':.5,
        'bin_value':'median of all non-NaN eligible framewise natural-log vigor samples; no minimum sample count or coverage fraction',
        'baseline':'median of finite bin medians in [-15,0); one scalar/vote per finite bin; all-NaN bins ignored',
        'formula':'delta_log_vigor = bin_median_log_vigor - median(finite baseline-bin medians)',
        'data_scaling':False,'data_clipping':False,'percentile_scaling':False,
        'colour_palette':'managua_r','colour_limits':[-.25,.25],'colour_saturation_only':True,
        'panels':panels,'undefined_trials':statistics[~statistics.defined][['panel','trial']].to_dict('records'),
        'data_files':[{'path':str(p),'sha256':digest(p)} for p in HERE.glob('*.csv')]}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'panels':panels,'undefined_trials':manifest['undefined_trials']},indent=2))

def render():
    manifest=json.loads((HERE/'data_manifest.json').read_text())
    for r in manifest['data_files']:assert digest(Path(r['path']))==r['sha256']
    fig=plt.figure(figsize=geometry.SIZE);tables=[]
    for i,(letter,name,fish,us) in enumerate(FISH):
        bins=pd.read_csv(HERE/f'Panel{letter}_median_bins.csv')
        runs=bins[bins.delta_log_vigor.notna()][['trial','start_s','end_s','delta_log_vigor']].rename(columns={'delta_log_vigor':'C'})
        tables.append((letter,runs))
        ax=fig.add_axes([geometry.LEFTS[i],.13,geometry.WIDTH,.75]);ax.set_facecolor('black')
        patches=[Rectangle((r.start_s,r.trial-.5),r.end_s-r.start_s,1) for r in runs.itertuples()]
        coll=PatchCollection(patches,facecolors=CMAP(COLOUR_NORM(runs.C.to_numpy())),edgecolors='none',antialiaseds=False)
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
        fig.text(geometry.LEFTS[i]-.035,.947,letter,fontsize=14,weight='bold',va='center')
        fig.text(geometry.LEFTS[i]+geometry.WIDTH/2,.947,name,fontsize=10,ha='center',va='center',color=COLOURS[letter])
        fig.text(geometry.LEFTS[i]+geometry.WIDTH/2,.909,fish,fontsize=7,ha='center',va='center',color='#555555')
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=COLOUR_NORM,cmap=CMAP),cax=fig.add_axes([.875,.21,.012,.59]),ticks=[-.25,0,.25])
    bar.set_label('Log vigor relative to baseline',fontsize=8,labelpad=4)
    bar.ax.tick_params(direction='out',length=2,width=.5,labelsize=7,pad=2)
    bar.outline.set_linewidth(.6);bar.solids.set_rasterized(False);bar.solids.set_edgecolor('face')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[t.get_window_extent(renderer) for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text()]
    assert all(b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1 for b in boxes)
    for ext in ['svg','png','pdf']:fig.savefig(STEM.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    geometry.NORM=COLOUR_NORM
    checks=geometry.verify(STEM.with_suffix('.svg'),tables)
    for row in checks:row['displayed_halfsecond_cells']=row.pop('sample_runs')
    validation={'variant':'Version4_DirectBinMedians','colourbar_label':'Log vigor relative to baseline',
        'colourbar_count':1,'palette':'managua_r','colour_limits':[-.25,.25],
        'data_scaling':False,'data_clipping':False,'panels':checks,
        'data_manifest_sha256':digest(HERE/'data_manifest.json'),
        'outputs':[{'path':str(STEM.with_suffix('.'+ext)),'sha256':digest(STEM.with_suffix('.'+ext))} for ext in ['svg','png','pdf']]}
    (HERE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')

if __name__=='__main__':build();render()
