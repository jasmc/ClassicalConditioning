"""Separate review candidate renderer; never registers or modifies selected/frozen figures."""
from pathlib import Path
import sys,json,re,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle,Polygon
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap, to_hex

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SAMPLES=REPO/'reviews/fgh_full_bouts_baseline_samples_20261009'
MEANS=REPO/'reviews/fgh_direct_bins_own_reference_20261009'
MEDIANS=REPO/'reviews/fgh_version4_bin_medians_20261009'
sys.path.insert(0,str(REPO/'reviews/fgh_v1_layout_freeze_20261009'))
from freeze_and_style import CMAP,COLOURS,digest
import shared_colourbar as geometry

SIZE=(9,5.1);LEFTS=[.09,.355,.62];WIDTH=.22
CS_COLOUR='#0d8136';CS_WIDTH=2.4;CS_ALPHA=.8
EXAMPLE_TRIALS=[9,17,63,66,93]
FISH=[('F','Delay','20221115_07',9),('G','3 s Trace','20230310_08',13),('H','Control','20221115_09',None)]
KINDS=[('C_BoutSamples',SAMPLES,'C',False),('D_BoutSamples',SAMPLES,'D',False),
       ('C_DirectBins',MEANS,'C',True),('Version4_DirectBinMedians',MEDIANS,'delta_log_vigor',True)]

def select_examples():
    p=pd.read_csv(MEDIANS/'PanelG_median_bins.csv')
    selections=[]
    for trial,phase in zip(EXAMPLE_TRIALS,['Pre','early Train','late Train','early Test','late Test']):
        q=p[p.trial.eq(trial)]
        baseline=q[q.start_s.ge(-15)&q.start_s.lt(0)]
        assert q.delta_log_vigor.notna().sum()>=79 and baseline.delta_log_vigor.notna().sum()>=29
        selections.append({'trial':trial,'phase':phase,'finite_display_bins':int(q.delta_log_vigor.notna().sum()),
                           'finite_baseline_bins':int(baseline.delta_log_vigor.notna().sum())})
    return {'fish':'20230310_08','panel':'G','trials':EXAMPLE_TRIALS,'selections':selections,
            'status':'provisional illustrative examples for a later panel E update',
            'reason':'match the legacy five phase positions, with at least 79/80 finite displayed bins and 29/30 finite baseline bins; not selected as maximal responders',
            'source':str(MEDIANS/'PanelG_median_bins.csv'),'source_sha256':digest(MEDIANS/'PanelG_median_bins.csv')}

def verify_annotations(path):
    root=ET.parse(path).getroot();ns={'s':'http://www.w3.org/2000/svg'}
    groups={g.get('id'):g for g in root.findall('.//s:g',ns)}
    def points(gid):
        p=groups[gid].find('s:path',ns)
        return np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',p.get('d'))]).reshape(-1,2),p
    for i,letter in enumerate('FGH'):
        for time in [0,10]:
            xy,p=points(f'CS_{letter}_{time}')
            expected=(LEFTS[i]+WIDTH*(time+20)/40)*SIZE[0]*72
            np.testing.assert_allclose(xy[:,0],expected,atol=2e-6,rtol=0)
            style=p.get('style','')
            assert f'stroke: {CS_COLOUR}' in style and f'stroke-width: {CS_WIDTH}' in style
            assert f'stroke-opacity: {CS_ALPHA}' in style
    for trial in EXAMPLE_TRIALS:
        xy,p=points(f'example_G_trial_{trial}')
        expected_y=(.12+.75*(trial-4.5)/90)*SIZE[1]*72
        np.testing.assert_allclose(xy[0,1],expected_y,atol=2e-6,rtol=0)
        assert xy[0,0]<xy[1:,0].max() and p.get('clip-path') is None
    assert all(f'phase_left_F_{name}' in groups for name in ['Pre','Train','Test'])
    assert not any(gid and gid.startswith('phase_right') for gid in groups)
    return {'six_CS_boundaries_verified':True,'CS_times_s':[0,10],'CS_colour':CS_COLOUR,
            'CS_linewidth_pt':CS_WIDTH,'CS_alpha':CS_ALPHA,'phase_labels':'left of F',
            'example_arrowheads_panel':'G','example_arrowheads_direction':'left',
            'example_arrowheads_trials':EXAMPLE_TRIALS,'arrowhead_trial_positions_verified':True}

def render(kind,source,value_col,binned,contrast=False,discrete=None):
    data=json.loads((source/'data_manifest.json').read_text())
    hashes={Path(r['path']):r['sha256'] for r in data['data_files']}
    is_version4=kind.startswith(('Version4_','Version5_','Version6_','Version10_'))
    norm=Normalize(-.25,.25,clip=True) if is_version4 else Normalize(-1,1,clip=True)
    cmap=CMAP
    if contrast:
        class SignedRootNorm(Normalize):
            def __call__(self,value,clip=None):
                d=(np.ma.asarray(value)-0)/.25
                return .5+.5*np.sign(d)*np.sqrt(np.minimum(abs(d),1))
            def inverse(self,value):
                d=2*np.asarray(value)-1
                return .25*np.sign(d)*abs(d)**2
        norm=SignedRootNorm(-.25,.25,clip=True)
    if discrete:
        cmap=ListedColormap(discrete['colours']);cmap.set_bad('black')
        thresholds=np.asarray(discrete['thresholds'])
        class FourClassNorm(Normalize):
            def __call__(self,value,clip=None):
                return np.ma.array(np.searchsorted(thresholds,np.ma.asarray(value),side='right'),mask=np.ma.getmaskarray(value))
        norm=FourClassNorm(1,len(discrete['colours']))

    fig=plt.figure(figsize=SIZE);tables=[];inputs=[]
    for i,(letter,name,fish,us) in enumerate(FISH):
        if is_version4:path=source/f"Panel{letter}_{'mean'}_bins.csv"
        elif binned:path=source/f'Panel{letter}_direct_bins.csv'
        else:path=source/f'Panel{letter}_{value_col}_sample_runs.csv'
        assert digest(path)==hashes[path]
        cells=pd.read_csv(path)
        runs=cells[cells[value_col].notna()][['trial','start_s','end_s',value_col]].rename(columns={value_col:'C'})
        tables.append((letter,runs));inputs.append({'panel':letter,'path':str(path),'sha256':hashes[path]})
        ax=fig.add_axes([LEFTS[i],.13,WIDTH,.75]);ax.set_facecolor('black')
        coll=PatchCollection([Rectangle((r.start_s,r.trial-.5),r.end_s-r.start_s,1) for r in runs.itertuples()],
                             facecolors=cmap(norm(runs.C.to_numpy())),edgecolors='none',antialiaseds=False)
        coll.set_gid(f'frozen_samples_{letter}');ax.add_collection(coll)
        ax.set(xlim=(-20,20),ylim=(94.5,4.5),xticks=[-20,0,20],yticks=list(range(10,91,10)))
        ax.tick_params(direction='out',length=2.5,width=.6,labelsize=7,top=False,right=False,pad=2)
        ax.set_xlabel('Time from CS onset (s)',fontsize=8,labelpad=4)
        for spine in ax.spines.values():spine.set_linewidth(.6);spine.set_color('#333333')
        for time in [0,10]:
            line=ax.axvline(time,color=CS_COLOUR,linewidth=CS_WIDTH,alpha=CS_ALPHA,zorder=4)
            line.set_gid(f'CS_{letter}_{time}')
        if us is not None:ax.plot([us]*2,[14.5,64.5],color='#964bad',linewidth=.6,linestyle=':',zorder=4.1)
        for y in [14.5,64.5]:ax.axhline(y,color='white',linewidth=.8,zorder=5)
        if i==0:
            for text,y in [('Pre',9.5),('Train',39.5),('Test',79.5)]:
                label=ax.text(-.28,y,text,transform=ax.get_yaxis_transform(),rotation=90,
                              va='center',ha='center',fontsize=10,weight='bold',color='black',clip_on=False)
                label.set_gid(f'phase_left_F_{text}')
        if i==1:
            for trial in EXAMPLE_TRIALS:
                arrow=Polygon([(1.006,trial),(1.064,trial-1.15),(1.064,trial+1.15)],
                              transform=ax.get_yaxis_transform(),facecolor='black',edgecolor='none',clip_on=False,zorder=6)
                arrow.set_gid(f'example_G_trial_{trial}');ax.add_patch(arrow)
        fig.text(LEFTS[i]-.035,.947,letter,fontsize=14,weight='bold',va='center')
        fig.text(LEFTS[i]+WIDTH/2,.947,name,fontsize=10,ha='center',va='center',color=COLOURS[letter])
        fig.text(LEFTS[i]+WIDTH/2,.909,fish,fontsize=7,ha='center',va='center',color='#555555')
    label='Log vigor relative to baseline' if is_version4 else 'Vigor relative to baseline'
    ticks=[-.25,0,.25] if is_version4 else [-1,-.5,0,.5,1]
    if discrete:
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(len(discrete['colours'])+1),len(discrete['colours'])),cmap=cmap),cax=fig.add_axes([.90,.21,.012,.59]),ticks=np.arange(len(discrete['colours']))+.5)
        bar.ax.set_yticklabels(discrete.get('band_labels',['Q1','Q2','Q3','Q4']))
        label=discrete.get('colourbar_label','Within-trial baseline quartile')
    else:
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.90,.21,.012,.59]),ticks=ticks)
    bar.set_label(label,fontsize=8,labelpad=4);bar.ax.tick_params(direction='out',length=2,width=.5,labelsize=7,pad=2)
    bar.outline.set_linewidth(.6);bar.solids.set_rasterized(False);bar.solids.set_edgecolor('face')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    textboxes=[t.get_window_extent(renderer) for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text()]
    assert all(b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1 for b in textboxes)
    stem=HERE/f'FGH_{kind}_legacy_layout'
    for ext in ['svg','png','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    geometry.SIZE=SIZE;geometry.LEFTS=LEFTS;geometry.WIDTH=WIDTH;geometry.NORM=norm;geometry.CMAP=cmap
    checks=geometry.verify(stem.with_suffix('.svg'),tables)
    for check in checks:check['display_rectangles']=check.pop('sample_runs')
    record={'variant':kind,'fish':{f[0]:f[2] for f in FISH},'numeric_values_unchanged':True,
            'source_data_manifest':str(source/'data_manifest.json'),'source_data_manifest_sha256':digest(source/'data_manifest.json'),
            'source_numeric_files':inputs,'panels':checks,'annotations':verify_annotations(stem.with_suffix('.svg')),
            'colour_mapping':discrete if discrete else ('signed square-root display contrast: u=.5+.5*sign(d)*sqrt(min(abs(d)/.25,1))' if contrast else 'linear'),
            'renderer_sha256':digest(Path(__file__)),
            'palette':'managua_r','colour_limits':[float(norm.vmin),float(norm.vmax)],'colourbar_label':label,'colourbar_count':1,
            'layout':{'size_inches':SIZE,'panel_lefts':LEFTS,'panel_width':WIDTH,'phase_labels':'left of F'},
            'outputs':[{'path':str(stem.with_suffix('.'+ext)),'sha256':digest(stem.with_suffix('.'+ext))} for ext in ['svg','png','pdf']]}
    (HERE/f'{kind}_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Rendered and verified',kind,flush=True)
