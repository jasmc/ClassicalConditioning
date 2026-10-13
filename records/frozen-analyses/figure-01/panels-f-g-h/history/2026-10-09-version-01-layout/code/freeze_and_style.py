"""Freeze selected Version 1 numeric signal; restyle F/G/H without recalculation."""
from pathlib import Path
import sys,json,shutil,re,xml.etree.ElementTree as ET
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import CenteredNorm,to_hex
from PIL import Image

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCE=REPO/'reviews/fgh_c_bout_vs_bins_20261008'
sys.path.insert(0,str(SOURCE))
from build_comparison import digest,FISH
FROZEN=HERE/'frozen-version1';OUT=HERE/'polished-layout'
FIGSIZE=(3.8,5.1);AXRECT=[.20,.13,.45,.75]
COLOURS={'F':'#ff0080','G':'#ef7900','H':'#26b7ed'}
CMAP=plt.get_cmap('managua_r').copy();CMAP.set_bad('black')
NORM=CenteredNorm(vcenter=0,halfrange=1,clip=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none'})

def snapshot():
    if FROZEN.exists():
        record=json.loads((FROZEN/'freeze.json').read_text())
        for item in record['snapshot_files']:assert digest(Path(item['path']))==item['sha256']
        return record
    FROZEN.mkdir()
    manifest=json.loads((SOURCE/'outputs/manifest.json').read_text())
    originals=[]
    for panel in 'FGH':
        runfile=SOURCE/'outputs'/f'Panel{panel}_display_sample_runs.csv'
        expected=next(r['sha256'] for r in manifest['outputs'] if Path(r['path']).name==runfile.name)
        assert digest(runfile)==expected
        shutil.copy2(runfile,FROZEN/runfile.name)
        for ext in ['svg','png','pdf']:
            p=SOURCE/'outputs'/f'C_BoutSamples_Panel{panel}.{ext}'
            expected=next(r['sha256'] for r in manifest['outputs'] if Path(r['path']).name==p.name)
            assert digest(p)==expected
            shutil.copy2(p,FROZEN/p.name)
        full=SOURCE/'outputs'/f'Panel{panel}_sample_data.parquet'
        expected=next(r['sha256'] for r in manifest['outputs'] if Path(r['path']).name==full.name)
        assert digest(full)==expected
        originals.append({'panel':panel,'path':str(full),'sha256':expected,
            'status':'source sample table hash-bound; immutable display runs snapshotted losslessly'})
    stats=pd.read_csv(SOURCE/'outputs/C_trial_baseline_statistics.csv')
    stats=stats[stats.version.eq('BoutSamples')].copy();assert len(stats)==270
    stats.to_csv(FROZEN/'sample_baseline_statistics.csv',index=False)
    shutil.copy2(SOURCE/'build_comparison.py',FROZEN/'selected_processing_code.py')
    shutil.copy2(SOURCE/'outputs/manifest.json',FROZEN/'source_comparison_manifest.json')
    record={'record_type':'Figure1_FGH_processing_and_numeric_signal_freeze',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'approval_date_local':'2026-10-08',
        'reviewer':'user','approval_evidence':'freeze version 1, but improve the layout of each panel',
        'panels':['F','G','H'],'selected_version':'C_BoutSamples',
        'processing':'frame angular speed -> shared detector -> eligible natural logs -> per-bout median within visible trial -> repeat on eligible samples -> sample baseline P10/P50/P90 -> C on samples -> sample interval display',
        'no_halfsecond_binning_anywhere':True,'display_interval_s':[-20,20],
        'baseline_interval_s':[-15,0],'baseline_voting_unit':'eligible displayed sample',
        'C_formula':'clip((x-P50)/((P90-P10)/2), -1, +1)',
        'palette':'managua_r','limits':[-1,1],'zero_coordinate':.5,'zero_colour':to_hex(CMAP(.5)),
        'undefined_trial':{'panel':'H','trial':16,'reason':'zero sample-baseline quantile range'},
        'freeze_scope':'selected single-fish recipe and numeric display values; layout may be refined without changing them',
        'original_sample_tables':originals,
        'snapshot_files':[{'path':str(p),'sha256':digest(p)} for p in FROZEN.iterdir() if p.is_file()]}
    (FROZEN/'freeze.json').write_text(json.dumps(record,indent=2))
    return record

def geometry(svg,runs):
    ns={'s':'http://www.w3.org/2000/svg'};root=ET.parse(svg).getroot()
    g=next(g for g in root.findall('.//s:g',ns) if g.get('id')=='frozen_sample_runs')
    paths=g.findall('s:path',ns);assert len(paths)==len(runs)
    xscale=AXRECT[2]*FIGSIZE[0]*72/40
    yscale=AXRECT[3]*FIGSIZE[1]*72/90
    left=AXRECT[0]*FIGSIZE[0]*72
    top=(1-AXRECT[1]-AXRECT[3])*FIGSIZE[1]*72
    for path,row in zip(paths,runs.itertuples()):
        xy=np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',path.get('d'))]).reshape(-1,2)
        np.testing.assert_allclose(np.ptp(xy[:,0]),(row.end_s-row.start_s)*xscale,atol=2e-6,rtol=0)
        np.testing.assert_allclose(np.min(xy[:,0]),left+(row.start_s+20)*xscale,atol=2e-6,rtol=0)
        np.testing.assert_allclose(np.ptp(xy[:,1]),yscale,atol=2e-6,rtol=0)
        np.testing.assert_allclose(np.min(xy[:,1]),top+(row.trial-5)*yscale,atol=2e-6,rtol=0)
        fill=re.search(r'fill:\s*(#[0-9a-f]+)',path.get('style',''))
        assert (fill.group(1) if fill else '#000000')==to_hex(CMAP(NORM(row.C)))
    return {'sample_run_count':len(paths),'all_frozen_run_values_and_intervals_preserved':True,
        'all_exported_positions_widths_heights_and_palette_fills_verified':True}

def panel(spec):
    letter,name,fish,*_=spec
    runs=pd.read_csv(FROZEN/f'Panel{letter}_display_sample_runs.csv')
    original=pd.read_csv(SOURCE/'outputs'/f'Panel{letter}_display_sample_runs.csv')
    pd.testing.assert_frame_equal(runs,original,check_exact=True)
    fig=plt.figure(figsize=FIGSIZE);ax=fig.add_axes(AXRECT)
    ax.set_facecolor('black')
    patches=[Rectangle((r.start_s,r.trial-.5),r.end_s-r.start_s,1) for r in runs.itertuples()]
    collection=PatchCollection(patches,facecolors=CMAP(NORM(runs.C.to_numpy())),
        edgecolors='none',antialiaseds=False)
    collection.set_gid('frozen_sample_runs');ax.add_collection(collection)
    ax.set_xlim(-20,20);ax.set_ylim(94.5,4.5)
    ax.set_xticks([-20,0,20]);ax.set_yticks(np.arange(10,91,10))
    ax.set_xlabel('Time from CS onset (s)',fontsize=8,labelpad=4)
    ax.set_ylabel('Global CS trial',fontsize=8,labelpad=4)
    ax.tick_params(direction='out',length=2.5,width=.6,labelsize=7,top=False,right=False,pad=2)
    for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.6);spine.set_color('#333333')
    for t in [0,10]:ax.axvline(t,color='#168241',linewidth=.65,alpha=.8)
    if spec[-1] is not None:
        ax.plot([spec[-1]]*2,[14.5,64.5],color='#964bad',linewidth=.6,linestyle=':')
    for y in [14.5,64.5]:ax.axhline(y,color='white',linewidth=.7)
    for text,y in [('Pre',9.5),('Train',39.5),('Test',79.5)]:
        ax.text(1.035,y,text,transform=ax.get_yaxis_transform(),rotation=90,
            va='center',ha='left',fontsize=7,color='#444444')
    fig.text(.045,.947,letter,fontsize=14,weight='bold',va='center')
    fig.text(AXRECT[0]+AXRECT[2]/2,.947,name,fontsize=10,ha='center',va='center',color=COLOURS[letter])
    fig.text(AXRECT[0]+AXRECT[2]/2,.909,fish,fontsize=7,ha='center',va='center',color='#555555')
    cax=fig.add_axes([.79,.21,.025,.59])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=cax,ticks=[-1,-.5,0,.5,1])
    bar.set_label('Scaled log vigor (C)',fontsize=8,labelpad=4)
    bar.ax.tick_params(direction='out',length=2,width=.5,labelsize=7,pad=2)
    bar.outline.set_linewidth(.6);bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    stem=OUT/f'Fig1_Panel{letter}_C_BoutSamples_polished'
    for ext in ['svg','png','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=300,facecolor='white')
    # Check every label is within the page before saving the validation record.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[t.get_window_extent(renderer) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    page=fig.bbox
    assert all(b.x0>=-1 and b.y0>=-1 and b.x1<=page.x1+1 and b.y1<=page.y1+1 for b in boxes)
    plt.close(fig)
    return {'panel':letter,'fish':fish,'source_runs_sha256':digest(FROZEN/f'Panel{letter}_display_sample_runs.csv'),
        'geometry':geometry(stem.with_suffix('.svg'),runs),
        'axes':{'xlim':[-20,20],'ylim':[94.5,4.5],'xticks':[-20,0,20],'yticks':list(range(10,91,10)),
            'spines':'four thin 0.6 pt spines; outward ticks','phase_boundaries':[14.5,64.5]},
        'outputs':[{'path':str(stem.with_suffix('.'+ext)),'sha256':digest(stem.with_suffix('.'+ext))} for ext in ['svg','png','pdf']]}

def main():
    frozen=snapshot();OUT.mkdir(exist_ok=True)
    refs=HERE/'references'
    ref=json.loads((refs/'pooled_style_freeze.json').read_text())
    expected=next(r['sha256'] for r in ref['source_bindings']['A'] if r['path'].endswith('.svg'))
    assert digest(refs/'pooled_style.svg')==expected
    assert digest(refs/'legacy_style.svg')=='4944009938fef65c6c3ae40bb9be91118708a7298e2a9367fd304a986fecb0ad'
    records=[panel(spec) for spec in FISH]
    images=[Image.open(OUT/f'Fig1_Panel{p}_C_BoutSamples_polished.png').convert('RGB') for p in 'FGH']
    overview=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)),'white')
    x=0
    for im in images:overview.paste(im,(x,0));x+=im.width
    overview.save(OUT/'FGH_C_BoutSamples_polished.png')
    report={'freeze_manifest':str(FROZEN/'freeze.json'),'freeze_manifest_sha256':digest(FROZEN/'freeze.json'),
        'status':'Version1 processing and values frozen; layout refinement only',
        'changes':'continuous tall global-trial axis, thin boxed spines, -20/0/20 time ticks, white phase boundaries, compact phase labels, coloured condition titles, right colourbar; remove processing text from figure face',
        'unchanged':'all sample C values, sample eligibility/intervals, baseline statistics, bout medians, C formula, palette and limits',
        'references':[{'path':str(p),'sha256':digest(p)} for p in refs.glob('*.svg')],
        'pooled_style_freeze':str(refs/'pooled_style_freeze.json'),'panels':records,
        'code_sha256':digest(Path(__file__))}
    (OUT/'layout_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'freeze':str(FROZEN/'freeze.json'),'panels':records},indent=2))

if __name__=='__main__':main()
