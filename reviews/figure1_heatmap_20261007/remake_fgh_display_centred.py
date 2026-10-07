"""User-authorized F/G/H correction: centre AFTER bout-summary binning."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image,ImageOps
from display_bin_centring import centre_trial_heatmap_bins,SIGNAL
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT,FISH,digest
SOURCE=ROOT/'fgh-rebuild-panel-e-contract-20261007'
OUT=ROOT/'fgh-trial-baseline-bin-centred-20261007'
OUT.mkdir(parents=True,exist_ok=True)
source_manifest=json.loads((SOURCE/'build_manifest.json').read_text())
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':9})
cmap=plt.get_cmap('managua_r').copy();cmap.set_bad('black');norm=Normalize(-.25,.25,clip=True)
reports=[];checks=[]
for spec in FISH:
    panel,name,fish,*_=spec
    report=next(p for p in source_manifest['panels'] if p['panel']==panel)
    path=Path(report['panel_data']);assert digest(path)==report['panel_data_sha256']
    old=pd.read_parquet(path)
    result=centre_trial_heatmap_bins(old)
    assert len(result)==7200
    np.testing.assert_array_equal(np.isfinite(result.signed_bout_log_bin),np.isfinite(old.signed_bout_log_bin))
    np.testing.assert_array_equal(result.eligible_frames,old.eligible_frames)
    for trial,g in result.groupby('trial'):
        baseline=g.loc[g.bin_center_s.ge(-15)&g.bin_center_s.lt(0),'signed_bout_log_bin'].dropna()
        assert len(baseline)>0
        residual=float(np.median(baseline));assert abs(residual)<1e-12
        offset=float(g.display_centre_offset_from_previous.iloc[0])
        np.testing.assert_allclose(g.signed_bout_log_bin,g.frame_baseline_centred_bin-offset,atol=1e-12,rtol=0,equal_nan=True)
        checks.append({'panel':panel,'fish':fish,'trial':int(trial),'finite_baseline_bins':len(baseline),
            'old_baseline_bin_median':offset,'new_baseline_bin_median':residual,
            'new_baseline_bin_mean':float(baseline.mean()),
            'new_negative_baseline_bins':int((baseline<0).sum()),
            'new_positive_baseline_bins':int((baseline>0).sum()),
            'reference_uncentred_log_bins':float(g.display_baseline_log_bin_median.iloc[0])})
    if 'framewise_log_bin' in result:result=result.rename(columns={'framewise_log_bin':'previous_frame_reference_centred_framewise_log_bin'})
    result=result.rename(columns={'baseline_log_median':'previous_frame_baseline_log_median'})
    data=OUT/f'Panel{panel}_trial_baseline_bin_centred.parquet';result.to_parquet(data,index=False)
    fig=plt.figure(figsize=(6.1,5.5))
    grid=fig.add_gridspec(3,1,height_ratios=[10,50,30],left=.17,right=.78,bottom=.16,top=.80,hspace=.10)
    matrix=result.pivot(index='trial',columns='bin_center_s',values='signed_bout_log_bin')
    for i,(phase,lo,hi) in enumerate([('Pre-Train',5,14),('Train',15,64),('Test',65,94)]):
        a=fig.add_subplot(grid[i]);values=matrix.reindex(range(lo,hi+1)).to_numpy()
        # Vector cells, including an explicit black underlay for empty bins.
        a.set_facecolor('black')
        a.pcolormesh(np.arange(-20,20.5,.5),np.arange(lo-.5,hi+1.5,1),np.ma.masked_invalid(values),shading='flat',cmap=cmap,norm=norm,rasterized=False,edgecolors='none')
        a.set_ylim(hi+.5,lo-.5);a.set_yticks([lo,hi]);a.tick_params(labelsize=8,length=2)
        a.set_ylabel(phase,fontsize=9)
        for sec in [0,10]:a.axvline(sec,color='#0d7f3c',lw=.8,ls='--' if sec==10 else '-')
        if phase=='Train' and spec[-1] is not None:a.axvline(spec[-1],color='#78358c',ls=':',lw=.9)
        a.set_xticks([-20,-10,0,10,20]);a.set_xlim(-20,20)
        if i<2:a.tick_params(labelbottom=False)
        else:a.set_xlabel('Time from measured CS onset (s)',fontsize=10)
    fig.text(.04,.955,panel,fontsize=19,weight='bold')
    fig.text(.17,.955,f'{name} · {fish}',fontsize=13,weight='bold')
    fig.text(.17,.905,'Tail bend angular speed · binned bout log vigor',fontsize=9)
    fig.text(.17,.865,'Each row: median baseline bins [−15, 0) s = 0',fontsize=9)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.825,.16,.023,.64]),ticks=[-.25,0,.25],extend='both')
    cb.ax.tick_params(labelsize=8);cb.set_label('Log-bin value − trial baseline-bin median',fontsize=9)
    fig.text(.17,.055,'0.5 s bins; finite eligible frames only; black = no contribution',fontsize=8)
    fig.text(.17,.026,'managua_r · colour saturation only; stored values uncapped',fontsize=8)
    stem=OUT/f'Fig1_Panel{panel}_{name.replace(" ","")}_trial-baseline-bin-centred'
    for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    rec={'panel':panel,'fish':fish,'signal':SIGNAL,'baseline_s':[-15,0],
        'baseline_reference':'median finite displayed 0.5 s bins from this trial only; equal bin weight',
        'processing':'unchanged reconstructed-acquisition metric/detector; eligible frame logs → bout median → finite-frame-weighted bin mean → subtract trial median baseline-bin value',
        'normalisation':'one additive log reference; no division or stored-value clipping',
        'colour_limits':[-.25,.25],'palette':'managua_r','source':str(path),'source_sha256':digest(path),
        'input_rebuild_manifest':str(SOURCE/'build_manifest.json'),'input_rebuild_manifest_sha256':digest(SOURCE/'build_manifest.json'),
        'input_raw_provenance':report['verified_input_artifacts'],'upstream_preprocessing_status':report['scope'],
        'missing_baseline_policy':'all trial bins NaN if no finite baseline bin; none of these trials trigger this',
        'panel_data':str(data),'panel_data_sha256':digest(data),'svg_sha256':digest(stem.with_suffix('.svg')),
        'selection_status':'user-directed correction; no freeze',
        'checks':'all 90 baseline bin medians zero within 1e-12; support and counts unchanged; every within-trial difference preserved'}
    stem.with_suffix('.svg.json').write_text(json.dumps(rec,indent=2));reports.append(rec)
pd.DataFrame(checks).to_csv(OUT/'all_270_trial_baseline_checks.csv',index=False)
thumbs=[]
for spec in FISH:
    panel,name,*_=spec
    with Image.open(OUT/f'Fig1_Panel{panel}_{name.replace(" ","")}_trial-baseline-bin-centred.png') as im:thumbs.append(ImageOps.contain(im.convert('RGB'),(900,820)))
gallery=Image.new('RGB',(2700,820),'white')
for i,im in enumerate(thumbs):gallery.paste(im,(900*i,0))
gallery.save(OUT/'F-G-H_corrected.png')
# Same palette/range/layout in before/after comparison, isolating reference change.
with Image.open(SOURCE/'F-G-H_rebuilt.png') as oldim:
    before=oldim.convert('RGB')
comparison=Image.new('RGB',(2700,1640),'white');comparison.paste(before,(0,0));comparison.paste(gallery,(0,820));comparison.save(OUT/'before_after_reference_correction.png')
legacy=[REPO/'legacy/scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py',REPO/'legacy/scripts/2_ExampleFishPlotting.py']
manifest={'panels':reports,'full_assembly_created':False,'other_panels_modified':False,'frozen':False,
    'legacy_files_reviewed':[{'path':str(p),'sha256':digest(p)} for p in legacy],
    'code':[{'path':str(p),'sha256':digest(p)} for p in [Path(__file__),Path(__file__).with_name('display_bin_centring.py')]],
    'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()]}
(OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
print(pd.DataFrame(checks).query('panel == "G" and trial == 93').to_string(index=False))
print('Checked all 270 trial baselines; preserved all 21600 bin support positions.');print(OUT)
