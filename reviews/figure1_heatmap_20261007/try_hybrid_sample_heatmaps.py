"""F/G/H only: sample display, post-bin baseline reference, C/D ranges."""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from hybrid_sample_display import hybrid_sample_values
from baseline_colour_mapping import baseline_colour_norm

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
from build_figure1_legacy_vigor_heatmaps import ROOT,FISH,digest
OUT=ROOT/'fgh-sample-display-bin-baseline-20261008'
V5=ROOT/'cadence-review-v5-20261007'
REF=ROOT/'fgh-all-options-baseline-colour-centred-20261007'
OUT.mkdir(parents=True,exist_ok=True)
reference_manifest=json.loads((REF/'colour_review_manifest.json').read_text())
reference_path=REF/'all_options_bins.parquet'
reference_record=next(r for r in reference_manifest['outputs'] if Path(r['path'])==reference_path)
assert digest(reference_path)==reference_record['sha256']
reference_bins=pd.read_parquet(reference_path)
reports=[];trial_reports=[];bin_tables=[]
writer=None
current_spec=None
matrices=None
fps=None

def capture(trial,ids,seconds,raw,valid,moving,bouts):
    global writer
    eligible=valid&moving&(bouts>0)&np.isfinite(raw)&(raw>0)
    log=np.log(np.where(eligible,raw,np.nan))
    p=pd.DataFrame({'trial':trial,'FrameID':ids,'time_s':seconds,'bout_id':bouts,
                    'eligible':eligible,'frame_log':log})
    p['bout_log']=p.loc[p.eligible].groupby('bout_id').frame_log.transform('median').reindex(p.index)
    samples,bins,counts,meta=hybrid_sample_values(seconds,p.bout_log.to_numpy())
    for name,values in samples.items():p[name]=values
    panel=current_spec[0]
    old=reference_bins[reference_bins.panel.eq(panel)&reference_bins.trial.eq(trial)].sort_values('bin_center_s')
    np.testing.assert_allclose(bins,old.uncentred_bout_log_bin,atol=1e-12,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(counts,old.eligible_frames)
    np.testing.assert_allclose(bins-meta['reference'],old.colour_centred_log_bin,atol=1e-12,rtol=0,equal_nan=True)
    range_errors={}
    for option in ['C','D']:
        scale=meta[option+'_scale']
        expected=np.clip((bins-meta['reference'])/scale,-1,1) if scale>0 else np.full(80,np.nan)
        # Narrow control ranges magnify sub-picounit summation differences.
        np.testing.assert_allclose(expected,old[option],atol=1e-10,rtol=0,equal_nan=True)
        range_errors[option+'_previous_bin_max_error']=float(np.nanmax(np.abs(expected-old[option].to_numpy()))) if scale>0 else np.nan
    # One actual acquired sample per cadence slot; no averaging or interpolation.
    slots=np.floor((seconds+20)*fps).astype(int)
    assert (np.diff(slots)>0).all() and slots.min()>=0 and slots.max()<matrices['C'].shape[1]
    for option in ['C','D']:
        matrices[option][trial-5,slots]=samples[option]
        if meta[option+'_scale']>0:
            np.testing.assert_array_equal(np.isfinite(samples[option]),eligible)
        else:assert np.isnan(samples[option]).all()
    schema_table=pa.Table.from_pandas(p,preserve_index=False)
    if writer is None:writer=pq.ParquetWriter(OUT/f'Panel{panel}_sample_values.parquet',schema_table.schema,compression='zstd')
    writer.write_table(schema_table)
    bin_tables.append(pd.DataFrame({'panel':panel,'trial':trial,'bin_center_s':np.arange(-19.75,20,.5),
                                   'uncentred_bout_log_bin':bins,'eligible_frames':counts,
                                   'baseline_log_bin_median':meta['reference'],
                                   'centred_bout_log_bin':bins-meta['reference']}))
    baseline=(seconds>=-15)&(seconds<0)&eligible
    trial_reports.append({'panel':panel,'trial':trial,**meta,**range_errors,'acquired_samples':len(seconds),
                          'eligible_samples':int(eligible.sum()),
                          'excluded_samples':int((~eligible).sum()),
                          'native_baseline_sample_median_after_bin_reference':float(np.nanmedian(samples['centred_log'][baseline])) if baseline.any() else np.nan,
                          'matched_previous_scalar_bins':80})

# Instrument a read-only callback; never execute the builder's main/render path.
source=REPO/'scripts/build_figure1_cadence_review_v5.py'
code=source.read_text(encoding='utf-8')
needle="        if panel=='F' and trial in TRIALS:"
assert code.count(needle)==1
code=code.replace(needle,'        capture(trial,ids[s],seconds,raw[s],valid[s],moving[s],boutids[s])\n'+needle)
namespace={'__file__':str(source),'__name__':'hybrid_sample_readonly','capture':capture}
exec(compile(code,str(source),'exec'),namespace)
plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
cmap=plt.get_cmap('managua_r').copy();cmap.set_bad('black')
norm=baseline_colour_norm(1.)

def render(spec,option,matrix):
    panel,name,fish,*_=spec
    title='Half baseline P10–P90 width' if option=='C' else 'Larger baseline P10/P90 distance'
    fig=plt.figure(figsize=(6.1,5.5))
    grid=fig.add_gridspec(3,1,height_ratios=[10,50,30],left=.17,right=.78,bottom=.16,top=.79,hspace=.10)
    for i,(phase,first,last) in enumerate([('Pre-Train',5,14),('Train',15,64),('Test',65,94)]):
        ax=fig.add_subplot(grid[i]);ax.set_facecolor('black')
        ax.imshow(matrix[first-5:last-4],cmap=cmap,norm=norm,aspect='auto',interpolation='nearest',
                  resample=False,extent=(-20,-20+matrix.shape[1]/fps,last+.5,first-.5))
        ax.set_ylim(last+.5,first-.5);ax.set_yticks([first,last]);ax.set_ylabel(phase)
        ax.set_xlim(-20,20);ax.set_xticks([-20,-10,0,10,20])
        if i<2:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Time from measured CS onset (s)')
        for time in [0,10]:ax.axvline(time,color='#0d7f3c',lw=.8,ls='--' if time==10 else '-')
        if phase=='Train' and spec[-1] is not None:ax.axvline(spec[-1],color='#78358c',lw=.8,ls=':')
    fig.text(.04,.955,panel,fontsize=19,weight='bold')
    fig.text(.17,.955,f'{name} · {fish}',fontsize=13,weight='bold')
    fig.text(.17,.90,'Time samples · repeated bout median log vigor',fontsize=9)
    fig.text(.17,.85,f'{option} · {title}\nReference: median of trial baseline 0.5 s bins [−15,0)',fontsize=8)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.825,.16,.023,.63]),ticks=[-1,0,1],extend='both')
    cb.set_label('Trial-scaled bout log vigor')
    fig.text(.17,.055,'Native sample display · black = excluded / undefined',fontsize=8)
    fig.text(.17,.026,'Bins set reference and scale only · managua_r · clipped ±1',fontsize=8)
    stem=OUT/f'{option}_Panel{panel}_{name.replace(" ","")}_sample_display'
    for ext in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=300)
    plt.close(fig)
    return stem.with_suffix('.png')

for spec in FISH:
    current_spec=spec;panel,name,fish,*_=spec
    print(f'Reconstructing native samples: {panel} {name}',flush=True)
    saved=V5/f'Fig1_Panel{panel}_{name.replace(" ","")}_presumed-cadence_v5.svg'
    meta=json.loads(saved.with_suffix('.svg.json').read_text())
    assert digest(saved)==meta['svg_sha256']
    assert digest(Path(meta['panel_data']))==meta['panel_data_sha256']
    for item in meta['input_artifacts']:assert digest(Path(item['path']))==item['sha256']
    fps=float(meta['fps'])
    matrices={option:np.full((90,int(np.ceil(40*fps))),np.nan,np.float32) for option in ['C','D']}
    oldbins,traces,events,provenance=namespace['fish_data'](spec)
    assert abs(provenance['fps']-fps)<1e-10
    writer.close();writer=None
    old=pd.read_parquet(meta['panel_data'])
    np.testing.assert_allclose(oldbins['Signed log vigor'],old['Signed log vigor'],atol=1e-12,rtol=0,equal_nan=True)
    np.savez_compressed(OUT/f'Panel{panel}_sample_display_grid.npz',
                        time_slot_left_s=-20+np.arange(matrices['C'].shape[1])/fps,
                        trials=np.arange(5,95),**matrices)
    for option,matrix in matrices.items():render(spec,option,matrix)
    report={'panel':panel,'fish':fish,'fps':fps,'sample_interval_ms':1000/fps,
            'native_grid_shape':list(matrices['C'].shape),'grid_assignment':'one sample per acquired-cadence slot; no mean/interpolation; exact times retained in Parquet',
            'verified_input_artifacts':provenance['input_artifacts'],
            'sample_data':str(OUT/f'Panel{panel}_sample_values.parquet'),
            'sample_data_sha256':digest(OUT/f'Panel{panel}_sample_values.parquet')}
    reports.append(report)
    print(f'Completed {panel}: sample matrices {matrices["C"].shape}; all 7200 reference bins match.',flush=True)
    del matrices,oldbins,traces,events;gc.collect()

pd.concat(bin_tables,ignore_index=True).to_parquet(OUT/'reference_bins.parquet',index=False)
pd.DataFrame(trial_reports).to_csv(OUT/'trial_baseline_and_support_audit.csv',index=False)
for option in ['C','D']:
    overview=Image.new('RGB',(2700,820),'white')
    for i,spec in enumerate(FISH):
        panel,name,*_=spec
        with Image.open(OUT/f'{option}_Panel{panel}_{name.replace(" ","")}_sample_display.png') as im:
            thumb=ImageOps.contain(im.convert('RGB'),(900,820))
            overview.paste(thumb,(900*i,0))
    overview.save(OUT/f'{option}_F-G-H_sample_display.png')
report={'scope':'F/G/H only; hybrid trial requested 8 October 2026; no other panels or shared preprocessing changed',
        'display':'native time samples carrying repeated bout median log vigor',
        'baseline_reference':'median of scalar 0.5 s finite-frame mean bins in each trial [-15,0)',
        'quantile_reference':'P10/P50/P90 of those same baseline bins, equal bin weights',
        'C':'(sample-P50)/((P90-P10)/2), clipped [-1,1]',
        'D':'(sample-P50)/max(P50-P10,P90-P50), clipped [-1,1]',
        'palette':'managua_r','norm':'CenteredNorm(vcenter=0, halfrange=1, clip=True)',
        'baseline_reference_bins_matched':21600,'panels':reports,
        'reference_source':reference_record,
        'code':[{'path':str(p),'sha256':digest(p)} for p in [Path(__file__),Path(__file__).with_name('hybrid_sample_display.py'),source]],
        'outputs':[{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.is_file() and p.name!='hybrid_build_manifest.json']}
(OUT/'hybrid_build_manifest.json').write_text(json.dumps(report,indent=2))
print(OUT,flush=True)
