"""Read-only scientific audit of the selected Figure 2 A/B heatmaps.

Writes new evidence exclusively; never changes inputs or figure state.
"""
from pathlib import Path
import base64, difflib, hashlib, io, json, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[1]
BASE = Path('J:/ClassicalConditioning Outputs/current/reviews')
TRACE = BASE/'figure-02-3strace'
OUT = TRACE/'matched-delay-statistics/ab-analysis-review'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p): return {'path':str(p),'sha256':sha(p)}
def summary(v):
    x=np.asarray(v,float);x=x[np.isfinite(x)]
    return {'n':len(x),'mean':float(x.mean()),'sd':float(x.std()),
            'median_abs':float(np.median(abs(x))), 'rms':float(np.sqrt(np.mean(x*x))),
            'p90_abs':float(np.quantile(abs(x),.9)),
            'fraction_abs_gt_025':float(np.mean(abs(x)>.25))}
def pooled(data):
    return data.groupby(['condition_id','acquisition_trial','bin_index']).delta_log_vigor.mean().reset_index(name='value')
def window_metrics(data,cols=('value',)):
    windows={'all':(-20,20),'baseline':(-15,0),'CS':(0,10),'early_CS':(0,9),
             'trace_gap':(10,13),'postUS_delay':(9,12),'postUS_trace':(13,16)}
    out=[]
    for c,g in data.groupby('condition_id'):
        for name,(lo,hi) in windows.items():
            part=g.loc[g.bin_index.between(lo+20,hi+19)]
            for col in cols:out.append({'condition':c,'window':name,'quantity':col,**summary(part[col])})
    return out

def main():
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Preserve prior review')
    state=REPO/'docs/paper/figures/figure-02/current-state.json'
    draft=load(state)['draft']
    for k in ['working_svg','review','legend','semantic_registry']:
        if sha(draft[k]['path'])!=draft[k]['sha256']:raise ValueError(k+' changed')
    ap=BASE/'figure-02-a-freeze-2026-10-14'
    selection=load(REPO/'configs/paper-figures/selections/figure2-A-pooled-freeze-20261014.json')
    for r in selection['data']:
        if sha(ap/Path(r['path']).name)!=r['sha256']:raise ValueError('Frozen A bytes changed')
    ac=load(BASE/'figure-02-a-pooled/calculation.json')
    amap={r['fish']:('control' if r['expected_us_s'] is None else 'delay') for r in ac['records']}
    a=pd.read_parquet(ap/'fish-bins.parquet');a['condition_id']=a.fish.map(amap)
    plan=load(TRACE/'matched-delay-statistics/plan.json')
    parts=[];records=[];inputs=[state,ap/'fish-bins.parquet',ap/'pooled-bins.parquet',TRACE/'matched-delay-statistics/plan.json']
    for r in plan['selected']:
        p=TRACE/'fish'/(r['recording_id']+'.json');record=load(p)
        bp=TRACE/'fish'/(r['recording_id']+'-bins.parquet')
        bound=next(x for x in record['outputs'] if x['path'].endswith('-bins.parquet'))
        if sha(bp)!=bound['sha256']:raise ValueError('Edited B bins')
        part=pd.read_parquet(bp);part['condition_id']=r['condition'];parts.append(part)
        records.append(record['heatmap_provenance']);inputs.extend([p,bp])
    b=pd.concat(parts,ignore_index=True)
    all_data={'A':a,'B':b};report={'inputs':[ref(p) for p in inputs], 'generator':ref(Path(__file__)),
        'selected_figure':draft['working_svg'],'scientific_changes':False,'hypothesis_tests':False}
    metrics=[];pooled_metrics=[];checks={};quality=[];fishrows=[]
    for panel,data in all_data.items():
        p=pooled(data);saved=pd.read_parquet(ap/'pooled-bins.parquet' if panel=='A' else TRACE/'matched-delay-statistics/B-pooled-bins.parquet')
        check=p.merge(saved,on=['condition_id','acquisition_trial','bin_index'])
        baseline=data.loc[data.bin_index.between(5,19)].groupby(['fish','acquisition_trial']).delta_log_vigor.median()
        finite=np.isfinite(data.bin_log_vigor)&np.isfinite(data.baseline_log_vigor)
        checks[panel]={'pooled_max_difference':float(abs(check.value-check.mean_delta_log_vigor).max()),
            'individual_trial_baseline_max_abs_median':float(abs(baseline).max()),
            'subtraction_max_error':float(abs(data.loc[finite,'delta_log_vigor']-(data.loc[finite,'bin_log_vigor']-data.loc[finite,'baseline_log_vigor'])).max()),
            'undefined_baseline_trials':int(data.groupby(['fish','acquisition_trial']).baseline_log_vigor.first().isna().sum()),
            'duplicate_fish_trial_bins':int(data.duplicated(['fish','acquisition_trial','bin_index']).sum())}
        metrics.extend([{'panel':panel,**x} for x in window_metrics(data,('delta_log_vigor','bin_log_vigor','eligible_frames'))])
        pooled_metrics.extend([{'panel':panel,**x} for x in window_metrics(p)])
        for c,g in data.groupby('condition_id'):
            counts=g.groupby(['acquisition_trial','bin_index']).delta_log_vigor.count()
            quality.append({'panel':panel,'condition':c,'fish':g.fish.nunique(),
                'median_contributors':float(counts.median()),'min_contributors':int(counts.min()),
                'finite_fraction':float(g.delta_log_vigor.notna().mean()),
                'median_eligible_frames_finite_bin':float(g.loc[g.delta_log_vigor.notna(),'eligible_frames'].median())})
        for (fish,c),g in data.groupby(['fish','condition_id']):
            for name,lo,hi in [('baseline',-15,0),('early_CS',0,9),('trace_gap',10,13)]:
                z=g.loc[g.bin_index.between(lo+20,hi+19)]
                fishrows.append({'panel':panel,'fish':fish,'condition':c,'window':name,
                    'mean_delta':float(z.delta_log_vigor.mean()),'median_delta':float(z.delta_log_vigor.median()),
                    'mean_abs_delta':float(abs(z.delta_log_vigor).mean()),
                    'median_log_vigor':float(z.bin_log_vigor.median()),'finite_bins':int(z.delta_log_vigor.notna().sum())})
    report['checks']=checks;report['coverage']=quality
    report['pooled_metrics']=pooled_metrics
    # Sample-size diagnostic: fixed-size whole-fish subsets, no replacement.
    rng=np.random.default_rng(20261019);resampling=[]
    for c,target in [('control',28),('trace',29)]:
        g=b.loc[b.condition_id.eq(c)];ids=sorted(g.fish.unique())
        mat=g.pivot(index='fish',columns=['acquisition_trial','bin_index'],values='delta_log_vigor').reindex(ids)
        for name,lo,hi in [('baseline',-15,0),('early_CS',0,9),('all',-20,20)]:
            use=(mat.columns.get_level_values('bin_index')>=lo+20)&(mat.columns.get_level_values('bin_index')<hi+20)
            v=mat.to_numpy()[:,use];rms=[];absmed=[]
            for i in range(1000):
                z=np.nanmean(v[rng.choice(len(ids),target,replace=False)],axis=0)
                rms.append(float(np.sqrt(np.nanmean(z*z))));absmed.append(float(np.nanmedian(abs(z))))
            resampling.append({'condition':c,'window':name,'subset_fish':target,'draws':1000,
                'rms_quantiles':np.quantile(rms,[.025,.5,.975]).tolist(),
                'median_abs_quantiles':np.quantile(absmed,[.025,.5,.975]).tolist()})
    report['matched_size_B_subsets']=resampling
    # Original versus verified additions; source plan is an intake history, not a new selection.
    additions=Path('F:/Digested Data/all3sTrace-full-v1/Metadata/trace3-additions-20261018-verified-run-plan.json')
    history=load(additions);added={r['recording_id'] for r in history['selected']};report['addition_plan']=ref(additions)
    subgroup=[]
    for panel,data in all_data.items():
        splits={'year':data.fish.str[:4]}
        if panel=='B':splits['intake']=data.fish.map(lambda f:'added' if f in added else 'original')
        for variable,values in splits.items():
            for value,g in data.groupby(values):
                subgroup.extend([{'panel':panel,'group_type':variable,'group':str(value),'fish':int(g.fish.nunique()),**x} for x in window_metrics(pooled(g))])
    report['subgroup_pooled_metrics']=subgroup
    a_source=(ap/'calculation-snapshot.py').read_text();b_source=(REPO/'src/classical_conditioning/figures/pooled_log_heatmap.py').read_text()
    report['calculation_source_diff']=''.join(difflib.unified_diff(a_source.splitlines(True),b_source.splitlines(True)))
    report['calibration']={p:{'detector_configs':list({json.dumps(r['detector'],sort_keys=True) for r in rows}),
        'cadence_ms_range':[min(r['cadence']['interval_ms'] for r in rows),max(r['cadence']['interval_ms'] for r in rows)],
        'expected_us_s_range':[min(r['expected_us_s'] for r in rows if r['expected_us_s'] is not None),max(r['expected_us_s'] for r in rows if r['expected_us_s'] is not None)],
        'paired_trial_counts':sorted({r['paired_training_trials'] for r in rows}),
        'frame_loss_flags':sum(bool(r['cadence']['missing_frame_ids'] or r['cadence']['buffer_capacity_exceeded']) for r in rows)} for p,rows in [('A',ac['records']),('B',records)]}
    # Authenticate actual raster colours against the selected numerical matrices.
    root=ET.fromstring(Path(draft['working_svg']['path']).read_bytes());lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black');norm=Normalize(-.25,.25,clip=False);raster=[]
    for panel,data in all_data.items():
        p=pooled(data)
        for i,c in enumerate(['control','delay' if panel=='A' else 'trace']):
            group=lookup[f'fig2__{panel.lower()}{i}__heatmap__vigor-cells'];image=next(n for n in group.iter() if n.tag.endswith('}image'))
            im=Image.open(io.BytesIO(base64.b64decode(image.get('{http://www.w3.org/1999/xlink}href').split(',',1)[1]))).convert('RGBA')
            matrix=p.loc[p.condition_id.eq(c)].pivot(index='acquisition_trial',columns='bin_index',values='value').reindex(index=range(5,95),columns=range(40)).to_numpy()
            rgba=np.asarray(cmap(norm(np.ma.masked_invalid(matrix)))*255,dtype=np.uint8)
            expected=np.asarray(Image.fromarray(rgba[::-1]).resize(im.size,Image.Resampling.NEAREST))
            raster.append({'panel':panel,'condition':c,'size':list(im.size),'exact_colour_match':bool(np.array_equal(np.asarray(im),expected)),
                'maximum_channel_difference':int(abs(np.asarray(im).astype(int)-expected.astype(int)).max())})
    report['raster_checks']=raster
    pd.DataFrame(metrics).to_csv(OUT/'individual-bin-summary.csv',index=False)
    pd.DataFrame(fishrows).to_csv(OUT/'fish-window-summary.csv',index=False)
    pd.DataFrame(pooled_metrics).to_csv(OUT/'pooled-summary.csv',index=False)
    pd.DataFrame(subgroup).to_csv(OUT/'subgroup-pooled-summary.csv',index=False)
    with (OUT/'audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['checks','coverage','pooled_metrics','matched_size_B_subsets','calibration','raster_checks']},indent=2))

if __name__=='__main__':main()
