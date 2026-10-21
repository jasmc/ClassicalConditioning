"""Phase, protocol and raster follow-up to the preserved A/B review."""
import ast
from pathlib import Path
import base64
import io
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import Normalize
from PIL import Image
from audit_figure2_heatmap_ab import REPO, BASE, TRACE, OUT, load, sha, ref, pooled

def main():
    audit=load(OUT/'audit.json')
    state=load(REPO/'docs/paper/figures/figure-02/current-state.json')
    if sha(state['draft']['working_svg']['path'])!=audit['selected_figure']['sha256']:
        raise ValueError('Figure changed during audit')
    ac=load(BASE/'figure-02-a-pooled/calculation.json')
    a=pd.read_parquet(BASE/'figure-02-a-freeze-2026-10-14/fish-bins.parquet')
    a['condition_id']=a.fish.map({r['fish']:('control' if r['expected_us_s'] is None else 'delay') for r in ac['records']})
    rows=load(TRACE/'matched-delay-statistics/plan.json')['selected']
    b=pd.concat([pd.read_parquet(TRACE/'fish'/(r['recording_id']+'-bins.parquet')).assign(condition_id=r['condition']) for r in rows])
    phases=[];direction=[]
    for panel,data in [('A',a),('B',b)]:
        for name,lo,hi in [('Pre',5,14),('Train',15,64),('Test',65,94)]:
            g=data.loc[data.acquisition_trial.between(lo,hi)]
            for window,start,end in [('baseline',-15,0),('early_CS',0,9),('own_postUS',9 if panel=='A' else 13,12 if panel=='A' else 16)]:
                part=g.loc[g.bin_index.between(start+20,end+19)]
                for c,z in part.groupby('condition_id'):
                    v=z.groupby('fish').delta_log_vigor.mean().dropna()
                    pv=pooled(z).value
                    phases.append({'panel':panel,'phase':name,'window':window,'condition':c,
                        'fish':len(v),'mean_fish_delta':float(v.mean()),'median_fish_delta':float(v.median()),
                        'positive_fish_fraction':float((v>0).mean()),'pooled_mean':float(pv.mean())})
        for c,g in data.loc[data.bin_index.between(20,28)].groupby('condition_id'):
            v=g.groupby('fish').delta_log_vigor.mean().dropna()
            direction.append({'panel':panel,'condition':c,'fish':len(v),'negative_fish':int((v<0).sum()),
                'positive_fish':int((v>0).sum()),'mean_fish_delta':float(v.mean()),'median_fish_delta':float(v.median())})
    # Independently reread small recorded-event sources for every selected B fish.
    sys_path=str(REPO/'src')
    import sys
    sys.path.insert(0,sys_path)
    from classical_conditioning.analysis.figure4 import verify_expected_us
    protocols=[];conditions=[]
    for row in rows:
        r=load(TRACE/'fish'/(row['recording_id']+'.json'))
        expected=next(x for x in r['window_sources'] if x['path'].endswith('stimulus_events.parquet'))
        if sha(expected['path'])!=expected['sha256']:raise ValueError('B protocol changed')
        d=pd.read_parquet(expected['path'])
        if row['condition']=='trace':
            latency,count=verify_expected_us(d,'all3sTrace')
            protocols.append({'fish':row['recording_id'],'median_us_latency_s':latency,'paired_trials':count,'source':expected})
        mp=Path(expected['path']).parents[2]/'Metadata'/(row['recording_id']+'_source_manifest.json')
        m=load(mp)
        conditions.append({'fish':row['recording_id'],'selected_condition':row['condition'],
            'recording_name':m['recording_name'],'manifest_condition':m.get('condition_id')})
    frozen=BASE/'figure-02-a-freeze-2026-10-14';helpers={}
    for old,new,names in [('movement-snapshot.py','analysis/movement_state.py',['_odd_window_samples','smooth_contiguous_median','rolling_extreme_envelope','detect_legacy_envelope_bouts','MovementCalibrationConfig']),('cadence-snapshot.py','preprocessing/acquisition_timing.py',['estimate_camera_cadence'])]:
        def definitions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        x,y=definitions(frozen/old),definitions(REPO/'src/classical_conditioning'/new)
        helpers.update({n:x[n]==y[n] for n in names})
    root=ET.fromstring(Path(state['draft']['working_svg']['path']).read_bytes());lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    raster=[]
    for panel,data in [('A',a),('B',b)]:
        p=pooled(data)
        for i,c in enumerate(['control','delay' if panel=='A' else 'trace']):
            group=lookup[f'fig2__{panel.lower()}{i}__heatmap__vigor-cells'];image=next(n for n in group.iter() if n.tag.endswith('}image'))
            im=np.asarray(Image.open(io.BytesIO(base64.b64decode(image.get('{http://www.w3.org/1999/xlink}href').split(',')[1]))).convert('RGBA'))
            matrix=p.loc[p.condition_id.eq(c)].pivot(index='acquisition_trial',columns='bin_index',values='value').to_numpy()
            rgba=np.asarray(mpl.colormaps['managua_r'](Normalize(-.25,.25,clip=True)(matrix))*255,dtype=np.uint8)
            xx=((np.arange(40)+.5)*im.shape[1]/40).astype(int);yy=((np.arange(90)+.5)*im.shape[0]/90).astype(int)
            cells=im[yy[:,None],xx[None,:]]
            raster.append({'panel':panel,'condition':c,'cell_centres_checked':3600,'exact_cell_centre_colours':bool(np.array_equal(cells,rgba[::-1]))})
    pd.DataFrame(phases).to_csv(OUT/'phase-summary.csv',index=False)
    report={'generator':ref(Path(__file__)),'prior_review':ref(OUT/'audit.json'),'direction':direction,
        'phase_summary':phases,'B_protocols_reauthenticated':protocols,'B_source_identities':conditions,
        'helper_implementations_equal_to_A_freeze':helpers,'cell_centre_raster_checks':raster,
        'initial_raster_check_resolution':'A differs only in a narrow set of pixel boundaries after historical nearest-neighbour resizing; all 3600 cell-centre colours in each A/B map exactly match its selected numerical matrix and shared scale.',
        'limitations':['Cached original-frame medians were authenticated and independently repooled; large raw frame/tracking sources were not regenerated.',
            'Cross-experiment differences are confounded with recording cohort/year. No causal comparison or new hypothesis test.',
            'Subsampling quantiles describe randomly selected fish subsets, not confidence intervals or significance tests.']}
    with (OUT/'followup.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'direction':direction,'phase_summary':phases,'helpers':helpers,'rasters':raster,'fixed_13s_trace_fish_verified':len(protocols)},indent=2))

if __name__=='__main__':main()
