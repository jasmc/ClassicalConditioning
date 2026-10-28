"""Draft D/E pair: one measured frame table and an explicit semantic layout contract.

This author-requested draft recomputes E; it does not replace any accepted freeze.
The existing E bout mask is joined by trial + FrameID, never by row position.
"""
from __future__ import annotations
import ast, base64, hashlib, io, json, re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, to_hex
from classical_conditioning.external_artifacts import resolve_artifact

ROLE_MAP={'trial-block-label':'phase.label','trial-number-label':'annotation.note','axis-spine':'axis.spine','axis-tick':'axis.tick','tick-label':'axis.tick_label','tail-angle-trace':'trace.tail_angle','raw-vigor-trace':'trace.raw_vigor','scaled-vigor-series':'trace.raw_vigor','heatmap-bin':'heatmap.vigor','axis-label':'axis.label'}
NS='http://www.w3.org/2000/svg'
ROOT=Path(__file__).resolve().parents[3]
TRIALS=(9,17,63,66,93)
STAGES=('Pre-Train','Early Train','Late Train','Early Test','Late Test')
FISH='20221115_07'
E_TRACE_LIMIT=30
PT=183/25.4*72/1800
STYLE_PATH=ROOT/'configs/paper-figures/figure-elements.json'
STYLE_SPEC=json.loads(STYLE_PATH.read_text())
FRAME_ROOT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-01/panels-a-b-c-d/versions/frozen/2026-10-05-ABCD')
FRAME_PATH=FRAME_ROOT/'Fig1_PanelD_TailAngle_pre15_frames.parquet'
EVENT_PATH=FRAME_ROOT/'Fig1_PanelD_events.parquet'
ELIGIBILITY_PATH=Path('reviews/fgh_full_bouts_baseline_samples_20261009/PanelF_complete_bout_sample_data.parquet')
SELECTION_PATH=ROOT/'records/frozen-analyses/figure-01/panel-e/2026-10-09-heatmap-rows/manifests/selection.json'
PALETTE_SELECTION_PATH=ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-09-version-12/manifests/selection.json'
PALETTE_PATH=ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-09-version-12/code/palette_functions_snapshot.py'
F_POINTER_PATH=ROOT/'configs/paper-figures/selections/figure1-fgh-version12-freeze-20261009.json'
def f_archive_path():return resolve_artifact('reviews/fgh_candidate_palette_versions_20261009/index.html')
EXPECTED={FRAME_PATH:'004861483e76a116544930769f5eaff85db1231e1d00d26c4e9d919d789efe32',EVENT_PATH:'c33e5e11889a5f8982ecee277585549363afa0cc870aab0ceb93eb8b92199007',ELIGIBILITY_PATH:'0ea6642119308fec55a6cb00c370b003bc6b3cc6c9f1a043dca1d3cc90bee030'}

def dependencies():return [Path(__file__),STYLE_PATH,FRAME_PATH,EVENT_PATH,resolve_artifact(ELIGIBILITY_PATH,expected_sha256=EXPECTED[ELIGIBILITY_PATH]),SELECTION_PATH,PALETTE_PATH,PALETTE_SELECTION_PATH,F_POINTER_PATH,f_archive_path(),ROOT/'records/frozen-analyses/index.json']
def sha(data):return hashlib.sha256(data).hexdigest()
def finite_json(x):
    if isinstance(x,np.ndarray):return [finite_json(v) for v in x.tolist()]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(int,np.integer)):return int(x)
    return x

def scale_bins(time,raw,eligible,edges):
    """Existing E recipe, applied to D's vigor and measured time without clipping."""
    good=np.asarray(eligible,bool)&np.isfinite(raw)&(raw>0)
    log=np.full(len(raw),np.nan);log[good]=np.log(raw[good])
    masks=[good&(time>=a)&(time<b) for a,b in zip(edges[:-1],edges[1:])]
    counts=np.array([m.sum() for m in masks]);med=np.array([np.median(log[m]) if m.any() else np.nan for m in masks])
    baseline=(edges[:-1]>=-15)&(edges[:-1]<0)&np.isfinite(med)
    z=np.full(len(med),np.nan);centre=den=None
    if baseline.sum()>=10:
        centre=float(np.median(med[baseline]));d=med-centre
        lo,hi=np.quantile(d[baseline],[.1,.9],method='linear');den=float(max(-lo,hi))
        if den>1e-12:z=.7*d/den
    return z,med,counts,good,dict(baseline_bins=int(baseline.sum()),baseline_log=centre,denominator_log=den)

def scale_frames(time,raw,eligible):
    """E black trace: unbinned raw vigor, raw-frame baseline and raw-frame scaling."""
    good=np.asarray(eligible,bool)&np.isfinite(raw)&(raw>0)
    baseline=good&(time>=-15)&(time<0)
    values=np.full(len(raw),np.nan)
    params=dict(baseline_frames=int(baseline.sum()),baseline_raw=None,denominator_raw=None,p10_raw=None,p90_raw=None,baseline_span_s=None)
    if baseline.any():
        centre=float(np.median(raw[baseline]))
        deviation=np.full(len(raw),np.nan);deviation[good]=raw[good]-centre
        lo,hi=np.quantile(deviation[baseline],[.1,.9],method='linear');den=float(max(-lo,hi))
        params.update(baseline_raw=centre,denominator_raw=den,p10_raw=float(lo),p90_raw=float(hi),baseline_span_s=[float(time[baseline].min()),float(time[baseline].max())])
        if den>1e-12:values[good]=.7*deviation[good]/den
    return values,good,params

def load_f_rows(selection,colour):
    """Copy selected F rows, including exact rendered RGBA bytes, without redefining F."""
    from PIL import Image
    pointer=json.loads(F_POINTER_PATH.read_text());path=f_archive_path();raw=path.read_bytes()
    if sha(raw)!=pointer['container']['sha256']:raise ValueError('F archive hash changed')
    page=raw.decode()
    def script(identifier):
        return json.loads(re.search(r'<script[^>]*id="'+re.escape(identifier)+r'"[^>]*>(.*?)</script>',page,re.S)[1])
    audit=script(selection['source_audit_id'])
    if sha(json.dumps(audit['cells'],ensure_ascii=True,allow_nan=False,sort_keys=True).encode())!=selection['source_cells_sha256']:
        raise ValueError('F numerical row audit changed')
    archive=script(pointer['freeze_manifest']['script_id'])
    entry=archive['files']['frozen-candidate.svg'];svg=base64.b64decode(entry['base64'])
    if sha(svg)!=pointer['exports'][0]['sha256'] or sha(svg)!=entry['sha256']:raise ValueError('Frozen F SVG changed')
    node=next(n for n in ET.fromstring(svg).iter() if n.get('id')=='fig1__f__heatmap__vigor-cells')
    png=base64.b64decode((node.get('href') or node.get('{http://www.w3.org/1999/xlink}href')).split(',')[1])
    pixels=np.asarray(Image.open(io.BytesIO(png)).convert('RGBA'))
    # Frozen SVG explicitly flips this raster vertically. Sample each cell centre.
    if node.get('transform')!='scale(1 -1) translate(0 -220.56)':raise ValueError('Unreviewed F raster transform')
    ys=((np.arange(90)+.5)*pixels.shape[0]/90).astype(int)
    xs=((np.arange(80)+.5)*pixels.shape[1]/80).astype(int)
    colours=pixels[ys[:,None],xs][::-1]
    rows={}
    for trial in range(5,95):
        cell=next(c for c in audit['cells'] if c['panel']=='F' and c['trial']==trial)
        median=np.asarray(cell['bin_median_log'],float)
        params=next(p for p in selection['trial_parameters'] if p['panel']=='F' and p['trial']==trial)
        z=np.full(80,np.nan)
        if params['defined']:
            bm=(np.asarray(selection['bin_edges_s'][:-1])>=-15)&(np.asarray(selection['bin_edges_s'][:-1])<0)&np.isfinite(median)
            centre=float(np.median(median[bm]));lo,hi=np.quantile((median-centre)[bm],[.1,.9],method='linear')
            np.testing.assert_allclose([centre,lo,hi,max(-lo,hi)],[params['baseline_log'],params['p10_log'],params['p90_log'],params['denominator_log']],atol=1e-12,rtol=0)
            z=.7*(median-params['baseline_log'])/params['denominator_log']
        rgba=colours[trial-5]
        if not np.array_equal(np.all(rgba[:,:3]==0,axis=1),np.isnan(z)) or not np.all(rgba[:,3]==255):
            raise ValueError('F raster missingness does not match stored rows')
        expected_rgb=np.asarray([[int(colour(v)[k:k+2],16) for k in (1,3,5)] for v in z])
        if np.max(abs(rgba[:,:3].astype(int)-expected_rgb))>1:raise ValueError('Frozen F pixels disagree with selected numerical normalization/palette')
        if trial in TRIALS:
            rows[trial]=dict(values=z,medians=median,counts=np.asarray(cell['eligible_frame_count']),params=params,colours=['#'+''.join(f'{int(c):02x}' for c in rgb[:3]) for rgb in rgba],rgba_sha256=sha(rgba.tobytes()),values_sha256=sha(z.astype('<f8').tobytes()))
    return rows,dict(archive_path=str(path),archive_sha256=sha(raw),frozen_svg_sha256=sha(svg),F_raster_png_sha256=sha(png),source_cells_sha256=selection['source_cells_sha256'],row_mapping={str(t):t-4 for t in TRIALS},all_90_rows_missingness_verified=True)

def validate_pair(registries,contract):
    """Validate declared identity/data relationships and numeric layout, not colour guesses."""
    for t in TRIALS:
        d=registries['D'][f'trial-{t}-trace'];e=registries['E'][f'trial-{t}-trace'];h=registries['E'][f'trial-{t}-heatmap']
        for key in ('fish','trial','frame_ids_sha256','time_values_sha256','time_coordinate'):
            if d['data_mapping'][key]!=e['data_mapping'][key]:
                raise ValueError(f'D/E shared mapping mismatch: trial {t}, {key}')
        if h['data_mapping']['fish']!=FISH or h['data_mapping']['trial']!=t or h['data_mapping']['F_display_row']!=t-4:
            raise ValueError('E band does not map to the corresponding F row')
        if d['layout']!=e['layout']:raise ValueError(f'D/E axis rectangles differ: trial {t}')
        if d['row_label']!=e['row_label']:raise ValueError(f'D/E names differ: trial {t}')
    if contract['D_body_size']!=contract['E_body_size']:raise ValueError('D/E body sizes differ')
    return dict(same_fish=True,same_trials=True,same_frame_ids=True,same_time_values=True,
                black_trace_uses_D_vigor=True,colour_bands_are_exact_F_rows=True,equal_axis_rectangles=True,equal_body_size_excluding_colourbar=True,
                row_labels_and_centres_match=True)

def build_pair():
    from classical_conditioning.external_artifacts import resolve_artifact
    index=json.loads((ROOT/'records/frozen-analyses/index.json').read_text())
    known={str(ROOT/i['path']):i['sha256'] for i in index['files']}
    sources=[]
    for p in dependencies():
        resolved=resolve_artifact(p,expected_sha256=EXPECTED.get(p))
        data=resolved.read_bytes();digest=sha(data)
        if p in (SELECTION_PATH,PALETTE_PATH,PALETTE_SELECTION_PATH) and digest!=known[str(p)]:raise ValueError('Frozen definition changed: '+str(p))
        if p in EXPECTED and digest!=EXPECTED[p]:raise ValueError('D/E scientific input changed: '+str(p))
        sources.append(dict(path=str(resolved),sha256=digest))
    selection=json.loads(SELECTION_PATH.read_text());assert selection['fish']==FISH and tuple(selection['trials'])==TRIALS
    f_selection=json.loads(PALETTE_SELECTION_PATH.read_text())
    edges=np.asarray(selection['bin_edges_s'],float)
    np.testing.assert_array_equal(edges,np.linspace(-20,20,81))
    d=pd.read_parquet(resolve_artifact(FRAME_PATH,expected_sha256=EXPECTED[FRAME_PATH]));old=pd.read_parquet(resolve_artifact(ELIGIBILITY_PATH,expected_sha256=EXPECTED[ELIGIBILITY_PATH]))
    old=old[old.trial.isin(TRIALS)]
    joined=d.merge(old[['trial','FrameID','eligible','raw_vigor','time_s']],left_on=['Trial number','FrameID'],right_on=['trial','FrameID'],how='left',validate='one_to_one',indicator=True)
    events=pd.read_parquet(resolve_artifact(EVENT_PATH,expected_sha256=EXPECTED[EVENT_PATH]))
    tree=ast.parse(PALETTE_PATH.read_text());ns={'np':np}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'authenticated_palette_functions','exec'),ns)
    lab=ns['rgb_lab']([to_rgb(c) for c in json.loads(PALETTE_SELECTION_PATH.read_text())['palette']]);u=np.linspace(0,1,513)
    rgb=np.clip(ns['lab_rgb'](np.stack([np.interp(u,[0,.5,1],lab[:,i]) for i in range(3)],axis=-1)),0,1)
    def colour(z):return to_hex(rgb[min(512,int(np.clip((z+1)/2,0,1)*513))]) if np.isfinite(z) else '#000000'
    f_rows,f_evidence=load_f_rows(f_selection,colour)
    registry={p:{} for p in ('D','E')};svgs={};numerical=[]
    contract=dict(figure='fig1',panels=['D','E'],fish=FISH,trials=list(TRIALS),D_body_size=[750,550],E_body_size=[750,550],E_colourbar_outside_body=True,x_limits_s=[-20,20],E_trace_y_limits=[-E_TRACE_LIMIT,E_TRACE_LIMIT],E_strip_colour_limits=[-1,1],E_band_time_coordinate='F accepted cadence-reconstructed timing',E_band_normalization='exact selected F baseline-bin normalization; independent of black trace',time_coordinate='measured time relative to CS onset',rows=[])
    for pid in ('D','E'):
        viewport_width=840 if pid=='D' else 885
        root=ET.Element('{'+NS+'}svg',{'viewBox':f'0 0 {viewport_width} 550','width':str(viewport_width),'height':'550','data-renderer':'aligned-example-pair'})
        defs=ET.SubElement(root,'{'+NS+'}defs')
        def add(kind,id,role,attrs,text=None,**science):
            attrs={k:str(v) for k,v in attrs.items()};attrs['id']=id
            role=ROLE_MAP.get(role,role)
            style_role=STYLE_SPEC['roles'][role]['style']
            defaults=STYLE_SPEC['styles'][style_role]
            # Styles dispatched from explicit meaning, never colour or artist order.
            if kind in ('line','path') and 'linewidth_pt' in defaults:
                attrs['stroke-width']=str(defaults['linewidth_pt']/PT)
                if 'color' in defaults:attrs['stroke']=defaults['color']
            resolved=dict(defaults)
            if 'stroke-width' in attrs:resolved['linewidth_pt']=float(attrs['stroke-width'])*PT
            if 'stroke-opacity' in attrs:resolved['stroke_opacity']=float(attrs['stroke-opacity'])
            if 'font-size' in attrs:resolved['font_size_pt']=float(attrs['font-size'])*PT;resolved['font_weight']=attrs.get('font-weight','normal')
            if kind=='rect':resolved['cmap']='authenticated V12 513-colour CIELAB palette';resolved['missing_color']='#000000'
            node=ET.SubElement(root,'{'+NS+'}'+kind,attrs);node.text=text
            semantic_role=science.pop('scientific_role',role)
            registry[pid][id]=dict(element_id=id,scientific_role=semantic_role,style_role=style_role,resolved_style=resolved,presentation_overrides='Author-approved Figure 1 legacy axis labels bold and tick pad 2pt; palette from accepted V12',svg_type=kind,artist_type='SVG.'+kind,figure_id='fig1',panel_id=pid.lower(),coordinate_system='svg_user_space',geometry=dict(attrs),required_in_svg=True,role_evidence='Explicit named artist in shared D/E renderer; source columns and layout assigned before rendering',classification_confidence='explicit',review_status='draft; not freeze-approved',**science)
            return node
        def text(id,role,x,y,value,size=7,anchor='end',**extra):
            return add('text',id,role,dict(x=x,y=y,fill='black',**{'font-family':'DejaVu Sans','font-size':size/PT,'text-anchor':anchor},**extra),value)
        def line(id,role,x1,y1,x2,y2,**extra):return add('line',id,role,dict(x1=x1,y1=y1,x2=x2,y2=y2)|{'stroke':'black','stroke-width':.5/PT}|extra)
        for i,(t,stage) in enumerate(zip(TRIALS,STAGES)):
            p=joined[joined['Trial number'].eq(t)].sort_values('FrameID');time=p['Time relative to CS onset (s)'].to_numpy(float);raw=p.Vigor.to_numpy(float)
            eligible=p.eligible.eq(True).to_numpy(bool)
            frame_values,good,params=scale_frames(time,raw,eligible)
            f_row=f_rows[t];z=f_row['values'];med=f_row['medians'];counts=f_row['counts']
            y=30+i*85;h=60;x=260;w=450
            mapping=dict(fish=FISH,trial=t,frame_ids_sha256=sha(p.FrameID.to_numpy(dtype='<i8').tobytes()),time_values_sha256=sha(time.astype('<f8').tobytes()),vigor_input_sha256=sha(raw.astype('<f8').tobytes()),time_coordinate=contract['time_coordinate'],source_table_sha256=EXPECTED[FRAME_PATH],units='degrees' if pid=='D' else 'rad/ms')
            layout=dict(x=x,y=y,width=w,height=h,x_limits_s=[-20,20],row_center=y+h/2)
            if pid=='D':contract['rows'].append(dict(trial=t,block=stage,axis=layout))
            label=dict(block=stage,trial=f'Trial {t}')
            text(f'trial-{t}-block','trial-block-label',185,y+22,stage)
            text(f'trial-{t}-name','trial-number-label',185,y+52,f'Trial {t}')
            ticks=[-200,0,200] if pid=='D' else [-E_TRACE_LIMIT,0,E_TRACE_LIMIT];limits=(-240,240) if pid=='D' else (-E_TRACE_LIMIT,E_TRACE_LIMIT)
            line(f'trial-{t}__spine-left','axis-spine',x,y,x,y+h)
            for k,v in enumerate(ticks):
                ty=y+h*(1-(v-limits[0])/(limits[1]-limits[0]))
                tick=line(f'trial-{t}__ytick_{k}','axis-tick',x,ty,x-2/PT,ty)
                registry[pid][tick.get('id')]['geometry'].update(dimension='y',side='left',kind='major',value=v)
                text(f'trial-{t}__ytick_{k}-label','tick-label',x-4/PT,ty+2.4/PT,str(v).replace('-', '−'))
            clip=ET.SubElement(defs,'{'+NS+'}clipPath',{'id':f'trial-{t}-clip'});ET.SubElement(clip,'{'+NS+'}rect',dict(x=str(x),y=str(y),width=str(w),height=str(h)))
            if pid=='D':
                values=np.rad2deg(p['Tail angle (rad)'].to_numpy(float));px=x+(time+20)/40*w
            else:
                values=frame_values
                if np.isfinite(values).any() and np.nanmax(abs(values))>E_TRACE_LIMIT:
                    raise ValueError('Frame-scaled vigor exceeds the reviewed full-range display')
                px=x+(time+20)/40*w
                zero_y=y+h/2
                line(f'trial-{t}__computational-zero','reference.signal.zero',x,zero_y,x+w,zero_y,
                     stroke='#777777',opacity='0.65',**{'stroke-width':str(.35/PT),'stroke-dasharray':f'{2/PT} {2/PT}'},
                     scientific_role='reference.signal.zero',reference_definition='V-B=0 at B, the median eligible raw baseline-frame vigor over [-15,0) s',
                     data_mapping=dict(fish=FISH,trial=t,baseline_raw_vigor=finite_json(params['baseline_raw']),time_coordinate=contract['time_coordinate']))
            py=y+h*(1-(values-limits[0])/(limits[1]-limits[0]))
            finite=np.isfinite(values);start=finite&~np.r_[False,finite[:-1]]
            commands=['M' if s else 'L' for s in start[finite]]
            path=' '.join(f'{c}{a:.5f},{b:.5f}' for c,a,b in zip(commands,px[finite],py[finite]))
            trace_style={'d':path,'fill':'none','stroke':'black','stroke-width':.6/PT,'clip-path':f'url(#trial-{t}-clip)'}
            if pid=='E':trace_style['stroke-opacity']='0.75'
            add('path',f'trial-{t}-trace','tail-angle-trace' if pid=='D' else 'scaled-vigor-series',trace_style,
                scientific_role='trace.scaled_vigor_series' if pid=='E' else 'trace.tail_angle',
                data_mapping=mapping if pid=='D' else dict(mapping,units='dimensionless baseline-frame spread units',input='eligible raw frames -> B=median raw baseline frames -> V-B -> 0.7/max(-P10,P90) of unbinned baseline-frame raw deviations',frame_times_s=finite_json(time),scaled_frame_values=finite_json(values)),
                normalization=({'baseline_s':[-15,0],'baseline_estimator':'median eligible raw frame vigor','frame_transform':'V-B; no logarithm','scale':'0.7/max(-P10,P90) of baseline-frame raw deviations; linear quantiles','parameters':params,'minimum_coverage':'nonempty eligible baseline and noncollapsed raw-deviation spread; actual frame count/span reported','temporal_binning':False,'smoothing':False}) if pid=='E' else None,
                layout=layout,display_y_limits=list(limits),value_to_y='y + height * (1 - (value - ymin) / (ymax - ymin))',row_label=label,protection={'scientific_data':'source hash + frame/time/value hashes; no resampling'},value_array_sha256=sha(values.astype('<f8').tobytes()))
            for k,event in enumerate(events[events['Trial number'].eq(t)].itertuples(index=False)):
                tx=x+(float(event[2])+20)/40*w;name=event.Event
                attrs={'stroke':'#702e78' if name=='actual US onset' else '#0d8136','stroke-width':.5/PT}
                if name=='CS offset':attrs['stroke-dasharray']=f'{2/PT} {2/PT}'
                node=line(f'trial-{t}-event-{k}',{'CS onset':'stimulus.cs.onset','CS offset':'stimulus.cs.offset','actual US onset':'stimulus.us.actual'}[name],tx,y,tx,y+h,**attrs)
                registry[pid][node.get('id')]['event']=dict(identity=name,time_s=float(event[2]),source_sha256=EXPECTED[EVENT_PATH])
            if pid=='E':
                group=ET.SubElement(root,'{'+NS+'}g',{'id':f'trial-{t}-heatmap'})
                registry[pid][group.get('id')]=dict(element_id=group.get('id'),scientific_role='heatmap.vigor',svg_type='g',data_mapping=dict(fish=FISH,trial=t,F_display_row=t-4,time_coordinate='F accepted cadence-reconstructed time',F_frozen_svg_sha256=f_evidence['frozen_svg_sha256'],F_raster_png_sha256=f_evidence['F_raster_png_sha256'],F_row_rgba_sha256=f_row['rgba_sha256'],F_row_values_sha256=f_row['values_sha256'],units='dimensionless baseline-bin spread units'),normalization=f_selection['scientific_definition'],mappable_ids=[f'trial-{t}-bin-{j}' for j in range(80)])
                for j,value in enumerate(z):
                    node=add('rect',f'trial-{t}-bin-{j}','heatmap-bin',dict(x=x+j*w/80,y=y+h+7,width=w/80,height=9,fill=f_row['colours'][j],stroke='none'),data_mapping=dict(fish=FISH,trial=t,F_display_row=t-4,bin_index=j,start_s=float(edges[j]),stop_s=float(edges[j+1]),eligible_frames=int(counts[j]),median_log_vigor=finite_json(med[j]),scaled_vigor=finite_json(value),units='dimensionless baseline-bin spread units',colour_source='exact F raster cell-centre RGB'),composite_id=f'trial-{t}-heatmap')
                    root.remove(node);group.append(node)
                numeric=dict(trial=t,parameters=params,F_band_parameters=f_row['params'],F_row_rgba_sha256=f_row['rgba_sha256'],F_row_values_sha256=f_row['values_sha256'],scaled_trace_frames=finite_json(values),scaled_trace_display_limits=list(limits),scaled_trace_frames_outside_display=int((np.isfinite(values)&(abs(values)>E_TRACE_LIMIT)).sum()),finite_scaled_trace_frames=int(np.isfinite(values).sum()),saturated_bins=int((abs(z)>1).sum()),missing_eligibility_frames=int(p._merge.eq('left_only').sum()),raw_vs_prior_max_difference=float(np.nanmax(abs(raw[good]-p.raw_vigor.to_numpy(float)[good]))),time_vs_prior_max_difference_s=float(np.nanmax(abs(time-p.time_s.to_numpy(float)))),scaled_bins=finite_json(z),median_log_bins=finite_json(med),eligible_counts=finite_json(counts))
                numerical.append(numeric)
        bottom=470
        line('time__spine-bottom','axis-spine',260,bottom,710,bottom)
        for j,v in enumerate([-20,-10,0,10,20]):
            tx=260+(v+20)/40*450;tick=line(f'time__xtick_{j}','axis-tick',tx,bottom,tx,bottom+2/PT)
            registry[pid][tick.get('id')]['geometry'].update(dimension='x',side='bottom',kind='major',value=v)
            text(f'time__xtick_{j}-label','tick-label',tx,bottom+10/PT,str(v).replace('-','−'),anchor='middle')
        text('time-label','axis-label',485,535,'Time relative to CS onset (s)',size=8,anchor='middle',**{'font-weight':'bold'})
        text('value-label','axis-label',24,245,'Tail angle (°)' if pid=='D' else 'Scaled vigor (baseline units)',size=8,anchor='middle',**{'font-weight':'bold','transform':'rotate(-90 24 245)'})
        if pid=='E':
            group=ET.SubElement(root,'{'+NS+'}g',{'id':'colour-scale'})
            for j in range(256):
                value=1-2*j/255;ET.SubElement(group,'{'+NS+'}rect',dict(x='754',y=str(105+j*300/256),width='12',height=str(300/256+.01),fill=colour(value)))
            registry[pid]['colour-scale']=dict(element_id='colour-scale',scientific_role='colorbar.scale',svg_type='g',mappable_ids=[f'trial-{t}-heatmap' for t in TRIALS],limits=[-1,1],numeric_clipping=False,missing_colour='#000000')
            for j,v in enumerate([1,.7,0,-.7,-1]):
                ty=105+(1-v)/2*300;tick=line(f'colour__ytick_{j}','axis-tick',766,ty,766+2/PT,ty)
                registry[pid][tick.get('id')]['geometry'].update(dimension='y',side='right',kind='major',value=v)
                label=f'{v:.1f}'.replace('-','−')
                text(f'colour__ytick_{j}-label','tick-label',780,ty+2.4/PT,label,anchor='start')
            text('colour-label','axis-label',866,255,'Scaled deviation from baseline',size=8,anchor='middle',**{'font-weight':'bold','transform':'rotate(-90 866 255)'})
        svgs[pid]=ET.tostring(root,encoding='utf-8')
    verification=validate_pair(registry,contract)
    # Verify actual serialized axis rectangles and names against the semantic map.
    for pid,data in svgs.items():
        nodes={n.get('id'):n for n in ET.fromstring(data).iter() if n.get('id')}
        for t in TRIALS:
            record=registry[pid][f'trial-{t}-trace'];rect=next(iter(nodes[f'trial-{t}-clip']))
            for key in ('x','y','width','height'):
                if float(rect.get(key))!=record['layout'][key]:raise ValueError('Exported axis geometry differs from registry')
            if nodes[f'trial-{t}-block'].text!=record['row_label']['block'] or nodes[f'trial-{t}-name'].text!=record['row_label']['trial']:
                raise ValueError('Exported row label differs from registry')
            if pid=='E':
                count=sum(value is not None for value in record['data_mapping']['scaled_frame_values'])
                if len(re.findall('[ML]',nodes[f'trial-{t}-trace'].get('d')))!=count:
                    raise ValueError('Black trace does not preserve one point per scaled frame')
                for j in range(80):
                    value=registry[pid][f'trial-{t}-bin-{j}']['data_mapping']['scaled_vigor']
                    if nodes[f'trial-{t}-bin-{j}'].get('fill')!=f_rows[t]['colours'][j]:raise ValueError('Exported band colour differs from exact F row')
    verification['serialized_geometry_labels_and_bin_colours']=True
    verification['serialized_one_point_per_scaled_frame']=True
    verification['black_trace_baseline_and_scaling_use_unbinned_frames']=True
    report=dict(status='author-requested exploratory draft; accepted freezes unchanged',sources=sources,shared_contract=contract,verification=verification,numerical=numerical,F_band_evidence=f_evidence,scientific_changes={'E':'Black trace scientific draft change: B is median eligible raw baseline frames in [-15,0); raw frame V-B is scaled using P10/P90 of unbinned baseline-frame raw deviations; no logarithm. No trace, baseline or scaling binning/smoothing. Shared y limits [-30,+30] contain all eligible frames. Colour bands copy the corresponding exact selected F rows (values and rendered RGB), with F cadence timing and baseline-bin normalization preserved. Black trace timing remains measured D time.','D':'Selected angle values and centring preserved; layout only'},element_registry=registry,renderer_snapshot_base64=base64.b64encode(Path(__file__).read_bytes()).decode(),registry_scope='D/E explicit semantic artists; not a complete whole-figure freeze registry')
    return svgs,registry,report
