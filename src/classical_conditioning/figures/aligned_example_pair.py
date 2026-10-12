"""Draft D/E pair: one measured frame table and an explicit semantic layout contract.

This author-requested draft recomputes E; it does not replace any accepted freeze.
The existing E bout mask is joined by trial + FrameID, never by row position.
"""
from __future__ import annotations
import ast, base64, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, to_hex
from classical_conditioning.external_artifacts import resolve_artifact

ROLE_MAP={'trial-block-label':'phase.label','trial-number-label':'annotation.note','axis-spine':'axis.spine','axis-tick':'axis.tick','tick-label':'axis.tick_label','tail-angle-trace':'trace.tail_angle','raw-vigor-trace':'trace.raw_vigor','heatmap-bin':'heatmap.vigor','axis-label':'axis.label'}
NS='http://www.w3.org/2000/svg'
ROOT=Path(__file__).resolve().parents[3]
TRIALS=(9,17,63,66,93)
STAGES=('Pre-Train','Early Train','Late Train','Early Test','Late Test')
FISH='20221115_07'
PT=183/25.4*72/1800
STYLE_PATH=ROOT/'configs/paper-figures/figure-elements.json'
STYLE_SPEC=json.loads(STYLE_PATH.read_text())
FRAME_ROOT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/frozen/2026-10-05-ABCD')
FRAME_PATH=FRAME_ROOT/'Fig1_PanelD_TailAngle_pre15_frames.parquet'
EVENT_PATH=FRAME_ROOT/'Fig1_PanelD_events.parquet'
ELIGIBILITY_PATH=resolve_artifact('reviews/fgh_full_bouts_baseline_samples_20261009/PanelF_complete_bout_sample_data.parquet')
SELECTION_PATH=ROOT/'records/frozen-analyses/figure-01/panel-e/2026-10-09-heatmap-rows/manifests/selection.json'
PALETTE_SELECTION_PATH=ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-09-version-12/manifests/selection.json'
PALETTE_PATH=ROOT/'records/frozen-analyses/figure-01/panels-f-g-h/2026-10-09-version-12/code/palette_functions_snapshot.py'
EXPECTED={FRAME_PATH:'004861483e76a116544930769f5eaff85db1231e1d00d26c4e9d919d789efe32',EVENT_PATH:'c33e5e11889a5f8982ecee277585549363afa0cc870aab0ceb93eb8b92199007',ELIGIBILITY_PATH:'0ea6642119308fec55a6cb00c370b003bc6b3cc6c9f1a043dca1d3cc90bee030'}

def dependencies():return [Path(__file__),STYLE_PATH,FRAME_PATH,EVENT_PATH,ELIGIBILITY_PATH,SELECTION_PATH,PALETTE_PATH,PALETTE_SELECTION_PATH,ROOT/'records/frozen-analyses/index.json']
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

def validate_pair(registries,contract):
    """Validate declared identity/data relationships and numeric layout, not colour guesses."""
    for t in TRIALS:
        d=registries['D'][f'trial-{t}-trace'];e=registries['E'][f'trial-{t}-trace'];h=registries['E'][f'trial-{t}-heatmap']
        for key in ('fish','trial','frame_ids_sha256','time_values_sha256','time_coordinate'):
            if d['data_mapping'][key]!=e['data_mapping'][key] or e['data_mapping'][key]!=h['data_mapping'][key]:
                raise ValueError(f'D/E shared mapping mismatch: trial {t}, {key}')
        if e['data_mapping']['vigor_input_sha256']!=h['data_mapping']['vigor_input_sha256']:
            raise ValueError('Raw/coloured vigor do not share a numeric input')
        if d['layout']!=e['layout']:raise ValueError(f'D/E axis rectangles differ: trial {t}')
        if d['row_label']!=e['row_label']:raise ValueError(f'D/E names differ: trial {t}')
    if contract['D_body_size']!=contract['E_body_size']:raise ValueError('D/E body sizes differ')
    return dict(same_fish=True,same_trials=True,same_frame_ids=True,same_time_values=True,
                raw_and_scaled_vigor_share_input=True,equal_axis_rectangles=True,equal_body_size_excluding_colourbar=True,
                row_labels_and_centres_match=True)

def build_pair():
    index=json.loads((ROOT/'records/frozen-analyses/index.json').read_text())
    known={str(ROOT/i['path']):i['sha256'] for i in index['files']}
    sources=[]
    for p in dependencies():
        data=p.read_bytes();digest=sha(data)
        if p in (SELECTION_PATH,PALETTE_PATH,PALETTE_SELECTION_PATH) and digest!=known[str(p)]:raise ValueError('Frozen definition changed: '+str(p))
        if p in EXPECTED and digest!=EXPECTED[p]:raise ValueError('D/E scientific input changed: '+str(p))
        sources.append(dict(path=str(p),sha256=digest))
    selection=json.loads(SELECTION_PATH.read_text());assert selection['fish']==FISH and tuple(selection['trials'])==TRIALS
    edges=np.asarray(selection['bin_edges_s'],float)
    np.testing.assert_array_equal(edges,np.linspace(-20,20,81))
    d=pd.read_parquet(FRAME_PATH);old=pd.read_parquet(ELIGIBILITY_PATH)
    old=old[old.trial.isin(TRIALS)]
    joined=d.merge(old[['trial','FrameID','eligible','raw_vigor','time_s']],left_on=['Trial number','FrameID'],right_on=['trial','FrameID'],how='left',validate='one_to_one',indicator=True)
    events=pd.read_parquet(EVENT_PATH)
    tree=ast.parse(PALETTE_PATH.read_text());ns={'np':np}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'authenticated_palette_functions','exec'),ns)
    lab=ns['rgb_lab']([to_rgb(c) for c in json.loads(PALETTE_SELECTION_PATH.read_text())['palette']]);u=np.linspace(0,1,513)
    rgb=np.clip(ns['lab_rgb'](np.stack([np.interp(u,[0,.5,1],lab[:,i]) for i in range(3)],axis=-1)),0,1)
    def colour(z):return to_hex(rgb[min(512,int(np.clip((z+1)/2,0,1)*513))]) if np.isfinite(z) else '#000000'
    registry={p:{} for p in ('D','E')};svgs={};numerical=[]
    contract=dict(figure='fig1',panels=['D','E'],fish=FISH,trials=list(TRIALS),D_body_size=[750,550],E_body_size=[750,550],E_colourbar_outside_body=True,x_limits_s=[-20,20],time_coordinate='measured time relative to CS onset',rows=[])
    for pid in ('D','E'):
        root=ET.Element('{'+NS+'}svg',{'viewBox':'0 0 840 550','width':'840','height':'550','data-renderer':'aligned-example-pair'})
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
            if 'font-size' in attrs:resolved['font_size_pt']=float(attrs['font-size'])*PT;resolved['font_weight']=attrs.get('font-weight','normal')
            if kind=='rect':resolved['cmap']='authenticated V12 513-colour CIELAB palette';resolved['missing_color']='#000000'
            node=ET.SubElement(root,'{'+NS+'}'+kind,attrs);node.text=text
            registry[pid][id]=dict(element_id=id,scientific_role=role,style_role=style_role,resolved_style=resolved,presentation_overrides='Author-approved Figure 1 legacy axis labels bold and tick pad 2pt; palette from accepted V12',svg_type=kind,artist_type='SVG.'+kind,figure_id='fig1',panel_id=pid.lower(),coordinate_system='svg_user_space',geometry=dict(attrs),required_in_svg=True,role_evidence='Explicit named artist in shared D/E renderer; source columns and layout assigned before rendering',classification_confidence='explicit',review_status='draft; not freeze-approved',**science)
            return node
        def text(id,role,x,y,value,size=7,anchor='end',**extra):
            return add('text',id,role,dict(x=x,y=y,fill='black',**{'font-family':'DejaVu Sans','font-size':size/PT,'text-anchor':anchor},**extra),value)
        def line(id,role,x1,y1,x2,y2,**extra):return add('line',id,role,dict(x1=x1,y1=y1,x2=x2,y2=y2)|{'stroke':'black','stroke-width':.5/PT}|extra)
        for i,(t,stage) in enumerate(zip(TRIALS,STAGES)):
            p=joined[joined['Trial number'].eq(t)].sort_values('FrameID');time=p['Time relative to CS onset (s)'].to_numpy(float);raw=p.Vigor.to_numpy(float)
            eligible=p.eligible.eq(True).to_numpy(bool)
            z,med,counts,good,params=scale_bins(time,raw,eligible,edges)
            y=30+i*85;h=60;x=260;w=450
            mapping=dict(fish=FISH,trial=t,frame_ids_sha256=sha(p.FrameID.to_numpy(dtype='<i8').tobytes()),time_values_sha256=sha(time.astype('<f8').tobytes()),vigor_input_sha256=sha(raw.astype('<f8').tobytes()),time_coordinate=contract['time_coordinate'],source_table_sha256=EXPECTED[FRAME_PATH],units='degrees' if pid=='D' else 'rad/ms')
            layout=dict(x=x,y=y,width=w,height=h,x_limits_s=[-20,20],row_center=y+h/2)
            if pid=='D':contract['rows'].append(dict(trial=t,block=stage,axis=layout))
            label=dict(block=stage,trial=f'Trial {t}')
            text(f'trial-{t}-block','trial-block-label',185,y+22,stage)
            text(f'trial-{t}-name','trial-number-label',185,y+52,f'Trial {t}')
            ticks=[-200,0,200] if pid=='D' else [0,.5,1];limits=(-240,240) if pid=='D' else (0,1)
            line(f'trial-{t}__spine-left','axis-spine',x,y,x,y+h)
            for k,v in enumerate(ticks):
                ty=y+h*(1-(v-limits[0])/(limits[1]-limits[0]))
                tick=line(f'trial-{t}__ytick_{k}','axis-tick',x,ty,x-2/PT,ty)
                registry[pid][tick.get('id')]['geometry'].update(dimension='y',side='left',kind='major',value=v)
                text(f'trial-{t}__ytick_{k}-label','tick-label',x-4/PT,ty+2.4/PT,str(v).replace('-', '−'))
            clip=ET.SubElement(defs,'{'+NS+'}clipPath',{'id':f'trial-{t}-clip'});ET.SubElement(clip,'{'+NS+'}rect',dict(x=str(x),y=str(y),width=str(w),height=str(h)))
            values=np.rad2deg(p['Tail angle (rad)'].to_numpy(float)) if pid=='D' else np.where(good,raw,np.nan)
            px=x+(time+20)/40*w;py=y+h*(1-(values-limits[0])/(limits[1]-limits[0]))
            finite=np.isfinite(values);start=finite&~np.r_[False,finite[:-1]]
            commands=['M' if s else 'L' for s in start[finite]]
            path=' '.join(f'{c}{a:.5f},{b:.5f}' for c,a,b in zip(commands,px[finite],py[finite]))
            add('path',f'trial-{t}-trace','tail-angle-trace' if pid=='D' else 'raw-vigor-trace',{'d':path,'fill':'none','stroke':'black','stroke-width':.6/PT,'clip-path':f'url(#trial-{t}-clip)'},data_mapping=mapping,layout=layout,row_label=label,protection={'scientific_data':'source hash + frame/time/value hashes; no resampling'},value_array_sha256=sha(values.astype('<f8').tobytes()))
            for k,event in enumerate(events[events['Trial number'].eq(t)].itertuples(index=False)):
                tx=x+(float(event[2])+20)/40*w;name=event.Event
                attrs={'stroke':'#702e78' if name=='actual US onset' else '#0d8136','stroke-width':.5/PT}
                if name=='CS offset':attrs['stroke-dasharray']=f'{2/PT} {2/PT}'
                node=line(f'trial-{t}-event-{k}',{'CS onset':'stimulus.cs.onset','CS offset':'stimulus.cs.offset','actual US onset':'stimulus.us.actual'}[name],tx,y,tx,y+h,**attrs)
                registry[pid][node.get('id')]['event']=dict(identity=name,time_s=float(event[2]),source_sha256=EXPECTED[EVENT_PATH])
            if pid=='E':
                group=ET.SubElement(root,'{'+NS+'}g',{'id':f'trial-{t}-heatmap'})
                registry[pid][group.get('id')]=dict(element_id=group.get('id'),scientific_role='heatmap.vigor',svg_type='g',data_mapping=dict(mapping,units='dimensionless baseline-spread units'),normalization=selection['scientific_definition']|{'timing':contract['time_coordinate'],'raw_source':'D stored Vigor; inherited E bout mask joined by trial/FrameID'},mappable_ids=[f'trial-{t}-bin-{j}' for j in range(80)])
                for j,value in enumerate(z):
                    node=add('rect',f'trial-{t}-bin-{j}','heatmap-bin',dict(x=x+j*w/80,y=y+h+7,width=w/80,height=9,fill=colour(value),stroke='none'),data_mapping=dict(fish=FISH,trial=t,bin_index=j,start_s=float(edges[j]),stop_s=float(edges[j+1]),eligible_frames=int(counts[j]),median_log_vigor=finite_json(med[j]),scaled_vigor=finite_json(value),units='dimensionless baseline-spread units'),composite_id=f'trial-{t}-heatmap')
                    root.remove(node);group.append(node)
                numeric=dict(trial=t,parameters=params,raw_cap_exceedances=int((raw[good]>1).sum()),saturated_bins=int((abs(z)>1).sum()),missing_eligibility_frames=int(p._merge.eq('left_only').sum()),raw_vs_prior_max_difference=float(np.nanmax(abs(raw[good]-p.raw_vigor.to_numpy(float)[good]))),time_vs_prior_max_difference_s=float(np.nanmax(abs(time-p.time_s.to_numpy(float)))),scaled_bins=finite_json(z),median_log_bins=finite_json(med),eligible_counts=finite_json(counts))
                numerical.append(numeric)
        bottom=470
        line('time__spine-bottom','axis-spine',260,bottom,710,bottom)
        for j,v in enumerate([-20,-10,0,10,20]):
            tx=260+(v+20)/40*450;tick=line(f'time__xtick_{j}','axis-tick',tx,bottom,tx,bottom+2/PT)
            registry[pid][tick.get('id')]['geometry'].update(dimension='x',side='bottom',kind='major',value=v)
            text(f'time__xtick_{j}-label','tick-label',tx,bottom+10/PT,str(v).replace('-','−'),anchor='middle')
        text('time-label','axis-label',485,535,'Time relative to CS onset (s)',size=8,anchor='middle',**{'font-weight':'bold'})
        text('value-label','axis-label',24,245,'Tail angle (°)' if pid=='D' else 'Raw vigor (rad/ms)',size=8,anchor='middle',**{'font-weight':'bold','transform':'rotate(-90 24 245)'})
        if pid=='E':
            group=ET.SubElement(root,'{'+NS+'}g',{'id':'colour-scale'})
            for j in range(256):
                value=1-2*j/255;ET.SubElement(group,'{'+NS+'}rect',dict(x='754',y=str(190+j*130/256),width='12',height=str(130/256+.01),fill=colour(value)))
            registry[pid]['colour-scale']=dict(element_id='colour-scale',scientific_role='colorbar.scale',svg_type='g',mappable_ids=[f'trial-{t}-heatmap' for t in TRIALS],limits=[-1,1],numeric_clipping=False,missing_colour='#000000')
            for j,v in enumerate([1,0,-1]):
                ty=190+(1-v)/2*130;tick=line(f'colour__ytick_{j}','axis-tick',766,ty,766+2/PT,ty)
                registry[pid][tick.get('id')]['geometry'].update(dimension='y',side='right',kind='major',value=v)
                text(f'colour__ytick_{j}-label','tick-label',780,ty+2.4/PT,('+' if v>0 else '')+str(v).replace('-','−'),anchor='start')
            text('colour-label','axis-label',832,255,'Scaled vigor',size=8,anchor='middle',**{'font-weight':'bold','transform':'rotate(-90 832 255)'})
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
                for j in range(80):
                    value=registry[pid][f'trial-{t}-bin-{j}']['data_mapping']['scaled_vigor']
                    if nodes[f'trial-{t}-bin-{j}'].get('fill')!=colour(np.nan if value is None else value):raise ValueError('Exported bin colour differs from stored scaled value')
    verification['serialized_geometry_labels_and_bin_colours']=True
    report=dict(status='author-requested exploratory draft; accepted freezes unchanged',sources=sources,shared_contract=contract,verification=verification,numerical=numerical,scientific_changes={'E':'D stored Vigor and measured time replace previous E raw values and cadence timing; retained E bout mask; same log/bin/baseline/scale recipe; recomputed colours','D':'selected angle values and centring preserved; layout only'},element_registry=registry,renderer_snapshot_base64=base64.b64encode(Path(__file__).read_bytes()).decode(),registry_scope='D/E explicit semantic artists; not a complete whole-figure freeze registry')
    return svgs,registry,report
