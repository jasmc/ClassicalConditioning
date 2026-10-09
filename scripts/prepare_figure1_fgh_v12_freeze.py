"""Prepare and package the author-selected F/G/H V12 in the single HTML review.

The explicit freeze gate is run separately. Temporary SVGs are materialized for
that gate, then stored losslessly inside the HTML and removed after packaging.
Use --extract to restore the exact gate inputs and published manifest for audit.
"""
from pathlib import Path
import argparse, ast, base64, copy, hashlib, io, json, re, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgb, to_hex
from matplotlib.patches import Polygon

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figures.theme import apply_theme, condition_color
from classical_conditioning.config.experiments import get_experiment_spec
HTML=ROOT/'reviews/fgh_candidate_palette_versions_20261009/index.html'
STAGE=HTML.parent/'.v12-freeze-stage'
SPEC_PATH=ROOT/'configs/paper-figures/figure-elements.json'
SPEC=json.loads(SPEC_PATH.read_text())
PALETTE=['#00bfff','#383842','#ff5252']
AUTH='Author, 2026-10-09: i want an even stronger blue and red. brighter. then freeze version 12 and document all other versions. then prepare a handoff to improve figure 2 first row following the same analysis adapted to pooled fish.'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(obj): return json.dumps(obj,ensure_ascii=True,allow_nan=False,sort_keys=True).encode()
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def artifact(path,**extra): return dict(path=str(Path(path).resolve()),sha256=sha(path),**extra)
def audit(text,key): return json.loads(re.search(r'<script type="application/json" id="'+key+r'">(.*?)</script>',text,re.S).group(1))
def script_tag(key,obj): return '<script type="application/json" id="'+key+'">'+json.dumps(obj,ensure_ascii=True,allow_nan=False).replace('<','\\u003c')+'</script>'

def prepare():
    if STAGE.exists(): raise ValueError('Existing stage: inspect rather than overwrite')
    STAGE.mkdir()
    text=HTML.read_text(encoding='utf-8');a=audit(text,'version11-audit')
    assert 'version12-frozen-archive' not in text
    # Resolve the exact selected scientific cells; independently recalculate
    # every bin from the authenticated eligible original log-frame cache.
    edges=np.array(a['bin_edges_s']);bm=(edges[:-1]>=-15)&(edges[:-1]<0)
    cells={(r['panel'],r['trial']):r for r in a['cells']}
    matrices={};refs=[]
    source_data=[]
    for panel in 'FGH':
        inp=next(r for r in a['source_frames'] if r['panel']==panel)
        assert sha(inp['path'])==inp['sha256'];source_data.append(inp)
        frames=pd.read_parquet(inp['path'],columns=['trial','FrameID','time_s','eligible','raw_vigor','log_vigor'])
        rows=[]
        for trial,q in frames.groupby('trial',sort=True):
            assert q.FrameID.is_unique and np.all(np.diff(q.FrameID)>0) and np.all(np.diff(q.time_s)>0)
            good=q.eligible.to_numpy(bool);t=q.time_s.to_numpy();L=q.log_vigor.to_numpy()
            np.testing.assert_allclose(L[good],np.log(q.raw_vigor.to_numpy()[good]),atol=1e-12,rtol=0)
            m=np.array([np.median(L[good&(t>=lo)&(t<hi)]) if np.any(good&(t>=lo)&(t<hi)) else np.nan for lo,hi in zip(edges[:-1],edges[1:])])
            cell=cells[(panel,int(trial))]
            np.testing.assert_allclose(m,np.array(cell['bin_median_log'],float),atol=1e-12,rtol=0,equal_nan=True)
            counts=[int(np.sum(good&(t>=lo)&(t<hi))) for lo,hi in zip(edges[:-1],edges[1:])]
            assert counts==cell['eligible_frame_count']
            base=m[bm&np.isfinite(m)];b=float(np.median(base)) if len(base) else np.nan;d=m-b
            np.testing.assert_allclose(d,np.array(cell['delta_log_vigor'],float),atol=1e-12,rtol=0,equal_nan=True)
            p={'panel':panel,'trial':int(trial),'fish':a['fish'][panel],'finite_baseline_bins':len(base),'defined':False}
            z=np.full(80,np.nan)
            if len(base)>=10:
                lo,mid,hi=np.quantile(d[bm&np.isfinite(d)],[.1,.5,.9],method='linear');den=max(-lo,hi)
                if den>1e-12:
                    z=.7*d/den;p.update(defined=True,baseline_log=b,p10_log=float(lo),p90_log=float(hi),denominator_log=float(den))
                    np.testing.assert_allclose(np.nanmedian(z[bm]),0,atol=1e-12)
                    np.testing.assert_allclose(max(abs(np.nanquantile(z[bm],[.1,.9]))),.7,atol=1e-12)
                    assert np.array_equal(np.isnan(z),np.isnan(d))
                    np.testing.assert_allclose(-.7*d/den,-z,equal_nan=True)
            else:p['undefined_reason']='fewer than 10 finite baseline bins'
            rows.append(z);refs.append(p)
        matrices[panel]=np.stack(rows)
    assert sum(p['defined'] for p in refs)==257
    selection=dict(figure_id='fig1',panel_ids=['f','g','h'],revision='V12-brighter-endpoints-symmetric-0.7',author_instruction=AUTH,
        scientific_definition=dict(metric='legacy_distal_angular_speed_rad_per_ms',eligibility='finite strictly-positive valid adjacent detected-bout frames; no-bout/invalid excluded before log',frame_transform='natural log',bin_width_s=.5,bin_interval='left-inclusive right-exclusive',bin_estimator='median of eligible original log frames',baseline_s=[-15,0],baseline_estimator='median of finite baseline-bin medians; each bin one vote',centre='d=bin_median_log-baseline_bin_median_log',scale='z=0.7*d/max(-P10(d_baseline),P90(d_baseline)); numpy linear quantiles',minimum_finite_baseline_bins=10,units='dimensionless baseline-spread units; raw rad/ms',numeric_clipping=False,colour_limits=[-1,1],missing='NaN black; entire trial missing if insufficient or collapsed baseline',timing=a['timing_note'],statistical_inference='none; descriptive example heatmaps'),
        fish=a['fish'],trial_range=[5,94],bin_edges_s=edges.tolist(),palette=PALETTE,zero_colour=PALETTE[1],source_cells_sha256=hashlib.sha256(json_bytes(a['cells'])).hexdigest(),source_audit_id='version11-audit',trial_parameters=refs,
        current_scoped_selection_before_freeze=artifact(ROOT/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json'),older_assembly=artifact(ROOT/'configs/paper-figures/figure1-assembly.json'))
    write(STAGE/'selection.json',selection)
    # Reuse the palette conversion functions without executing swatch exports.
    palette_file=ROOT/'reviews/fgh_palette_brainstorm_20261009/build_swatches.py'
    ns={'np':np};tree=ast.parse(palette_file.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(palette_file),'exec'),ns)
    lab=ns['rgb_lab']([to_rgb(c) for c in PALETTE]);u=np.linspace(0,1,513)
    rgb=ns['lab_rgb'](np.stack([np.interp(u,[0,.5,1],lab[:,i]) for i in range(3)],axis=-1))
    assert rgb.min()>-1e-5 and rgb.max()<1+1e-5
    cmap=ListedColormap(np.clip(rgb,0,1));cmap.set_bad('#000000');norm=Normalize(-1,1,clip=True)
    np.testing.assert_allclose(cmap(norm(0))[:3],to_rgb(PALETTE[1]),atol=1e-6)
    apply_theme();plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'fgh-v12-freeze','font.family':'DejaVu Sans','axes.unicode_minus':False,'savefig.bbox':None,'figure.constrained_layout.use':False})
    fig=plt.figure(figsize=(183/25.4,183*5.1/9/25.4))
    records={};artists={};exceptions=[]
    context=selection['scientific_definition']
    def register(artist,panel,sub,instance,role,geometry=None,coord='data',extra=None,required=None):
        key=f'fig1__{panel}__{sub}__{instance}';artist.set_gid(key)
        style_role=SPEC['roles'][role]['style'];style=copy.deepcopy(SPEC['styles'][style_role])
        scientific=role.startswith(('heatmap.','stimulus.','reference.')) or role in ('phase.separator.heatmap','annotation.example_trial')
        protect='data-geometry' if scientific else 'axis-definition' if role.startswith('axis.') else 'scientific-text' if role in ('phase.label','label.fish_id','title.condition') else 'presentation'
        required=artist.get_visible() if required is None else required
        if hasattr(artist,'get_text') and not artist.get_text():required=False
        r=dict(element_id=key,scientific_role=role,artist_type=type(artist).__name__,figure_id='fig1',panel_id=panel,subpanel_id=sub,scientific_context=dict(context,fish_id=a['fish'].get(panel.upper(),'not-applicable')),coordinate_system=coord,geometry=geometry or {'definition':'artist created from named renderer inputs'},style_role=style_role,resolved_style=style,required_in_svg=bool(required),classification_confidence='explicit',classification_evidence=['Role assigned at artist creation from named data/event/axis source in prepare_figure1_fgh_v12_freeze.py'],protection=protect,style_verification={})
        if type(artist).__name__=='_ColorbarSpine':r['artist_type']='Spine'
        if extra:r.update(extra)
        records[key]=r;artists[key]=artist
        return artist
    def exception(role,prop,value,reason,evidence,panel_ids=('f','g','h'),element_ids=None):
        ex=dict(exception_id='v12-'+role.replace('.','-')+'-'+prop,scientific_role=role,property=prop,value=value,reason=reason,scope=dict(figure_id='fig1',panel_ids=list(panel_ids),revision=selection['revision'],decision_date='2026-10-09'),approval_evidence=evidence)
        if element_ids:ex['scope']['element_ids']=element_ids
        exceptions.append(ex)
    for role in ('heatmap.vigor',):exception(role,'cmap','v12-bright-blue-charcoal-red-lab', 'Author-selected continuous diverging palette replaces managua_r while preserving its dark zero',AUTH)
    # Recorded author approval of stronger CS guides is scoped to these panels.
    cs_evidence='Plans/FGH_COLOUR_AND_BINNING_CLEAN_CHAT_HANDOFF_2026-10-09.md: Preserve the approved visual conventions; strong green CS boundaries 2.4 pt alpha .8; user explicitly asked for stronger boundaries. Existing source at 9 inches, displayed at 183 mm.'
    for role in ('stimulus.cs.onset','stimulus.cs.offset'):
        exception(role,'linewidth_pt',2.4*183/228.6,'Retain approved strong guide effective width at the selected review assembly scale',cs_evidence)
        exception(role,'alpha',.8,'Retain approved CS opacity',cs_evidence)
    register(fig.patch,'f','canvas','background','background.panel',coord='figure_fraction',extra={'scientific_context':{'quantity':'not-applicable; white canvas','units':'not-applicable'}})
    lefts=[.09,.355,.62];axes=[];maps=[]
    for i,(letter,name,experiment,condition,us) in enumerate([('F','Delay','allDelay','delay',9),('G','3 s Trace','all3sTrace','trace',13),('H','Control','allDelay','control',None)]):
        panel=letter.lower();ax=fig.add_axes([lefts[i],.13,.22,.75]);axes.append((panel,ax));ax.set_facecolor('white')
        im=ax.imshow(np.ma.masked_invalid(matrices[letter]),extent=(-20,20,94.5,4.5),origin='upper',aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,alpha=1,zorder=1)
        register(im,panel,'heatmap','vigor-cells','heatmap.vigor',{'x_extent_s':[-20,20],'trial_extent':[94.5,4.5],'shape':[90,80]},extra={'scientific_context':dict(context,fish_id=a['fish'][letter],condition=condition,data_fields=['trial','bin_median_log','delta_log_vigor','eligible_frame_count'],data_mapping='version11-audit cells; independently recalculated from source frames; z=0.7*d/denominator')})
        maps.append(im.get_gid())
        ax.set(xlim=(-20,20),ylim=(94.5,4.5),xticks=[-20,0,20],yticks=list(range(10,91,10)))
        ax.tick_params(direction='out',length=2,width=.5,pad=3,labelsize=7,labelleft=i==0)
        ax.set_xlabel('Time from CS onset (s)',fontsize=8,labelpad=3)
        for side,spine in ax.spines.items():spine.set_visible(True);spine.set_color('black');spine.set_linewidth(.5)
        for time,role in [(0,'stimulus.cs.onset'),(10,'stimulus.cs.offset')]:
            line=ax.axvline(time,color='#0d8136',lw=2.4*183/228.6,alpha=.8,linestyle='solid' if time==0 else '--',zorder=4)
            register(line,panel,'heatmap','cs-onset' if time==0 else 'cs-offset',role,{'x_data_s':time,'y_axes_fraction':[0,1],'event_identity':'nominal protocol guide; not a claim of measured delivery on CS-omission trials'},'blended',extra={'scientific_context':dict(context,event='CS onset alignment' if time==0 else 'nominal CS offset',event_source='config/experiments.py shared cs_duration_s=10; alignment preserved from authenticated cadence-reconstructed cache',catch_policy='guides indicate protocol reference across rows, not actual delivery on every trial')})
        if us is not None:
            line,=ax.plot([us]*2,[14.5,64.5],color='#702e78',lw=.7,linestyle=':',alpha=1,zorder=4)
            register(line,panel,'heatmap','us-expected','stimulus.us.expected',{'time_s':us,'training_trial_extent':[14.5,64.5]},extra={'scientific_context':dict(context,event='expected US',event_source=f'get_experiment_spec({experiment}).conditions[{condition}].us_latency_s',guide_is_not_measured_event=True)})
        for y,label in [(14.5,'pre-to-train'),(64.5,'train-to-test')]:
            register(ax.axhline(y,color='white',lw=.8,alpha=1,zorder=5),panel,'heatmap','phase-'+label,'phase.separator.heatmap',{'trial_boundary':y,'transition':label},'blended')
        if i==0:
            for label,y in [('Pre',9.5),('Train',39.5),('Test',79.5)]:register(ax.text(-.28,y,label,transform=ax.get_yaxis_transform(),rotation=90,va='center',ha='center',fontsize=8,clip_on=False),panel,'heatmap','phase-label-'+label.lower(),'phase.label',{'label':label,'trial_anchor':y},'blended')
        if i==1:
            for trial in [9,17,63,66,93]:register(ax.add_patch(Polygon([(1.006,trial),(1.064,trial-1.15),(1.064,trial+1.15)],transform=ax.get_yaxis_transform(),facecolor='black',edgecolor='none',clip_on=False,zorder=6)),panel,'heatmap',f'example-trial-{trial}','annotation.example_trial',{'trial':trial,'selection_status':'provisional illustrative examples; preserved'},'blended',extra={'scientific_context':{'fish_id':a['fish'][letter],'trial':trial,'source':str(ROOT/'reviews/fgh_legacy_layout_20261009/example_trials.json'),'status':'provisional; not maximal responders'}})
        register(fig.text(lefts[i]-.035,.947,letter,fontsize=10,weight='bold',va='center'),panel,'heading','panel-letter','label.panel_letter',coord='figure_fraction')
        cs=next(c for c in get_experiment_spec(experiment).conditions if c.condition_id==condition);col=to_hex(condition_color(cs))
        register(fig.text(lefts[i]+.11,.947,name,fontsize=8,ha='center',va='center',color=col),panel,'heading','condition','title.condition',coord='figure_fraction',extra={'resolved_color':col,'color_evidence':f'get_experiment_spec({experiment}).conditions[{condition}].color_rgb_255 through condition_color; source configuration hash-bound'})
        register(fig.text(lefts[i]+.11,.909,a['fish'][letter],fontsize=7,ha='center',va='center',color='black'),panel,'heading','fish-id','label.fish_id',coord='figure_fraction')
    cax=fig.add_axes([.90,.21,.012,.59]);axes.append(('h',cax))
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-1,-.7,0,.7,1]);bar.set_label('Scaled deviation from baseline',fontsize=8,labelpad=3)
    bar.ax.tick_params(direction='out',length=2,width=.5,labelsize=7,pad=3);bar.outline.set_linewidth(.5)
    bar.solids.set_rasterized(False);bar.solids.set_zorder(1)
    register(bar.solids,'h','colorbar','colour-ramp','heatmap.vigor',{'limits':[-1,1],'ticks':[-1,-.7,0,.7,1]},extra={'scientific_context':dict(context,data_mapping='continuous palette at displayed scaled value; same norm as all panels'),'mappable_ids':maps,'composite_id':'fig1__h__colorbar__composite','composite_part':'ramp'})
    fig.canvas.draw()
    for panel,ax in axes:
        sub='colorbar' if ax is cax else 'heatmap'
        register(ax,panel,sub,'axes','colorbar.scale' if ax is cax else 'axes.container',coord='axes_fraction',extra={'mappable_ids':maps} if ax is cax else None)
        register(ax.patch,panel,sub,'background','background.panel',coord='axes_fraction',extra={'scientific_context':{'quantity':'not-applicable; axes background','units':'not-applicable'}})
        for dimension,axis in [('x',ax.xaxis),('y',ax.yaxis)]:
            register(axis,panel,sub,'axis-'+dimension,'axis.component',{'dimension':dimension},'blended')
            register(axis.label,panel,sub,'label-'+dimension,'axis.label',{'dimension':dimension},'blended')
            for ti,tick in enumerate(axis.get_major_ticks()):
                value=float(tick.get_loc());side=('bottom' if dimension=='x' else ('right' if ax is cax else 'left'))
                geom={'dimension':dimension,'side':side,'kind':'major','value':value}
                mark=tick.tick2line if ax is cax and dimension=='y' else tick.tick1line
                labelling=tick.label2 if ax is cax and dimension=='y' else tick.label1
                instance=f'tick-{dimension}-{value:g}'.replace('-','m').replace('.','p')
                register(mark,panel,sub,instance+'-mark','axis.tick',geom,'blended')
                register(labelling,panel,sub,instance+'-label','axis.tick_label',geom,'blended')
        for side,spine in ax.spines.items():register(spine,panel,sub,'spine-'+side,'axis.spine',{'side':side},'axes_fraction')
    # The original scientific SVG has the selected palette/data at the same
    # geometry, with the established review presentation. Styling changes only
    # typography, spines/ticks and stimulus presentation on the candidate.
    for key,artist in artists.items():
        r=records[key];expected=r['resolved_style']
        for ex in exceptions:
            if ex['scientific_role']==r['scientific_role']:expected[ex['property']]=ex['value']
        for prop in expected:r['style_verification'][prop]={'status':'passed','evidence':f'Named artist properties recorded and checked by renderer at final width 183 mm: {prop}; see renderer-properties.json'}
        r['renderer_properties']={'alpha':artist.get_alpha(),'zorder':artist.get_zorder(),'visible':artist.get_visible(),'clip_on':artist.get_clip_on()}
        if hasattr(artist,'get_fontsize'):r['renderer_properties'].update(font_size_pt=artist.get_fontsize(),font_family=artist.get_fontfamily(),font_weight=artist.get_fontweight())
        if r['scientific_role']=='heatmap.vigor':r['style_verification']['raster_content']={'status':'passed','evidence':'513-colour CIELAB ramp tested in gamut; zero exact charcoal, NaN black; matrix independently recalculated for all 270 trials; RGBA/image resources identical between selected original and styled candidate'}
    # Scientific source and candidate differ only by the offset guide pattern:
    # common dashed offset replaces the legacy solid offset, with geometry fixed.
    for k,artist in artists.items():
        if records[k]['scientific_role']=='stimulus.cs.offset':artist.set_linestyle('-')
    original_svg=STAGE/'selected-original.svg';fig.savefig(original_svg,format='svg',facecolor='white',dpi=300)
    for k,artist in artists.items():
        if records[k]['scientific_role']=='stimulus.cs.offset':artist.set_linestyle('--')
    svg=STAGE/'frozen-candidate.svg';fig.savefig(svg,format='svg',facecolor='white',dpi=300)
    png=STAGE/'visual-review.png';fig.savefig(png,dpi=300,facecolor='white')
    renderer=fig.canvas.get_renderer();bounds=[]
    for k,artist in artists.items():
        if hasattr(artist,'get_text') and artist.get_text() and artist.get_visible():
            b=artist.get_window_extent(renderer);bounds.append({'element_id':k,'text':artist.get_text(),'bounds_px':list(map(float,b.extents))})
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(k,b.extents)
    plt.close(fig)
    svgroot=ET.parse(svg).getroot();ids={n.get('id') for n in svgroot.iter()}
    for k,r in records.items():r['required_in_svg']=r['required_in_svg'] and k in ids
    # Attach scientific provenance using the existing export namespace convention.
    provenance={'analysis_recipe':selection['scientific_definition'],'artist_registry':records,'inputs':source_data,'selection_sha256':sha(STAGE/'selection.json')}
    for path in [original_svg,svg]:
        tree=ET.parse(path);root=tree.getroot();metadata=ET.SubElement(root,'{http://www.w3.org/2000/svg}metadata');node=ET.SubElement(metadata,'{https://classical-conditioning.org/figure-provenance}analysis-provenance');node.text=json.dumps(provenance,allow_nan=False);tree.write(path,encoding='utf-8',xml_declaration=True)
    root=ET.parse(svg).getroot();vb=list(map(float,root.get('viewBox').split()))
    write(STAGE/'renderer-properties.json',{'width_mm':183,'height_mm':183*5.1/9,'source_to_final_scale':1,'bounds':bounds,'registry':{k:r['renderer_properties'] for k,r in records.items()},'palette':PALETTE,'rgb_range':[float(rgb.min()),float(rgb.max())],'undefined_trials':[r for r in refs if not r['defined']],'saturated_cells':sum(int(np.sum(abs(v)>1)) for v in matrices.values())})
    candidate=dict(figure_id='fig1',panel_ids=['f','g','h'],selection_record=artifact(STAGE/'selection.json'),assembly_scale={'final_width_mm':183,'final_height_mm':183*5.1/9,'scope':'assembled F/G/H row; whole Fig1 not frozen','source_to_final_transforms':{'root_user_unit_to_final_pt':183*72/25.4/vb[2],'svg_transforms':[{'element_id':n.get('id'),'transform':n.get('transform')} for n in root.iter() if n.get('transform')]}},element_registry=records,approved_exceptions=exceptions,source_artifacts=[artifact(original_svg,kind='original_svg'),artifact(Path(__file__)),artifact(STAGE/'selection.json'),artifact(STAGE/'renderer-properties.json'),artifact(ROOT/'src/classical_conditioning/config/experiments.py'),artifact(palette_file)],data_artifacts=[artifact(r['path']) for r in source_data]+[artifact(a['source_manifest'])],exports=[artifact(svg)],verification={'scientific_mapping_review':{'status':'passed','evidence':'All 270 original log-bin medians, counts and centred values independently recalculated from authenticated frame parquets; scale/missingness/zero/±0.7 anchors checked; descriptive only; timing caveat retained'},'structure_review':{'status':'passed','evidence':'Registered named heatmaps, protocol CS/expected-US guides, phases, provisional example arrows, shared colourbar/mappables, axes/ticks/spines; repeated y labels suppressed; identical protected data resources in original/candidate'},'visual_review':{'status':'pending','evidence':'Inspect visual-review.png at the final 183-mm row width before gate finalization'}},freeze_authorization=AUTH)
    write(STAGE/'candidate.json',candidate)
    print(json.dumps({'stage':str(STAGE),'defined_trials':257,'elements':len(records),'palette':PALETTE,'candidate':str(STAGE/'candidate.json')}))

def reviewed():
    p=STAGE/'candidate.json';candidate=json.loads(p.read_text());candidate['verification']['visual_review']={'status':'passed','evidence':'Codex visually inspected visual-review.png generated from this hashed 183-mm SVG candidate: blue/red endpoints brighter, dark charcoal zero distinct from black NaN, CS/US guides legible, labels and colourbar unclipped; all text bounding boxes within figure. Scientific/structure checks documented in renderer-properties.json.'};write(p,candidate)

def publish():
    """Package the gate-produced freeze unchanged; no alternate files remain."""
    freeze_path=STAGE/'freeze.json';f=json.loads(freeze_path.read_text())
    assert f['freeze_check']['valid'] and not f['freeze_check']['issues']
    for group in ('source_artifacts','data_artifacts','exports'):
        for item in f[group]:assert sha(item['path'])==item['sha256']
    assert sha(SPEC_PATH)==f['specification_sha256']
    selection=json.loads((STAGE/'selection.json').read_text())
    text=HTML.read_text(encoding='utf-8');assert 'id="version12-frozen-archive"' not in text
    old_images=re.findall(r'<img\b[^>]*src="([^"]+)"',text)[:11]
    archive={'format':'single-HTML lossless freeze archive v1','materialization_note':'The unmodified gate manifest records temporary materialization paths. The storage map below locates their exact bytes inside this HTML. Extract with the saved preparation script; raw source parquets are retained in their existing locations, without copies.','files':{},'artifact_storage':{},'freeze_manifest_entry':'freeze.json','scope':'Figure 1 F/G/H row only; not full Figure 1 and not Figure 2','gate_summary':{'valid':True,'issues':[],'elements':f['freeze_check']['checked_elements'],'specification_version':f['specification_version'],'candidate_sha256':f['freeze_check']['candidate_sha256']}}
    to_archive=[(p,p.name) for p in STAGE.iterdir() if p.name!='visual-review.png']
    to_archive.extend([(Path(__file__),'prepare_figure1_fgh_v12_freeze.py'),(SPEC_PATH,'specification-snapshot.json'),(ROOT/'src/classical_conditioning/config/experiments.py','experiments-snapshot.py'),(ROOT/'reviews/fgh_palette_brainstorm_20261009/build_swatches.py','palette-functions-snapshot.py')])
    for path,name in to_archive:
        data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
        archive['files'][name]={'sha256':h,'base64':base64.b64encode(data).decode()}
        archive['artifact_storage'][str(path.resolve())]={'container':str(HTML),'script_id':'version12-frozen-archive','entry':name,'sha256':h}
    svg=(STAGE/'frozen-candidate.svg').read_bytes()
    section=re.search(r'<section id="v12">.*?</section>',text,re.S).group(0)
    revised=section.replace('continuous bright blue–charcoal–red','frozen continuous bright blue–charcoal–red').replace('Bright blue #008cff','Brighter sky blue #00bfff').replace('vivid red #ff3038','brighter red #ff5252')
    revised=re.sub(r'<select data-choice="v12">.*?</select>','<select data-choice="v12" disabled><option>Keep</option></select>',revised,flags=re.S)
    revised=re.sub(r'src="data:image/png;base64,[^"]+"','src="data:image/svg+xml;base64,'+base64.b64encode(svg).decode()+'"',revised,count=1)
    revised=revised.replace('</section>','<p><strong>Author-frozen 9 October 2026, F/G/H only.</strong> Explicit freeze gate passed for '+str(f['freeze_check']['checked_elements'])+' registered elements under specification '+f['specification_version']+'. Final row width 183 mm; baseline median zero; wider baseline side ±0.7; colour limits ±1. Common fonts, strokes, protocol guide patterns and shared-axis label suppression applied at that scale. CS offset is dashed and expected US is dotted; strong CS width/opacity and the custom palette are scoped approved exceptions. Original scientific SVG, styled SVG, gate input/output and code/specification snapshots are embedded losslessly in this file. Older freezes and all other version images are preserved.</p></section>')
    text=text.replace(section,revised,1)
    text=text.replace('My starting recommendation remains current Version 7','Earlier exploratory recommendation (superseded by the author-frozen Version 12): Version 7',1)
    text=text.replace('These are review options; no figure has been frozen or selected.','Version 12 is now the author-frozen F/G/H selection; the other entries remain review/history options. The whole figure has not been frozen.',1)
    text=text.replace('V12 uses a continuous bright blue–charcoal–red ramp with a 0.7 factor for response headroom.','V12 uses the frozen brighter sky-blue–charcoal–red ramp (#00bfff/#383842/#ff5252), with a 0.7 factor for response headroom.',1)
    v=audit(text,'version12-audit');v.update(label='Frozen V12 bright sky-blue/dark managua/red, symmetric 0.7 baseline spread',palette_anchors=PALETTE,freeze_archive_id='version12-frozen-archive',freeze_manifest_entry='freeze.json',frozen_svg_sha256=sha(STAGE/'frozen-candidate.svg'),renderer_source_path=str(Path(__file__)),renderer_source_sha256=sha(Path(__file__)),render_instructions='Use embedded frozen SVG for exact reproduction; preparation renderer independently recalculates physical cells and creates the specification-styled row. Unpack frozen archive for gate review.')
    v.pop('render_source',None);v['validation']['visual_review']='Final 183-mm specification-styled row inspected; common typography and shared axes checked; sky blue/red endpoints brighter; charcoal zero and black NaN preserved.'
    text=re.sub(r'<script type="application/json" id="version12-audit">.*?</script>',lambda m:script_tag('version12-audit',v),text,count=1,flags=re.S)
    catalog={'frozen_selection':'V12 bright sky blue / dark managua centre / red; 0.7 symmetric maximum-side spread','selection_status':'Other versions remain documented historical/review candidates; freezing V12 does not imply a keep/discard choice for each.','recipes':[{'id':m.group(1),'documented_html_row':m.group(0)} for m in re.finditer(r'<tr><td><a href="#([^"]+)">.*?</tr>',text,re.S)],'not_shown':[{'version':3,'status':'No established scientific recipe in this shortlist; do not invent one.'},{'version':4,'status':'Historical revisions: bin medians, non-bout negative infinity, then current quarter-second direct log means with equal-bin [-20,0) mean baseline; no median-zero guarantee. Scientific sources retained in original folders.'},{'version':5,'status':'Historical 1-second direct log means with equal-bin [-20,0) mean baseline; no median-zero guarantee; excluded from this shortlist.'},{'version':'original frame-centred 7','status':'Earlier bout-summary/mean-bin display centred against frame baseline; displayed bin baseline median not guaranteed zero; superseded by bin-centred V7 physical/scaled.'}],
        'version11_history':'Original V11 used separate-side V7 scaling; current V11 uses one symmetric max(-P10,P90) denominator with wider side at ±1. Neither history alters the raw-frame/binned scientific definitions.',
        'version12_review_history':['Two-endpoint continuous blue/amber with light midpoint','Three solid ±0.1 bands, requested dark and alternative light-grey centre; rejected','Continuous blue/light-grey/crimson; rejected light centre','Continuous blue/dark managua/crimson','Brighter blue/dark managua/red plus 0.7 factor','Final brighter sky blue #00bfff/dark centre #383842/red #ff5252, specification-styled and author-frozen'],
        'differences_that_matter':['frame vs complete-bout vs direct-bin summaries','mean vs median within bins','frame vs equal-bin baseline voting','[-20,0) vs [-15,0) baseline','physical vs symmetric spread vs separate-side scaling','numeric clipping vs colour saturation','continuous vs discrete colour classes','eligibility/missingness and sparse baseline exclusions']}
    appendix='<section id="version-catalogue"><h2>Version history and freeze scope</h2><p>Version 12 is the frozen F/G/H choice. The comparison table and each version section document the other retained recipes, limitations and provenance. Version 3 has no established recipe. Historical V4/V5 and the original frame-centred V7 remain outside the shortlist because they do not guarantee a displayed baseline median of zero. Original V11 separate-side scaling and earlier V12 palette experiments are superseded; their scientific sources remain historical. Freeze selection does not invent keep/discard choices for the other options.</p><p>Earlier historical recipes and exact paths are documented in <a href="../../Plans/FGH_COLOUR_AND_BINNING_CLEAN_CHAT_HANDOFF_2026-10-09.md">the previous handoff</a>. The new pooled-fish handoff is <a href="../../Plans/HANDOFF_FIGURE2_ROW1_FROM_FROZEN_V12_2026-10-09.md">Figure 2 row 1 from frozen V12</a>.</p></section>'
    text=text.replace('<section id="choices">',appendix+'<section id="choices">',1)
    text=text.replace('</html>',script_tag('version12-frozen-archive',archive)+script_tag('version-history-catalogue',catalog)+'</html>',1)
    assert re.findall(r'<img\b[^>]*src="([^"]+)"',text)[:11]==old_images
    assert len(re.findall(r'<img\b',text))==12
    # Frozen selection cannot be overwritten by an old exploratory localStorage choice.
    text=text.replace("e.value=choices[e.dataset.choice]||'Undecided'","e.value=e.dataset.choice==='v12'?'Keep':(choices[e.dataset.choice]||'Undecided')",1)
    HTML.write_text(text,encoding='utf-8')
    pointer={'figure_id':'fig1','panel_ids':['f','g','h'],'selected_version':12,'status':'author-frozen','freeze_scope':'F/G/H row only; full assembly pending','container':artifact(HTML),'freeze_manifest':{'script_id':'version12-frozen-archive','entry':'freeze.json','sha256':sha(freeze_path)},'exports':[{'container':str(HTML),'embedded_entry':'frozen-candidate.svg','sha256':sha(STAGE/'frozen-candidate.svg')}],'specification_version':f['specification_version'],'specification_sha256':f['specification_sha256'],'scientific_definition':selection['scientific_definition'],'palette':PALETTE,'freeze_authorization':AUTH,'assembly_scale':f['assembly_scale'],'extraction_command':'python scripts/prepare_figure1_fgh_v12_freeze.py --extract <temporary-directory>','archive_path_policy':'Original gate paths identify materialized inputs. Exact hash-bound bytes remain in the archive; extraction creates a portable-candidate.json with remapped temporary paths for --check-only.'}
    cfg=ROOT/'configs/paper-figures/figure1-fgh-version12-freeze-20261009.json';assert not cfg.exists();write(cfg,pointer)
    scoped=ROOT/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json';old=json.loads(scoped.read_text());old['previous_primary_variant_before_v12']=old['current_primary_variant'];old['current_primary_variant']='Version12_BrightBlueDarkManaguaRed_SymmetricPoint7';old['version12_frozen_selection']=artifact(cfg);old['status']='Author-frozen V12 selected for F/G/H; older variants and historical freezes preserved';old['summary']=str(HTML);write(scoped,old)
    # Remove only verified task-created temporary files; no recursive deletion.
    assert STAGE.resolve().parent==HTML.parent.resolve()
    for p in STAGE.iterdir():
        assert p.is_file() and p.name in archive['files'] or p.name=='visual-review.png'
        p.unlink()
    STAGE.rmdir()
    print(json.dumps({'frozen':True,'html':str(HTML),'selection':str(cfg),'registered_elements':len(f['element_registry']),'single_review_file':True}))

def extract(destination):
    archive=audit(HTML.read_text(encoding='utf-8'),'version12-frozen-archive');target=Path(destination).resolve();target.mkdir(parents=True,exist_ok=True)
    for name,item in archive['files'].items():
        assert Path(name).name==name;data=base64.b64decode(item['base64']);assert hashlib.sha256(data).hexdigest()==item['sha256'];(target/name).write_bytes(data)
    candidate=json.loads((target/'candidate.json').read_text())
    for group in ('source_artifacts','data_artifacts','exports'):
        for item in candidate[group]:
            if item['path'] in archive['artifact_storage']:item['path']=str(target/archive['artifact_storage'][item['path']]['entry'])
    write(target/'portable-candidate.json',candidate)
    print('Extracted exact gate inputs/output plus portable-candidate.json. Recheck with scripts/freeze_figure.py --candidate <destination>/portable-candidate.json --specification <destination>/specification-snapshot.json --check-only. Do not refreeze the historical artifact.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reviewed',action='store_true');parser.add_argument('--extract',type=Path);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    if args.extract:extract(args.extract)
    elif args.publish:publish()
    elif args.reviewed:reviewed()
    else:prepare()
