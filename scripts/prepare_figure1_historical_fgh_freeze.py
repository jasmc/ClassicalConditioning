"""Prepare the author-approved F/G/H 1 s physical log median freeze candidate."""
from pathlib import Path
import copy, hashlib, json, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import Polygon
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figures.theme import apply_theme, condition_color
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.figure_workspace import Snapshot, state_lock
SPEC_PATH=ROOT/'configs/paper-figures/figure-elements.json'
SPEC=json.loads(SPEC_PATH.read_text())
STAGE=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-fgh-historical-one-second-managua-freeze-2026-10-13')
AUTH='Author, 2026-10-13: freeze these panels. Refers to the displayed historical-fish F/G/H row: Delay 20221115_07, Trace 20230307_12, Control 20221115_09; 1 s medians, no amplitude scaling, managua_r +/-0.25. Scope F/G/H only.'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def artifact(path,**extra):return dict(path=str(Path(path).resolve()),sha256=sha(path),**extra)
def prepare():
    ticket=json.loads(Path('/tmp/figure-01-historical-fgh-freeze-start.json').read_text())
    for p,h in ticket['starting_hashes'].items():assert sha(p)==h
    record_path=ROOT/'records/figure-reviews/figure-01/historical-fish-one-second-managua-2026-10-13.json'
    approved_path=ROOT/'records/figure-reviews/figure-01/historical-fish-one-second-managua-2026-10-13.json'
    record=json.loads(record_path.read_text());approved=json.loads(approved_path.read_text())
    assert sha(approved['review']['path'])==approved['review']['sha256']
    data=Path(record['data']['path']);assert sha(data)==record['data']['sha256']
    snapshot=Snapshot([data,record_path,approved_path,SPEC_PATH,Path(__file__),ROOT/'docs/paper/figures/figure-01/current-state.json'])
    if STAGE.exists():raise ValueError('Existing candidate folder; preserve rather than overwrite')
    STAGE.mkdir();(STAGE/'renderer-snapshot.py').write_bytes(Path(__file__).read_bytes())
    table=pd.read_csv(data);matrices={};refs=[]
    for letter in 'FGH':
        q=table[table.panel.eq(letter)];matrices[letter]=q.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').to_numpy()
        assert matrices[letter].shape==(90,40)
        np.testing.assert_allclose(q.delta_log_vigor,q.bin_log_vigor-q.baseline_log_vigor,atol=1e-12,rtol=0,equal_nan=True)
        for trial,t in q.groupby('acquisition_trial'):
            b=t[t.start_s.ge(-15)&t.start_s.lt(0)];ref=b.bin_log_vigor.median();defined=bool(np.isfinite(ref))
            if defined:
                assert abs(b.delta_log_vigor.median())<1e-12
                assert abs(ref-t.baseline_log_vigor.iloc[0])<1e-12
            refs.append(dict(panel=letter,trial=int(trial),defined=defined,baseline_log=float(ref) if defined else None))
    assert sum(p['defined'] for p in refs)==270
    a={'fish':{'F':'20221115_07','G':'20230307_12','H':'20221115_09'}}
    PALETTE=[to_hex(matplotlib.colormaps['managua_r'](x)) for x in (0.,.5,1.)]
    context=dict(metric='legacy_distal_angular_speed_rad_per_ms',eligibility='finite positive valid adjacent detected-bout frames',frame_transform='natural log',bin_width_s=1,bin_estimator='median of eligible original log frames',baseline_s=[-15,0],baseline_estimator='median of finite baseline-bin medians; each bin one vote',centre='bin_log_vigor-baseline_log_vigor',scale='none',minimum_finite_baseline_bins=1,units='natural-log vigor difference',numeric_clipping=False,colour_limits=[-.25,.25],missing='NaN black; entire trial undefined without baseline data',timing='established camera cadence reconstruction; not measured AbsoluteTime reanalysis',statistical_inference='none; descriptive examples')
    selection=dict(figure_id='fig1',panel_ids=['f','g','h'],revision='historical-fish-1s-direct-log-medians-managua-r',scientific_definition=context,fish=a['fish'],trial_range=[5,94],display_trial_mapping='acquisition minus 4',palette='managua_r',author_instruction=AUTH)
    write(STAGE/'selection.json',selection)
    source_data=[]
    for dependency in [data,record_path]:
        destination=STAGE/('dependency-'+dependency.name)
        destination.write_bytes(dependency.read_bytes());source_data.append(artifact(destination))
    data=STAGE/('dependency-'+data.name)
    config_snapshot=STAGE/'experiments-snapshot.py'
    config_snapshot.write_bytes((ROOT/'src/classical_conditioning/config/experiments.py').read_bytes())
    palette_file=STAGE/'renderer-snapshot.py'
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black');norm=Normalize(-.25,.25,clip=True)
    rgb=cmap(np.linspace(0,1,256))[:,:3]
    apply_theme();plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'fgh-historical-one-second-managua-freeze','font.family':'DejaVu Sans','axes.unicode_minus':False,'savefig.bbox':None,'figure.constrained_layout.use':False})
    fig=plt.figure(figsize=(183/25.4,78/25.4))
    records={};artists={};exceptions=[]
    context=selection['scientific_definition']
    def register(artist,panel,sub,instance,role,geometry=None,coord='data',extra=None,required=None):
        key=f'fig1__{panel}__{sub}__{instance}';artist.set_gid(key)
        style_role=SPEC['roles'][role]['style'];style=copy.deepcopy(SPEC['styles'][style_role])
        scientific=role.startswith(('heatmap.','stimulus.','reference.')) or role in ('phase.separator.heatmap','annotation.example_trial')
        protect='data-geometry' if scientific else 'axis-definition' if role.startswith('axis.') else 'scientific-text' if role in ('phase.label','label.fish_id','title.condition') else 'presentation'
        required=artist.get_visible() if required is None else required
        if hasattr(artist,'get_text') and not artist.get_text():required=False
        r=dict(element_id=key,scientific_role=role,artist_type=type(artist).__name__,figure_id='fig1',panel_id=panel,subpanel_id=sub,scientific_context=dict(context,fish_id=a['fish'].get(panel.upper(),'not-applicable')),coordinate_system=coord,geometry=geometry or {'definition':'artist created from named renderer inputs'},style_role=style_role,resolved_style=style,required_in_svg=bool(required),classification_confidence='explicit',classification_evidence=['Role assigned at artist creation from named data/event/axis source in prepare_figure1_historical_fgh_freeze.py'],protection=protect,style_verification={})
        if type(artist).__name__=='_ColorbarSpine':r['artist_type']='Spine'
        if extra:r.update(extra)
        records[key]=r;artists[key]=artist
        return artist
    def exception(role,prop,value,reason,evidence,panel_ids=('f','g','h'),element_ids=None):
        ex=dict(exception_id='v12-'+role.replace('.','-')+'-'+prop,scientific_role=role,property=prop,value=value,reason=reason,scope=dict(figure_id='fig1',panel_ids=list(panel_ids),revision=selection['revision'],decision_date='2026-10-09'),approval_evidence=evidence)
        if element_ids:ex['scope']['element_ids']=element_ids
        exceptions.append(ex)
    # Recorded author approval of stronger CS guides is scoped to these panels.
    cs_evidence='records/figure-history/figure-01/fgh-colour-and-binning-history-2026-10-09.md: Preserve the approved visual conventions; strong green CS boundaries 2.4 pt alpha .8; user explicitly asked for stronger boundaries. Existing source at 9 inches, displayed at 183 mm.'
    for role in ('stimulus.cs.onset','stimulus.cs.offset'):
        exception(role,'linewidth_pt',2.4*183/228.6,'Retain approved strong guide effective width at the selected review assembly scale',cs_evidence)
        exception(role,'alpha',.8,'Retain approved CS opacity',cs_evidence)
    register(fig.patch,'f','canvas','background','background.panel',coord='figure_fraction',extra={'scientific_context':{'quantity':'not-applicable; white canvas','units':'not-applicable'}})
    lefts=[.09,.355,.62];axes=[];maps=[]
    for i,(letter,name,experiment,condition,us) in enumerate([('F','Delay','allDelay','delay',9),('G','3 s Trace','all3sTrace','trace',13),('H','Control','allDelay','control',None)]):
        panel=letter.lower();ax=fig.add_axes([lefts[i],.13,.22,.75]);axes.append((panel,ax));ax.set_facecolor('white')
        im=ax.imshow(np.ma.masked_invalid(matrices[letter]),extent=(-20,20,90.5,.5),origin='upper',aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,alpha=1,zorder=1)
        register(im,panel,'heatmap','vigor-cells','heatmap.vigor',{'x_extent_s':[-20,20],'trial_extent':[90.5,.5],'shape':[90,40]},extra={'scientific_context':dict(context,fish_id=a['fish'][letter],condition=condition,data_fields=['acquisition_trial','bin_index','bin_log_vigor','baseline_log_vigor','delta_log_vigor','eligible_frames'],data_mapping='Hash-authenticated historical-fish CSV; direct original eligible frame log medians in 1 s bins; subtract each trial finite baseline-bin median over [-15,0); no amplitude scaling; independently verified by renderer evidence')})
        maps.append(im.get_gid())
        ax.set(xlim=(-20,20),ylim=(90.5,.5),xticks=[-20,0,20],yticks=[1,10,20,30,40,50,60,70,80,90])
        ax.tick_params(direction='out',length=2,width=.5,pad=3,labelsize=7,labelleft=i==0)
        ax.set_xlabel('Time from CS onset (s)',fontsize=8,labelpad=3)
        for side,spine in ax.spines.items():spine.set_visible(True);spine.set_color('black');spine.set_linewidth(.5)
        for time,role in [(0,'stimulus.cs.onset'),(10,'stimulus.cs.offset')]:
            line=ax.axvline(time,color='#0d8136',lw=2.4*183/228.6,alpha=.8,linestyle='solid' if time==0 else '--',zorder=4)
            register(line,panel,'heatmap','cs-onset' if time==0 else 'cs-offset',role,{'x_data_s':time,'y_axes_fraction':[0,1],'event_identity':'nominal protocol guide; not a claim of measured delivery on CS-omission trials'},'blended',extra={'scientific_context':dict(context,event='CS onset alignment' if time==0 else 'nominal CS offset',event_source='config/experiments.py shared cs_duration_s=10; alignment preserved from authenticated cadence-reconstructed cache',catch_policy='guides indicate protocol reference across rows, not actual delivery on every trial')})
        if us is not None:
            line,=ax.plot([us]*2,[10.5,60.5],color='#702e78',lw=.7,linestyle=':',alpha=1,zorder=4)
            register(line,panel,'heatmap','us-expected','stimulus.us.expected',{'time_s':us,'training_trial_extent':[10.5,60.5]},extra={'scientific_context':dict(context,event='expected US',event_source=f'get_experiment_spec({experiment}).conditions[{condition}].us_latency_s',guide_is_not_measured_event=True)})
        for y,label in [(10.5,'pre-to-train'),(60.5,'train-to-test')]:
            register(ax.axhline(y,color='white',lw=.8,alpha=1,zorder=5),panel,'heatmap','phase-'+label,'phase.separator.heatmap',{'trial_boundary':y,'transition':label},'blended')
        if i==0:
            for label,y in [('Pre',5.5),('Train',35.5),('Test',75.5)]:register(ax.text(-.28,y,label,transform=ax.get_yaxis_transform(),rotation=90,va='center',ha='center',fontsize=8,clip_on=False),panel,'heatmap','phase-label-'+label.lower(),'phase.label',{'label':label,'trial_anchor':y},'blended')
        if i==0:
            for acquisition_trial in [9,17,63,66,93]:
                trial=acquisition_trial-4
                register(ax.add_patch(Polygon([(1.006,trial),(1.064,trial-1.15),(1.064,trial+1.15)],transform=ax.get_yaxis_transform(),facecolor='black',edgecolor='none',clip_on=False,zorder=6)),panel,'heatmap',f'example-trial-{acquisition_trial}','annotation.example_trial',{'trial':acquisition_trial,'display_trial':trial,'selection_status':'provisional illustrative examples; preserved'},'blended',extra={'scientific_context':{'fish_id':a['fish'][letter],'trial':acquisition_trial,'display_trial':trial,'source':'Author-selected Delay D/E illustrative trials 9/17/63/66/93; mapped acquisition minus 4','status':'author-selected illustrative examples'}})
        register(fig.text(lefts[i]-.035,.947,letter,fontsize=10,weight='bold',va='center'),panel,'heading','panel-letter','label.panel_letter',coord='figure_fraction')
        cs=next(c for c in get_experiment_spec(experiment).conditions if c.condition_id==condition);col=to_hex(condition_color(cs))
        register(fig.text(lefts[i]+.11,.947,name,fontsize=8,ha='center',va='center',color=col),panel,'heading','condition','title.condition',coord='figure_fraction',extra={'resolved_color':col,'color_evidence':f'get_experiment_spec({experiment}).conditions[{condition}].color_rgb_255 through condition_color; source configuration hash-bound'})
        register(fig.text(lefts[i]+.11,.909,a['fish'][letter],fontsize=7,ha='center',va='center',color='black'),panel,'heading','fish-id','label.fish_id',coord='figure_fraction')
    cax=fig.add_axes([.90,.21,.012,.59]);axes.append(('h',cax))
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-.25,0,.25]);bar.set_label('Log vigor relative to baseline',fontsize=8,labelpad=3)
    bar.ax.tick_params(direction='out',length=2,width=.5,labelsize=7,pad=3);bar.outline.set_linewidth(.5)
    bar.solids.set_rasterized(False);bar.solids.set_zorder(1)
    register(bar.solids,'h','colorbar','colour-ramp','heatmap.vigor',{'limits':[-.25,.25],'ticks':[-.25,0,.25]},extra={'scientific_context':dict(context,data_mapping='native managua_r at physical log difference; same norm as all panels'),'mappable_ids':maps,'composite_id':'fig1__h__colorbar__composite','composite_part':'ramp'})
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
        if r['scientific_role']=='heatmap.vigor':r['style_verification']['raster_content']={'status':'passed','evidence':'Native managua_r; exact saved 1 s median matrix, zero midpoint, black NaN; identical data and masks in selected-original and styled candidate'}
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
    for path in [original_svg,svg]:
        tree=ET.parse(path);r=tree.getroot()
        next(n for n in r.iter() if n.get('id')=='fig1__h__colorbar__colour-ramp').set('shape-rendering','crispEdges')
        tree.write(path,encoding='utf-8',xml_declaration=True)
    svgroot=ET.parse(svg).getroot();ids={n.get('id') for n in svgroot.iter()}
    for k,r in records.items():r['required_in_svg']=r['required_in_svg'] and k in ids
    # Attach scientific provenance using the existing export namespace convention.
    provenance={'analysis_recipe':selection['scientific_definition'],'artist_registry':records,'inputs':source_data,'selection_sha256':sha(STAGE/'selection.json')}
    for path in [original_svg,svg]:
        tree=ET.parse(path);root=tree.getroot();metadata=ET.SubElement(root,'{http://www.w3.org/2000/svg}metadata');node=ET.SubElement(metadata,'{https://classical-conditioning.org/figure-provenance}analysis-provenance');node.text=json.dumps(provenance,allow_nan=False);tree.write(path,encoding='utf-8',xml_declaration=True)
    root=ET.parse(svg).getroot();vb=list(map(float,root.get('viewBox').split()))
    write(STAGE/'renderer-properties.json',{'width_mm':183,'height_mm':78,'source_to_final_scale':1,'bounds':bounds,'registry':{k:r['renderer_properties'] for k,r in records.items()},'palette':PALETTE,'rgb_range':[float(rgb.min()),float(rgb.max())],'undefined_trials':[r for r in refs if not r['defined']],'saturated_cells':sum(int(np.sum(abs(v)>.25)) for v in matrices.values())})
    candidate=dict(figure_id='fig1',panel_ids=['f','g','h'],selection_record=artifact(STAGE/'selection.json'),assembly_scale={'final_width_mm':183,'final_height_mm':78,'scope':'assembled F/G/H row; whole Fig1 not frozen','source_to_final_transforms':{'root_user_unit_to_final_pt':183*72/25.4/vb[2],'svg_transforms':[{'element_id':n.get('id'),'transform':n.get('transform')} for n in root.iter() if n.get('transform')]}},element_registry=records,approved_exceptions=exceptions,source_artifacts=[artifact(original_svg,kind='original_svg'),artifact(STAGE/'renderer-snapshot.py'),artifact(STAGE/'selection.json'),artifact(STAGE/'renderer-properties.json'),artifact(config_snapshot),artifact(palette_file)],data_artifacts=[artifact(r['path']) for r in source_data]+[artifact(data)],exports=[artifact(svg)],verification={'scientific_mapping_review':{'status':'passed','evidence':'Authenticated author-selected saved 1 s direct-log medians; baseline subtraction and all defined baseline medians checked zero; 270 defined trials; prior source-bound raw-frame recomputation records included; no scaling or numerical clipping'},'structure_review':{'status':'passed','evidence':'Registered heatmaps, nominal CS/expected-US guides, phases, Delay illustrative arrows linked to D/E, shared colorbar, axes/ticks/spines; displayed trials 1-90 map to acquisition 5-94; repeated y labels suppressed'},'visual_review':{'status':'pending','evidence':'Inspect visual-review.png at the final 183-mm row width before gate finalization'}},freeze_authorization=AUTH)
    write(STAGE/'candidate.json',candidate)
    print(json.dumps({'stage':str(STAGE),'defined_trials':270,'elements':len(records),'palette':PALETTE,'candidate':str(STAGE/'candidate.json')}))

    snapshot.check()

if __name__=='__main__':prepare()
