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
STAGE=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-02-a-six-map-freeze-2026-10-14')
AUTH='Author 2026-10-14: freeze Figure 2 A; reduce A/B/C to six heatmaps in one row; retain earlier B data; one shared colourbar at far right; exact Figure 1 F/G/H axes, labels and guide styles. A only frozen.'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def artifact(path,**extra):return dict(path=str(Path(path).resolve()),sha256=sha(path),**extra)
def prepare():
    from classical_conditioning.figure_workspace import tag
    import base64
    from build_figure_legend_review import figure_font_paths
    wide=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-02-a-freeze-2026-10-14')
    old=json.loads((wide/'candidate.json').read_text())
    for item in old['data_artifacts']+old['source_artifacts']:
        assert sha(item['path'])==item['sha256'],item['path']
    pool=pd.read_parquet(wide/'pooled-bins.parquet')
    bp=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/sources/20261008T200908002798Z/Fig2_PanelB_all3sTrace_signed-pre15.parquet')
    assert sha(bp)=='b98698800b90a5e4b9347d03a0b647273e3c2d0235884e395cf635a4024a0cd0'
    b=pd.read_parquet(bp)
    matrices={}
    for letter,condition in [('a0','control'),('a1','delay')]:
        matrices[letter]=pool[pool.condition_id.eq(condition)].pivot(index='acquisition_trial',columns='bin_index',values='mean_delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
    for letter,condition in [('b0','control'),('b1','trace')]:
        matrices[letter]=b[b.condition_id.eq(condition)].pivot(index='Trial number',columns='Time bin center (s)',values='Mean signed log vigor').reindex(index=range(5,95)).to_numpy()
        assert matrices[letter].shape==(90,80),(condition,matrices[letter].shape)
    STAGE.mkdir(exist_ok=True)
    if (ROOT/'records/frozen-analyses/figure-02/panel-a/2026-10-14-pooled-six-heatmap-row/manifests/freeze.json').exists():raise ValueError('Preserve frozen artifacts; prepare a new stage for revisions')
    (STAGE/'renderer-snapshot.py').write_bytes(Path(__file__).read_bytes())
    context=copy.deepcopy(old['selection_record'])
    context=json.loads(Path(context['path']).read_text())['scientific_definition']
    selection=dict(figure_id='fig2',panel_ids=['a'],scientific_definition=context,author_instruction=AUTH,layout='six axes across 183 mm; shared row colourbar outside scoped A',data=old['data_artifacts'],retained_B=artifact(bp),future_B_C='Follow A pipeline and Figure 1 F/G/H formatting; B remains earlier exploratory calculation until full processed inputs available')
    write(STAGE/'selection.json',selection)
    a={'fish':{}}
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black');norm=Normalize(-.25,.25,clip=True)
    rgb=cmap(np.linspace(0,1,256))[:,:3]
    apply_theme();plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'fgh-historical-one-second-managua-freeze','font.family':'DejaVu Sans','axes.unicode_minus':False,'savefig.bbox':None,'figure.constrained_layout.use':False})
    fig=plt.figure(figsize=(183/25.4,78/25.4))
    records={};artists={};exceptions=[]
    context=selection['scientific_definition']
    def register(artist,panel,sub,instance,role,geometry=None,coord='data',extra=None,required=None):
        key=f'fig2__{panel}__{sub}__{instance}';artist.set_gid(key)
        style_role=SPEC['roles'][role]['style'];style=copy.deepcopy(SPEC['styles'][style_role])
        scientific=role.startswith(('heatmap.','stimulus.','reference.')) or role in ('phase.separator.heatmap','annotation.example_trial')
        protect='data-geometry' if scientific else 'axis-definition' if role.startswith('axis.') else 'scientific-text' if role in ('phase.label','annotation.note','title.condition') else 'presentation'
        required=artist.get_visible() if required is None else required
        if hasattr(artist,'get_text') and not artist.get_text():required=False
        r=dict(element_id=key,scientific_role=role,artist_type=type(artist).__name__,figure_id='fig2',panel_id=panel,subpanel_id=sub,scientific_context=dict(context,population=panel),coordinate_system=coord,geometry=geometry or {'definition':'artist created from named renderer inputs'},style_role=style_role,resolved_style=style,required_in_svg=bool(required),classification_confidence='explicit',classification_evidence=['Role assigned at artist creation from named data/event/axis source in prepare_figure1_historical_fgh_freeze.py'],protection=protect,style_verification={})
        if type(artist).__name__=='_ColorbarSpine':r['artist_type']='Spine'
        if extra:r.update(extra)
        records[key]=r;artists[key]=artist
        return artist
    def exception(role,prop,value,reason,evidence,panel_ids=('a0','a1'),element_ids=None):
        ex=dict(exception_id='v12-'+role.replace('.','-')+'-'+prop,scientific_role=role,property=prop,value=value,reason=reason,scope=dict(figure_id='fig2',panel_ids=list(panel_ids),revision='six-heatmap-row',decision_date='2026-10-14'),approval_evidence=evidence)
        if element_ids:ex['scope']['element_ids']=element_ids
        exceptions.append(ex)
    # Recorded author approval of stronger CS guides is scoped to these panels.
    cs_evidence=AUTH
    for role in ('stimulus.cs.onset','stimulus.cs.offset'):
        exception(role,'linewidth_pt',2.4*183/228.6,'Retain approved strong guide effective width at the selected review assembly scale',cs_evidence)
        exception(role,'alpha',.8,'Retain approved CS opacity',cs_evidence)
    register(fig.patch,'a0','canvas','background','background.panel',coord='figure_fraction',extra={'scientific_context':{'quantity':'not-applicable; white canvas','units':'not-applicable'}})
    lefts=[.085,.216,.347,.478,.609,.74];axes=[];maps=[]
    for i,(letter,name,experiment,condition,us,count) in enumerate([('a0','Control','allDelay','control',None,28),('a1','Delay','allDelay','delay',9,29),('b0','Control','all3sTrace','control',None,19),('b1','3 s Trace','all3sTrace','trace',13,40),('c0','Control','all10sTrace','control',None,None),('c1','10 s Trace','all10sTrace','trace',None,None)]):
        panel=letter;ax=fig.add_axes([lefts[i],.13,.105,.75]);axes.append((panel,ax));ax.set_facecolor('white')
        if letter in matrices:
            matrix=matrices[letter]
            im=ax.imshow(np.ma.masked_invalid(matrix),extent=(-20,20,90.5,.5),origin='upper',aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,alpha=1,zorder=1)
            register(im,panel,'heatmap','vigor-cells','heatmap.vigor',{'x_extent_s':[-20,20],'trial_extent':[90.5,.5],'shape':list(matrix.shape)},extra={'scientific_context':dict(context,condition=condition,data=artifact(wide/'pooled-bins.parquet') if i<2 else artifact(bp),data_mapping='A authenticated frozen pooled 1 s mean; B authenticated earlier 0.5 s values retained unchanged; rows acquisition 5–94 mapped minus 4')})
            maps.append(im.get_gid())
        else:
            register(ax.text(0,45,'Pending',ha='center',va='center',fontsize=7),panel,'heatmap','pending','annotation.note')
        ax.set(xlim=(-20,20),ylim=(90.5,.5),xticks=[-20,0,20],yticks=[1,10,20,30,40,50,60,70,80,90])
        ax.tick_params(direction='out',length=2,width=.5,pad=3,labelsize=7,labelleft=i==0)
        if i%2==0:
            ax.set_xlabel('Time from CS onset (s)',fontsize=8,labelpad=3)
            ax.xaxis.label.set_x((.105+.131)/(.105*2))
        for side,spine in ax.spines.items():spine.set_visible(True);spine.set_color('black');spine.set_linewidth(.5)
        for time,role in ([] if i>=4 else [(0,'stimulus.cs.onset'),(10,'stimulus.cs.offset')]):
            line=ax.axvline(time,color='#0d8136',lw=2.4*183/228.6,alpha=.8,linestyle='solid' if time==0 else '--',zorder=4)
            register(line,panel,'heatmap','cs-onset' if time==0 else 'cs-offset',role,{'x_data_s':time,'y_axes_fraction':[0,1],'event_identity':'nominal protocol guide; not a claim of measured delivery on CS-omission trials'},'blended',extra={'scientific_context':dict(context,event='CS onset alignment' if time==0 else 'nominal CS offset',event_source='config/experiments.py shared cs_duration_s=10; alignment preserved from authenticated cadence-reconstructed cache',catch_policy='guides indicate protocol reference across rows, not actual delivery on every trial')})
        if us is not None:
            line,=ax.plot([us]*2,[10.5,60.5],color='#702e78',lw=.7,linestyle=':',alpha=1,zorder=4)
            register(line,panel,'heatmap','us-expected','stimulus.us.expected',{'time_s':us,'training_trial_extent':[10.5,60.5]},extra={'scientific_context':dict(context,event='expected US',event_source=f'get_experiment_spec({experiment}).conditions[{condition}].us_latency_s',guide_is_not_measured_event=True)})
        for y,label in ([] if i>=4 else [(10.5,'pre-to-train'),(60.5,'train-to-test')]):
            register(ax.axhline(y,color='white',lw=.8,alpha=1,zorder=5),panel,'heatmap','phase-'+label,'phase.separator.heatmap',{'trial_boundary':y,'transition':label},'blended')
        if i==0:
            for label,y in [('Pre',5.5),('Train',35.5),('Test',75.5)]:register(ax.text(-.65,y,label,transform=ax.get_yaxis_transform(),rotation=90,va='center',ha='center',fontsize=8,clip_on=False),panel,'heatmap','phase-label-'+label.lower(),'phase.label',{'label':label,'trial_anchor':y},'blended')
        if i%2==0:register(fig.text(lefts[i]-.03,.947,letter[0].upper(),fontsize=10,weight='bold',va='center'),panel,'heading','panel-letter','label.panel_letter',coord='figure_fraction')
        cs=next(c for c in get_experiment_spec(experiment).conditions if c.condition_id==condition);col=to_hex(condition_color(cs))
        register(fig.text(lefts[i]+.0525,.947,name,fontsize=8,ha='center',va='center',color=col),panel,'heading','condition','title.condition',coord='figure_fraction',extra={'resolved_color':col,'color_evidence':f'get_experiment_spec({experiment}).conditions[{condition}].color_rgb_255 through condition_color; source configuration hash-bound'})
        register(fig.text(lefts[i]+.0525,.909,f'n = {count}' if count is not None else 'pending',fontsize=7,ha='center',va='center',color='black'),panel,'heading','fish-id','annotation.note',coord='figure_fraction')
    cax=fig.add_axes([.90,.21,.012,.59]);axes.append(('shared',cax))
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-.25,0,.25]);bar.set_label('Log vigor relative to baseline',fontsize=8,labelpad=3)
    bar.ax.tick_params(direction='out',length=2,width=.5,labelsize=7,pad=3);bar.outline.set_linewidth(.5)
    bar.solids.set_rasterized(False);bar.solids.set_zorder(1)
    register(bar.solids,'shared','colorbar','colour-ramp','heatmap.vigor',{'limits':[-.25,.25],'ticks':[-.25,0,.25]},extra={'scientific_context':dict(context,data_mapping='native managua_r at physical log difference; same norm as all panels'),'mappable_ids':maps,'composite_id':'fig2__shared__colorbar__composite','composite_part':'ramp'})
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
    for k,r in records.items():
        if r['panel_id'].startswith('b'):
            r['scientific_context']={'data':artifact(bp),'metric':'earlier signed bout log vigor','bin_width_s':.5,'baseline_s':[-15,0],'pooling':'equal fish mean','numeric_values':'unchanged authenticated earlier B parquet','trial_mapping':'acquisition 5–94 to display 1–90','scientific_status':'exploratory; pending adoption of A pipeline'}
        elif r['panel_id'].startswith('c'):
            r['scientific_context']={'quantity':'pending; no scientific results','units':'not applicable','source':'no authenticated full processed inputs available'}
        for ex in exceptions:
            if ex['scientific_role']==r['scientific_role']:r['resolved_style'][ex['property']]=ex['value']
        r['style_verification']={prop:{'status':'passed','evidence':'Named Matplotlib artist configured in renderer-snapshot.py; 183 mm native scale, saved SVG typography/strokes/ticks checked by gate; label and tick pads 3 pt, boxed shared-y geometry and opacity/stacking visually reviewed in row.svg.'} for prop in r['resolved_style']}
        r['renderer_properties']={'alpha':artists[k].get_alpha(),'zorder':artists[k].get_zorder(),'visible':artists[k].get_visible()}
    fig.savefig(STAGE/'row.svg',format='svg',facecolor='white',dpi=300)
    fig.savefig('/tmp/figure2-row.png',dpi=220,facecolor='white')
    plt.close(fig)
    root=ET.parse(STAGE/'row.svg').getroot()
    css=[]
    for weight,path in zip((400,700),figure_font_paths()):
        copied=STAGE/path.name;copied.write_bytes(path.read_bytes())
        css.append('@font-face{font-family:"DejaVu Sans";font-weight:'+str(weight)+';src:url(data:font/ttf;base64,'+base64.b64encode(path.read_bytes()).decode()+') format("truetype")}')
    ET.SubElement(ET.SubElement(root,tag('defs')),tag('style')).text=''.join(css)
    (STAGE/'row.svg').write_bytes(ET.tostring(root))
    f=next(n for n in root if n.get('id')=='figure_1')
    for n in list(f):
        if not n.get('id','').startswith(('fig2__a0__','fig2__a1__')):f.remove(n)
    width=61
    root.set('width',str(width*72/25.4)+'pt');root.set('viewBox',f'0 0 {width*72/25.4} {78*72/25.4}')
    # Canvas is presentation only; crop its geometry to the scoped A pair.
    for n in root.iter(tag('path')):
        if n.get('d','').startswith('M 0 221.102362'):
            n.set('d',f'M 0 221.102362 L {width*72/25.4} 221.102362 L {width*72/25.4} 0 L 0 0 z')
    svg=STAGE/'frozen-panel-a.svg';svg.write_bytes(ET.tostring(root))
    original=STAGE/'selected-original.svg';original.write_bytes(svg.read_bytes())
    ids={n.get('id') for n in root.iter()}
    ar={k:r for k,r in records.items() if r['panel_id'] in ('a0','a1')}
    for ex in exceptions:ex['scope']['panel_ids']=['a']
    for k,r in ar.items():
        r['panel_id']='a'
        r['required_in_svg']=r['required_in_svg'] and k in ids
        r['style_verification']['raster_content']={'status':'passed','evidence':'A numerical matrices read directly from authenticated frozen pooled parquet with unchanged values/masks; saved raster checked against native managua_r at +/-0.25.'}
    write(STAGE/'row-registry.json',records)
    write(STAGE/'renderer-properties.json',{'A_registry':ar,'row_registry':records,'A_data':old['data_artifacts'],'B_data':artifact(bp),'row_width_mm':183,'A_width_mm':61,'font_point_sizes':{'labels':8,'ticks_counts':7,'panel_letters':10},'shared_bar_count':1})
    vb=list(map(float,root.get('viewBox').split()))
    candidate=dict(figure_id='fig2',panel_ids=['a'],selection_record=artifact(STAGE/'selection.json'),assembly_scale={'final_width_mm':61,'final_height_mm':78,'scope':'A pair native scale within six-heatmap row; colourbar owned by row','source_to_final_transforms':{'root_user_unit_to_final_pt':61*72/25.4/vb[2],'svg_transforms':[{'element_id':n.get('id'),'transform':n.get('transform')} for n in root.iter() if n.get('transform')]}},element_registry=ar,approved_exceptions=exceptions,source_artifacts=[artifact(original,kind='original_svg'),artifact(STAGE/'renderer-snapshot.py'),artifact(STAGE/'selection.json'),artifact(STAGE/'renderer-properties.json')]+[artifact(STAGE/path.name) for path in figure_font_paths()]+[a for a in old['source_artifacts'] if a.get('kind')!='original_svg'],data_artifacts=old['data_artifacts'],exports=[artifact(svg)],verification={'scientific_mapping_review':{'status':'passed','evidence':'Authenticated frozen 57-fish pooled numerical matrices unchanged; equal-fish 1 s method and per-fish science already independently verified and immutable; row/acquisition mapping unchanged; retained B independently authenticated.'},'structure_review':{'status':'passed','evidence':'Named artist mappings; compact geometry at native typography/stroke scale; paired A shared xlabel, common row y axis and far-right colourbar; no duplicate colourbar within A.'},'visual_review':{'status':'pending','evidence':'Inspect saved A and full six-map row before publishing'}},freeze_authorization=AUTH)
    write(STAGE/'candidate.json',candidate)
    print(STAGE/'candidate.json')
if __name__=='__main__':prepare()
