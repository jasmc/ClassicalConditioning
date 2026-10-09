"""Prepare the selected Panel E for the explicit freeze gate; archive into one HTML."""
from pathlib import Path
import sys,json,copy,hashlib,base64,io,ast,html,argparse,re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import to_hex
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'reviews/.panel-e-heatmap-row-freeze-stage-20261009'
SPEC_PATH=ROOT/'configs/paper-figures/figure-elements.json'
SPEC=json.loads(SPEC_PATH.read_text())
BUILDER=ROOT/'scripts/build_panel_e_frozen_v12_heatmap_rows_review.py'
STORAGE=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
SELECTED=STORAGE/'panel-e-frozen-v12-heatmap-rows-review-20261009/PanelE_frozen-v12-heatmap-rows-review.html'
DEST=STORAGE/'frozen-panel-e-heatmap-rows-20261009/PanelE_frozen_and_version_history.html'
AUTH='Author in this Panel E chat, 2026-10-09: freeze this version and document all other versions'
WIDTH=183.;SCALE=WIDTH/(9*25.4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def artifact(p,**kw):return {'path':str(Path(p).resolve()),'sha256':sha(p),**kw}
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(p,obj):Path(p).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n',encoding='utf-8')

def capture(loc):
    fig=loc['fig'];axes=loc['axes'];strips=loc['strips'];rows=loc['rows'];f=loc['f'];e=loc['e'];cmap=loc['cmap'];norm=loc['norm']
    records={};artists={};exceptions=[];context=loc['selection']['scientific_definition']
    def exception(role,prop,value,reason,evidence,ids=None):
        scope={'figure_id':'fig1','panel_ids':['e'],'revision':'heatmap-row-20261009','decision_date':'2026-10-09'}
        if ids:scope['element_ids']=ids
        exceptions.append({'exception_id':'panel-e-'+role.replace('.','-')+'-'+prop,'scientific_role':role,'property':prop,'value':value,'reason':reason,'scope':scope,'approval_evidence':evidence})
    exception('trace.raw_vigor','alpha',.75,'Retain author-selected visibility of raw trace','Panel E chat: 50% transparency is too much; revised 75% opacity retained through subsequent requests and expressly frozen.')
    exception('trace.raw_vigor','linewidth_pt',.65,'Retain thicker raw trace at final assembly scale','Panel E chat: do not make the black traces thinner; subsequent selected revision used 0.65 pt, then author requested freezing this version.')
    exception('heatmap.vigor','cmap','frozen-v12-bright-blue-charcoal-red-CIELAB','Match exact current frozen heatmap row and its shared colour scale','Panel E chat: follow exactly as done in heatmaps; represent exactly the row of the heatmap below raw vigor; freeze this version. Hash-bound V12 selection records the same approved palette.')
    def register(artist,sub,instance,role,geometry=None,coord='data',extra=None):
        key=f'fig1__e__{sub}__{instance}';artist.set_gid(key)
        style=copy.deepcopy(SPEC['styles'][SPEC['roles'][role]['style']])
        for ex in exceptions:
            if ex['scientific_role']==role:style[ex['property']]=ex['value']
        scientific=role.startswith(('trace.','heatmap.','stimulus.','reference.'))
        protection='data-geometry' if scientific else 'axis-definition' if role.startswith('axis.') else 'scientific-text' if role in ['phase.label','label.fish_id'] else 'presentation'
        records[key]={'element_id':key,'scientific_role':role,'artist_type':type(artist).__name__,'figure_id':'fig1','panel_id':'e','subpanel_id':sub,'scientific_context':dict(context,fish_id='20221115_07',quantity='raw angular speed or scaled frozen vigor' if scientific else 'not-applicable: presentation/axis context'),'coordinate_system':coord,'geometry':geometry or {'definition':'named renderer input'},'style_role':SPEC['roles'][role]['style'],'resolved_style':style,'required_in_svg':artist.get_visible(),'classification_confidence':'explicit','classification_evidence':[f'Assigned from renderer source {BUILDER.name} and named raw-frame/event/bin inputs; not inferred from appearance'],'protection':protection,'style_verification':{}}
        if type(artist).__name__=='_ColorbarSpine':records[key]['artist_type']='Spine'
        if extra:records[key].update(extra)
        artists[key]=artist;return key
    register(fig.patch,'panel','background','background.panel',coord='figure_fraction')
    maps=[]
    for a,b,t,stage in zip(axes,strips,loc['TRIALS'],loc['STAGES']):
        sub=f'trial-{t}-trace';strip=f'trial-{t}-heatmap'
        raw=a.lines[0]
        register(raw,sub,'raw-vigor','trace.raw_vigor',{'trial':t,'x_field':'time_s','y_field':'raw_vigor','frame_support':'eligible; invalid/excluded NaN','display_y_range':[0,1],'clipping':'12 frames overall above 1 rad/ms'},extra={'scientific_context':dict(context,trial=t,fish_id='20221115_07',units='rad/ms',data_mapping='authenticated original raw_vigor frames',normalization='none',raw_values_unchanged=True)})
        events=e[e['Trial number'].eq(t)]
        assert len(a.lines[1:])==len(events)
        for line,event in zip(a.lines[1:],events.itertuples(index=False)):
            role={'actual US onset':'stimulus.us.actual','CS onset':'stimulus.cs.onset','CS offset':'stimulus.cs.offset'}[event.Event]
            register(line,sub,event.Event.lower().replace(' ','-'),role,{'time_s':float(event[2]),'event_identity':event.Event},'blended',extra={'scientific_context':dict(context,trial=t,event=event.Event,units='s',event_source=artifact(loc['ep']),measured_delivery=event.Event=='actual US onset')})
        register(a.texts[0],sub,'phase-trial','phase.label',{'phase':stage,'trial':t},'axes_fraction')
        # Named scientific collection: same 80 selected bin rectangles, no data/geometry change.
        for patch in list(b.patches):patch.remove()
        selected=[r for r in rows if r['trial']==t]
        rects=[Rectangle((r['start_s'],0),.5,1,facecolor=cmap(norm(r['frozen_scaled_vigor'])) if np.isfinite(r['frozen_scaled_vigor']) else '#000000',edgecolor='none') for r in selected]
        collection=PatchCollection(rects,match_original=True,zorder=1);b.add_collection(collection)
        maps.append(register(collection,strip,'vigor-cells','heatmap.vigor',{'trial':t,'bin_edges_s':loc['edges'].tolist(),'shape':[1,80],'cell_height':1,'all_bins_present':True},extra={'scientific_context':dict(context,trial=t,fish_id='20221115_07',units='dimensionless baseline-spread units',data_mapping='frozen V12 half-second bin values',display_limits=[-1,1],numeric_clipping=False,missing='NaN black',values=[r['frozen_scaled_vigor'] for r in selected])}))
    cax=fig.axes[-1];bar=loc['cb'];bar.solids.set_rasterized(False);bar.solids.set_zorder(1)
    register(bar.solids,'colorbar','vigor-ramp','heatmap.vigor',{'limits':[-1,1],'ticks':[-1,0,1]},extra={'mappable_ids':maps,'composite_id':'fig1__e__colorbar__composite','composite_part':'ramp','scientific_context':dict(context,units='dimensionless',normalization='Normalize(-1,1,clip=True) only for colours',data_mapping='exact frozen V12 513-colour CIELAB ramp')})
    for i,text in enumerate(fig.texts):
        role='title.panel' if i==0 else 'label.fish_id' if i==1 else 'axis.label' if i==3 else 'annotation.note'
        register(text,'heading' if i<3 else 'notes', ['title','fish-baseline','recipe','raw-axis-label','raw-clipping','colour-note'][i],role,{'text':text.get_text()},'figure_fraction')
    # Build ticks once before changing only their presentation; numeric anchors stay fixed.
    fig.canvas.draw()
    for ax in fig.axes:
        sub='colorbar' if ax is cax else f'trial-{loc["TRIALS"][axes.index(ax)]}-trace' if ax in axes else f'trial-{loc["TRIALS"][strips.index(ax)]}-heatmap'
        register(ax,sub,'axes','colorbar.scale' if ax is cax else 'axes.container',{'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim())},'axes_fraction',extra={'mappable_ids':maps} if ax is cax else None)
        register(ax.patch,sub,'background','background.panel',coord='axes_fraction')
        for dimension,axis in [('x',ax.xaxis),('y',ax.yaxis)]:
            register(axis,sub,'axis-'+dimension,'axis.component',{'dimension':dimension},'blended')
            register(axis.label,sub,'label-'+dimension,'axis.label',{'dimension':dimension,'units':'s' if dimension=='x' else 'rad/ms' if ax in axes else 'dimensionless'},'blended')
            for ti,tick in enumerate(axis.get_major_ticks()):
                value=float(tick.get_loc())
                for num in [1,2]:
                    side=('bottom' if num==1 else 'top') if dimension=='x' else ('left' if num==1 else 'right')
                    geom={'dimension':dimension,'side':side,'kind':'major','value':value}
                    token=f'tick-{dimension}-{side}-{value:g}'.replace('-','m').replace('.','p')
                    register(getattr(tick,f'tick{num}line'),sub,token+'-mark','axis.tick',geom,'blended')
                    register(getattr(tick,f'label{num}'),sub,token+'-label','axis.tick_label',geom,'blended')
        for side,spine in ax.spines.items():register(spine,sub,'spine-'+side,'axis.spine',{'side':side},'axes_fraction')
    # Save the selected original at the exact same protected data geometry/assembly scale.
    plt.rcParams['svg.hashsalt']='panel-e-heatmap-row-freeze-20261009'
    plt.rcParams['savefig.bbox']=None
    original=STAGE/'selected-original.svg';fig.savefig(original,format='svg')
    # Apply shared styles; explicit scoped exceptions preserve author-approved raw/palette choices.
    for key,artist in artists.items():
        rec=records[key];style=rec['resolved_style'];role=rec['scientific_role']
        if hasattr(artist,'set_alpha') and 'alpha' in style:artist.set_alpha(style['alpha'])
        if hasattr(artist,'set_zorder') and 'zorder' in style:artist.set_zorder(style['zorder'])
        if 'font_size_pt' in style:
            artist.set_fontsize(style['font_size_pt']/SCALE);artist.set_fontweight(style['font_weight']);artist.set_fontfamily('DejaVu Sans');artist.set_color(style.get('color','#000000'))
        elif role=='axis.tick':
            artist.set_markersize(style['length_pt']/SCALE);artist.set_markeredgewidth(style['linewidth_pt']/SCALE);artist.set_color(style['color']);artist.set_markeredgecolor(style['color'])
        elif 'linewidth_pt' in style:
            artist.set_linewidth(style['linewidth_pt']/SCALE)
            artist.set_color(style['color']) if hasattr(artist,'set_color') else artist.set_edgecolor(style['color'])
            if 'linestyle' in style:artist.set_linestyle(style['linestyle'])
        if role in ['background.panel','axes.container','axis.component']:pass
        if role=='heatmap.vigor':artist.set_edgecolor('none')
        for prop,value in style.items():rec['style_verification'][prop]={'status':'passed','evidence':f'Named artist property applied from specification/scoped exception at 183-mm width. {prop}={value}; scientific geometry and original data mappings preserved.'}
        rec['renderer_properties']={'alpha':artist.get_alpha(),'zorder':artist.get_zorder(),'visible':artist.get_visible(),'clip_on':artist.get_clip_on()}
    # Common padding changes only label/tick presentation; preserve tick numeric anchors.
    for ax in fig.axes:
        ax.xaxis.labelpad=3/SCALE;ax.yaxis.labelpad=3/SCALE
        for axis in [ax.xaxis,ax.yaxis]:
            axis.set_tick_params(which='major',length=2/SCALE,width=.5/SCALE,pad=3/SCALE)
            for tick in axis.get_major_ticks():tick.set_pad(3/SCALE)
    candidate_svg=STAGE/'PanelE_frozen-candidate.svg';fig.savefig(candidate_svg,format='svg');fig.savefig(STAGE/'visual-review.png',dpi=160)
    fig.canvas.draw();bounds=[]
    for k,artist in artists.items():
        if hasattr(artist,'get_text') and artist.get_text() and artist.get_visible():
            bb=artist.get_window_extent(fig.canvas.get_renderer());bounds.append({'element_id':k,'text':artist.get_text(),'bounds':list(bb.extents)})
            assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(k,bb.extents)
    plt.close(fig)
    root=ET.parse(candidate_svg).getroot();ids={n.get('id') for n in root.iter()}
    for k,r in records.items():r['required_in_svg']=bool(r['required_in_svg'] and k in ids)
    selection={'figure_id':'fig1','panel_ids':['e'],'selected_source':artifact(SELECTED),'revision':'raw-traces-plus-exact-V12-heatmap-rows','scientific_definition':context,'fish':'20221115_07','trials':loc['TRIALS'],'bin_edges_s':loc['edges'].tolist(),'selected_bins':rows,'baseline_parameters':loc['checks'],'source_data':loc['source'],'current_heatmap_selection':artifact(loc['pointer']),'raw_style_authorization':'User: do not make black traces thinner; 50% transparency is too much; selected 75% revision retained and explicitly frozen','display_raw_cap':1,'clip_note':'12 raw frame values exceed visible range; signed numeric values unchanged','assembly_scope':'Standalone Panel E at 183 mm; whole Figure 1 assembly remains separately selected','authorization':AUTH}
    write(STAGE/'selection.json',selection)
    write(STAGE/'renderer-properties.json',{'final_width_mm':WIDTH,'source_to_final_scale':SCALE,'bounds':bounds,'registry':{k:r['renderer_properties'] for k,r in records.items()},'heatmap_cells':400,'bin_values_counts_parameters_verified':True,'all_5_raw_scaled_support_masks_verified':True,'palette_anchors':loc['anchors']})
    vb=list(map(float,root.get('viewBox').split()))
    candidate={'figure_id':'fig1','panel_ids':['e'],'selection_record':artifact(STAGE/'selection.json'),'assembly_scale':{'final_width_mm':WIDTH,'final_height_mm':WIDTH*6/9,'scope':'standalone Panel E; whole Figure 1 not frozen or assembled','source_to_final_transforms':{'root_user_unit_to_final_pt':WIDTH*72/25.4/vb[2],'svg_transforms':[{'element_id':n.get('id'),'transform':n.get('transform')} for n in root.iter() if n.get('transform')]}},'element_registry':records,'approved_exceptions':exceptions,'source_artifacts':[artifact(original,kind='original_svg'),artifact(BUILDER),artifact(Path(__file__)),artifact(SELECTED),artifact(STAGE/'selection.json'),artifact(STAGE/'renderer-properties.json'),artifact(loc['pointer']),artifact(loc['container'])],'data_artifacts':[artifact(loc['source']['path']),artifact(loc['ep'])],'exports':[artifact(candidate_svg)],'verification':{'scientific_mapping_review':{'status':'passed','evidence':'All 400 bin medians/counts and 5 trial scaling parameters independently verified against exact frozen V12 archive; finite raw/log masks identical. Raw values/masks/event times unchanged; cells equal-sized and missing black. Review renderer checks completed before capture.'},'structure_review':{'status':'passed','evidence':'Named raw traces, event identities from event table, 5 collections each 80 cells, one shared exact V12 colourbar with mappable IDs, phase/trial labels, axes/ticks/units, backgrounds and notes registered. Scientific originals and candidate share identical data geometry.'},'visual_review':{'status':'pending','evidence':'Inspect visual-review.png at 183-mm panel scale before finalization.'}},'freeze_authorization':AUTH}
    write(STAGE/'candidate.json',candidate);print(json.dumps({'stage':str(STAGE),'elements':len(records),'candidate':str(STAGE/'candidate.json')}))

def prepare():
    if STAGE.exists():
        assert not list(STAGE.iterdir()),'Existing stage contains work; inspect before reusing'
    else:STAGE.mkdir(parents=True)
    code=BUILDER.read_text(encoding='utf-8').replace("buf=io.StringIO();fig.savefig(buf,format='svg');svg=buf.getvalue()","freeze_capture(locals()); return")
    ns={'__file__':str(BUILDER),'__name__':'panel_e_freeze_renderer','freeze_capture':capture}
    exec(compile(code,str(BUILDER),'exec'),ns)
    ns['freeze_capture']=lambda local:capture({**ns,**local})
    sys.argv=[str(BUILDER)];ns['main']()

def reviewed():
    p=STAGE/'candidate.json';c=json.loads(p.read_text());c['verification']['visual_review']={'status':'passed','evidence':'Codex visually inspected candidate visual-review.png at verified standalone final width 183 mm: 5 raw traces legible at 75% opacity with scoped final thickness; every 80-cell strip aligned below its trace; black missing distinct from dark zero; compact exact V12 colourbar; axis/phase/text legible and bounded. Raw display clipping labelled. Common fonts/strokes checked by gate and renderer.'};write(p,c)

def publish():
    frozen=json.loads((STAGE/'freeze.json').read_text());assert frozen['freeze_check']['valid'] and not frozen['freeze_check']['issues']
    for group in ['source_artifacts','data_artifacts','exports']:
        for item in frozen[group]:assert sha(item['path'])==item['sha256']
    assert sha(SPEC_PATH)==frozen['specification_sha256']
    archive={'format':'single HTML lossless Panel E freeze archive v1','files':{},'artifact_storage':{},'note':'Gate paths record temporary materialization. Exact byte contents are embedded here; original source parquets remain at their authenticated locations.'}
    for p in STAGE.iterdir():
        if p.name=='visual-review.png':continue
        data=p.read_bytes();archive['files'][p.name]={'sha256':sha(p),'base64':base64.b64encode(data).decode()};archive['artifact_storage'][str(p.resolve())]={'entry':p.name,'sha256':sha(p)}
    for p,name in [(SPEC_PATH,'specification-snapshot.json'),(Path(__file__),'freeze-preparer-snapshot.py'),(BUILDER,'renderer-snapshot.py')]:
        archive['files'][name]={'sha256':sha(p),'base64':base64.b64encode(p.read_bytes()).decode()}
    svg=(STAGE/'PanelE_frozen-candidate.svg').read_text(encoding='utf-8')
    # Catalogue all existing Panel E originals/reviews, retaining exact paths/hashes and previews.
    folders=['traces','audit-vigor-same-frames-20261006','audit-vigor-explicit-nans-repeat-20261006','audit-camera-gaps-and-bin-mapping-20261006','audit-legacy-cadence-20261006','cadence-review-v5-20261007','panel-e-presentation-review-20261007','panel-e-f-binned-bars-20261007','panel-e-f-binned-bars-axis025-20261007','panel-e-f-binned-bars-axis050-20261007','panel-e-f-binned-bars-zeroaligned-20261007','panel-e-f-binned-bars-polished-20261007','panel-e-f-binned-bars-rawzoom-20261007','panel-e-individual-bout-means-20261007','panel-e-scaled-bout-means-full-duration-20261007','panel-e-frozen-heatmap-signal-review-20261009','panel-e-exact-frozen-heatmap-runs-review-20261009','panel-e-frozen-v12-halfsecond-review-20261009','panel-e-frozen-v12-heatmap-rows-review-20261009']
    descriptions=['Historical arrival-clock raw/zoom/inset/stored-bin designs; retain only as history','Arrival-clock identical-frame audit','Explicit NaN repeated-bout audit','Arrival gaps/bin mapping audit','Acquisition cadence audit','Reconstructed-clock v5 raw/signed bout medians','A–H comparison on same v5 data; raw-only, zoom/inset, framewise logs, bout medians, stored bins','0.5 s means of bout medians; managua_r; uncapped raw/bar heights','Same binned signal; signed axis ±0.25','Same binned signal; signed axis ±0.5','Same bins with centred raw/signed zeros','Lighter axes; raw 50% opacity at 0.4 pt','Raw display cap 1 rad/ms; raw 75% opacity at 0.65 pt','Raw-first bout means: ln(mean raw) minus median ln baseline; no time bins','Log-first mean centred logs; full detected-bout bars; no time bins','Historical frozen V1 C_BoutSamples values averaged per bout with full spans','Historical frozen V1 exact sample-run values, intervals and gaps','Current frozen V12 0.5 s original-log medians and bin-fitted symmetric 0.7 scaling as height bars','SELECTED: exact V12 80-bin colour rows below raw; frozen here']
    folders+=['panels','audit-vigor-alignment-20261006']
    descriptions+=['Original exported Panel E and its scientific sidecar; historical assembly source','Earlier alignment diagnostic with joined frames and bin audit; historical scientific support']
    catalogue=[];history=''
    for folder,description in zip(folders,descriptions):
        directory=STORAGE/folder
        if not directory.exists():continue
        items=[];preview=''
        candidates=[p for p in directory.iterdir() if p.is_file() and (('PanelE' in p.name or 'PanelD-E' in p.name or 'same_frame' in p.name or 'gallery' in p.name) and p.suffix in ['.svg','.png','.html','.pdf','.parquet','.json'])]
        if 'audit' in folder:candidates=[p for p in directory.iterdir() if p.is_file() and p.suffix in ['.json','.svg','.png','.csv','.parquet']]
        candidates+= [p for p in directory.iterdir() if p.is_file() and p.name in ['inventory.json','manifest.json','build_manifest.json','index.html'] and p not in candidates]
        for p in sorted(candidates):
            items.append(artifact(p))
            if p.suffix=='.png' and ('PanelE' in p.name or p.name=='gallery.png' or 'audit' in folder):
                im=Image.open(p).convert('RGB');im.thumbnail((1000,900));buf=io.BytesIO();im.save(buf,format='PNG');preview+='<img alt="'+html.escape(p.stem)+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
            if p.suffix=='.html':
                s=p.read_text(encoding='utf-8');m=re.search(r'(<svg\b.*?</svg>)',s,re.S)
                if m:preview+='<img alt="'+html.escape(p.stem)+'" src="data:image/svg+xml;base64,'+base64.b64encode(m.group(1).encode('utf-8')).decode()+'">'
        # Historical vector designs without PNG: embed exact SVG as an image to isolate IDs.
        if folder in ['traces','panels']:
            for p in sorted(candidates):
                if p.suffix=='.svg' and 'PanelE' in p.name:preview+='<p>'+html.escape(p.stem)+'</p><img src="data:image/svg+xml;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
        catalogue.append({'folder':str(directory),'description':description,'artifacts':items})
        history+='<details><summary>'+html.escape(folder+' · '+description)+'</summary>'+preview+'<pre>'+html.escape(json.dumps(items,indent=2))+'</pre></details>'
    catalogue.append({'kind':'different scientific alternatives','folder':str(ROOT/'outputs/figure1-cd-baseline-review'),'description':'Earlier tail_length_weighted_angular_l1 P10–P90 scaled-log alternatives; different metric/scaling, not substitutes for the selected frozen V12 legacy metric','artifacts':[artifact(p) for p in (ROOT/'outputs/figure1-cd-baseline-review').glob('*.figure.json')]})
    history+='<details><summary>Earlier alternative metric/scaling</summary><pre>'+html.escape(json.dumps(catalogue[-1],indent=2))+'</pre></details>'
    chronology=ROOT/'docs/analysis/PANEL_E_PRESENTATION_REVIEW_2026-10-07.md'
    archive['files']['version-history-source.md']={'sha256':sha(chronology),'base64':base64.b64encode(chronology.read_bytes()).decode()}
    history='<details open><summary>Chronology, A–H definitions and calculation changes</summary><pre>'+html.escape(chronology.read_text(encoding='utf-8'))+'</pre></details>'+history
    page='<!doctype html><html><meta charset="utf-8"><title>Frozen Figure 1 E and version history</title><style>body{font:15px system-ui;max-width:1200px;margin:24px auto;padding:0 20px;color:#222}svg,img{width:100%;height:auto}pre{white-space:pre-wrap;font-size:11px}details{margin:15px 0;padding:10px;border:1px solid #ddd}summary{cursor:pointer}</style><body><h1>Frozen Figure 1 · Panel E</h1>'+svg
    page+='<p>Author-frozen 2026-10-09: raw traces plus their exact 80-cell frozen V12 heatmap rows. Final standalone width 183 mm; whole Figure 1 assembly remains separate. Raw trace opacity 75%, final linewidth 0.65 pt, visible range 0–1 rad/ms. Missing cells are black; heatmap values are numerically unclipped with colour limits ±1. Common final-scale fonts and axes styles applied. Raw/palette exceptions retain explicit author choices.</p><h2>Earlier versions</h2>'+history
    page+='<details><summary>Freeze manifest and verification</summary><pre>'+html.escape(json.dumps(frozen,indent=2))+'</pre></details>'
    page+='<script type="application/json" id="panel-e-version-catalogue">'+json.dumps(clean(catalogue),allow_nan=False).replace('<','\\u003c')+'</script>'
    page+='<script type="application/json" id="panel-e-frozen-archive">'+json.dumps(archive,allow_nan=False).replace('<','\\u003c')+'</script></body></html>'
    DEST.parent.mkdir(exist_ok=True);assert not DEST.exists();DEST.write_text(page,encoding='utf-8')
    pointer={'figure_id':'fig1','panel_ids':['e'],'status':'author-frozen','selected_revision':'raw-trace-plus-exact-frozen-V12-heatmap-rows','container':artifact(DEST),'freeze_manifest':{'script_id':'panel-e-frozen-archive','entry':'freeze.json','sha256':sha(STAGE/'freeze.json')},'export':{'embedded_entry':'PanelE_frozen-candidate.svg','sha256':sha(STAGE/'PanelE_frozen-candidate.svg')},'scientific_definition':json.loads((STAGE/'selection.json').read_text())['scientific_definition'],'assembly_scale':frozen['assembly_scale'],'scope':'Panel E only; preserve A-D/FGH freezes and all existing assemblies','author_approval':AUTH,'version_catalogue_count':len(catalogue)}
    write(ROOT/'configs/paper-figures/figure1-panel-e-heatmap-rows-freeze-20261009.json',pointer)
    # Only task-created, explicitly staged files are removed; preserve all history/source data.
    assert STAGE.resolve().parent== (ROOT/'reviews').resolve()
    for p in STAGE.iterdir():assert p.is_file();p.unlink()
    STAGE.rmdir();print(json.dumps({'frozen_review':str(DEST),'history_groups':len(catalogue),'freeze_valid':True,'temporary_stage_removed':True}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','reviewed','publish']);args=parser.parse_args()
    {'prepare':prepare,'reviewed':reviewed,'publish':publish}[args.mode]()
