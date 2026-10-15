"""Presentation-only shared D/E scale and compact G/H/I trial layout.

Reuses authenticated SVG science and the selected D/E semantic mappings. D's
ordinate is remapped affinely onto E's existing wider range; data coordinates
remain invertible and no statistical result is recalculated. H retains its ratio
axis pending the author-requested later conversion to G's log difference.
"""
from pathlib import Path
import sys,json,copy,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from classical_conditioning.figure_workspace import Snapshot,tag,sha,file_hash,publish_state,measured_presentation
from build_figure_legend_review import svg_editing_page
STATE=ROOT/'docs/paper/figures/figure-02/current-state.json'
EVIDENCE=ROOT/'records/figure-reviews/figure-02/compact-rows-20261015.json'
TEMP=Path('/tmp/figure2-compact-rows.svg')
TICKET=Path('/tmp/fig2-rows-ticket.json')
def enc(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def index(root):return {n.get('id'):n for n in root.iter() if n.get('id')}
def prepare():
    snapshot=Snapshot([]);snapshot.hashes=json.loads(TICKET.read_text());snapshot.hashes[str(Path(__file__).resolve())]=file_hash(__file__)
    state=json.loads(STATE.read_text());d=state['draft']
    for k in ('working_svg','review','legend','semantic_registry'):
        a=d[k];assert file_hash(a['path'])==a['sha256'],'Preserve edited input '+a['path']
    registryfile=ROOT/'records/frozen-analyses/figure-02/panel-d/2026-10-09-version-b/manifests/freeze.json'
    snapshot.hashes[str(registryfile)]=file_hash(registryfile);snapshot.check()
    root=ET.parse(d['working_svg']['path']).getroot();look=index(root)
    # Capture every root subtree and explicit science artist before mutations.
    before={n.get('id'):sha(ET.tostring(n)) for n in root}
    da=look['edit-D-axis__d-delay__y'];ea=look['edit-E-axis__e-trace__y']
    def axis_mapping(axis):
        points=[]
        for tick in axis:
            text=next(tick.iter(tag('text')),None);mark=next(tick.iter(tag('use')),None)
            if text is not None and mark is not None:
                try:points.append((float(text.text),float(mark.get('y'))))
                except (TypeError,ValueError):pass
        v=np.asarray(points);slope=(v[-1,1]-v[0,1])/(v[-1,0]-v[0,0]);offset=v[0,1]-slope*v[0,0]
        np.testing.assert_allclose(v[:,1],slope*v[:,0]+offset,atol=1e-6,rtol=0)
        return float(slope),float(offset),points
    ds,do,dp=axis_mapping(da);es,eo,ep=axis_mapping(ea);factor=es/ds;shift=eo-factor*do
    frozen=json.loads(registryfile.read_text())['element_registry'];roles=('trajectory.fish','observation.fish','summary.median','uncertainty.iqr','reference.signal.zero')
    science=[]
    for parent in list(look['panel-D'].iter()):
        for node in list(parent):
            source=node.get('data-source-id');r=frozen.get(source,{})
            if r.get('scientific_role') in roles:
                body=sha(ET.tostring(node));position=list(parent).index(node);parent.remove(node)
                wrapper=ET.Element(tag('g'),{'id':node.get('id')+'-common-y','transform':f'matrix(1 0 0 {factor:.14g} 0 {shift:.14g})','data-presentation-role':'shared-y-coordinate-map'})
                wrapper.append(node);parent.insert(position,wrapper)
                science.append({'element_id':node.get('id'),'scientific_role':r['scientific_role'],'unchanged_payload_sha256':body})
    assert len(science)>100
    # Bring E's exact ticks to the left of D, retaining D's scientific axis title.
    title=copy.deepcopy(look['edit-D-axis-title__d-delay__y']);newaxis=copy.deepcopy(ea)
    for parent in newaxis.iter():
        for node in list(parent):
            if node.get('id')=='edit-E-axis-title__e-trace__y':parent.remove(node)
    names={n.get('id'):'shared-DEF-'+n.get('id') for n in newaxis.iter() if n.get('id')}
    for n in newaxis.iter():
        if n.get('id'):n.set('id',names[n.get('id')])
        for k,v in list(n.attrib.items()):
            for old,new in names.items():v=v.replace('url(#'+old+')','url(#'+new+')').replace('#'+old,'#'+new)
            n.set(k,v)
    newaxis.append(title)
    for p in root.iter():
        if da in list(p):p.insert(list(p).index(da),newaxis);p.remove(da);break
    # E uses the same scale and position but carries no repeated y labels/spine.
    for p in list(look['panel-E'].iter()):
        for n in list(p):
            if n is ea or n.get('id')=='edit-E-spine__e-trace__left':p.remove(n)
    # F remains pending, with no fabricated observations or statistical guides.
    # Align its empty plot body with the D/E data-axis body.
    units=1800/(183*72/25.4)
    f=look['panel-F'];frame=next(f.iter(tag('rect')))
    clip=next(n for n in look['panel-E'].iter(tag('clipPath')));body=next(clip.iter(tag('rect')))
    ymin,yheight=float(body.get('y')),float(body.get('height'))
    frame.set('y',str(830+ymin*523/150.80315));frame.set('height',str(yheight*523/150.80315))
    pending=look['fig2__f__placeholder__pending'];pending.set('y',str(float(frame.get('y'))+float(frame.get('height'))/2))
    # Keep G's full scientific contents, uniformly reduce the complete source.
    g=look['panel-G'];g.attrib.pop('transform',None);gsvg=next(g.iter(tag('svg')))
    gsvg.set('x','45');gsvg.set('y','1530');gsvg.set('width','540');gsvg.set('height',str(540*277.795276/518.740157))
    for parent in list(gsvg.iter()):
        for n in list(parent):
            if n.get('data-source-id','').endswith('panel-letter'):parent.remove(n)
    # External panel letters retain the same row style as D/E/F.
    letter=ET.SubElement(g,tag('text'),{'id':'fig2__g__compact-row__panel-letter','x':'63','y':'1510','font-family':'DejaVu Sans','font-size':str(10*units),'font-weight':'bold'});letter.text='G'
    h=look['panel-H'];h.attrib.pop('transform',None);hsvg=next(h.iter(tag('svg')))
    hsvg.set('x','615');hsvg.set('y','1530');hsvg.set('width','540');hsvg.set('height',str(540*259.2/417.6))
    for n in list(h):
        if n.tag==tag('text'):n.set('x','633');n.set('y','1510')
    # Pending I follows the same clean placeholder family, occupying row column 3.
    i=look['panel-I'];i.attrib.pop('transform',None)
    rect=next(i.iter(tag('rect')));rect.set('x','1185');rect.set('y','1530');rect.set('width','540');rect.set('height',str(540*259.2/417.6))
    for n in i.iter(tag('text')):
        if n.text=='I':n.set('x','1203');n.set('y','1510')
        elif n.text=='Pending':n.set('x','1455');n.set('y',str(1530+float(rect.get('height'))/2))
    root.set('viewBox','0 0 1800 1930');root.set('height',str(1930/1800*183)+'mm')
    for n in root.findall(tag('rect')):n.set('height','1930')
    # Root canvas resize and D–I are the only changes. A/B/C stays byte-identical.
    assert sha(ET.tostring(look['panel-ABC']))==before['panel-ABC']
    svg=ET.tostring(root);TEMP.write_bytes(svg)
    html=svg_editing_page(svg,scope_label='Figure 2',legend=Path(d['legend']['path']).read_text()).encode();TEMP.with_suffix('.html').write_bytes(html)
    evidence=json.loads(Path(d['semantic_registry']['path']).read_text())
    evidence['compact_rows']={'scope':list('DEFGHI'),'author_direction':'2026-10-15: shared D/E/F Y-axis, very small G with H/I in a single row; H will adopt G measure later','source_svg':d['working_svg'],'generator':{'path':str(Path(__file__).resolve()),'sha256':file_hash(__file__)},'candidate_sha256':sha(svg),'D_y_remapping':{'old_value_to_y':[ds,do],'common_value_to_y':[es,eo],'svg_y_transform':[factor,shift],'old_ticks':dp,'common_ticks':ep,'transformed_artists':science,'validation':'All data artist subtrees byte-identical inside new affine wrappers. Tick affine fits agree <1e-6 source units. Numeric values retained by inverse coordinate mapping.'},'DEF_shared_y':{'display':'ticks/title beside D only; F pending','range':[-.65,.65],'status':'verified'},'GHI_row':{'columns_x':[45,615,1185],'top_y':1530,'G_scale_relative_to_prior':.3,'G_complete_statistics_preserved':True,'H':'existing ratio and axis retained; later author-requested conversion to G log difference pending','shared_y_status':'pending H measure update; distinct y axes explicitly retained for current unlike quantities'},'preserved_ABC_sha256':before['panel-ABC'],'effective_presentation':measured_presentation(svg,width_mm=183),'visual_review':{'status':'pending'}}
    TEMP.with_suffix('.json').write_bytes(enc(evidence));Path('/tmp/fig2-rows-publish-ticket.json').write_bytes(enc(snapshot.hashes));print(TEMP)
def publish():
    snapshot=Snapshot([]);snapshot.hashes=json.loads(Path('/tmp/fig2-rows-publish-ticket.json').read_text());snapshot.check()
    s=json.loads(STATE.read_text());d=s['draft'];e=json.loads(TEMP.with_suffix('.json').read_text());change=e['compact_rows'];svg=TEMP.read_bytes();html=TEMP.with_suffix('.html').read_bytes()
    assert change['visual_review']['status']=='passed' and sha(svg)==change['candidate_sha256'] and svg in html
    assert file_hash(__file__)==change['generator']['sha256']
    history=Path(d['working_svg']['path']).parent/'history/figure2-before-compact-rows-20261015';history.mkdir(parents=True,exist_ok=True)
    for key in ('working_svg','review'):
        src=Path(d[key]['path']);dst=history/src.name
        if dst.exists():assert dst.read_bytes()==src.read_bytes()
        else:
            with dst.open('xb') as f:f.write(src.read_bytes())
    old=history/'current-state.json'
    if not old.exists():old.write_bytes(STATE.read_bytes())
    eb=enc(e);d['working_svg']['sha256']=sha(svg);d['review']['sha256']=sha(html);d['semantic_registry']={'path':str(EVIDENCE),'sha256':sha(eb)}
    d['assembly_scale']['final_height_mm']=1930/1800*183
    d['status']='Compact three-row presentation draft; D/E shared Y, F pending; G/H/I single row; H conversion to log difference pending'
    d['consistency_requests'].append({'id':'fig2-def-ghi-compact-shared-axes','scope':list('DEFGHI'),'author_direction':change['author_direction'],'status':'partial','verified':['D/E common range and left-only ticks/title','F placeholder aligned with D/E body','G/H/I in one row; complete G scaled to 30%','A/B/C unchanged'],'pending':['Convert H to G measure before sharing G/H/I Y-axis; I data pending'],'evidence':{'path':str(EVIDENCE),'sha256':sha(eb)}})
    d['reproduction']={'generator':change['generator'],'command':'.venv/bin/python scripts/layout_figure2_compact_rows.py; review exact SVG; --publish with pre-edit ticket'}
    publish_state(STATE,s,snapshot,artifacts=[(EVIDENCE,eb)],evidence_updates=[(Path(d['working_svg']['path']),svg),(Path(d['review']['path']),html)])
    print(d['review']['path'])
if __name__=='__main__':publish() if '--publish' in sys.argv else prepare()
