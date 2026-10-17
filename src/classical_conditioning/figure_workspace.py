"""Editable SVG copies, measured presentation evidence and guarded figure state.

This module never freezes an artifact or infers a scientific role. Scientific
registries are supplied from hash-verified selected records; anonymous IDs are
structural editing handles only. Current state is published last under a lock.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from contextlib import contextmanager
import tempfile
import xml.etree.ElementTree as ET

from classical_conditioning.figure_freeze import _transform, _multiply, NUMBER

SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', XLINK)
IDENTITY = (1., 0., 0., 1., 0., 0.)
DRAWABLE = {'path','rect','line','polyline','polygon','circle','ellipse','image','use','text'}

def tag(name): return '{'+SVG+'}'+name
def sha(data): return hashlib.sha256(data).hexdigest()
def file_hash(path): return sha(Path(path).read_bytes()) if Path(path).is_file() else None

class StaleFigureState(RuntimeError):
    """Another editor changed a bound file; reconcile instead of overwriting."""

class Snapshot:
    def __init__(self, paths):
        self.hashes = {str(Path(p).resolve()): file_hash(p) for p in paths}

    def check(self,ignore=()):
        ignored={str(Path(p).resolve()) for p in ignore}
        changed = [p for p,h in self.hashes.items() if p not in ignored and file_hash(p) != h]
        if changed:
            raise StaleFigureState('Figure inputs changed during work: '+', '.join(changed))

def bind_loaded_state(snapshot,state,state_bytes,path):
    """Bind the bytes actually parsed, avoiding a read-then-snapshot race."""
    snapshot.hashes[str(Path(path).resolve())]=sha(state_bytes) if state_bytes is not None else None
    if state:
        records=state.get('scientific_selection',{}).get('records',[])+list(state.get('preserved_figures',{}).values())
        for scope in ('frozen_display','draft'):
            records=records+[state[scope][key] for key in ('working_svg','effective_presentation','review','legend') if state[scope].get(key)]
        for item in records:snapshot.hashes[str(Path(item['path']).resolve())]=item['sha256']
    snapshot.check()

@contextmanager
def state_lock(path):
    lock = Path(str(path)+'.lock')
    try:
        descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise StaleFigureState('Another figure publisher holds '+str(lock)) from error
    try:
        with os.fdopen(descriptor,'w') as stream:stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()

def publish_state(path, state, snapshot, artifacts=(), evidence_updates=(), removals=()):
    """Exclusive new artifacts; atomic state replacement is the commit point.

    All writers must capture/check the same state file. Existing output copies
    are never replaced. On failure, remove only artifacts created by this call.
    """
    path=Path(path)
    if str(path.resolve()) not in snapshot.hashes:
        raise ValueError('State path must be captured in the starting snapshot')
    created=[];previous={};removed={}
    with state_lock(path):
        snapshot.check()
        try:
            for destination in removals:
                destination=Path(destination)
                if str(destination.resolve()) not in snapshot.hashes or snapshot.hashes[str(destination.resolve())] is None:
                    raise ValueError('Removal must bind an existing artifact in the starting snapshot')
                removed[destination]=destination.read_bytes()
            for destination,data in artifacts:
                destination=Path(destination)
                if not destination.parent.is_dir():raise FileNotFoundError(destination.parent)
                with destination.open('xb') as stream:stream.write(data)
                created.append(destination)
                if file_hash(destination)!=sha(data):raise OSError('Saved-byte mismatch: '+str(destination))
            snapshot.check(ignore=created)
            for destination,data in artifacts:
                if file_hash(destination)!=sha(data):raise StaleFigureState('New artifact changed before state publication: '+str(destination))
            for destination,data in evidence_updates:
                destination=Path(destination)
                if str(destination.resolve()) not in snapshot.hashes:
                    raise ValueError('Evidence update was not captured in the starting snapshot')
                previous[destination]=destination.read_bytes() if destination.exists() else None
                descriptor,name=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent)
                try:
                    with os.fdopen(descriptor,'wb') as stream:stream.write(data)
                    os.replace(name,destination)
                    if file_hash(destination)!=sha(data):raise OSError('Saved-byte mismatch: '+str(destination))
                finally:
                    if Path(name).exists():Path(name).unlink()
            snapshot.check(ignore=[*created,*previous])
            for destination in removed:destination.unlink()
            descriptor,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
            try:
                with os.fdopen(descriptor,'w') as stream:
                    json.dump(state,stream,indent=2,allow_nan=False);stream.write('\n')
                os.replace(name,path)
            finally:
                if Path(name).exists():Path(name).unlink()
        except BaseException:
            for destination,data in removed.items():
                if not destination.exists():
                    with destination.open('xb') as stream:stream.write(data)
            for destination,data in previous.items():
                if data is None:destination.unlink(missing_ok=True)
                else:destination.write_bytes(data)
            for item in created:item.unlink()
            raise

def _declarations(value):
    result={}
    for piece in value.split(';'):
        if ':' in piece:
            key,val=piece.split(':',1);result[key.strip()]=val.strip()
    return result

def _inline_css(root):
    """Resolve supported simple SVG selectors inside each isolated source.

    Inline styles remain authoritative, including !important. Unsupported CSS
    is rejected rather than silently rendering an editable copy differently.
    """
    rules=[];parents={c:p for p in root.iter() for c in p}
    for style in list(root.iter(tag('style'))):
        css=re.sub(r'/\*.*?\*/','',style.text or '',flags=re.S)
        if re.sub(r'[^{}]+\{[^{}]*\}','',css).strip():raise ValueError('Unsupported SVG stylesheet')
        for selectors,body in re.findall(r'([^{}]+)\{([^{}]*)\}',css):
            for selector in selectors.split(','):
                selector=selector.strip()
                if not re.fullmatch(r'\*|[.#]?[A-Za-z_][\w-]*',selector):
                    raise ValueError('Unsupported SVG selector: '+selector)
                specificity=100 if selector.startswith('#') else 10 if selector.startswith('.') else 0
                rules.append((specificity,len(rules),selector,_declarations(body)))
        parents[style].remove(style)
    for node in root.iter():
        values={};importance={}
        for _,_,selector,declarations in sorted(rules):
            matches=(selector=='*' or (selector.startswith('.') and selector[1:] in node.get('class','').split())
                     or (selector.startswith('#') and selector[1:]==node.get('id')) or selector==node.tag.rsplit('}',1)[-1])
            if matches:
                for key,value in declarations.items():
                    important='!important' in value
                    if important or not importance.get(key):values[key]=value;importance[key]=important
        for key,value in _declarations(node.get('style','')).items():
            if '!important' in value or not importance.get(key):values[key]=value
        if values:node.set('style',';'.join(k+':'+v for k,v in values.items()))

def editable_svg(data, *, kind, source_records, registries=None, width_mm=183.):
    """Expand embedded SVG panels into nested vector trees without restyling.

    Namespaces resource IDs/references and assigns structural IDs to anonymous
    nodes. Frozen source bytes/hashes remain recorded separately and untouched.
    """
    root=ET.fromstring(data);registries=registries or {};records=[];semantic={}
    outer_parents={c:p for p in root.iter() for c in p}
    for image in list(root.iter(tag('image'))):
        href=image.get('href',image.get('{'+XLINK+'}href',''))
        if not href.startswith('data:image/svg+xml;base64,'):continue
        original=base64.b64decode(href.split(',',1)[1],validate=True)
        panel=ET.fromstring(original);parent=outer_parents[image]
        group=parent
        while not group.get('id','').startswith('panel-') and group in outer_parents:group=outer_parents[group]
        pid=group.get('id','panel-unknown').removeprefix('panel-')
        prefix='edit-'+pid
        _inline_css(panel)
        mapping={}
        for index,node in enumerate(panel.iter()):
            old=node.get('id');new=prefix+'-'+(old or 'node-'+str(index))
            if old:
                if old in mapping:raise ValueError('Duplicate source ID: '+old)
                mapping[old]=new;node.set('data-source-id',old)
            node.set('id',new)
            if node.tag.rsplit('}',1)[-1] in DRAWABLE:
                records.append(dict(element_id=new,panel=pid,source_element_id=old,source_node_index=index,
                                    svg_type=node.tag.rsplit('}',1)[-1],classification='structural handle; no inferred scientific role'))
        for node in panel.iter():
            for key,value in list(node.attrib.items()):
                if key in {'id','data-source-id'}:continue
                value=re.sub(r'url\(#([^)]+)\)',lambda m:'url(#'+mapping[m[1]]+')',value)
                if key.rsplit('}',1)[-1]=='href' and value.startswith('#'):value='#'+mapping[value[1:]]
                node.set(key,value)
        for old,record in registries.get(pid,{}).items():
            if old not in mapping:
                raise ValueError('Scientific registry ID absent from SVG: '+pid+' / '+old)
            if old in mapping:
                semantic[mapping[old]]=dict(record,element_id=mapping[old],source_element_id=old,
                                           import_evidence='Source registry imported without reclassification; review status retained')
                copied=semantic[mapping[old]]
                if 'mappable_ids' in record:
                    copied['source_mappable_ids']=record['mappable_ids']
                    copied['mappable_ids']=[mapping[i] for i in record['mappable_ids'] if i in mapping]
                    missing=[i for i in record['mappable_ids'] if i not in mapping]
                    if missing:raise ValueError('Scientific registry mappable IDs absent: '+str(missing))
                if record.get('composite_id'):
                    copied['source_composite_id']=record['composite_id']
                    copied['composite_id']=prefix+'-'+record['composite_id']
        for key in ('x','y','width','height','preserveAspectRatio'):
            if key in image.attrib:panel.set(key,image.get(key))
        panel.set('overflow','hidden')
        parent.insert(list(parent).index(image),panel);parent.remove(image)
    _inline_css(root)
    existing={n.get('id') for n in root.iter() if n.get('id')}
    for index,node in enumerate(root.iter()):
        if not node.get('id') and node.tag.rsplit('}',1)[-1] in DRAWABLE|{'g','svg'}:
            new='assembly-node-'+str(index)
            if new in existing:raise ValueError('Structural ID collision')
            node.set('id',new);existing.add(new)
    all_ids=[n.get('id') for n in root.iter() if n.get('id')]
    if len(all_ids)!=len(set(all_ids)):raise ValueError('Duplicate composed SVG IDs')
    # Preserve source viewport dimensions: changing them to fractional mm can
    # introduce renderer rounding/re-sampling despite identical vector geometry.
    # Intended physical width is recorded independently in provenance/evidence.
    root.set('aria-label',kind)
    metadata=dict(schema_version=1,kind=kind,whole_figure_frozen=False,source_assembly_sha256=sha(data),
                  final_width_mm=width_mm,panel_sources=source_records,structural_inventory=records,
                  scientific_registry=semantic,semantic_limitations='Only supplied scientifically evidenced registries are imported. Remaining elements have structural handles only; this is not a complete whole-figure freeze registry. Historical source styles are not evidence of compliance at the current assembly scale.')
    ET.SubElement(root,tag('metadata'),{'id':'editable-copy-provenance'}).text=json.dumps(metadata,allow_nan=False)
    return ET.tostring(root,encoding='utf-8'),metadata

def _props(node):
    values=dict(node.attrib);values.update(_declarations(node.get('style','')))
    return {k:v.replace('!important','').strip() for k,v in values.items()}

def _point(matrix,x,y):
    a,b,c,d,e,f=matrix;return (a*x+c*y+e,b*x+d*y+f)

def _viewport(node):
    box=[float(v) for v in node.get('viewBox').split()]
    x,y,w,h=[float(node.get(k,'0')) for k in ('x','y','width','height')]
    sx,sy=w/box[2],h/box[3];fit=node.get('preserveAspectRatio','xMidYMid meet')
    if fit=='none':return (sx,0,0,sy,x-box[0]*sx,y-box[1]*sy)
    if fit not in {'xMidYMid meet','xMinYMin meet'}:raise ValueError('Unsupported editable viewport fit: '+fit)
    scale=min(sx,sy);mx=(w-box[2]*scale)/2 if fit.startswith('xMid') else 0
    my=(h-box[3]*scale)/2 if fit.startswith('xMid') else 0
    return (scale,0,0,scale,x+mx-box[0]*scale,y+my-box[1]*scale)

def measured_presentation(data, *, width_mm=183.):
    """Measure final text, stroke/tick scales and coordinate maps from SVG.

    Supports affine display transforms in drafts. Nonuniform stroke widths are
    reported as ranges or direction-specific values, never one guessed scalar.
    The explicit freeze gate remains stricter; this is not freeze approval.
    """
    root=ET.fromstring(data);ids={n.get('id'):n for n in root.iter() if n.get('id')}
    box=list(map(float,root.get('viewBox').split()));unit=width_mm*72/25.4/box[2]
    texts=[];strokes=[];ticks=[];maps=[];issues=[]
    def walk(node,matrix,props,opacity,context,in_resource=False):
        local=_props(node);props={**props,**local};opacity*=float(local.get('opacity','1'))
        if local.get('transform','')!=node.get('transform','').strip():
            raise ValueError('Resolve CSS transforms into SVG transform attributes before measurement')
        if local.get('vector-effect','none')not in {'none','non-scaling-stroke'}:
            raise ValueError('Unsupported measured SVG vector effect')
        for property_name in ('mask','filter','mix-blend-mode'):
            if local.get(property_name,'none') not in {'none','normal'}:
                raise ValueError('Unsupported measured SVG property requires resolution: '+property_name)
        matrix=_multiply(matrix,_transform(node.get('transform','')))
        if node is not root and node.tag==tag('svg'):matrix=_multiply(matrix,_viewport(node))
        context=context+[node.get('data-source-id',node.get('id',''))]
        resource=in_resource or node.tag in {tag('defs'),tag('clipPath'),tag('metadata')}
        name=node.tag.rsplit('}',1)[-1];target=node
        if not resource and name in DRAWABLE:
            a,b,c,d,_,_=matrix;sx=math.hypot(a,b);sy=math.hypot(c,d)
            maps.append(dict(element_id=node.get('id'),svg_type=name,source_to_assembly_matrix=list(matrix),root_unit_to_final_pt=unit))
            if name=='text':
                size=props.get('font-size')
                if size and re.fullmatch(NUMBER+r'(?:px)?',size):
                    value=float(size.removesuffix('px'));anchor=_point(matrix,float(node.get('x','0')),float(node.get('y','0')))
                    texts.append(dict(element_id=node.get('id'),text=''.join(node.itertext()),font_family=props.get('font-family'),weight=props.get('font-weight','normal'),
                                      font_height_pt=value*sy*unit,font_width_scale_pt=value*sx*unit,anchor_pt=[v*unit for v in anchor],context=context))
                else:issues.append(dict(element_id=node.get('id'),property='font-size',reason='Not resolved to numeric SVG user units'))
            paint=props
            if name=='use':
                ref=node.get('href',node.get('{'+XLINK+'}href','')).removeprefix('#')
                target=ids.get(ref)
                if target is None:raise ValueError('Broken use reference: '+ref)
                if target.get('transform'):raise ValueError('Transformed use templates require explicit resolution before measurement')
                # Referenced element properties override properties inherited by use.
                paint={**props,**{k:v for k,v in _props(target).items()if v!='inherit'}}
            if paint.get('stroke','none')!='none':
                width=float(paint.get('stroke-width','1').removesuffix('px'));det=abs(a*d-b*c)
                trace=a*a+b*b+c*c+d*d;disc=max(0.,trace*trace-4*det*det)
                hi=math.sqrt((trace+math.sqrt(disc))/2);lo=det/hi if hi else 0
                record=dict(element_id=node.get('id'),context=context,color=paint['stroke'],effective_opacity=opacity*float(paint.get('stroke-opacity','1')),
                            width_range_pt=[width*lo*unit,width*hi*unit])
                non_scaling=paint.get('vector-effect')=='non-scaling-stroke'
                if non_scaling:record.update(width_range_pt=[width*72/96,width*72/96],normal_width_pt=width*72/96,vector_effect='non-scaling-stroke')
                nums=[float(v) for v in re.findall(NUMBER,target.get('d',''))] if name in {'path','use'} else []
                if name=='line':nums=[float(node.get(k)) for k in ('x1','y1','x2','y2')]
                if len(nums)==4 and (name=='line' or re.fullmatch(r'\s*M\s*'+NUMBER+r'[ ,]+'+NUMBER+r'\s*L\s*'+NUMBER+r'[ ,]+'+NUMBER+r'\s*',target.get('d',''))):
                    dx,dy=nums[2]-nums[0],nums[3]-nums[1];length=math.hypot(dx,dy)
                    mapped=math.hypot(a*dx+c*dy,b*dx+d*dy)
                    if mapped and not non_scaling:record['normal_width_pt']=width*det*length/mapped*unit
                    scheme_tick=any(s.startswith('time-axis-') for s in context) and name=='line' and dx==0
                    if re.search(r'xtick|ytick', ' '.join(context)) or scheme_tick:
                        x=float(node.get('x','0')) if name=='use' else 0;y=float(node.get('y','0')) if name=='use' else 0
                        start=_point(matrix,nums[0]+x,nums[1]+y);end=_point(matrix,nums[2]+x,nums[3]+y)
                        ticks.append(dict(record,length_pt=mapped*unit,start_pt=[v*unit for v in start],end_pt=[v*unit for v in end]))
                strokes.append(record)
        for child in node:walk(child,matrix,props,opacity,context,resource)
    walk(root,IDENTITY,{'fill':'black','stroke':'none','stroke-width':'1','font-weight':'normal'},1.,[])
    spacing=[]
    for tick in ticks:
        # Structural matching only: common explicit tick/axis group, no role guess.
        keys=[s for s in tick['context'] if re.fullmatch(r'[xy]tick_\d+',s) or s.startswith('time-axis-')]
        candidates=[t for t in texts if any(k in t['context'] for k in keys)]
        if candidates:
            nearest=min(candidates,key=lambda t:math.dist(t['anchor_pt'],tick['end_pt']))
            spacing.append(dict(tick_element_id=tick['element_id'],label_element_id=nearest['element_id'],
                                endpoint_to_text_anchor_pt=math.dist(nearest['anchor_pt'],tick['end_pt']),
                                interpretation='Measured anchor distance, not glyph-edge padding; padding/legibility still needs renderer or visual review'))
    from classical_conditioning.figures.layout_frames import assess
    frames = assess(data)
    return dict(layout_frames=frames,svg_sha256=sha(data),final_width_mm=width_mm,texts=texts,strokes=strokes,ticks=ticks,tick_label_spacing=spacing,
                coordinate_maps=maps,unresolved=issues,freeze_compliance='not evaluated; use explicit freeze gate with scientific registry and review evidence')

def draft_axis_findings(report):
    """Compare measured axis styling with the author-requested 2/.5-pt ticks.

    Tick identity comes from explicit SVG axis/tick groups. These findings do
    not classify scientific marks or grant a freeze exception.
    """
    findings=[]
    for tick in report['ticks']:
        if not math.isclose(tick['length_pt'],2.,abs_tol=.025):
            findings.append(dict(element_id=tick['element_id'],property='tick_length_pt',expected=2.,measured=tick['length_pt']))
        width=tick.get('normal_width_pt')
        if width is not None and not math.isclose(width,.5,abs_tol=.025):
            findings.append(dict(element_id=tick['element_id'],property='tick_width_pt',expected=.5,measured=width))
    for stroke in report['strokes']:
        context=' '.join(stroke['context'])
        # D legacy spine IDs were verified against its selected renderer.
        spine=('__spine-' in context or '__outline' in context or
               ('edit-D' in context and any(('patch_'+str(i)) in stroke['context'] for i in [3,4,6,7,9,10,12,13,15,16])))
        if spine:
            actual=stroke.get('normal_width_pt')
            if actual is None:actual=stroke['width_range_pt']
            if isinstance(actual,float) and not math.isclose(actual,.5,abs_tol=.025):
                findings.append(dict(element_id=stroke['element_id'],property='spine_width_pt',expected=.5,measured=actual))
    return findings

def verify_mapping_baseline(data, baseline):
    """Fail on unreviewed geometry/text/resource/ancestor-transform changes.

    Compared IDs and coordinate matrices prevent an ancestor-only movement from
    bypassing protection. For intentional layout changes, build and review a new
    baseline; this function cannot authorize them.
    """
    current=measured_presentation(data,width_mm=baseline['final_width_mm'])
    if current['coordinate_maps']!=baseline['coordinate_maps']:
        raise StaleFigureState('SVG coordinate mapping changed; review display transforms before adoption')
    if current['svg_sha256']!=baseline['svg_sha256']:
        raise StaleFigureState('SVG bytes changed; review scientific/text/resource and presentation changes before adoption')
    return current
