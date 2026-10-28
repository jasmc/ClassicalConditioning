"""Source-bound, read-only figure browsing. No renderer or analysis is executed."""
from __future__ import annotations

import base64
import copy
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import hashlib
from html.parser import HTMLParser
import html
import io
import json
import os
from pathlib import Path
import re
import zipfile
from urllib.parse import unquote, unquote_to_bytes, urlsplit, quote
from classical_conditioning.external_artifacts import resolve_artifact, relocated_external_directory

VISUAL = {'.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.pdf'}
MIME = {'.svg':'image/svg+xml', '.png':'image/png', '.jpg':'image/jpeg',
        '.jpeg':'image/jpeg', '.gif':'image/gif', '.webp':'image/webp', '.pdf':'application/pdf'}
GENERATED = {'INDEX.html', 'CONTENTS.html', 'START-HERE.html', 'REVIEW-LIBRARY.md'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def packed(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()


def natural(value):
    # Tagged components remain comparable for labels beginning with a number.
    return tuple((0, int(x)) if x.isdigit() else (1, x.casefold())
                 for x in re.split(r'(\d+)', str(value)) if x)


def scope_key(scope):
    if scope == 'assembly': return (0, ())
    if scope.startswith('panel-'): return (1, tuple(scope[6:].split('-')))
    if scope.startswith('panels-'): return (2, tuple(scope[7:].split('-')))
    return (3, natural(scope))


def location(path, root):
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts)>2 and parts[0]=='figures' and re.fullmatch(r'figure-\d+',parts[1]):
        fig=int(parts[1][7:]);scope=parts[2]
        if scope not in {'assembly'} and not re.fullmatch(r'panels?-[a-z](?:-[a-z])*',scope):
            scope='assembly'
        return [fig],scope
    return [],'shared'


def supporting(path):
    return any(x in path.parts for x in ('sources','assets','code-snapshot','reading-originals',
                                         'reproduction','repository-history','transactions'))


def source_files(outputs, archive):
    """Exclude only generated browsing snapshots and this task's control records."""
    paths=[]
    outputs=relocated_external_directory(outputs);archive=relocated_external_directory(archive)
    for storage,root in [('outputs',outputs),('archive',archive)]:
        if not root.is_dir():raise ValueError('Unavailable storage: '+str(root))
        if str(root).startswith('/Volumes/') and not Path('/Volumes/JOAQUIM').is_mount():
            raise ValueError('JOAQUIM is unavailable')
        for base,dirs,names in os.walk(root):
            # Git object databases and maintenance evidence are reproduction/control,
            # not another artwork tree. Their original inventories remain evidence.
            dirs[:]=sorted(d for d in dirs if d not in {'.git','review-library'} and
                          not (storage=='archive' and Path(base)==root/'repository-history' and
                               d.startswith(('storage-consolidation-','ordered-figure-library-'))))
            for name in sorted(names):
                p=Path(base)/name
                if p.is_symlink():raise ValueError('Source symlink requires review: '+str(p))
                if p.is_file():paths.append((storage,p))
    return paths


class SavedHTML(HTMLParser):
    """Read actual markup; do not execute historical scripts/notebooks."""
    def __init__(self,text):
        super().__init__(convert_charrefs=True)
        self.text=text; self.starts=[0]+[m.end() for m in re.finditer('\n',text)]
        self.visuals=[];self.scripts={};self.heading='';self.heading_tag=None;self.heading_text=[]
        self.sections=[];self.svg_depth=0;self.svg_start=0;self.svg_info=None
        self.script=None;self.script_body=[];self.figure=None;self.caption=False
        self.div_depth=0;self.legend_depth=None;self.legend_text=[];self.blocks=[];self.in_style=False
        self.feed(text);self.close()
    def position(self):
        line,col=self.getpos();return self.starts[line-1]+col
    def context(self):
        return dict(title=self.heading,section=self.sections[-1] if self.sections else '')
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='style':self.in_style=True
        if tag in {'div','section','article','figure'}:self.blocks.append(dict(tag=tag,start=len(self.visuals),text=[]))
        if tag=='div':
            self.div_depth+=1
            if self.legend_depth is None and ('legend' in a.get('class','').split() or a.get('aria-label','').lower()=='associated legend'):
                self.legend_depth=self.div_depth;self.legend_text=[]
        if self.legend_depth is not None and tag in {'p','br'}:self.legend_text.append('\n')
        if tag=='script':self.script=a.get('id','');self.script_body=[]
        if tag in {'section','article'}:self.sections.append(a.get('id',''))
        if re.fullmatch('h[1-6]',tag):self.heading_tag=tag;self.heading_text=[]
        if tag=='figure':self.figure=[]
        if tag=='figcaption':self.caption=True
        if tag=='svg':
            if not self.svg_depth:self.svg_start=self.position();self.svg_info=self.context()
            self.svg_depth+=1
        if tag=='img' and not self.svg_depth:
            item=dict(self.context(),src=a.get('src',''),alt=a.get('alt',''),caption='')
            self.visuals.append(item)
            if self.figure is not None:self.figure.append(item)
    def handle_endtag(self,tag):
        if tag=='style':self.in_style=False
        if tag in {'div','section','article','figure'} and self.blocks:
            if self.blocks[-1]['tag']==tag:
                block=self.blocks.pop();saved=' '.join(''.join(block['text']).split())
                if len(self.visuals)-block['start']==1 and len(saved)>=40 and not self.visuals[-1].get('caption'):
                    self.visuals[-1]['caption']=saved;self.visuals[-1]['caption_binding']='saved source explanation'
        if tag=='div':
            if self.legend_depth==self.div_depth:
                if self.visuals and not self.visuals[-1].get('caption'):self.visuals[-1]['caption']=''.join(self.legend_text).strip()
                self.legend_depth=None;self.legend_text=[]
            self.div_depth-=1
        if tag=='script':
            if self.script:self.scripts[self.script]=''.join(self.script_body)
            self.script=None;self.script_body=[]
        if tag in {'section','article'} and self.sections:self.sections.pop()
        if tag==self.heading_tag:
            self.heading=' '.join(''.join(self.heading_text).split());self.heading_tag=None
        if tag=='svg' and self.svg_depth:
            self.svg_depth-=1
            if not self.svg_depth:
                end=self.text.index('>',self.position())+1
                item=dict(self.svg_info,raw=self.text[self.svg_start:end].encode(),ext='.svg',caption='')
                self.visuals.append(item)
                if self.figure is not None:self.figure.append(item)
        if tag=='figcaption':self.caption=False
        if tag=='figure':self.figure=None
    def handle_data(self,data):
        if self.script is not None:self.script_body.append(data)
        if self.script is None and not self.in_style and not self.svg_depth:
            for block in self.blocks:block['text'].append(data+' ')
        if self.heading_tag:self.heading_text.append(data)
        if self.legend_depth is not None:self.legend_text.append(data)
        if self.caption and self.figure is not None:
            for item in self.figure:item['caption']+=data


def panel_views(raw):
    """Isolate declared ownership groups, retaining ancestors and resource defs.

    No bounding boxes are inferred from pictures or labels. Historical SVGs
    without an explicit layout specification remain whole saved views.
    """
    from classical_conditioning.figures.layout_frames import read_spec
    try:spec=read_spec(raw)
    except (ValueError, ET.ParseError):return []
    if spec is None:return []
    root=ET.fromstring(raw);view=list(map(float,root.get('viewBox','').split()))
    if len(view)!=4:return []
    sx=view[2]/spec['final_size_mm'][0];sy=view[3]/spec['final_size_mm'][1]
    found=[]
    for allocation in spec['allocations']:
        letter=allocation['id']
        if allocation['kind']!='panel' or not re.fullmatch('[A-Z]',letter):continue
        owned=[allocation]
        for shared in spec['allocations']:
            name=shared['id'];prefix=name.split('-')[0]
            if shared['kind']=='shared' and ('colorbar' in name.lower() or 'colourbar' in name.lower()):
                if not re.fullmatch('[A-Z]+',prefix) or letter in prefix:owned.append(shared)
        wanted={i for a in owned for i in a['group_ids']}
        ids={n.get('id') for n in root.iter()}
        if not wanted.issubset(ids):raise ValueError('Declared panel groups are missing')
        def keep(node):
            if node.get('id') in wanted or node.tag.rsplit('}',1)[-1] in {'defs','style'}:return copy.deepcopy(node)
            result=ET.Element(node.tag,dict(node.attrib));result.text=node.text;result.tail=node.tail
            for child in node:
                saved=keep(child)
                if saved is not None:result.append(saved)
            return result if len(result) else None
        saved=keep(root)
        if saved is None:continue
        boxes=[a['box_mm'] for a in owned];x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
        width=max(b[0]+b[2] for b in boxes)-x;height=max(b[1]+b[3] for b in boxes)-y
        saved.set('viewBox',f'{view[0]+x*sx} {view[1]+y*sy} {width*sx} {height*sy}')
        saved.set('width',str(width)+'mm');saved.set('height',str(height)+'mm')
        saved.set('aria-label','Panel '+letter+' from authenticated assembly')
        found.append(dict(panel=letter,raw=ET.tostring(saved,encoding='utf-8'),groups=sorted(wanted),box_mm=[x,y,width,height]))
    return found


class Catalogue:
    def __init__(self,outputs,archive,repo,stage):
        self.outputs=relocated_external_directory(outputs);self.archive=relocated_external_directory(archive);self.repo=Path(repo);self.stage=Path(stage)
        self.roots={'outputs':self.outputs,'archive':self.archive,'repo':self.repo}
        self.guards={};self.inventory=[];self.entries=[];self.issues=[];self.payloads={}
        self.path_rows={};self.physical={};self.figures=defaultdict(set);self.current={};self.review_companions={}
        self.reading_derivatives={};self.origins={};self.derivative_records={};self.repair_originals={};self.migration=None
        w=self.archive/'repository-history/storage-consolidation-2026-10-26'
        if (w/'manifest-v3.json').is_file():
            self.migration=json.loads(self.read(w/'manifest-v3.json'))
            source_rows={r['source']:r for r in self.migration['files']}
            for repair in self.migration.get('link_repairs',[]):
                if Path(repair['source']).suffix.lower()!='.html':continue
                original=Path(repair['original_destination']);expected=source_rows[repair['source']]['sha256']
                if not original.is_file():
                    original=resolve_artifact(original,expected_sha256=expected)
                ref=self.ref(original)
                if ref['sha256']!=expected:raise ValueError('Link-repair original changed')
                self.repair_originals[str(original)]=dict(reading=repair['reading_path'],sha256=expected)
        for folder in ('historical-reading-derivatives','historical-reading-derivatives-more'):
            p=w/folder/'changes.json'
            if p.is_file():
                for r in json.loads(self.read(p)):
                    self.reading_derivatives[r['reading']]=r['original'];self.derivative_records[r['reading']]=r
    def read(self,path):
        path=Path(path)
        if not path.is_file():path=resolve_artifact(path,repo=self.repo)
        before=path.stat();raw=path.read_bytes();after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Source changed while reading: '+str(path))
        h=sha(raw);prior=self.guards.get(str(path))
        if prior and prior['sha256']!=h:raise ValueError('Source advanced: '+str(path))
        self.guards[str(path)]=dict(path=str(path),sha256=h,bytes=len(raw))
        return raw
    def ref(self,path):
        if str(path) not in self.guards:self.read(path)
        return dict(self.guards[str(path)])
    def materialize(self,raw,ext):
        h=sha(raw);key='version-assets/'+h+ext.lower()
        if key not in self.payloads:
            p=self.stage/key;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and p.read_bytes()!=raw:raise ValueError('Prepared payload collision')
            if not p.exists():p.write_bytes(raw)
            self.payloads[key]=dict(path=key,source=str(p),sha256=h,bytes=len(raw))
        return key
    def discover_states(self):
        for p in sorted((self.repo/'docs/paper/figures').rglob('*state.json')):
            d=json.loads(self.read(p));fig=d.get('figure_id','')
            m=re.match(r'fig(?:ure-)?0*(\d+)(?:$|[-_])',fig)
            if not m:continue
            number=int(m[1]);rel=p.relative_to(self.repo/'docs/paper/figures')
            scopes=[x for x in rel.parts if re.fullmatch(r'panels?-[a-z](?:-[a-z])*',x)]
            scope=scopes[-1] if scopes else 'assembly';selected=d.get(d.get('default_display','draft'),{})
            if not isinstance(selected,dict) or not selected.get('working_svg'):continue
            historical=d.get('default_display','draft')!='draft'
            svg=selected['working_svg'];art=resolve_artifact(svg['path'],expected_sha256=svg['sha256'],repo=self.repo,allow_historical=historical)
            raw=self.read(art)
            if sha(raw)!=svg['sha256']:raise ValueError('Selected artwork changed: '+str(art))
            legend=selected.get('legend');bound=None
            if legend:
                lp=resolve_artifact(legend['path'],expected_sha256=legend['sha256'],repo=self.repo,allow_historical=historical)
                bound=self.ref(lp)
                if bound['sha256']!=legend['sha256']:
                    bound=self.ref(resolve_artifact(lp,expected_sha256=legend['sha256']))
            key=(str(art),number,scope)
            self.current[key]=dict(state=self.ref(p),legend=bound,status='Selected working view',scope=scope)
            if selected.get('review'):
                review=selected['review'];observed=self.ref(resolve_artifact(review['path'],expected_sha256=review['sha256'],repo=self.repo,allow_historical=historical))
                if observed['sha256']!=review['sha256']:raise ValueError('Selected review changed')
                self.review_companions[review['path']]=observed
            self.figures[str(art)].add(number)
    def legend(self,path,selected=None,caption=''):
        if selected and selected.get('legend'):
            ref=selected['legend'];return dict(ref,text=self.read(ref['path']).decode(errors='replace'),binding='scoped state')
        # An explicitly named paired legend outranks a bundle-wide legend.
        choices=[(path.with_name(path.stem+'-legend.md'),'paired filename'),
                 (path.with_name(path.stem+'_legend.md'),'paired filename'),
                 (path.parent/'legend.md','bundle legend')]
        for p,binding in choices:
            if (str(p) in self.known_files if hasattr(self,'known_files') else p.is_file()):
                if binding=='bundle legend' and caption.strip():continue
                return dict(self.ref(p),text=self.read(p).decode(errors='replace'),binding=binding)
        if caption.strip():return dict(text=' '.join(caption.split()),binding='saved caption')
        return dict(text='',binding='not recorded')
    def add(self,path,storage,owner,scope,*,raw=None,ext=None,title=None,locator=None,
            family=None,legend=None,role='version',member=None,source_ref=None):
        source_ref=source_ref or self.origins.get(str(path)) or self.ref(path);ext=ext or path.suffix.lower()
        selected=next((v for (p,f,s),v in self.current.items() if p==str(path) and f in owner and s==scope),None) if not locator else None
        ref=dict(source_ref)
        if locator:ref['locator']=locator
        if member:ref['member']=member
        # Identity includes origin and association; visual equality never merges versions.
        identity=sha(packed(dict(source=ref,figures=owner,scope=scope,role=role)))[:24]
        if any(r['id']==identity for r in self.entries):return
        artifact=raw if raw is not None else None
        if artifact is not None:
            preview=self.materialize(artifact,ext);artifact_sha=sha(artifact)
        else:preview=str(path);artifact_sha=source_ref['sha256']
        relative=path.relative_to(self.roots[storage]);parts=relative.parts
        if not family and owner==[1] and 'assembly' in parts and re.search(r'figure[-_]0?1[-_](?:working[-_])?v\d+',relative.as_posix(),re.I):
            family='Presentation drafts'
        if not family:
            if 'versions' in parts:
                i=parts.index('versions');family=parts[i+1] if len(parts)>i+2 else 'Saved versions'
            elif 'current' in parts:family='Working presentation'
            elif 'studies' in parts:family='Study illustrations'
            else:family='Saved artwork'
        label=title or path.stem.replace('_',' ')
        bindings=[]
        for parent in [path.parent,*list(path.parents)[1:3]]:
            if not parent.is_relative_to(self.roots[storage]):continue
            for name in ('manifest.json','evidence-manifest.json','source-inventory.json','scientific-selection.json',path.stem+'.figure.json'):
                candidate=parent/name
                if (str(candidate) in self.known_files if hasattr(self,'known_files') else candidate.is_file()):bindings.append(self.ref(candidate))
        self.entries.append(dict(source_bindings=bindings,id=identity,figures=owner,scope=scope,family=family,label=label,
            source=ref,artifact_sha256=artifact_sha,preview=preview,format=ext,role=role,
            legend=legend or self.legend(path,selected),selected=bool(selected),
            status=selected['status'] if selected else ('Supporting source/reference' if role!='version' else 'Retained version'),
            state=selected.get('state') if selected else None))
    def payload(self,text):
        """Decode a data URL without executing its source page."""
        header,body=text.split(',',1);mime=header[5:].split(';')[0]
        ext=next((e for e,m in MIME.items() if m==mime),None)
        if not ext:return None
        raw=base64.b64decode(body,validate=True) if ';base64' in header else unquote_to_bytes(body)
        return raw,ext
    def members(self,obj,script,path,storage,owner,scope,location=''):
        if isinstance(obj,dict):
            if isinstance(obj.get('base64'),str) and obj.get('sha256'):
                raw=base64.b64decode(obj['base64'],validate=True)
                if sha(raw)!=obj['sha256']:raise ValueError('Corrupt embedded member: '+str(path)+'#'+location)
                ext=Path(location).suffix.lower()
                row=dict(container=str(path),member=script+'/'+location,sha256=sha(raw),bytes=len(raw),role='reproduction dependency')
                if ext in VISUAL:
                    self.add(path,storage,owner,scope,raw=raw,ext=ext,title=Path(location).name,
                             locator='script:'+script+'/'+location,family='Embedded archive · '+path.stem,member=location)
                    row['role']='embedded artwork'
                self.inventory.append(row)
            else:
                for k,v in obj.items():self.members(v,script,path,storage,owner,scope,location+'/'+str(k))
        elif isinstance(obj,list):
            for i,v in enumerate(obj):self.members(v,script,path,storage,owner,scope,location+'/'+str(i))
    def html(self,path,storage,owner,scope,raw,role):
        text=raw.decode(errors='replace');doc=SavedHTML(text);meta={}
        for name,body in doc.scripts.items():
            try:obj=json.loads(body)
            except json.JSONDecodeError:
                if body.strip().startswith('UEsD'):
                    self.zip(path,storage,owner,scope,base64.b64decode(body.strip(),validate=True),locator='script:'+name)
                continue
            if name=='comparison-provenance':meta=obj
            self.members(obj,name,path,storage,owner,scope)
        cards={c.get('id'):c for c in meta.get('cards',[]) if isinstance(c,dict)}
        recipes=set()
        if 'version-history-catalogue' in doc.scripts:
            history=json.loads(doc.scripts['version-history-catalogue'])
            recipes={r['id'] for r in history.get('recipes',[]) if isinstance(r,dict) and r.get('id')}
        for index,item in enumerate(doc.visuals):
            src=item.get('src','');artifact=item.get('raw');ext=item.get('ext');origin=None
            if src.startswith('data:'):
                result=self.payload(src)
                if result is None:continue
                artifact,ext=result
            elif src:
                u=urlsplit(src)
                if u.scheme not in ('','file'):
                    self.inventory.append(dict(container=str(path),member='image:'+str(index),role='remote reference',reference=src));continue
                target=Path(os.path.abspath(path.parent/unquote(u.path)))
                if not target.is_file():
                    self.issues.append(dict(source=str(path),locator='image:'+str(index),reason='Missing local image',reference=src));continue
                if not any(target.is_relative_to(root) for root in self.roots.values()):
                    self.issues.append(dict(source=str(path),reason='Image outside declared roots',reference=src));continue
                ext=target.suffix.lower()
                if ext not in VISUAL:continue
                origin=self.ref(target)
                # Reuse this exact file rather than a permanent preview copy.
            if not artifact and not origin:continue
            card=cards.get(item.get('section'),{})
            title=card.get('title') or item.get('alt') or item.get('title') or path.stem+' · image '+str(index+1)
            family=card.get('family')
            if not family and item.get('section') in recipes:family='recipe'
            if family:family={'recipe':'Scientific recipes','draft':'Presentation drafts','history':'Historical revisions','latest':'Recorded comparison'}.get(family,family)
            caption=card.get('legend') or item.get('caption','')
            legend=self.legend(path,caption=caption)
            if legend['binding']=='saved caption' and item.get('caption_binding'):legend['binding']=item['caption_binding']
            unit_owner,unit_scope=owner,scope
            declared=re.match(r'^Figure\s+0*([1-4])\b',item.get('title',''),re.I)
            if declared and int(declared[1]) not in owner:
                unit_owner=[int(declared[1])];unit_scope='assembly' if scope=='assembly' else 'shared'
            self.add(path,storage,unit_owner,unit_scope,raw=artifact,ext=ext,title=title,
                     locator='visual:'+str(index)+':'+item.get('section',''),family=family,
                     legend=legend,role=role,source_ref=self.origins.get(str(path)) or self.ref(path))
            if item.get('raw') and legend['text']:
                paired=path.with_suffix('.svg')
                if str(paired) in self.known_files:
                    saved=self.read(paired).strip();saved=re.sub(rb'^<\?xml[^>]*\?>\s*',b'',saved)
                    if saved==item['raw'].strip():self.entries[-1]['bound_representation']=self.ref(paired)
            if origin:
                self.entries[-1].update(preview=origin['path'],artifact_sha256=origin['sha256'],resource=origin)
    def zip(self,path,storage,owner,scope,raw=None,locator='zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(raw) if raw is not None else path) as z:
                names=set()
                for info in z.infolist():
                    if info.is_dir():continue
                    if info.filename in names:raise ValueError('Duplicate archive member: '+str(path))
                    names.add(info.filename)
                    if Path(info.filename).suffix.lower() not in VISUAL:continue
                    value=z.read(info)
                    self.inventory.append(dict(container=str(path),member=locator+'/'+info.filename,sha256=sha(value),bytes=len(value),role='archive artwork'))
                    self.add(path,storage,owner,scope,raw=value,ext=Path(info.filename).suffix.lower(),
                             title=info.filename,locator=locator+'/'+info.filename,family='Archive · '+path.stem,member=info.filename)
        except zipfile.BadZipFile as error:raise ValueError('Invalid archive: '+str(path)) from error
    def discover(self):
        self.discover_states();source_list=source_files(self.outputs,self.archive)
        self.known_files={str(p) for _,p in source_list};self.source_names={(storage,str(p)) for storage,p in source_list if p.name not in GENERATED}
        for storage,path in source_list:
            relative=path.relative_to(self.roots[storage]);owner,scope=location(path,self.roots[storage]);ext=path.suffix.lower()
            row=dict(path=str(path),storage=storage,relative_path=relative.as_posix(),role='supporting file')
            self.path_rows[str(path)]=row;self.inventory.append(row)
            if path.name.startswith('._') or path.name in GENERATED:
                row['role']='filesystem metadata' if path.name.startswith('._') else 'generated navigation';continue
            if str(path) in self.repair_originals:
                ref=self.ref(path);row.update(role='link-repair original with canonical reading derivative',sha256=ref['sha256'],bytes=ref['bytes'],reading_path=self.repair_originals[str(path)]['reading'])
                self.physical.setdefault(ref['sha256'],[]).append(str(path));continue
            if str(path) in self.reading_derivatives:
                record=self.derivative_records[str(path)];ref=self.ref(path)
                if ref['sha256']!=record['reading_sha256']:raise ValueError('Reading derivative advanced: '+str(path))
                original=self.ref(Path(record['original']))
                if original['sha256']!=record['original_sha256']:raise ValueError('Historical original changed')
                self.origins[str(path)]=original
                row['role']='authenticated reading derivative'
            if str(path) in self.reading_derivatives.values():
                raw=self.read(path);row.update(role='historical original with authenticated reading derivative',sha256=sha(raw),bytes=len(raw))
                self.physical.setdefault(sha(raw),[]).append(str(path));continue
            if ext in {'.json','.md'}:
                bound=self.ref(path);row.update(sha256=bound['sha256'],bytes=bound['bytes'])
            if ext not in VISUAL|{'.html','.zip'}:continue
            raw=self.read(path);row.update(sha256=sha(raw),bytes=len(raw));self.physical.setdefault(sha(raw),[]).append(str(path))
            if str(path) in self.review_companions:
                row['role']='selected HTML companion';continue
            role='supporting illustration' if supporting(relative) or 'studies' in relative.parts else 'version'
            row['role']=role
            if not owner:
                # A shared source stays shared; explicit image names/headings below
                # may establish its figure membership, never current authority.
                matches=re.findall(r'figure[-_ ]?0*([1-4])\b',relative.as_posix(),re.I)
                owner=sorted({int(n) for n in matches})
                if not owner:row['role']='shared/reference artwork'
            if ext in VISUAL:
                self.add(path,storage,owner,scope,role=role,source_ref=dict(path=str(path),sha256=sha(raw),bytes=len(raw)))
                if ext=='.svg' and len(owner)==1 and scope=='assembly' and role=='version':
                    for panel in panel_views(raw):
                        target_scope='panel-'+panel['panel'].lower()
                        self.add(path,storage,owner,target_scope,raw=panel['raw'],ext='.svg',
                                 title=path.stem+' · Panel '+panel['panel'],locator='layout-panel:'+panel['panel'])
                        entry=self.entries[-1];entry['scope_derivation']={k:v for k,v in panel.items() if k!='raw'}
                        selected=self.current.get((str(path),owner[0],'assembly'))
                        if selected and not any(f==owner[0] and sc==target_scope for _,f,sc in self.current):
                            entry.update(selected=True,status='Selected assembly · panel view',state=selected['state'],legend=self.legend(path,selected))
            elif ext=='.html':self.html(path,storage,owner,scope,raw,role)
            else:self.zip(path,storage,owner,scope)
        # Shared bundles record membership from their existing saved filenames and
        # headings, then link the one canonical collection from every owner.
        for entry in self.entries:
            if not entry['figures']:
                matches=re.findall(r'figure[-_ ]?0*([1-4])\b',entry['label'],re.I)
                entry['figures']=sorted({int(n) for n in matches})
            if not entry['figures'] and entry['role']=='version':
                entry['role']='shared/reference artwork'
        explanations=defaultdict(list)
        for entry in self.entries:
            binding=entry.get('resource') or entry.get('bound_representation')
            if binding and entry['legend']['text']:
                explanations[binding['path']].append(dict(entry['legend'],context_source=entry['source']))
        for entry in self.entries:
            if not entry['legend']['text'] and not entry['source'].get('locator'):
                candidates=explanations.get(entry['source']['path'],[])
                unique={c['text']:c for c in candidates}
                if len(unique)==1:entry['legend']=next(iter(unique.values()))
                elif unique:entry['related_explanations']=list(unique.values())
        # Active working files can legitimately advance. A browsing snapshot
        # keeps its observed pixels immutable alongside its observed legend.
        active_sources={p for p,_,_ in self.current}
        for entry in self.entries:
            if entry['preview'] in active_sources:
                raw=self.read(entry['preview'])
                if sha(raw)!=entry['artifact_sha256']:raise ValueError('Active preview source advanced')
                entry['preview']=self.materialize(raw,entry['format']);entry['snapshot_preview']=True
        # Reuse exact immutable payloads without merging version identities.
        for entry in self.entries:
            if not Path(entry['preview']).is_absolute() and not entry.get('snapshot_preview') and not entry['selected']:
                candidates=[p for p in self.physical.get(entry['artifact_sha256'],[]) if Path(p).suffix.lower()==entry['format']]
                if candidates:entry['preview']=min(candidates,key=lambda p:(not p.startswith(str(self.outputs)),len(p),p))
        used={e['preview'] for e in self.entries if not Path(e['preview']).is_absolute()}
        for key in list(self.payloads):
            if key not in used:Path(self.payloads.pop(key)['source']).unlink()
        self.reconcile()
        for entry in self.entries:
            if not entry['legend']['text'] and entry['role']=='version':
                self.issues.append(dict(source=entry['source']['path'],id=entry['id'],reason='No saved legend association'))
        # Duplicate labels retain separate identities and receive a short suffix.
        counts=Counter((tuple(e['figures']),e['scope'],e['family'],e['label']) for e in self.entries)
        for e in self.entries:
            if counts[(tuple(e['figures']),e['scope'],e['family'],e['label'])]>1:e['label']+=' · '+e['id'][:8]
        self.entries.sort(key=lambda e:(e['figures'] or [99],scope_key(e['scope']),natural(e['family']),natural(e['label']),e['id']))
        return dict(schema_version=1,versions=self.entries,inventory=self.inventory,issues=self.issues,
                    documented_gaps=[dict(figure=1,scope='panels-f-g-h',family='Scientific recipes',label='V3',reason='No established saved recipe/artwork; documented in the retained F–H comparison')],
                    coverage=self.coverage())
    def reconcile(self):
        if not self.migration:return
        # Member identities and historical source copies are accounted by exact
        # payload hash, while their full original origin stays in this ledger.
        for row in self.migration['files']:
            if Path(row['source']).suffix.lower() not in VISUAL|{'.html'}:continue
            item=dict(original=row['source'],sha256=row['sha256'],bytes=row['bytes'])
            if row.get('mutable_navigation') or Path(row['source']).name in GENERATED:
                item['role']='historical generated navigation'
            elif row['sha256'] in self.physical:
                item.update(role='authenticated retained identity',retained=self.physical[row['sha256']])
            elif row.get('action')=='delete':item.update(role='previous authorized disposition',reason=row.get('reason'))
            else:
                # Link-repaired readers keep their original in reading-originals,
                # but a moved historical path may need the authenticated resolver.
                from classical_conditioning.external_artifacts import resolve_artifact
                try:
                    p=resolve_artifact(row['source'],expected_sha256=row['sha256']);ref=self.ref(p)
                    item.update(role='authenticated recovery identity',retained=[str(p)])
                except (OSError,ValueError) as error:
                    item.update(role='unresolved',reason=str(error));self.issues.append(item)
            self.inventory.append(item)
        for row in self.migration.get('embedded',[]):
            if Path(row['member']).suffix.lower() not in VISUAL:continue
            item=dict(row)
            if row['sha256'] in self.physical:item.update(role='authenticated embedded alias',retained=self.physical[row['sha256']])
            elif any(e['artifact_sha256']==row['sha256'] for e in self.entries):item['role']='authenticated extracted embedded artwork'
            else:item['role']='unresolved embedded artwork';self.issues.append(item)
            self.inventory.append(item)
    def coverage(self):
        return dict(physical_files=len(self.path_rows),artwork_views=len(self.entries),
                    roles=dict(Counter(r['role'] for r in self.inventory)),issues=len(self.issues),
                    status='complete' if not self.issues else 'coverage exceptions require review',
                    scientific_selection_changed=False)

BROWSER_CSS = """
body{margin:0;background:#f7f5ef;color:#20272b;font:17px/1.5 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:24px}a{color:#125b76}h1{margin:.5em 0}
.controls{position:sticky;top:0;background:#f7f5ef;padding:12px 0;z-index:2;border-bottom:1px solid #d4d8d5}
.buttons{display:flex;flex-wrap:wrap;gap:7px;margin:8px 0}button,input{font:inherit}
button{padding:7px 12px;border:1px solid #9eaaa9;border-radius:6px;background:white;color:#20272b;cursor:pointer}
button[aria-pressed=true]{background:#125b76;color:white;border-color:#125b76}button:focus-visible{outline:3px solid #d58324;outline-offset:2px}
button:disabled{opacity:.45;cursor:default}.version-buttons{max-height:220px;overflow:auto;padding:3px}
.workspace{background:white;padding:20px;border:1px solid #d7dcd9;border-radius:8px;margin:18px 0}
#artwork img{display:block;max-width:100%;height:auto;margin:auto}#artwork iframe{width:100%;height:78vh;border:0}
#legend{white-space:pre-wrap;max-width:100ch}#status{font-size:.9em;color:#46595c}
details{margin:16px 0}summary{cursor:pointer}small{overflow-wrap:anywhere}input{padding:8px;max-width:90%;width:500px}
[hidden]{display:none!important}li{margin:.5em 0}@media(max-width:700px){main{padding:12px}.workspace{padding:12px}.version-buttons{max-height:170px}}
@media print{.controls,nav,details{display:none}.workspace{border:0;padding:0}body{background:white}#artwork img{width:auto;max-width:100%}}
"""

BROWSER_JS = r"""
const data=JSON.parse(document.getElementById('version-catalogue').textContent);
const scopes=document.getElementById('scopes'),families=document.getElementById('families'),versions=document.getElementById('versions');
const related=document.getElementById('related');
const art=document.getElementById('artwork'),legend=document.getElementById('legend'),title=document.getElementById('version-title'),status=document.getElementById('status'),sources=document.getElementById('sources');
const prev=document.getElementById('previous'),next=document.getElementById('next'),current=document.getElementById('current'),search=document.getElementById('search');
let scope='',family='',selected=null;
function button(label,pressed,action){const b=document.createElement('button');b.type='button';b.textContent=label;b.setAttribute('aria-pressed',String(pressed));b.onclick=action;return b;}
function available(){return data.versions.filter(v=>v.scope===scope&&v.family===family);}
function filtered(){const q=search.value.trim().toLowerCase();return available().filter(v=>(v.label+' '+v.status).toLowerCase().includes(q));}
function link(text,href,download=false){const a=document.createElement('a');a.textContent=text;a.href=href;if(download)a.setAttribute('download','');return a;}
function address(){const p=new URLSearchParams({scope,family});if(selected)p.set('version',selected.id);const hash='#'+p.toString();if(location.hash!==hash)history.pushState(null,'',hash);}
function render(push=true){
 scopes.replaceChildren(...data.scopes.map(s=>button(s.label,s.id===scope,()=>chooseScope(s.id))));
 const fs=[...new Set(data.versions.filter(v=>v.scope===scope).map(v=>v.family))];
 families.replaceChildren(...fs.map(f=>button(f,f===family,()=>{family=f;selected=available().find(v=>v.selected)||null;render();})));
 versions.replaceChildren(...filtered().map(v=>{const b=button(v.label,selected?.id===v.id,()=>{selected=v;render();});b.dataset.version=v.id;return b;}));
 art.replaceChildren();sources.replaceChildren();legend.textContent='';
 related.replaceChildren(...(data.scopes.find(s=>s.id===scope)?.related||[]).map(s=>button('Open joint '+data.scopes.find(x=>x.id===s).label,false,()=>chooseScope(s))));
 if(selected){
  title.textContent=selected.label;status.textContent=selected.status+' · '+selected.scope_label;
  if(selected.format==='.pdf'){const el=document.createElement('iframe');el.title=selected.label;el.src=selected.url;art.append(el);}
  else{const el=document.createElement('img');el.alt=selected.label;el.src=selected.url;art.append(el);}
  legend.textContent=selected.legend.text||'No saved legend association was recorded for this view. The exact original is linked below.';
  sources.append(link('Original source',selected.source_url),document.createTextNode(' · '),link('Download saved artwork',selected.url,true));
  if(selected.legend_url)sources.append(document.createTextNode(' · '),link('Legend source',selected.legend_url));
  const small=document.createElement('small');small.textContent=' Source identity '+selected.source.sha256+'; artwork '+selected.artifact_sha256+'.';sources.append(document.createElement('br'),small);
 }else{title.textContent='Choose a saved version';status.textContent='No working version is selected for this family. Browsing does not select science.';}
 const list=filtered(),i=selected?list.findIndex(v=>v.id===selected.id):-1;prev.disabled=i<=0;next.disabled=i<0||i>=list.length-1;
 current.disabled=!data.versions.some(v=>v.scope===scope&&v.selected);
 document.getElementById('count').textContent=filtered().length+' version buttons in this family';
 if(push)address();
}
function chooseScope(s){scope=s;const vs=data.versions.filter(v=>v.scope===scope);selected=vs.find(v=>v.selected)||null;family=selected?.family||vs[0]?.family||'';search.value='';render();}
function move(n){const list=filtered(),i=selected?list.findIndex(v=>v.id===selected.id):-1;if(i>=0&&i+n>=0&&i+n<list.length){selected=list[i+n];render();versions.querySelector('[aria-pressed=true]')?.focus();}}
prev.onclick=()=>move(-1);next.onclick=()=>move(1);
current.onclick=()=>{selected=data.versions.find(v=>v.scope===scope&&v.selected)||null;if(selected)family=selected.family;search.value='';render();};
search.oninput=()=>render(false);
document.addEventListener('keydown',e=>{if(e.altKey||e.ctrlKey||e.metaKey||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();move(e.key==='ArrowRight'?1:-1);}});
function restore(){const p=new URLSearchParams(location.hash.slice(1));scope=data.scopes.some(s=>s.id===p.get('scope'))?p.get('scope'):(data.scopes[0]?.id||'assembly');
 const vs=data.versions.filter(v=>v.scope===scope),wanted=vs.find(v=>v.id===p.get('version'));
 selected=wanted||(!p.has('family')?vs.find(v=>v.selected):null)||null;
 family=selected?.family||(vs.some(v=>v.family===p.get('family'))?p.get('family'):vs[0]?.family||'');render(false);}
window.addEventListener('popstate',restore);window.addEventListener('hashchange',restore);restore();
"""


def scope_label(value):
    if value=='assembly':return 'Assembly'
    if value.startswith('panel-'):return 'Panel '+value[6:].upper()
    if value.startswith('panels-'):return 'Panels '+ '–'.join(value[7:].upper().split('-'))
    return 'Shared work and references'


def figure_page(number, catalogue, outputs, *, reading_html='', figure_links=()):
    page_path=outputs/f'figures/figure-{number:02}/INDEX.html' if number else outputs/'shared/INDEX.html'
    def url(path):return quote(Path(os.path.relpath(path,page_path.parent)).as_posix(),safe='/._-')
    entries=[]
    for entry in catalogue['versions']:
        if (number and number not in entry['figures']) or (not number and entry['figures']):continue
        e=dict(entry);preview=Path(e['preview'])
        if not preview.is_absolute():preview=outputs/'review-library'/preview
        e.update(url=url(preview),source_url=url(Path(e['source']['path'])),scope_label=scope_label(e['scope']),
                 legend_url=url(Path(e['legend']['path'])) if e['legend'].get('path') else None)
        entries.append(e)
    scope_set={e['scope'] for e in entries}
    for s in list(scope_set):
        if s.startswith('panels-'):scope_set.update('panel-'+c for c in s[7:].split('-'))
    scopes=sorted(scope_set,key=scope_key)
    metadata=dict(figure=number,scopes=[dict(id=s,label=scope_label(s),related=[g for g in scopes if s.startswith('panel-') and g.startswith('panels-') and s[6:] in g[7:].split('-')]) for s in scopes],versions=entries)
    nav=' · '.join(f'<a href="{url(outputs/f"figures/figure-{n:02}/INDEX.html")}">Figure {n}</a>' for n in figure_links)
    # All files remain discoverable, including sources that are not artwork versions.
    inventory=[]
    for row in catalogue['inventory']:
        if not row.get('path') or 'relative_path' not in row:continue
        p=Path(row['path'])
        if number and f'figure-{number:02}' not in p.parts:continue
        if not number and 'shared' not in p.parts:continue
        inventory.append('<li><a href="'+url(p)+'">'+html.escape(row['relative_path'])+'</a> <small>'+html.escape(row['role'])+'</small></li>')
    gaps=[r for r in catalogue['documented_gaps'] if r['figure']==number]
    gap_html=''.join('<li>'+html.escape(r['label']+': '+r['reason'])+'</li>' for r in gaps)
    issues=[]
    entryids={r['id'] for r in entries}
    for row in catalogue['issues']:
        if row.get('id') in entryids or f'figure-{number:02}' in row.get('source',row.get('original','')):
            issues.append('<li>'+html.escape(row.get('reason',row.get('role','Unresolved'))+': '+row.get('source',row.get('original',''))+((' #'+row['locator']) if row.get('locator') else ''))+'</li>')
    encoded=json.dumps(metadata,ensure_ascii=False).replace('<','\\u003c')
    page_title='Figure '+str(number) if number else 'Shared and reference artwork'
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{page_title} · versions and panels</title><style>{BROWSER_CSS}</style>
<main><nav><a href="{url(outputs/'START-HERE.html')}">All figures</a> · {nav}</nav><h1>{page_title}</h1>
<p>Choose a scope, family and saved version. One version is shown at a time; current artwork remains controlled by scoped repository state.</p>
<div class="controls"><div id="scopes" class="buttons" role="group" aria-label="Figure scope"></div><div id="families" class="buttons" role="group" aria-label="Version family"></div>
<label>Find a version <input id="search" type="search"></label><div id="versions" class="buttons version-buttons" role="group" aria-label="Saved versions"></div>
<button id="previous" type="button">Previous version</button> <button id="next" type="button">Next version</button> <button id="current" type="button">Selected working view</button> <small id="count" aria-live="polite"></small></div>
<div id="related" class="buttons" aria-label="Related joint-panel versions"></div><section class="workspace" aria-label="Selected version"><h2 id="version-title"></h2><p id="status"></p><div id="artwork"></div><h3>Associated legend / saved context</h3><div id="legend"></div><details><summary>Original and downloads</summary><p id="sources"></p></details></section>
{reading_html}<details><summary>All retained files for this figure</summary><ul>{''.join(inventory)}</ul></details>
<details><summary>Coverage and documented gaps ({len(issues)} exceptions)</summary><ul>{gap_html}{''.join(issues)}</ul><p><a href="{url(outputs/'review-library/versions.json')}">Complete version and source inventory</a></p></details>
<script type="application/json" id="version-catalogue">{encoded}</script><script>{BROWSER_JS}</script></main></html>'''
    return content.encode()
