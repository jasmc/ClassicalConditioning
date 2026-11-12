"""Scoped presentation-only CS guide emphasis in selected Figure 1/2 drafts."""
from pathlib import Path
import sys,json,re,copy,xml.etree.ElementTree as ET
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());sys.path[:0]=[str(ROOT/'src')]
from classical_conditioning.figure_workspace import Snapshot,sha,publish_state,measured_presentation
from classical_conditioning.figures.review_html import svg_editing_page
from classical_conditioning.figures.presentation_profiles import load_presentation_profile

def encode(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def ref(p):return {'path':str(p.resolve()),'sha256':sha(p.read_bytes())}
def paths(f):
 state=ROOT/f'docs/paper/figures/figure-{f}/current-state.json'
 out=Path(f'/Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-{f}/assembly/versions/figure-{f}-cs-guides-20261031')
 return state,out

def prepare(f):
 state,out=paths(f);assert out.parent.is_dir()
 s=json.loads(state.read_bytes());d=s['draft'];svg=Path(d['working_svg']['path']);html=Path(d['review']['path']);legend=Path(d['legend']['path'])
 for p,k in [(svg,'working_svg'),(html,'review'),(legend,'legend')]:assert ref(p)['sha256']==d[k]['sha256']
 regref=d['semantic_registry'];regpath=Path(regref['path']) if 'path' in regref else None
 if regpath:assert ref(regpath)['sha256']==regref['sha256']
 snap=Snapshot([state,svg,html,legend,Path(__file__),*([regpath] if regpath else [])]);reg=json.loads(regpath.read_bytes()) if regpath else copy.deepcopy(regref)
 profile=load_presentation_profile('nature-main')
 root=ET.fromstring(svg.read_bytes());old=copy.deepcopy(root);changes=[];identities=[]
 for n in root.iter():
  identity=n.get('id','')
  group=('heatmap__cs-onset' in identity or 'heatmap__cs-offset' in identity)
  direct=('heatmap__vigor-cells-CS-onset' in identity or 'heatmap__vigor-cells-CS-offset' in identity)
  if not (group or direct):continue
  identities.append(identity)
  targets=[n] if direct else [x for x in n.iter() if x.tag.endswith(('path','line'))]
  for x in targets:
   i=list(root.iter()).index(x)
   style=x.get('style','');style=re.sub(r'(?:^|;)\s*(?:stroke|stroke-width|stroke-opacity|vector-effect)\s*:[^;]*','',style)
   x.set('style',style+';stroke:rgb(13,129,54) !important;stroke-opacity:1 !important;stroke-width:1.3333333333333333px !important;vector-effect:non-scaling-stroke !important;')
   # C's inherited tuple-valued stroke is invalid SVG; replace explicitly.
   if direct:x.set('stroke','rgb(13,129,54)')
   changes.append({'node_index':i,'guide_id':identity,'attributes':['style']+(['stroke'] if direct else [])})
 assert len(identities)==(6 if f=='01' else 12)
 check=copy.deepcopy(root);nodes=list(check.iter());before=list(old.iter())
 for c in changes:
  for key in c['attributes']:
   value=before[c['node_index']].get(key)
   if value is None:nodes[c['node_index']].attrib.pop(key,None)
   else:nodes[c['node_index']].set(key,value)
 assert ET.tostring(check)==ET.tostring(old),'Unauthorized artwork change'
 w=float(root.get('width').removesuffix('mm'));raw=ET.tostring(root,encoding='utf-8');measured=measured_presentation(raw,width_mm=w)
 csstrokes=[r for r in measured['strokes'] if any(i in r['context'] for i in identities)]
 assert len(csstrokes)==len(changes)
 assert all(abs(r['normal_width_pt']-1)<1e-10 and r['effective_opacity']==1 for r in csstrokes)
 evidence={'scope':'CS onset/offset guides in heatmaps only','width_pt':1,'stroke_opacity':1,'color':'rgb(13,129,54)','onset':'solid, unchanged timing','offset':'existing dash pattern preserved','guide_ids':identities,'authorized_attribute_changes':changes,'all_other_svg_bytes_preserved':True,'raster_payloads_preserved':True,'native_width_mm':w,'source':ref(svg),'scientific_data_recomputed':False}
 reg['cs_guide_emphasis_20261031']=evidence
 out.mkdir(exist_ok=False)
 for name,data in [('figure.svg',raw),('figure.html',svg_editing_page(raw,scope_label=f'Figure {int(f)}',legend=legend.read_text()).encode()),('legend.md',legend.read_bytes()),('semantic-registry.json',encode(reg)),('effective-presentation.json',encode(measured)),('evidence.json',encode(evidence)),('starting-state.json',state.read_bytes()),('renderer-source.py',Path(__file__).read_bytes())]:(out/name).write_bytes(data)
 snap.include([p for p in out.iterdir() if p.is_file()]);snap.check();(out/'starting-hashes.json').write_bytes(encode(snap.hashes));print(out)

def publish(f):
 state,out=paths(f);snap=Snapshot([]);snap.hashes=json.loads((out/'starting-hashes.json').read_text());snap.include([out/'visual-review.json']);snap.check();v=json.loads((out/'visual-review.json').read_text());assert v['passed'] and v['svg_sha256']==ref(out/'figure.svg')['sha256']
 s=json.loads(state.read_bytes());d=s['draft'];d['working_svg']=ref(out/'figure.svg');d['review']=ref(out/'figure.html');d['legend']=ref(out/'legend.md');d['semantic_registry']=ref(out/'semantic-registry.json');d['effective_presentation']=ref(out/'effective-presentation.json')
 d.setdefault('consistency_requests',[]).append({'id':'heatmap-CS-guide-emphasis-20261031','scope':'heatmap CS guides','properties':{'width_pt':1,'stroke_opacity':1},'status':'verified for scoped guides; inherited broader coverage unchanged','evidence':ref(out/'evidence.json'),'visual_review':ref(out/'visual-review.json')})
 d['cs_guide_emphasis']={'evidence':ref(out/'evidence.json'),'previous_state':ref(out/'starting-state.json')}
 publish_state(state,s,snap);print('Published',f)
if __name__=='__main__':
 for f in ['01','02']:(prepare if sys.argv[1]=='prepare' else publish)(f)
