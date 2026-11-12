"""Prepare/guardedly select the author-approved Figure 1 robust heatmap update."""
from pathlib import Path
import sys,json,io,base64,copy,re,hashlib
import xml.etree.ElementTree as ET
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import pandas as pd
import matplotlib as mpl
from PIL import Image
from classical_conditioning.figure_workspace import Snapshot,publish_state,sha
from classical_conditioning.figures.review_html import svg_editing_page
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme
STATE=ROOT/'docs/paper/figures/figure-01/current-state.json'
LEGEND=STATE.parent/'legend.md'
OUT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-01/assembly/versions/figure-01-robust-baseline-20261031')
XL='{http://www.w3.org/1999/xlink}href'
def ref(p):return {'path':str(p.resolve()),'sha256':sha(p.read_bytes())}
def encode(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def sampling(old,values,cmap):
 h,w=old.shape[:2];colors=cmap(mpl.colors.Normalize(-.25,.25)(values[::-1]),bytes=True)
 ys=np.floor((np.arange(90)+.5)*h/90).astype(int);xs=np.floor((np.arange(40)+.5)*w/40).astype(int)
 cols=np.all(old[ys,:,None,:]==colors[:,None,:,:],axis=3).sum(axis=0).argmax(axis=1)
 rows=np.all(old[:,xs,None,:]==colors.transpose(1,0,2)[None,:,:,:],axis=3).sum(axis=1).argmax(axis=1)
 assert np.array_equal(old,colors[rows[:,None],cols[None,:]]),'Original raster does not match selected source'
 return rows,cols

def prepare():
 assert OUT.parent.is_dir(),'External storage unavailable'
 s=json.loads(STATE.read_bytes());d=s['draft'];svg=Path(d['working_svg']['path']);html=Path(d['review']['path'])
 for p,k in [(svg,'working_svg'),(html,'review')]:assert ref(p)['sha256']==d[k]['sha256']
 candidate=Path(d['semantic_registry']['FGH_frozen_registry']['path']);assert ref(candidate)['sha256']==d['semantic_registry']['FGH_frozen_registry']['sha256']
 c=json.loads(candidate.read_text());csv=candidate.parent/Path(c['data_artifacts'][0]['path']).name
 assert ref(csv)['sha256']==c['data_artifacts'][0]['sha256']
 hdata=Path(d['scientific_example_selection']['data']['path']);assert ref(hdata)['sha256']==d['scientific_example_selection']['data']['sha256']
 decision=ROOT/'docs/paper/decisions.md'
 snap=Snapshot([STATE,LEGEND,svg,html,candidate,csv,hdata,decision,Path(__file__)])
 root=ET.fromstring(svg.read_bytes());lookup={n.get('id'):n for n in root.iter()}
 protected={k:sha(ET.tostring(lookup[k])) for k in ['panel-A','panel-B','panel-C','panel-D']}
 frames={n.get('id'):dict(n.attrib) for n in root.iter() if n.get('id','').startswith('layout-data-')}
 raw=pd.read_csv(csv);hd=pd.read_parquet(hdata)
 matrices={};tables=[];evidence={'recipe':'baseline-mad-robust-z-v1','baseline_s':[-15,0],'colour_limits':[-2,2],'sources':[ref(csv),ref(hdata)],'panels':{},'protected_subtrees':protected,'linked_E_bins':0,'scientific_status':'author-approved method; unfrozen scientific draft'}
 profile=load_presentation_profile('nature-main')
 with mpl.rc_context():
  theme=apply_theme(profile=profile);cmap=mpl.colormaps[theme.single_fish_scaled_vigor_cmap].copy();cmap.set_bad('black')
  for sourceid,fish,display,table in [('h','20221115_09','F',raw),('f','20221115_07','G',raw),('g','20230419_10','H',hd)]:
   sub=table.loc[table.fish.eq(fish)].copy();m=sub.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
   baseline=m[:,5:20];b=np.nanmedian(baseline,axis=1);scale=np.nanmedian(np.abs(baseline-b[:,None]),axis=1)/.6744897501960817
   valid=np.isfinite(scale)&(scale>0);z=np.full_like(m,np.nan);z[valid]=(m[valid]-b[valid,None])/scale[valid,None]
   assert np.allclose(np.nanmedian(z[valid,5:20],axis=1),0,atol=1e-12)
   assert np.allclose(np.nanmedian(np.abs(z[valid,5:20]),axis=1)/.6744897501960817,1,atol=1e-12)
   image=lookup[f'edit-FGH-fig1__{sourceid}__heatmap__vigor-cells'];old=np.asarray(Image.open(io.BytesIO(base64.b64decode(image.get(XL).split(',',1)[1]))).convert('RGBA'))
   rows,cols=sampling(old,m,cmap);rgba=cmap(mpl.colors.Normalize(-2,2)(np.ma.masked_invalid(z[::-1])),bytes=True)
   pixels=rgba[rows[:,None],cols[None,:]];buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='PNG');image.set(XL,'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
   image.set('data-normalization','baseline-mad-robust-z-v1');matrices[display]=z
   evidence['panels'][display]={'fish':fish,'source_raster_authenticated':True,'undefined_rows':np.flatnonzero(~valid).tolist(),'minimum_baseline_bins':int(np.isfinite(baseline).sum(axis=1).min()),'normalized_matrix_sha256':sha(np.asarray(z,dtype='<f8').tobytes()),'sampling_preserved':True}
   sub['robust_z']=z.reshape(-1);tables.append(sub)
   evidence['panels'][display]['scales']=scale.tolist()
  for node in root.iter():
   match=re.fullmatch(r'edit-E-trial-(\d+)-bin-(\d+)',node.get('id',''))
   if match:
    trial,binid=map(int,match.groups());v=matrices['G'][trial-5,binid];rgba=cmap(mpl.colors.Normalize(-2,2)(np.ma.masked_invalid(v)),bytes=True)
    node.set('fill','#'+''.join(f'{int(x):02x}' for x in rgba[:3]));evidence['linked_E_bins']+=1
  assert evidence['linked_E_bins']==200
  for node in root.iter():
   if node.tag.endswith('text'):
    t=''.join(node.itertext())
    if t in ['-0.25','−0.25','0.25']:node.text='−2' if t.startswith(('-','−')) else '2'
    if t=='Log vigor relative to baseline':node.text='Robust z-score'
 for k,h in protected.items():assert sha(ET.tostring(lookup[k]))==h
 assert frames=={n.get('id'):dict(n.attrib) for n in root.iter() if n.get('id','').startswith('layout-data-')}
 # All non-band E nodes retain their source geometry and bytes.
 orig=ET.fromstring(svg.read_bytes());oldlookup={n.get('id'):n for n in orig.iter()}
 for n in lookup['layout-E'].iter():
  if n.get('id','').startswith('edit-E-trial-') and '-bin-' in n.get('id',''):continue
  if n.tag.endswith(('path','line','text')):assert ET.tostring(n)==ET.tostring(oldlookup[n.get('id')])
 meta=ET.SubElement(root,'{http://www.w3.org/2000/svg}metadata',{'id':'robust-baseline-normalization-20261031'});meta.text=json.dumps(evidence,allow_nan=False)
 legend=LEGEND.read_text().replace('No amplitude scaling is applied. Colours saturate at ±0.25 log units; black denotes missing data.','Within each fish/trial, divide the centred bins by MAD / 0.6744897501960817, where MAD is the median absolute deviation of finite baseline-bin values over [−15, 0) s. Baseline median is zero and normal-calibrated MAD is one. Robust z-scores are dimensionless; colours saturate at ±2 without numerical clipping. Missing values and rows with undefined/nonpositive baseline spread are black.').replace('Colour bands copy the exact 1-s heatmap rows in g; their logarithmic normalization differs from the black traces.','Colour bands copy the exact robust-normalized 1-s heatmap rows in g; their normalization differs from the black traces.')
 legend+='\nThe E colour bands and F–H heatmaps implement the author-approved 31 October robust-normalization decision as an unfrozen scientific draft; previous scoped freezes remain preserved.\n'
 data=ET.tostring(root,encoding='utf-8');page=svg_editing_page(data,scope_label='Figure 1',legend=legend).encode()
 OUT.mkdir(exist_ok=False)
 for name,bytes_ in [('figure-01.svg',data),('figure-01.html',page),('legend.md',legend.encode()),('evidence.json',encode(evidence)),('starting-state.json',STATE.read_bytes()),('previous-legend.md',LEGEND.read_bytes()),('starting-hashes.json',encode(snap.hashes))]: (OUT/name).write_bytes(bytes_)
 pd.concat(tables).to_parquet(OUT/'normalized-bins.parquet',index=False)
 snap.check();print(OUT)

def publish():
 snap=Snapshot([]);snap.hashes=json.loads((OUT/'starting-hashes.json').read_text());snap.check()
 visual=json.loads((OUT/'visual-review.json').read_text());assert visual['svg_sha256']==ref(OUT/'figure-01.svg')['sha256'] and visual['passed']
 s=json.loads(STATE.read_bytes());d=s['draft'];d['working_svg']=ref(OUT/'figure-01.svg');d['review']=ref(OUT/'figure-01.html');d['legend']=ref(OUT/'legend.md');d['status']='Unfrozen scientific draft; F–H and linked E bands use author-approved robust baseline normalization'
 d['robust_heatmap_normalization']={'recipe':'baseline-mad-robust-z-v1','evidence':ref(OUT/'evidence.json'),'data':ref(OUT/'normalized-bins.parquet'),'visual_review':ref(OUT/'visual-review.json'),'previous_state':ref(OUT/'starting-state.json'),'scope':['F','G','H','E colour bands'],'scientific_freezes_replaced':False}
 d['semantic_registry']['current_normalization_override']=d['robust_heatmap_normalization'];d['semantic_registry']['scope']+='; robust normalization overrides historical heatmap units/scales for F–H and E bands; see current_normalization_override'
 publish_state(STATE,s,snap,evidence_updates=[(LEGEND,(OUT/'legend.md').read_bytes())]);print('Published',d['review']['path'])
if __name__=='__main__':(prepare if sys.argv[1]=='prepare' else publish)()
