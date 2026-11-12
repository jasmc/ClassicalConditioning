"""Update only Figure 2 A-C to the approved pooled-baseline MAD recipe."""
from pathlib import Path
import sys,json,io,base64,copy
import xml.etree.ElementTree as ET
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
import matplotlib as mpl
from PIL import Image
from classical_conditioning.external_artifacts import resolve_artifact
from classical_conditioning.figure_workspace import Snapshot,publish_state,sha,measured_presentation
from classical_conditioning.figures.review_html import svg_editing_page
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme
from scripts.paper.figure_01.panels_f_g_h.apply_robust_normalization import sampling
STATE=ROOT/'docs/paper/figures/figure-02/current-state.json';LEGEND=STATE.parent/'legend.md';README=STATE.parent/'README.md';METHODS=STATE.parent/'current-methods.md'
OUT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-02/assembly/versions/figure-02-robust-baseline-20261031')
XL='{http://www.w3.org/1999/xlink}href'
def ref(p):return {'path':str(p.resolve()),'sha256':sha(p.read_bytes())}
def encode(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def resolve(r):return resolve_artifact(r['path'],expected_sha256=r['sha256'],allow_historical=True)
def findrefs(x,suffix):
 found=[]
 if isinstance(x,dict):
  if str(x.get('path','')).endswith(suffix):found.append(x)
  for v in x.values():found+=findrefs(v,suffix)
 elif isinstance(x,list):
  for v in x:found+=findrefs(v,suffix)
 return found

def prepare():
 assert OUT.parent.is_dir()
 s=json.loads(STATE.read_bytes());d=s['draft'];svg=resolve(d['working_svg']);html=resolve(d['review']);registry=resolve(d['semantic_registry']);reg=json.loads(registry.read_bytes())
 selection=ROOT/'configs/paper-figures/selections/figure2-A-pooled-freeze-20261014.json';aselect=json.loads(selection.read_text())
 ap=resolve(aselect['scientific_definition']['data']);completion=resolve(d['matched_trace3_scientific_draft']['statistics']);bref=findrefs(json.loads(completion.read_text()),'B-pooled-bins.parquet')[0];bp=resolve(bref)
 cp=resolve(findrefs(reg,'C-pooled-bins.parquet')[0]);paths=[ap,bp,cp]
 snap=Snapshot([STATE,LEGEND,README,METHODS,svg,html,registry,selection,completion,*paths,Path(__file__),ROOT/'docs/paper/decisions.md'])
 root=ET.fromstring(svg.read_bytes());lookup={n.get('id'):n for n in root.iter()}
 protected={k:sha(ET.tostring(lookup[k])) for k in ['panel-D','panel-E','panel-F','panel-G','panel-H','panel-I']}
 frameattrs={n.get('id'):dict(n.attrib) for n in root.iter() if n.get('id','').startswith(('layout-data-','layout-frame-'))}
 mappings={};tables=[];scales=[];diagnostics={}
 profile=load_presentation_profile('nature-main')
 with mpl.rc_context():
  theme=apply_theme(profile=profile);cmap=mpl.colormaps[theme.single_fish_scaled_vigor_cmap].copy();cmap.set_bad('black')
  for letter,path in zip('ABC',paths):
   tab=pd.read_parquet(path);assert not tab.duplicated(['condition_id','acquisition_trial','bin_index']).any()
   for i,condition in enumerate(['control','delay' if letter=='A' else 'trace']):
    sub=tab.loc[tab.condition_id.eq(condition)].copy().sort_values(['acquisition_trial','bin_index']);m=sub.pivot(index='acquisition_trial',columns='bin_index',values='mean_delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
    b=np.nanmedian(m[:,5:20],axis=1);scale=np.nanmedian(abs(m[:,5:20]-b[:,None]),axis=1)/.6744897501960817;valid=np.isfinite(scale)&(scale>0);z=np.full_like(m,np.nan);z[valid]=(m[valid]-b[valid,None])/scale[valid,None]
    assert np.allclose(np.nanmedian(z[valid,5:20],axis=1),0,atol=1e-12)
    assert np.allclose(np.nanmedian(abs(z[valid,5:20]),axis=1)/.6744897501960817,1,atol=1e-12)
    assert np.array_equal(np.isnan(m[valid]),np.isnan(z[valid]))
    key=f'fig2__{letter.lower()}{i}__heatmap__vigor-cells';image=lookup[key];old=np.asarray(Image.open(io.BytesIO(base64.b64decode(image.get(XL).split(',',1)[1]))).convert('RGBA'))
    if letter=='C':
     rgba=cmap(mpl.colors.Normalize(-.25,.25)(np.ma.masked_invalid(m)),bytes=True);assert np.array_equal(old,rgba)
     pixels=cmap(mpl.colors.Normalize(-2,2)(np.ma.masked_invalid(z)),bytes=True)
    else:
     rows,cols=sampling(old,m,cmap);rgba=cmap(mpl.colors.Normalize(-2,2)(np.ma.masked_invalid(z[::-1])),bytes=True);pixels=rgba[rows[:,None],cols[None,:]]
    buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='PNG');image.set(XL,'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode());image.set('data-normalization','baseline-mad-robust-z-v1')
    mappings[key]={'element_id':key,'panel':letter,'condition':condition,'source':ref(path),'quantity':'robust z-score of pooled log-vigor difference','units':'dimensionless','recipe':'baseline-mad-robust-z-v1','baseline_s':[-15,0],'limits':[-2,2],'matrix_sha256':sha(np.asarray(z,dtype='<f8').tobytes())}
    sub['robust_z']=z.reshape(-1);sub['panel']=letter;tables.append(sub)
    for trial,med,sd in zip(range(5,95),b,scale):scales.append({'panel':letter,'condition':condition,'acquisition_trial':trial,'baseline_median':med,'normal_calibrated_MAD':sd})
    diagnostics[key]={'undefined_display_trials':(np.flatnonzero(~valid)+1).tolist(),'minimum_finite_baseline_bins':int(np.isfinite(m[:,5:20]).sum(axis=1).min()),'source_raster_authenticated':True,'saturation_fraction':float(np.nanmean(abs(z)>2))}
  for n in lookup['panel-ABC'].iter():
   if n.tag.endswith('text'):
    t=''.join(n.itertext())
    if t in ['-0.25','−0.25','0.25']:n.text='−2' if t.startswith(('-','−')) else '2'
    if t=='Log vigor relative to baseline':n.text='Robust z-score'
 for k,h in protected.items():assert sha(ET.tostring(lookup[k]))==h
 assert frameattrs=={n.get('id'):dict(n.attrib) for n in root.iter() if n.get('id','').startswith(('layout-data-','layout-frame-'))}
 evidence={'recipe':'baseline-mad-robust-z-v1','scope':['A','B','C'],'normalization_order':'individual fish/trial centring -> equal-fish mean -> pooled condition/trial baseline median/MAD normalization','sources':[ref(p) for p in paths],'elements':mappings,'diagnostics':diagnostics,'protected_D_to_I_subtrees':protected,'geometry_preserved':True,'freeze_replaced':False}
 meta=ET.SubElement(root,'{http://www.w3.org/2000/svg}metadata',{'id':'robust-baseline-normalization-20261031'});meta.text=json.dumps(evidence,allow_nan=False)
 legend=LEGEND.read_text().replace('then averaged with equal fish weight. Colours saturate at ±0.25; black is missing.','then averaged with equal fish weight. Separately for each pooled condition/trial row, subtract the median of its finite pooled baseline bins over [−15,0) s and divide by MAD / 0.6744897501960817, where MAD is their median absolute deviation. These robust z-scores are dimensionless: baseline median = 0 and normal-calibrated MAD = 1. Colours saturate at ±2 without numerical clipping; black is missing or undefined baseline spread.')
 assert legend!=LEGEND.read_text()
 legend+='\nA–C now implement the author-approved robust heatmap normalization as an unfrozen scientific draft; earlier scoped freezes remain preserved. D–I outcomes and statistics are unchanged.\n'
 svgbytes=ET.tostring(root,encoding='utf-8');page=svg_editing_page(svgbytes,scope_label='Figure 2',legend=legend).encode();OUT.mkdir(exist_ok=False)
 for name,data in [('figure-02.svg',svgbytes),('figure-02.html',page),('legend.md',legend.encode()),('evidence.json',encode(evidence)),('starting-state.json',STATE.read_bytes()),('previous-legend.md',LEGEND.read_bytes()),('renderer-source.py',Path(__file__).read_bytes())]: (OUT/name).write_bytes(data)
 pd.concat(tables).to_parquet(OUT/'normalized-bins.parquet',index=False);pd.DataFrame(scales).to_csv(OUT/'baseline-scales.csv',index=False)
 reg['robust_baseline_normalization']=evidence
 def update(x):
  if isinstance(x,dict):
   if x.get('element_id') in mappings:x.update(mappings[x['element_id']]);x['data']=ref(OUT/'normalized-bins.parquet')
   for v in list(x.values()):update(v)
  elif isinstance(x,list):
   for v in x:update(v)
 update(reg);(OUT/'semantic-registry.json').write_bytes(encode(reg));(OUT/'effective-presentation.json').write_bytes(encode(measured_presentation(svgbytes,width_mm=136)))
 snap.include([p for p in OUT.iterdir() if p.is_file()]);snap.check();(OUT/'starting-hashes.json').write_bytes(encode(snap.hashes));print(json.dumps(diagnostics,indent=2))

def publish():
 snap=Snapshot([]);snap.hashes=json.loads((OUT/'starting-hashes.json').read_text());snap.include([OUT/'visual-review.json']);snap.check();visual=json.loads((OUT/'visual-review.json').read_text());assert visual['passed'] and visual['svg_sha256']==ref(OUT/'figure-02.svg')['sha256']
 s=json.loads(STATE.read_bytes());d=s['draft'];d['working_svg']=ref(OUT/'figure-02.svg');d['review']=ref(OUT/'figure-02.html');d['legend']=ref(OUT/'legend.md');d['legend']['panel_ids']=list('ABCDEFGHI');d['semantic_registry']=ref(OUT/'semantic-registry.json');d['effective_presentation']=ref(OUT/'effective-presentation.json');d['status']='Unfrozen scientific draft; A–C robust baseline z-scores after equal-fish pooling; D–I unchanged'
 d['robust_heatmap_normalization']={'recipe':'baseline-mad-robust-z-v1','evidence':ref(OUT/'evidence.json'),'data':ref(OUT/'normalized-bins.parquet'),'visual_review':ref(OUT/'visual-review.json'),'previous_state':ref(OUT/'starting-state.json'),'scope':list('ABC'),'scientific_freezes_replaced':False}
 if 'presentation_lock' in s:s['presentation_lock']={'status':'unlocked-new-scientific-draft','previous_lock':s['presentation_lock']}
 readme=README.read_text().replace('## Panel composition','## Current robust-normalized draft\n\nThe 31 October working assembly applies the [standing robust normalization decision](../../decisions.md#2026-10-31--robust-baseline-normalization-for-all-heatmaps)\nto A–C after equal-fish pooling, with shared dimensionless colour limits ±2.\nD–I outcomes/statistics are unchanged. Earlier scoped freezes remain immutable;\nthe new A heatmap is an unfrozen scientific derivative, not a replacement freeze.\n\n## Panel composition').replace('pooled with equal fish weight; managua_r at −0.25 to +0.25','pooled with equal fish weight, then robust baseline z-scored; managua_r at −2 to +2').replace('A remains frozen. B/C now contain','The accepted original A remains frozen; its current robust-normalized derivative is unfrozen. B/C contain')
 methods=METHODS.read_text();anchor='## Outcome, aggregation and uncertainty';methods=methods.replace(anchor,'## Heatmap normalization (author direction, 31 October 2026)\n\nA–C retain individually baseline-centred 1-s log-vigor bins and equal-fish\narithmetic means. Each pooled condition/trial row then follows the\n[approved median/MAD robust recipe](../../decisions.md#2026-10-31--robust-baseline-normalization-for-all-heatmaps)\nover [−15,0) s. Shared colour limits are ±2 dimensionless robust z-score units.\nState binds the saved normalized data and verification. All six pooled\nheatmaps have defined baseline scales. D–I retain their existing outcomes,\nunits, inference and source bytes; accepted historical A is preserved.\n\n'+anchor)
 publish_state(STATE,s,snap,evidence_updates=[(LEGEND,(OUT/'legend.md').read_bytes()),(README,readme.encode()),(METHODS,methods.encode())]);print('Published',d['review']['path'])
if __name__=='__main__':(prepare if sys.argv[1]=='prepare' else publish)()
