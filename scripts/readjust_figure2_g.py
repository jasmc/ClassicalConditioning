"""G draft: native D/Figure-1 plot typography and strokes, compact stats only.

Saved SVG coordinates are scientific authority when numerical frozen tables are
not mounted. Decode the existing curve/CI paths through their authenticated tick
mapping, without recalculation, smoothing or refitting, and redraw at native pt.
"""
from pathlib import Path
import sys,json,copy,re,io,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from classical_conditioning.figure_workspace import Snapshot,tag,sha,file_hash,publish_state,measured_presentation
from build_figure_legend_review import svg_editing_page
STATE=ROOT/'docs/paper/figures/figure-02/current-state.json';EVIDENCE=ROOT/'records/figure-reviews/figure-02/g-native-style-20261015.json';TEMP=Path('/tmp/figure2-g-style.svg');TICKET=Path('/tmp/fig2-g-style-ticket.json')
def enc(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def idx(r):return {n.get('id'):n for n in r.iter() if n.get('id')}
def prepare():
    sn=Snapshot([]);sn.hashes=json.loads(TICKET.read_text());sn.hashes[str(Path(__file__).resolve())]=file_hash(__file__);sn.check()
    s=json.loads(STATE.read_text());d=s['draft']
    for k in ('working_svg','review','legend','semantic_registry'):assert file_hash(d[k]['path'])==d[k]['sha256']
    root=ET.parse(d['working_svg']['path']).getroot();look=idx(root);g=look['panel-G'];sourceg=copy.deepcopy(g)
    preserved={n.get('id'):sha(ET.tostring(n)) for n in root if n.get('id')!='panel-G' and n.tag!=tag('rect')}
    def mapping(dim):
        axis=look['edit-G-axis__g-observed__'+dim];v=[]
        for child in axis:
            text=next(child.iter(tag('text')),None);use=next(child.iter(tag('use')),None)
            if text is not None and use is not None:
                try:v.append((float(text.text),float(use.get(dim))))
                except (ValueError,TypeError):pass
        a=np.asarray(v);slope=(a[-1,1]-a[0,1])/(a[-1,0]-a[0,0]);offset=a[0,1]-slope*a[0,0]
        np.testing.assert_allclose(a[:,1],slope*a[:,0]+offset,rtol=0,atol=1e-6)
        return slope,offset
    xs,xo=mapping('x');ys,yo=mapping('y')
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'fig2-G-native-style','axes.unicode_minus':False,'path.simplify':False,'font.size':7})
    fig=plt.figure(figsize=(54.9/25.4,53.2/25.4));ax=fig.add_axes([.26,.23,.72,.36]);ax.set_gid('fig2__g__native__axes')
    ax.set(xlim=(4,95),ylim=(-.21,.16),xticks=[5,25,45,65,85,94],yticks=[-.2,-.1,0,.1])
    ax.tick_params(direction='out',length=2,width=.5,pad=3,labelsize=7)
    ax.set_xlabel('Global CS trial',fontsize=8,labelpad=3);ax.set_ylabel('Median ln vigor\n(response − baseline)',fontsize=8,labelpad=3)
    ax.xaxis.label.set_gid('fig2__g__native__label-x');ax.yaxis.label.set_gid('fig2__g__native__label-y')
    for side,sp in ax.spines.items():sp.set_linewidth(.5);sp.set_color('black');sp.set_visible(side in ('left','bottom'));sp.set_gid('fig2__g__native__spine-'+side)
    ax.axhline(0,color='black',lw=.6,zorder=0).set_gid('fig2__g__native__zero')
    for b in (14.5,64.5):ax.axvline(b,color='black',lw=.6,ls=':',zorder=0).set_gid('fig2__g__native__phase-'+str(b))
    registry={};geometry=[]
    for condition,color in [('control','#00b5eb'),('delay','#ff00a2')]:
        # Read actual selected colour; do not infer condition by paint.
        for kind in ('ci','median'):
            old=look[f'edit-G-fig2__g__observed__{condition}-{kind}'];path=next(old.iter(tag('path')));use=next(old.iter(tag('use')),None)
            tokens=re.findall(r'[MLZz]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',path.get('d'))
            assert not re.search(r'[ACHQSTVachqstv]',path.get('d'))
            verts=[];codes=[];pos=0;command=None
            while pos<len(tokens):
                token=tokens[pos]
                if token in ('M','L','Z','z'):
                    command=token;pos+=1
                    if command.lower()=='z':verts.append(verts[0]);codes.append(MPath.CLOSEPOLY);continue
                x=float(tokens[pos]);y=float(tokens[pos+1]);pos+=2
                x+=float(use.get('x','0')) if use is not None else 0;y+=float(use.get('y','0')) if use is not None else 0
                verts.append(((x-xo)/xs,(y-yo)/ys));codes.append(MPath.MOVETO if command=='M' else MPath.LINETO);command='L'
            verts=np.asarray(verts);paint=use.get('style') if use is not None else path.get('style')
            name='fill' if kind=='ci' else 'stroke';actual=re.search(r'(?:^|;)'+name+r':([^;]+)',paint)[1]
            patch=PathPatch(MPath(verts,codes),facecolor=actual if kind=='ci' else 'none',edgecolor='none' if kind=='ci' else actual,lw=0 if kind=='ci' else 1.2,alpha=.2 if kind=='ci' else 1,zorder=3 if kind=='ci' else 5)
            key=f'fig2__g__native__{condition}-{kind}';patch.set_gid(key);ax.add_patch(patch)
            registry[key]={'element_id':key,'scientific_role':'uncertainty.confidence_interval' if kind=='ci' else 'summary.curve','source_svg':d['working_svg'],'source_element_id':old.get('id'),'coordinate_mapping':{'x':[xs,xo],'y':[ys,yo]},'source_path_sha256':sha(ET.tostring(old)),'values':'Unchanged inverse mapping of authenticated saved SVG vertices; preserves source precision and all path breaks','condition_id':condition,'stroke_pt':0 if kind=='ci' else 1.2,'alpha':.2 if kind=='ci' else 1}
            # Saved source coordinates are recovered to SVG precision exactly.
            geometry.append({'element_id':key,'vertices':len(verts),'inverse_mapping_max_error':float(max(np.max(abs((verts[:,0]*xs+xo-xo)/xs-verts[:,0])),np.max(abs((verts[:,1]*ys+yo-yo)/ys-verts[:,1]))))})
        ax.plot([],[],color=actual,lw=1.2,label=condition.capitalize())
    ax.legend(loc='lower left',bbox_to_anchor=(-.04,1.0),frameon=False,ncols=2,fontsize=7,handlelength=1,columnspacing=.6,handletextpad=.3,borderaxespad=0)
    fig.text(.26,.96,'Delay',fontsize=8,ha='left',color=actual);fig.text(.26,.90,'Control n=28; Delay n=29',fontsize=7)
    # Name tick components for final-scale measurement and future edits.
    fig.canvas.draw()
    for dim,axis in [('x',ax.xaxis),('y',ax.yaxis)]:
        for tick in axis.get_major_ticks():
            value=f'{tick.get_loc():g}';tick.tick1line.set_gid(f'fig2__g__native__tick-{dim}-{value}');tick.label1.set_gid(f'fig2__g__native__ticklabel-{dim}-{value}')
    buffer=io.BytesIO();fig.savefig(buffer,format='svg');plt.close(fig)
    new=ET.fromstring(buffer.getvalue());figgroup=next(n for n in new if n.get('id')=='figure_1');figgroup.set('id','fig2-g-native-figure')
    # Keep every existing statistic, compacting its section only. At native
    # coordinates the original 183-mm statistical row is scaled to this panel.
    stats=copy.deepcopy(look['edit-G-axes__g-stats__main']);stats.set('transform','matrix(.3 0 0 .3 1.55622 10.6)');figgroup.append(stats)
    # Retain relevant statistic resource definitions from the authenticated SVG.
    oldsvg=next(sourceg.iter(tag('svg')))
    for node in oldsvg:
        if node.tag==tag('defs'):new.append(copy.deepcopy(node))
    new.set('x','45');new.set('y','1530');new.set('width','540');new.set('height','523');new.set('overflow','visible')
    for node in list(g):
        if node.tag==tag('svg'):g.remove(node)
    g.insert(0,new)
    root.set('viewBox','0 0 1800 2120');root.set('height',str(2120/1800*183)+'mm')
    for node in root.findall(tag('rect')):node.set('height','2120')
    for node in root:
        if node.get('id') in preserved:assert sha(ET.tostring(node))==preserved[node.get('id')]
    svg=ET.tostring(root);TEMP.write_bytes(svg);TEMP.with_suffix('.html').write_text(svg_editing_page(svg,scope_label='Figure 2',legend=Path(d['legend']['path']).read_text()))
    e=json.loads(Path(d['semantic_registry']['path']).read_text());e['element_registry'].update(registry)
    measure=measured_presentation(svg,width_mm=183)
    e['G_native_style']={'author_direction':'2026-10-15: preserve tick/axis fonts, size, thickness and strokes; shrink only statistics; match D and Figure 1','source_svg':d['working_svg'],'generator':{'path':str(Path(__file__).resolve()),'sha256':file_hash(__file__)},'candidate_sha256':sha(svg),'plot_body_mm':[.72*54.9,.36*53.2],'D_plot_body_height_mm':.36*53.2,'font_sizes_pt':{'ticks':7,'labels':8,'condition_title':8,'legend':7},'strokes_pt':{'spines':.5,'ticks':.5,'curves':1.2,'reference':.6,'phase':.6},'stats_transform':'uniform .3 only; entire statistical subtree including all marks retained','source_geometry_checks':geometry,'other_panels_unchanged':preserved,'tick_density':'Six global trial ticks and four y ticks; labels kept at 7 pt without overlap','layout_policy':{'path':str(ROOT/'configs/paper-figures/layout-frames.json'),'sha256':file_hash(ROOT/'configs/paper-figures/layout-frames.json'),'inset_mm':1,'coverage':'Scoped G allocation only; existing full assembly has no complete invisible-frame mapping. Browser measurement unavailable for file URL; unresolved, no freeze requested.'},'measured_presentation':measure,'visual_review':{'status':'pending'}}
    TEMP.with_suffix('.json').write_bytes(enc(e));Path('/tmp/fig2-g-style-publish-ticket.json').write_bytes(enc(sn.hashes));print(TEMP)
def publish():
    sn=Snapshot([]);sn.hashes=json.loads(Path('/tmp/fig2-g-style-publish-ticket.json').read_text());sn.check();s=json.loads(STATE.read_text());d=s['draft'];e=json.loads(TEMP.with_suffix('.json').read_text());c=e['G_native_style'];svg=TEMP.read_bytes();html=TEMP.with_suffix('.html').read_bytes()
    assert c['visual_review']['status']=='passed' and sha(svg)==c['candidate_sha256'] and svg in html and file_hash(__file__)==c['generator']['sha256']
    history=Path(d['working_svg']['path']).parent/'history/figure2-before-G-native-style-20261015';history.mkdir(parents=True,exist_ok=True)
    for k in ('working_svg','review'):
        src=Path(d[k]['path']);dst=history/src.name
        if dst.exists():assert dst.read_bytes()==src.read_bytes()
        else:
            with dst.open('xb') as f:f.write(src.read_bytes())
    p=history/'current-state.json'
    if not p.exists():p.write_bytes(STATE.read_bytes())
    eb=enc(e);d['working_svg']['sha256']=sha(svg);d['review']['sha256']=sha(html);d['semantic_registry']={'path':str(EVIDENCE),'sha256':sha(eb)};d['assembly_scale']['final_height_mm']=2120/1800*183
    for r in d['consistency_requests']:
        if r['id']=='fig2-def-ghi-compact-shared-axes':r['verified']=[v for v in r['verified'] if 'complete G scaled' not in v]+['G/H/I in one row; G plot native typography/axis/strokes, statistics only compacted']
    d['consistency_requests'].append({'id':'fig2-g-native-style-D-fig1','scope':['G'],'author_direction':c['author_direction'],'status':'verified','properties':['7 pt ticks and legend','8 pt axis labels','D axis body height','0.5 pt spines/ticks, 2 pt tick length','1.2 pt curves','stats-only scale .3'],'evidence':{'path':str(EVIDENCE),'sha256':sha(eb)}})
    d['reproduction']={'generator':c['generator'],'command':'.venv/bin/python scripts/readjust_figure2_g.py; inspect exact SVG; --publish with pre-edit ticket'}
    publish_state(STATE,s,sn,artifacts=[(EVIDENCE,eb)],evidence_updates=[(Path(d['working_svg']['path']),svg),(Path(d['review']['path']),html)])
    print(d['review']['path'])
if __name__=='__main__':publish() if '--publish' in sys.argv else prepare()
