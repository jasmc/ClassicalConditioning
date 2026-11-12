"""Disposable, source-bound comparison preview; never changes paper selection."""
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import io,json,re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from classical_conditioning.figures.theme import apply_theme,style_axes
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.preview import write_preview
from classical_conditioning.figure_workspace import sha,Snapshot
from classical_conditioning.analysis.review.figure4_profiles import GROUPS
from scripts.paper.figure_04.exports.summarize_spline_comparison import CANDIDATES

def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(Path(p).read_bytes())}

def render(directory,output):
    d=Path(directory)
    bound=[d/'comparison-specification.json',d/'execution-start.json',*[d/(a+'-model-data.csv')for a in ['delay','3strace']],*sorted((d/'fits').glob('*-0-curves.csv'))]
    if(d/'comparison-summary.json').exists():bound.append(d/'comparison-summary.json')
    guard=Snapshot([*bound,Path(__file__)])
    input_records=[ref(p)for p in bound]
    spec=json.loads((d/'comparison-specification.json').read_text())
    curves={a:{c:pd.read_csv(d/'fits'/f'{a}-{c}-0-curves.csv')for c in CANDIDATES if (d/'fits'/f'{a}-{c}-0-curves.csv').exists()}for a in ['delay','3strace']}
    observed={a:pd.read_csv(d/(a+'-model-data.csv')).groupby(['display_group','time_s']).signed.mean().reset_index()for a in curves}
    vals=np.concatenate([c.fitted.to_numpy()for data in curves.values()for c in data.values()]+[v.signed.to_numpy()for v in observed.values()])
    low=min(-.30,np.floor((vals.min()-.01)/.05)*.05);high=max(.15,np.ceil((vals.max()+.01)/.05)*.05)
    colors=['#777777','#56B4E9','#0072B2','#D55E00','#E69F00','#009E73'];styles=['--',':','-.','-','--','-']
    labels=['Fixed spline: 6 df','Fixed spline: 12 df','Fixed spline: 20 df','Penalized GAMM: k=20','Penalized GAMM: k=30','Penalized GAMM: k=20 + AR(1)']
    profile=load_presentation_profile('nature-main');width,height=183.,155.;mapping={}
    with mpl.rc_context():
        theme=apply_theme(profile=profile);mpl.rcParams.update({'svg.fonttype':'none','figure.constrained_layout.use':False,'savefig.bbox':None,'svg.hashsalt':'fig4-spline-comparison'})
        fig=plt.figure(figsize=(width/25.4,height/25.4));handles=[]
        for col,a in enumerate(['delay','3strace']):
            members=pd.read_csv(d/(a+'-model-data.csv')).groupby('display_group').fish_id.nunique()
            for row,g in enumerate(GROUPS):
                x=15+91.5*col;top=14+43*row;ax=fig.add_axes([x/width,(height-top-28)/height,72/width,28/height]);identity=a+'-'+str(row);ax.set_gid(identity)
                ax.set_xlim(-20,20);ax.set_ylim(low,high);ax.set_xticks([-20,-10,0,10,20]);ax.set_yticks(np.arange(low,high+.001,.1))
                ax.axhline(0,color='#aaaaaa',lw=.35,zorder=0);ax.axvline(0,color=theme.cs_color,lw=.5,zorder=0);ax.axvline(10,color=theme.cs_color,ls='--',lw=.4,zorder=0);ax.axvline(9 if a=='delay'else 13,color=theme.us_color,ls=':',lw=.5,zorder=0)
                p=observed[a].loc[observed[a].display_group.eq(g)];ax.scatter(p.time_s,p.signed,color='#333333',s=3,alpha=.35,lw=0,gid=identity+'-observed')
                for i,c in enumerate(CANDIDATES):
                    if c not in curves[a]:continue
                    p=curves[a][c].loc[curves[a][c].display_group.eq(g)]
                    line,=ax.plot(p.time_s,p.fitted,color=colors[i],ls=styles[i],lw=.75 if i<3 else .9,gid=identity+'-'+c)
                    if col==0 and row==0:handles.append(line)
                title=('Delay'if a=='delay'else'3sTrace')+' · '+g+f' (n={members[g]})'
                ax.set_title(title,pad=5,fontsize=6)
                style_axes(ax,theme=theme,xlabel='Time from CS (s)'if row==2 else None,ylabel='Signed bout ln vigor')
                mapping[identity]={'assay':a,'group':g,'xlim':[-20,20],'ylim':[float(low),float(high)],'source':ref(d/(a+'-model-data.csv')),
                    'curves':{c:ref(d/'fits'/f'{a}-{c}-0-curves.csv')for c in curves[a]}}
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.018),ncol=2,fontsize=5.5,frameon=False,handlelength=3,columnspacing=2)
        stream=io.BytesIO();fig.savefig(stream,format='svg',metadata={'Date':None});png=io.BytesIO();fig.savefig(png,format='png',dpi=300);plt.close(fig)
    root=ET.fromstring(stream.getvalue());root.set('width','183mm');root.set('height','155mm')
    node=ET.SubElement(root,'{http://www.w3.org/2000/svg}metadata',{'id':'editable-copy-provenance'})
    node.text=json.dumps({'scientific_registry':{'status':'Exploratory alternative-model comparison; no selection or calibration','panels':mapping,
        'models':dict(zip(CANDIDATES,labels)),'observed_points':'Equal-fish means of observed fish-level catch medians; all models use identical observed rows.'}})
    legend=('Exploratory Figure 4 spline comparison. Columns show Delay and 3sTrace; rows show controls, learner-flagged and nonlearner-flagged fish. '
        'Gray dots are equal-fish means of the unchanged observed fish-level catch medians in 0.5 s bins. '
        'All models use the same [−20,+20) s window, each trial’s [−20,0) baseline, coverage ≥0.9, configured catches and retrospective group membership. No-bout intensity remains missing. '
        'Fixed models use group-specific unpenalized cubic B-splines with 6, 12 or 20 basis columns plus group intercepts. '
        'GAMMs use group-specific penalized cubic regression splines with k=20 or 30 and REML-estimated smoothing. '
        'Fish random effects match the current assay: intercepts for Delay; intercepts and time slopes for 3sTrace. The AR(1) variant additionally models residual half-second dependence within fish with actual missing-bin gaps. '
        'Solid green marks CS onset, dashed green CS offset at 10 s; dotted violet marks expected paired-training US (omitted from the catch protocol). '
        'Lines are full-data point fits; no confidence bands or calibrated peak/onset claims are supplied. An AR(1) full-data line may be shown even when a CV refit failed; incomplete candidates are excluded from predictive ranking and a missing line indicates a failed full-data fit. Five-fold prediction holds out whole fish, conditional on saved retrospective labels; it does not independently validate those labels. '
        'The locked paper presentation is unchanged. This comparison does not select or freeze a scientific method.')
    inputs=input_records
    sources=[ref(Path(__file__)),ref(ROOT/'scripts/paper/figure_04/exports/compare_splines.R'),ref(ROOT/'scripts/paper/figure_04/exports/run_spline_comparison.mjs')]
    guard.check()
    manifest=write_preview(ET.tostring(root),legend,Path(output),inputs=inputs,command='render_spline_comparison.py <comparison-directory> <fresh-preview-directory>',sources=sources,
        parameters={'model_colors':dict(zip(CANDIDATES,colors)),'shared_ylim':[float(low),float(high)],'native_size_mm':[width,height],'presentation_profile':profile.inputs})
    (Path(output)/'comparison.png').write_bytes(png.getvalue())
    saved=ET.fromstring((Path(output)/'preview.svg').read_bytes());ns='{http://www.w3.org/2000/svg}';nodes={n.get('id'):n for n in saved.iter()if n.get('id')};count=0
    for col,a in enumerate(['delay','3strace']):
        for row,g in enumerate(GROUPS):
            x0=(15+91.5*col)*72/25.4;top=(14+43*row)*72/25.4
            def mapped(t,v):return np.column_stack([x0+(np.asarray(t)+20)/40*(72*72/25.4),top+(high-np.asarray(v))/(high-low)*(28*72/25.4)])
            identity=a+'-'+str(row)
            for c,data in curves[a].items():
                p=data.loc[data.display_group.eq(g)]
                path=next(nodes[identity+'-'+c].iter(ns+'path'))
                numbers=re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',path.get('d'))
                actual=np.asarray(list(map(float,numbers))).reshape(-1,2)
                np.testing.assert_allclose(actual,mapped(p.time_s,p.fitted),atol=1e-5,rtol=0);count+=len(actual)
            uses=list(nodes[identity+'-observed'].iter(ns+'use'))
            actual=np.array([[float(u.get('x')),float(u.get('y'))]for u in uses]);p=observed[a].loc[observed[a].display_group.eq(g)]
            np.testing.assert_allclose(actual,mapped(p.time_s,p.signed),atol=1e-5,rtol=0)
    verification={'passed':True,'svg_sha256':sha((Path(output)/'preview.svg').read_bytes()),'curve_vertices_checked':count,'observed_centers_checked':480,
        'source_mappings':'Every plotted curve and observed point checked against assay/group-specific source values and common axes.',
        'scope':'Numeric artwork verification only; no confidence calibration or full Nature compliance claim.'}
    raw=(json.dumps(verification,indent=2)+'\n').encode();(Path(output)/'numeric-verification.json').write_bytes(raw)
    record=json.loads(manifest.read_text());record['artifacts']+=[{'path':'comparison.png','sha256':sha(png.getvalue())},{'path':'numeric-verification.json','sha256':sha(raw)}]
    manifest.write_text(json.dumps(record,indent=2)+'\n');guard.check();print(manifest)

if __name__=='__main__':render(sys.argv[1],sys.argv[2])
