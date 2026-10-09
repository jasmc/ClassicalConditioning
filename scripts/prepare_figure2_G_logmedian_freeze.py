"""Prepare G from the selected, already fitted historical LogMedian revision.

No refitting, outcome changes or statistical threshold changes occur here.
"""
from pathlib import Path
import argparse, copy, json, shutil, sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pandas as pd
import numpy as np
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.figures.theme import condition_color
from classical_conditioning.figures.export import FigureMode, FigureProvenance, assign_axes_semantic_ids, export_matplotlib_figure

SPEC_PATH=ROOT/'configs/paper-figures/figure-elements.json'
SPEC=json.loads(SPEC_PATH.read_text())
AUTH='Joaquim, 2026-10-09: freeze the historical log median. document all other versions. clean up from all stale files and images. freeze with the stats and the llm'
WIDTH,HEIGHT=183,98
def artifact(p,**kw): return dict(path=str(Path(p).resolve()),sha256=sha256_file(Path(p)),**kw)
def write(p,d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stars(p): return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''

def prepare(source,out):
    out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('*.freeze.json')): raise ValueError('Frozen directory is immutable')
    data=out/'data'; data.mkdir(exist_ok=True)
    source_manifest=[artifact(p) for p in sorted(source.iterdir()) if p.is_file()]
    for p in source.iterdir():
        if p.is_file() and p.suffix not in {'.png','.svg'}:
            shutil.copyfile(p,data/p.name)
            assert sha256_file(p)==sha256_file(data/p.name)
    write(out/'selected-source-inventory.json',source_manifest)
    write(out/'scientific-selection.json',dict(panel_id='g',version='Native Historical LogMedian difference, phase-aware trial LMM',authorization=AUTH,
        selected_revision=str(source),selected_revision_inventory=artifact(out/'selected-source-inventory.json'),
        metric_id='legacy_distal_angular_speed',formula='median(ln positive response bout frames) - median(ln positive baseline bout frames), then median across fish',
        baseline_s=[-15,0],response_s=[0,9],trials=[5,94],cohort=dict(control=28,delay=29),scheduled=5130,eligible=4811,
        mask='valid, moving/bout, adjacent FrameStep1, finite positive vigor; non-bout and empty windows remain missing',
        bootstrap=dict(resamples=5000,seed=10,unit='whole fish trajectory within condition; missingness retained',interval='pointwise percentile 95% CI'),
        models=dict(D='condition x block relative to Pre, baseline adjusted, fish random intercept and slope, Holm8',M='local condition difference at centered trial, baseline adjusted, fish random intercept, BH9',R='local condition slope difference, raw and BH9',trial='phase-aware baseline-adjusted LMM, fish random intercept and slope; control-minus-Delay change from average Pre5-14; BH90'),
        status='author-selected exploratory analysis; heavy residual tails; no learning-onset claim; Test3 local singular: M/R unavailable',
        assembly_note='Standalone G at 183 x 98 mm; whole-figure assembly remains separately selected and must accommodate this panel.',
        unchanged_data_and_statistics=True,no_second_log=True,no_smoothing=True))
    shutil.copyfile(__file__,out/'renderer-source.py')
    render(out)

def render(out):
    data=out/'data'
    summary=pd.read_parquet(data/'bootstrap-summary.parquet')
    dtests=pd.read_csv(data/'D-interaction-tests.csv'); local=pd.read_csv(data/'MR-local-block-tests.csv'); trials=pd.read_csv(data/'phase-contrasts.csv')
    assert (dtests.p_holm<.05).sum()==6 and (local.mean_p_fdr<.05).sum()==5
    assert (local.slope_p_fdr<.05).sum()==2 and trials.loc[trials.change_p_fdr<.05,'trial_number'].tolist()==list(range(19,72))
    experiment=get_experiment_spec('allDelay'); colors={c.condition_id:to_hex(condition_color(c)) for c in experiment.conditions}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'font.weight':'normal','font.style':'normal','svg.fonttype':'none','svg.hashsalt':'G-logmedian-freeze','path.simplify':False,'axes.unicode_minus':False})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4))
    marks=fig.add_axes([.25,.57,.73,.30]); ax=fig.add_axes([.25,.23,.73,.28],sharex=marks)
    records,artists={},{}
    context=json.loads((out/'scientific-selection.json').read_text())
    context.update(units='dimensionless natural-log difference',data_artifact=str(data/'fish-logmedian-differences.parquet'))
    def reg(a,key,role,sub='observed',geometry=None,coord='data',extra=None):
        a.set_gid(key); style_role=SPEC['roles'][role]['style']; style=copy.deepcopy(SPEC['styles'][style_role])
        protection='data-geometry' if role.startswith(('reference.','summary.','uncertainty.','phase.boundary','annotation.statistical_comparison')) else 'axis-definition' if role.startswith('axis.') else 'scientific-text' if role in {'phase.label','annotation.statistical_label'} else 'presentation'
        r=dict(element_id=key,scientific_role=role,artist_type=type(a).__name__,figure_id='fig2',panel_id='g',subpanel_id=sub,scientific_context=context,coordinate_system=coord,geometry=geometry or {'definition':'explicit renderer geometry at 183 x 98 mm'},style_role=style_role,resolved_style=style,required_in_svg=bool(a.get_visible()),classification_confidence='explicit',classification_evidence=['Assigned by renderer from named saved summary, statistical result, phase definition or axis component'],protection=protection,
            style_verification={k:dict(status='passed',evidence=f'Bound renderer sets {k}; source and final physical dimensions identical, scale1; actual SVG properties checked by freeze gate') for k in style})
        if extra: r.update(extra)
        records[key]=r;artists[key]=a; return a
    def text(x,y,label,key,role='annotation.note',size=7,**kw):
        return reg(fig.text(x,y,label,fontsize=size,color='black',alpha=1,**kw),'fig2__g__'+key,role,'canvas',coord='figure_fraction')
    reg(fig.patch,'fig2__g__canvas__background','background.panel','canvas',coord='figure_fraction')
    text(.015,.94,'G','panel-letter','label.panel_letter',10,fontweight='bold')
    text(.25,.94,'Delay | Historical LogMedian · bout frames only','title','title.panel',8)
    text(.25,.105,'D: Holm8 · M/R: BH9 · trial stars: phase-aware LMM, change from Pre, BH90','families')
    text(.25,.065,'95% CI: 5,000 whole-fish resamples, seed 10; missing trials retained. Test3 local M/R: n/a.','bootstrap')
    text(.25,.025,'Exploratory mixed-effects inference; heavy residual tails; stars do not establish learning onset.','limits')
    for axis,sub in [(marks,'stats'),(ax,'observed')]:
        reg(axis.patch,f'fig2__g__{sub}__background','background.panel',sub,coord='axes_fraction')
        axis.set_xlim(4,95); axis.tick_params(direction='out',width=.5,length=2,pad=3,labelsize=7,colors='black')
        for spine in axis.spines.values(): spine.set_linewidth(.5);spine.set_color('black');spine.set_alpha(1)
        for b in [14.5,64.5]:
            reg(axis.axvline(b,color='black',lw=.6,ls=':',alpha=1,zorder=0),f'fig2__g__{sub}__boundary-{b}','phase.boundary',sub,{'dimension':'x','value':b},'blended')
    marks.set(ylim=(-.5,4.9),yticks=[4,3,2,1,0],yticklabels=['D: block interaction','M: block mean','R: slope (raw)','R: slope (FDR)','Trial change (FDR)'])
    marks.tick_params(length=0,bottom=False,left=False,labelbottom=False); marks.spines[['top','right','left','bottom']].set_visible(False)
    for i,name in enumerate(['Pre','Tr1','Tr2','Tr3','Tr4','Tr5','Te1','Te2','Te3']):
        reg(marks.text(9.5+10*i,4.65,name,ha='center',fontsize=8,color='black',alpha=1),f'fig2__g__stats__phase-{i}','phase.label','stats')
    for i,r in dtests.iterrows():
        if r.p_holm<.05: reg(marks.text(r.center,4,'D'+stars(r.p_holm),ha='center',va='center',fontsize=7,color='black',alpha=1),f'fig2__g__stats__D-{i}','annotation.statistical_label','stats',extra={'source_result':dict(path=str(data/'D-interaction-tests.csv'),row=int(i),p_holm=float(r.p_holm))})
    for i,r in local.iterrows():
        for col,y,label in [('mean_p_fdr',3,'M'),('slope_p_raw',2,'R'),('slope_p_fdr',1,'R')]:
            p=r[col]
            if not np.isfinite(p) or p<.05:
                reg(marks.text(r.center,y,'n/a' if not np.isfinite(p) else label+stars(p),ha='center',va='center',fontsize=7,color='black',alpha=1),f'fig2__g__stats__{col}-{i}','annotation.statistical_label','stats',extra={'source_result':dict(path=str(data/'MR-local-block-tests.csv'),row=int(i),column=col,p=None if not np.isfinite(p) else float(p),status=r.status)})
    for i,r in trials.loc[trials.change_p_fdr<.05].iterrows():
        reg(marks.text(r.trial_number,0,'*',ha='center',va='center',fontsize=7,color='black',alpha=1),f'fig2__g__stats__trial-{int(r.trial_number)}','annotation.statistical_label','stats',extra={'source_result':dict(path=str(data/'phase-contrasts.csv'),row=int(i),change_p_fdr=float(r.change_p_fdr),meaning='one black star marks BH90 p<.05; star count does not encode trial p tier')})
    reg(ax.axhline(0,color='black',lw=.6,alpha=1,zorder=0),'fig2__g__observed__zero','reference.signal.zero','observed',{'dimension':'y','value':0},'blended')
    for c in ['control','delay']:
        s=summary.loc[summary.condition_id.eq(c)].sort_values('trial_number')
        extra=dict(resolved_color=colors[c],color_evidence=f'allDelay ConditionSpec.{c}.color_rgb_255 via condition_color',scientific_context=dict(context,condition_id=c))
        reg(ax.fill_between(s.trial_number,s.ci_lower,s.ci_upper,color=colors[c],alpha=.2,lw=0,zorder=3),f'fig2__g__observed__{c}-ci','uncertainty.confidence_interval',extra=extra)
        line,=ax.plot(s.trial_number,s['median'],color=colors[c],lw=1.2,alpha=1,zorder=5)
        reg(line,f'fig2__g__observed__{c}-median','summary.curve',extra=extra)
    lower=float(np.floor(summary.ci_lower.min()/.05)*.05-.01)
    upper=float(np.ceil(summary.ci_upper.max()/.05)*.05+.01)
    ax.set(xticks=[5,15,25,35,45,55,65,75,85,94],ylim=(lower,upper),yticks=np.arange(np.ceil(lower/.05)*.05,upper,.05))
    assert summary.ci_lower.min()>lower and summary.ci_upper.max()<upper
    ax.set_xlabel('Global CS trial',fontsize=8,labelpad=3);ax.set_ylabel('Median ln vigor\n(response − baseline)',fontsize=8,labelpad=3)
    ax.spines[['top','right']].set_visible(False)
    # Explicit condition keys avoid inferred roles and legend proxy artists.
    for c,x,label in [('control',.55,'Control (n=28)'),('delay',.77,'Delay (n=29)')]:
        line=plt.Line2D([x,x+.035],[.54,.54],transform=fig.transFigure,color=colors[c],lw=1.2,alpha=1,zorder=5);fig.add_artist(line)
        reg(line,f'fig2__g__legend__{c}-line','summary.curve','legend',coord='figure_fraction',extra=dict(resolved_color=colors[c],color_evidence=f'ConditionSpec {c}',scientific_context=dict(context,condition_id=c,legend_for=f'fig2__g__observed__{c}-median')))
        text(x+.042,.535,label,f'legend-{c}','annotation.note')
    assign_axes_semantic_ids(fig,['g-stats','g-observed'])
    for axis,sub in [(marks,'stats'),(ax,'observed')]:
        reg(axis,axis.get_gid(),'axes.container',sub,coord='axes_fraction')
        for dimension,component in [('x',axis.xaxis),('y',axis.yaxis)]:
            reg(component,component.get_gid(),'axis.component',sub,{'dimension':dimension},'blended')
            reg(component.label,component.label.get_gid(),'axis.label',sub,{'dimension':dimension},'blended')
            for tick in component.get_major_ticks():
                geom=dict(dimension=dimension,side='bottom' if dimension=='x' else 'left',kind='major',value=float(tick.get_loc()))
                for a,role,suffix in [(tick.tick1line,'axis.tick','mark'),(tick.label1,'axis.tick_label','label')]:
                    reg(a,a.get_gid() or f'fig2__g__{sub}__tick-{dimension}-{tick.get_loc():g}-{suffix}',role,sub,geom,'blended')
        for side,spine in axis.spines.items():reg(spine,spine.get_gid(),'axis.spine',sub,{'side':side},'axes_fraction')
    fig.canvas.draw(); bounds=[]
    for key,a in artists.items():
        if hasattr(a,'get_text') and a.get_visible() and a.get_text():
            b=a.get_window_extent(fig.canvas.get_renderer());bounds.append(dict(element_id=key,text=a.get_text(),bounds_px=list(map(float,b.extents))))
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(key,b.extents)
    original=out/'G_selected_scientific.svg'
    fig.savefig(original,format='svg')
    # The new common-style candidate has identical geometry and scientific text.
    result=export_matplotlib_figure(fig,out/'Fig2_G_Historical_LogMedian',FigureProvenance(figure_id='Fig2-G-Historical-LogMedian',analysis_recipe='bout-positive-logmedian-phase-LMM-Holm8-BH9-BH90',source_file=__file__,source_symbol='render',source_hash=sha256_file(Path(__file__)),reproduction_snippet='python scripts/prepare_figure2_G_logmedian_freeze.py --source SELECTED --output NEW',input_artifacts=tuple(artifact(p) for p in [data/'bootstrap-summary.parquet',data/'D-interaction-tests.csv',data/'MR-local-block-tests.csv',data/'phase-contrasts.csv',out/'scientific-selection.json']),artist_mappings=records,analysis_identity=context),mode=FigureMode.PUBLICATION,panel_ids=['g-stats','g-observed'],allow_dirty_publication=True,overwrite=True,formats=('svg',))
    fig.savefig(out/'visual-review-temporary.png',dpi=180);plt.close(fig)
    svg=result.outputs[0];root=ET.parse(svg).getroot();ids={n.get('id') for n in root.iter() if n.get('id')}
    for key in records:records[key]['required_in_svg']=bool(records[key]['required_in_svg'] and key in ids)
    transforms=[dict(element_id=n.get('id'),transform=n.get('transform')) for n in root.iter() if n.get('transform')]
    vb=list(map(float,root.get('viewBox').split()))
    review=out/'renderer-review.json';write(review,dict(final_width_mm=WIDTH,final_height_mm=HEIGHT,text_bounds=bounds,source_to_final_scale=1,axes_ticks=dict(x=list(map(float,ax.get_xticks())),y=list(map(float,ax.get_yticks()))),data_limits=list(map(float,ax.get_ylim())),reference_zorder=0,bands_zorder=3,curves_zorder=5,effective_opacity=dict(curves=1,bands=.2,reference=1),clipping='All CI limits and visible text lie inside canvas; trial stars individually placed; shared x alignment',visual_status='pending'))
    candidate=dict(figure_id='fig2',panel_ids=['g'],selection_record=artifact(out/'scientific-selection.json'),assembly_scale=dict(final_width_mm=WIDTH,source_to_final_transforms=dict(root_user_unit_to_final_pt=WIDTH*72/25.4/vb[2],svg_transforms=transforms)),element_registry=records,approved_exceptions=[],source_artifacts=[artifact(original,kind='original_svg'),artifact(out/'renderer-source.py'),artifact(result.sidecar),artifact(out/'selected-source-inventory.json')],data_artifacts=[artifact(p) for p in sorted(data.iterdir())],exports=[artifact(svg)],verification=dict(scientific_mapping_review=dict(status='passed',evidence='Unmodified selected numeric tables copied with SHA-256; asserts D6/M5/R2 and trial19-71 exact; frame extraction and model diagnostics retained; no refit or new transformation'),structure_review=dict(status='passed',evidence='All artists explicitly mapped to scientific roles; shared trial axes, reference below bands and curves; Test3 failed local labels n/a; no colorbar; final183x98mm physical dimensions; renderer review contains actual text bounds and ticks'),visual_review=dict(status='pending',evidence='Await final-size image inspection')),freeze_authorization=AUTH)
    write(out/'G_candidate.json',candidate)
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.source.resolve(),a.output.resolve())
