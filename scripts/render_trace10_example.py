"""Render the legacy-designated fixed-10sTrace example without editing Figure 1.

Uses the selected original-frame 1 s heatmap method and the Nature profile.
One editable SVG and its minimal SVG/legend review are published exclusively
with a scoped state; raw data, processing and the parent figure remain read-only.
"""
from pathlib import Path
import base64
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,findfont
import numpy as np
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot,publish_state,tag,sha
from classical_conditioning.figures.pooled_log_heatmap import analyze_recording
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme,style_axes
from classical_conditioning.figures.layout_frames import load_policy

FISH='20230307_04'
PROJECT=Path(r'F:\Digested Data\all10sTrace-full-v1')
OUT=Path('J:/ClassicalConditioning Outputs/current/reviews/figure-01-10strace-example-20261019')
SCOPE=REPO/'docs/paper/figures/figure-01/10strace-example/current-state.json'
PARENT=REPO/'docs/paper/figures/figure-01/current-state.json'
LEGACY=[REPO/'legacy/scripts/2_ExampleFishPlotting.py',REPO/'legacy/scripts/5_NormalizedVigorPlotting.py']


def encoded(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def ref(p):return {'path':str(p.resolve()),'sha256':sha256_file(p)}


def main():
    if not OUT.parent.is_dir():raise FileNotFoundError('Authenticated JOAQUIM review storage required')
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Preserve existing example artifacts')
    if SCOPE.exists():raise FileExistsError('Existing scoped example state requires reconciliation')
    inputs=[PARENT,SCOPE,*LEGACY,Path(__file__),REPO/'src/classical_conditioning/figures/pooled_log_heatmap.py']
    snap=Snapshot(inputs)
    parent_before=sha256_file(PARENT)
    for p in LEGACY:
        if FISH not in p.read_text(encoding='utf-8'):raise ValueError('Legacy example identity is no longer present')
    print('AUTHENTICATE AND CALCULATE '+FISH,flush=True)
    matrix,bins,info=analyze_recording(PROJECT,FISH,'','all10sTrace','trace')
    snap.hashes.update(info['sources'])
    if matrix.shape!=(90,40) or info['paired_training_trials']!=46:raise ValueError('Unexpected fish protocol/matrix')
    profile=load_presentation_profile('nature-main')
    snap.hashes.update({str((REPO/p).resolve()):h for p,h in profile.inputs.items()})
    policy,policy_hash=load_policy();inset=float(policy['selected_inset_mm'])
    width,height=80.,96.
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        fig=plt.figure(figsize=(width/25.4,height/25.4),layout=None)
        ax=fig.add_axes([20/width,13/height,44/width,72/height])
        cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black')
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',interpolation='nearest',
            extent=(-20,20,90.5,.5),cmap=cmap,vmin=-.25,vmax=.25,rasterized=True)
        image.set_gid(FISH+'__direct-1s-log-frame-median')
        for t,style,name in [(0,'-','onset'),(10,'--','offset')]:
            line=ax.axvline(t,color=theme.cs_color,lw=theme.cs_linewidth,ls=style)
            line.set_gid('CS-'+name)
        line,=ax.plot([20,20],[10.5,60.5],color=theme.us_color,lw=theme.us_linewidth,ls=':',clip_on=False)
        line.set_gid('expected-training-US-20s')
        for y in (10.5,60.5):
            line=ax.axhline(y,color='white',lw=theme.axes_linewidth);line.set_gid('phase-boundary-'+str(y))
        ax.set(xlim=(-20,20),ylim=(90.5,.5),xticks=[-20,0,10,20],yticks=[1,10,30,60,90])
        style_axes(ax,theme=theme,xlabel='Time from CS onset (s)',ylabel='Trial')
        ax.set_title('10sTrace · '+FISH,fontsize=theme.axes_titlesize,pad=theme.axes_titlepad)
        for label,y in [('Pre',5.5),('Train',35.5),('Test',75.5)]:
            text=ax.text(-.34,y,label,transform=ax.get_yaxis_transform(),rotation=90,
                         ha='center',va='center',fontsize=theme.font_size)
            text.set_gid('phase-label-'+label)
        ca=fig.add_axes([68/width,29/height,2/width,40/height])
        cb=fig.colorbar(image,cax=ca,ticks=[-.25,0,.25]);cb.ax.set_title('Δ ln vigor',fontsize=theme.font_size,pad=theme.axes_titlepad)
        cb.ax.tick_params(labelsize=theme.tick_labelsize,length=theme.tick_major_size,width=theme.tick_major_width)
        cb.outline.set_linewidth(theme.axes_linewidth)
        svg=io.BytesIO();fig.savefig(svg,format='svg');plt.close(fig)
    root=ET.fromstring(svg.getvalue())
    defs=ET.SubElement(root,tag('defs'));css=[];fonts=[]
    for weight in ('normal','bold'):
        p=Path(findfont(FontProperties(family='Arial',weight=weight),fallback_to_default=False));fonts.append(ref(p))
        css.append('@font-face{font-family:Arial;font-weight:'+('400' if weight=='normal' else '700')+';src:url(data:font/ttf;base64,'+base64.b64encode(p.read_bytes()).decode()+') format("truetype")}')
    ET.SubElement(defs,tag('style')).text=''.join(css)
    for image in root.iter(tag('image')):image.set('style',image.get('style','')+';image-rendering:pixelated')
    # Frames are invisible, physically specified, and bound to the native canvas.
    ET.SubElement(root,tag('metadata'),{'id':'layout-frames'}).text=json.dumps({
        'policy':policy,'width_mm':width,'height_mm':height,
        'allocations':[{'id':'single-fish-example','box_mm':[0,0,width,height],
                        'content_inset_mm':inset,'group_ids':['figure_1']}],
        'verification':'exact saved-render containment requires visual review'})
    legend=(f'10sTrace example {FISH}, explicitly marked as the 10sTrace fish in the legacy plotting code. '
            'Acquisition trials 5–94 are displayed as 1–90. Eligible positive detected-bout vigor is '
            'natural-log transformed; direct 1-s medians are centred on the median finite baseline-bin '
            'medians over [−15,0) s. No amplitude scaling; colours saturate at ±0.25, black is missing. '
            'Green lines mark CS onset and 10-s offset; the violet dotted training guide at the right '
            'edge marks expected US onset at 20 s. Boundaries separate Pre, Train and Test. '
            'Provisional descriptive example, not a cohort-wide best-learner classification or a new Figure 1 selection.')
    report={'recording_id':FISH,'status':'provisional standalone example; not frozen',
            'legacy_recommendation_sources':[ref(p) for p in LEGACY],
            'calculation':info,'generator':ref(Path(__file__)),'profile_inputs':profile.inputs,
            'font_resources':fonts,'parent_state_sha256_before':parent_before,
            'Figure1_selection_changed':False,'full_cohort_ranking':'not completed; batch still processing'}
    ET.SubElement(root,tag('metadata'),{'id':'scientific-provenance'}).text=json.dumps(report)
    svg_bytes=ET.tostring(root,encoding='utf-8')
    page=('<!doctype html><html><meta charset="utf-8"><style>body{margin:24px;background:white;color:black;font:14px Arial,sans-serif}main{max-width:800px;margin:auto}svg{display:block;width:min(100%,520px);height:auto;margin:auto}p{line-height:1.45}</style><main>'
          +svg_bytes.decode()+'<p>'+legend+'</p></main></html>').encode()
    OUT.mkdir(exist_ok=True);SCOPE.parent.mkdir(parents=True,exist_ok=True)
    import pyarrow as pa,pyarrow.parquet as pq
    buf=io.BytesIO();pq.write_table(pa.Table.from_pandas(bins,preserve_index=False),buf,compression='zstd')
    files=[(OUT/'single-fish.svg',svg_bytes),(OUT/'single-fish.html',page),
           (OUT/'legend.md',(legend+'\n').encode()),(OUT/'bins.parquet',buf.getvalue()),
           (OUT/'calculation.json',encoded(report))]
    state={'figure_id':'fig1-10strace-example','recording_id':FISH,'default_display':'draft',
           'whole_figure_frozen':False,'parent_figure_selection_changed':False,
           'draft':{'working_svg':{'path':str(files[0][0]),'sha256':sha(svg_bytes)},
                    'review':{'path':str(files[1][0]),'sha256':sha(page)},
                    'legend':{'path':str(files[2][0]),'sha256':sha(files[2][1])},
                    'calculation':{'path':str(files[4][0]),'sha256':sha(files[4][1])}},
           'verification':{'visual_review':'pending','scientific_method':'supported selected 1-s heatmap pipeline; source hashes rechecked'}}
    publish_state(SCOPE,state,snap,artifacts=files)
    if sha256_file(PARENT)!=parent_before:raise ValueError('Parent Figure 1 changed during standalone build')
    print(json.dumps({'fish':FISH,'review':str(OUT/'single-fish.html'),'finite_cells':info['finite_cells'],'defined_trials':info['defined_trials']}),flush=True)


if __name__=='__main__':main()
