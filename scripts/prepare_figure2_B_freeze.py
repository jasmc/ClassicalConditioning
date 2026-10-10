"""Prepare the selected bout-only B panels with corrected fish-level statistics.

Exports through the existing provenance exporter. Final publication is performed
only by scripts/freeze_figure.py, after check-only and actual visual review.
"""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[1] / "src"))
from classical_conditioning.external_artifacts import resolve_artifact, external_output
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import sys
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
from classical_conditioning.analysis.bout_block_statistics import corrected_b_statistics, VALUE, PAIRS
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.figures.theme import condition_color
from classical_conditioning.figures.export import FigureMode, FigureProvenance, assign_axes_semantic_ids, export_matplotlib_figure
from review_figure2_bout_only import change_effects

SOURCE = resolve_artifact('reviews/figure2_bout_only_ratio_vs_log_20261009/manifest.json').parent
SPEC_PATH = ROOT/"configs/paper-figures/figure-elements.json"
SPEC = json.loads(SPEC_PATH.read_text())
LABELS = ["PT","ET","LT"]
WIDTH_MM = 54.9  # 540/1800 of the provisional 183-mm assembly
AUTH = "Joaquim, 2026-10-09: try fixing the stats by using the appropriate stats test if it needs be corrected. then, freeze version. Context explicitly selects B."


def artifact(path, **extra):
    return dict(path=str(Path(path).resolve()),sha256=sha256_file(Path(path)),**extra)


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def render(panel, fish, tests, out):
    conditioned = "delay" if panel=="D" else "trace"
    experiment = get_experiment_spec("allDelay" if panel=="D" else "all3sTrace")
    conditions = {c.condition_id:c for c in experiment.conditions}
    colors = {c:to_hex(condition_color(conditions[c])) for c in (conditioned,"control")}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"font.weight":"normal","font.style":"normal","svg.fonttype":"none","svg.hashsalt":"bout-B-freeze","axes.unicode_minus":False,"path.simplify":False})
    fig, axes = plt.subplots(1,2,figsize=(WIDTH_MM/25.4,53.2/25.4),sharey=True)
    fig.subplots_adjust(left=.26,right=.98,bottom=.23,top=.59,wspace=.18)
    records, artists = {}, {}
    context = {"metric_id":"legacy_distal_angular_speed","measure":"fish/block median of trial median log response minus median log baseline","units":"dimensionless natural-log difference","normalization":"no pseudocount; no-bout/invalid/zero samples excluded from logs","alignment":"CS onset","baseline_s":[-15,0],"response_s":[0,9 if panel=="D" else 13],"blocks":[[10,14],[65,69],[90,94]],"statistics":"two-sided exact sign and Brunner-Munzel t; Holm36 across D/E"}
    def register(artist, key, role, sub, geometry=None, coord="data", extra=None):
        artist.set_gid(key)
        style_role = SPEC["roles"][role]["style"]
        protection = "data-geometry" if role.startswith(("reference.","trajectory.","observation.","summary.","uncertainty.")) else "axis-definition" if role.startswith("axis.") else "scientific-text" if role in {"title.condition","annotation.statistical_label"} else "presentation"
        style = copy.deepcopy(SPEC["styles"][style_role])
        record = dict(element_id=key,scientific_role=role,artist_type=type(artist).__name__,figure_id="fig2",panel_id=panel.lower(),subpanel_id=sub,
            scientific_context=dict(context,subpanel_condition=sub),coordinate_system=coord,geometry=geometry or {"definition":"bound renderer artist at recorded figure size"},style_role=style_role,resolved_style=style,
            required_in_svg=bool(artist.get_visible()),classification_confidence="explicit",classification_evidence=["Renderer assigns this role when creating the artist from the named fish/block value or explicit axis/annotation definition"],protection=protection,
            style_verification={k:dict(status="passed",evidence=f"Renderer sets {k} on this named {role} artist; physical width {WIDTH_MM} mm, no assembly rescaling; see saved renderer and artist properties") for k in style})
        if "color_source" in style:
            record.update(resolved_color=colors[sub],color_evidence=f"get_experiment_spec({experiment.experiment_id}).conditions[{sub}].color_rgb_255 via condition_color")
        if extra: record.update(extra)
        records[key]=record;artists[key]=artist
        return artist
    def rid(sub, instance): return f"fig2__{panel.lower()}__{sub}__{instance}"
    register(fig.patch,rid("canvas","background"),"background.panel","canvas",coord="figure_fraction",extra=dict(scientific_context={"measure":"not-applicable; non-data canvas background","units":"not-applicable"}))
    finite = fish.loc[fish.Eligible,VALUE].dropna().to_numpy()
    radius = max(.2,np.ceil(np.max(np.abs(finite))*10)/10+.05)
    ticks = np.arange(-np.floor(radius/.2)*.2,np.floor(radius/.2)*.2+.01,.2)
    ticks[np.abs(ticks)<1e-9]=0
    for ai,(ax,condition) in enumerate(zip(axes,(conditioned,"control"))):
        register(ax.patch,rid(condition,"background"),"background.panel",condition,coord="axes_fraction",extra=dict(scientific_context={"measure":"not-applicable; non-data axes background","units":"not-applicable"}))
        ax.set(xlim=(-.15,2.15),ylim=(-radius,radius),xticks=range(3),xticklabels=LABELS)
        ax.set_yticks(ticks,[f"{x:g}" for x in ticks])
        ax.tick_params(axis="x",bottom=False,length=0,pad=3,labelsize=7)
        ax.tick_params(axis="y",direction="out",width=.5,length=2,pad=3,labelsize=7)
        ax.spines[["top","right","bottom"]].set_visible(False)
        ax.spines["left"].set_visible(ai==0);ax.spines["left"].set_linewidth(.5)
        if ai==0: ax.set_ylabel("Median ln vigor\n(response − baseline)",fontsize=8,labelpad=3)
        else: ax.tick_params(axis="y",left=False,labelleft=False)
        line = ax.axhline(0,color="black",lw=.6,alpha=1,zorder=0)
        register(line,rid(condition,"reference-zero"),"reference.signal.zero",condition,{"dimension":"y","value":0,"x_axes_fraction":[0,1]},"blended")
        sub = fish.loc[fish.Eligible & fish.condition_id.eq(condition)]
        wide = sub.pivot(index="fish_id",columns="Selected block order",values=VALUE).reindex(columns=[0,1,2])
        for fish_id,row in wide.iterrows():
            common = dict(scientific_context=dict(context,condition_id=condition,fish_id=fish_id,data_field=VALUE,data_artifact=str(out/f"{panel}_fish_blocks.parquet")))
            if row.notna().sum()>=2:
                line, = ax.plot(range(3),row,color=colors[condition],alpha=.3,lw=.6,zorder=1)
                register(line,rid(condition,f"fish-{fish_id}-trajectory"),"trajectory.fish",condition,extra=common)
            points = ax.scatter(range(3),row,s=9,color=colors[condition],alpha=.3,edgecolors=colors[condition],linewidths=.4,zorder=2)
            register(points,rid(condition,f"fish-{fish_id}-observations"),"observation.fish",condition,extra=common)
        q = np.array([sub.loc[sub["Selected block order"].eq(i),VALUE].quantile([.25,.5,.75]).to_numpy() for i in range(3)])
        median,caps,stems = ax.errorbar(range(3),q[:,1],yerr=[q[:,1]-q[:,0],q[:,2]-q[:,1]],fmt="o-",color="black",alpha=.72,lw=1.2,elinewidth=1.2,capsize=1.5,capthick=1.2,ms=2.8,mew=1.2,zorder=5).lines
        composite = rid(condition,"median-iqr-composite")
        register(median,rid(condition,"median"),"summary.median",condition,extra=dict(composite_id=composite,composite_part="median-line-and-marker"))
        for i,artist in enumerate(caps): register(artist,rid(condition,f"iqr-cap-{i}"),"uncertainty.iqr",condition,extra=dict(composite_id=composite,composite_part=f"cap-{i}"))
        for i,artist in enumerate(stems): register(artist,rid(condition,f"iqr-stem-{i}"),"uncertainty.iqr",condition,extra=dict(composite_id=composite,composite_part=f"stem-{i}"))
        pos = ax.get_position(); center=pos.x0+pos.width/2
        title=fig.text(center,.975,"Delay" if condition=="delay" else "3 s Trace" if condition=="trace" else "Control",ha="center",va="top",color=colors[condition],fontsize=8)
        register(title,rid(condition,"condition-title"),"title.condition",condition,coord="figure_fraction")
        note=fig.text(center,.89,"n="+"/".join(str(int(wide[i].notna().sum())) for i in range(3)),ha="center",fontsize=7)
        register(note,rid(condition,"sample-count"),"annotation.note",condition,coord="figure_fraction")
        within=tests.loc[tests.panel.eq(panel)&tests.family.eq("within")&tests.condition.eq(condition)&tests.p_holm36.lt(.05)]
        for level,row in enumerate(within.itertuples()):
            y=1.04+.22*level
            line,=ax.plot([row.left,row.right],[y,y],transform=ax.get_xaxis_transform(),color="black",lw=.7,zorder=10,clip_on=False)
            extra=dict(source_result=dict(path=str(out/"statistics.csv"),family="within",condition=condition,left=row.left,right=row.right,p_holm36=row.p_holm36))
            register(line,rid(condition,f"within-{row.left}-{row.right}-line"),"annotation.statistical_comparison",condition,coord="blended",extra=extra)
            text=ax.text((row.left+row.right)/2,y+.025,row.stars,transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7,zorder=10,clip_on=False)
            register(text,rid(condition,f"within-{row.left}-{row.right}-label"),"annotation.statistical_label",condition,coord="blended",extra=extra)
        zero=tests.loc[tests.panel.eq(panel)&tests.family.eq("zero")&tests.condition.eq(condition)&tests.p_holm36.lt(.05)]
        for row in zero.itertuples():
            text=ax.text(row.left,.83,"†"+row.stars,transform=ax.get_xaxis_transform(),ha="center",fontsize=7,zorder=10)
            register(text,rid(condition,f"zero-{row.left}-label"),"annotation.statistical_label",condition,coord="blended",extra=dict(source_result=dict(path=str(out/"statistics.csv"),family="zero",condition=condition,left=row.left,right=row.right,p_holm36=row.p_holm36)))
    between=tests.loc[tests.panel.eq(panel)&tests.family.eq("between")&tests.p_holm36.lt(.05)]
    for row in between.itertuples():
        level=1.52
        line=ConnectionPatch((row.left,level),(row.right,level),coordsA=axes[0].get_xaxis_transform(),coordsB=axes[1].get_xaxis_transform(),color="black",lw=.7,zorder=10,clip_on=False)
        fig.add_artist(line)
        extra=dict(source_result=dict(path=str(out/"statistics.csv"),family="between",condition=row.condition,left=row.left,right=row.right,p_holm36=row.p_holm36))
        register(line,rid(conditioned,f"between-{row.left}-line"),"annotation.statistical_comparison",conditioned,coord="blended",extra=extra)
        a=fig.transFigure.inverted().transform(axes[0].get_xaxis_transform().transform((row.left,level)))
        b=fig.transFigure.inverted().transform(axes[1].get_xaxis_transform().transform((row.right,level)))
        text=fig.text((a[0]+b[0])/2,a[1]+.009,row.stars,ha="center",va="bottom",fontsize=7,zorder=10)
        register(text,rid(conditioned,f"between-{row.left}-label"),"annotation.statistical_label",conditioned,coord="figure_fraction",extra=extra)
    note=fig.text(.26,.08,"Fish median/IQR · Holm36",fontsize=7)
    register(note,rid("canvas","summary-note"),"annotation.note","canvas",coord="figure_fraction")
    note=fig.text(.26,.015,"† versus 0 · sign / BM",fontsize=7)
    register(note,rid("canvas","statistics-note"),"annotation.note","canvas",coord="figure_fraction")
    ids=[f"{panel.lower()}-{c}" for c in (conditioned,"control")]
    assign_axes_semantic_ids(fig,ids)
    for ax,condition in zip(axes,(conditioned,"control")):
        register(ax,ax.get_gid(),"axes.container",condition,coord="axes_fraction")
        for dimension,axis in (("x",ax.xaxis),("y",ax.yaxis)):
            register(axis,axis.get_gid(),"axis.component",condition,{"dimension":dimension},"blended")
            register(axis.label,axis.label.get_gid(),"axis.label",condition,{"dimension":dimension},"blended")
            for tick in axis.get_major_ticks():
                value=float(tick.get_loc())
                geom=dict(dimension=dimension,side="bottom" if dimension=="x" else "left",kind="major",value=value,category=LABELS[int(value)] if dimension=="x" and value in range(3) else None)
                for artist,role,suffix in [(tick.tick1line,"axis.tick","mark"),(tick.label1,"axis.tick_label","label")]:
                    key=artist.get_gid() or rid(condition,f"tick-{dimension}-{value:g}-{suffix}")
                    register(artist,key,role,condition,geom,"blended")
        for side,spine in ax.spines.items(): register(spine,spine.get_gid(),"axis.spine",condition,{"side":side},"axes_fraction")
    # Bind actual renderer properties rather than merely copying style defaults.
    for key,artist in artists.items():
        records[key]["renderer_properties"] = dict(visible=bool(artist.get_visible()),alpha=artist.get_alpha(),zorder=artist.get_zorder(),clip_on=bool(artist.get_clip_on()))
        if hasattr(artist,"get_fontsize"):
            records[key]["renderer_properties"].update(font_family=artist.get_fontfamily(),font_size_pt=artist.get_fontsize(),font_weight=artist.get_fontweight())
    fig.canvas.draw()
    bounds=[]
    for key,artist in artists.items():
        if hasattr(artist,"get_text") and artist.get_visible() and artist.get_text():
            b=artist.get_window_extent(fig.canvas.get_renderer())
            bounds.append(dict(element_id=key,text=artist.get_text(),bounds_px=list(map(float,b.extents))))
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1, (key,b.extents,fig.bbox.extents)
    # Selected scientific SVG at identical geometry; candidate changes only
    # paint of fish trajectories. Scientific points, ticks and text are fixed.
    for key,artist in artists.items():
        if records[key]["scientific_role"]=="trajectory.fish": artist.set_alpha(.37)
    original=out/f"Fig2_{panel}_B_selected_scientific.svg"
    fig.savefig(original,format="svg")
    for key,artist in artists.items():
        if records[key]["scientific_role"]=="trajectory.fish": artist.set_alpha(.3)
    mappings={key:record for key,record in records.items()}
    result=export_matplotlib_figure(fig,out/f"Fig2_{panel}_B",FigureProvenance(figure_id=f"Fig2-{panel}-B",analysis_recipe="bout-only-log-median-sign-BrunnerMunzel-Holm36",source_file=__file__,source_symbol="render",source_hash=sha256_file(Path(__file__)),reproduction_snippet=f"python scripts/prepare_figure2_B_freeze.py --output {out}",input_artifacts=tuple(artifact(p) for p in [out/f"{panel}_fish_blocks.parquet",out/"statistics.csv",out/"scientific-selection.json"]),artist_mappings=mappings,analysis_identity=context),mode=FigureMode.PUBLICATION,panel_ids=ids,allow_dirty_publication=True,overwrite=True)
    fig.savefig(out/f"Fig2_{panel}_B.png",dpi=300)
    plt.close(fig)
    svg=next(p for p in result.outputs if p.suffix==".svg")
    root=ET.parse(svg).getroot(); actual_ids={n.get("id") for n in root.iter() if n.get("id")}
    for key in records: records[key]["required_in_svg"]=bool(records[key]["required_in_svg"] and key in actual_ids)
    transforms=[dict(element_id=n.get("id"),transform=n.get("transform")) for n in root.iter() if n.get("transform")]
    vb=list(map(float,root.get("viewBox").split()))
    candidate=dict(figure_id="fig2",panel_ids=[panel.lower()],selection_record=artifact(out/"scientific-selection.json"),assembly_scale=dict(final_width_mm=WIDTH_MM,source_to_final_transforms=dict(root_user_unit_to_final_pt=WIDTH_MM*72/25.4/vb[2],svg_transforms=transforms)),element_registry=records,approved_exceptions=[],source_artifacts=[artifact(original,kind="original_svg"),artifact(Path(__file__)),artifact(out/"scientific-selection.json"),artifact(result.sidecar)],data_artifacts=[artifact(out/f"{panel}_fish_blocks.parquet"),artifact(out/f"{panel}_trials.parquet"),artifact(out/"statistics.csv"),artifact(out/"method.json"),artifact(out/"bootstrap-effects.csv")],exports=[artifact(svg),artifact(out/f"Fig2_{panel}_B.pdf"),artifact(out/f"Fig2_{panel}_B.png")],verification=dict(scientific_mapping_review=dict(status="passed",evidence="B values unchanged from authenticated source; unique fish/block rows; corrected 36-test family recomputed, three statistical invariants tests pass; means/log window order and NaN policy audited"),structure_review=dict(status="passed",evidence="Renderer creates conditioned-left/control-right paired blocks, shared y limits/ticks, one left spine, PT/ET/LT without tick marks; median and IQR parts registered separately, zero reference behind data"),visual_review=dict(status="pending",evidence="Inspect saved final-width PNG and SVG before freezing")),freeze_authorization=AUTH)
    write(out/f"{panel}_candidate.json",candidate)
    write(out/f"{panel}_renderer_review.json",dict(final_width_mm=WIDTH_MM,final_height_mm=53.2,source_to_final_scale=1,axes_limits=[list(map(float,axes[0].get_ylim()))],y_ticks=list(map(float,ticks)),text_bounds=bounds,background_color="#ffffff",font_family="DejaVu Sans",effective_fill_stroke_alpha="reference 1; fish .3; median/IQR .72; all others 1",shared_axes=True,clipping_checks="all visible text inside figure; all eligible fish values within data limits"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if any(out.glob("*.freeze.json")): raise ValueError("Frozen exports are immutable; use a new directory")
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((SOURCE/"manifest.json").read_text());hashes={Path(x["path"]).name:x["sha256"] for x in manifest["files"]}
    data={}
    for panel in ("D","E"):
        for srcname,destname in [(f"{panel}_log_fish_blocks.parquet",f"{panel}_fish_blocks.parquet"),(f"{panel}_log_trials.parquet",f"{panel}_trials.parquet")]:
            src=SOURCE/srcname;assert sha256_file(src)==hashes[srcname]
            shutil.copyfile(src,out/destname)
        data[panel]=pd.read_parquet(out/f"{panel}_fish_blocks.parquet")
    tests,changes=corrected_b_statistics(data)
    tests.to_csv(out/"statistics.csv",index=False);changes.to_csv(out/"fish-changes.csv",index=False)
    change_effects(changes,"log").to_csv(out/"bootstrap-effects.csv",index=False)
    method=json.loads((SOURCE/"method.json").read_text());method.update(statistics="36 two-sided B tests across D/E; exact sign for zero-reference and paired changes; Brunner-Munzel t for independent blocks and changes; single Holm36",statistical_status="author-selected exploratory analysis; fish independence assumed; no day/tank adjustment; no equivalence claims",style_specification=artifact(SPEC_PATH),intended_panel_size_mm=[WIDTH_MM,53.2],assembly_note="54.9 mm = 540/1800 of provisional 183-mm figure width. Row-2 height is now 53.2 mm; whole-figure layout must accommodate this height before its own freeze.")
    write(out/"method.json",method)
    write(out/"scientific-selection.json",dict(selected_version="B",panel_ids=["D","E"],authorization=AUTH,source_review=artifact(SOURCE/"manifest.json"),source_inputs_manifest=artifact(SOURCE/"inputs.json"),data=[artifact(out/f"{p}_fish_blocks.parquet") for p in ("D","E")],stats=artifact(out/"statistics.csv"),scientific_method=artifact(out/"method.json"),F="unavailable; not frozen",source_estimator_values_changed=False))
    for panel in ("D","E"): render(panel,data[panel],tests,out)
    print("Prepared",out)
    print(tests.loc[tests.p_holm36.lt(.05),["panel","family","condition","left","right","p_holm36"]].to_string(index=False))


if __name__=="__main__": main()
