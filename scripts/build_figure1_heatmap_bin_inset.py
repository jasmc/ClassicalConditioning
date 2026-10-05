"""Panel E review: raw frame vigor with a short inset of exact heatmap bins."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from build_figure1_trace_panels import OUTPUT, FISH, digest, draw
from classical_conditioning.figures.theme import apply_theme, heatmap_cmap

TIME = "Time relative to CS onset (s)"
ORANGE = "#c85a17"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",type=int,choices=(17,63),default=17)
    args=parser.parse_args()
    trial=args.trial
    stage="Early Train" if trial==17 else "Late Train"
    root=OUTPUT.parent
    meta=json.loads((OUTPUT/"Fig1_PanelE_RawVigor_legacy-vigor_v1.svg.json").read_text(encoding="utf-8"))
    for key in ("frames","events"):
        if digest(Path(meta[key]))!=meta[f"{key}_sha256"]:
            raise ValueError(f"Changed source {key}")
    frames=pd.read_parquet(meta["frames"])
    events=pd.read_parquet(meta["events"])
    heat_path=root/"heatmaps/Fig1_PanelF_Delay_legacy-vigor_v2.parquet"
    heat_meta=json.loads(heat_path.with_suffix(".svg.json").read_text(encoding="utf-8"))
    if heat_meta["recording_id"]!=FISH or digest(heat_path)!=heat_meta["panel_data_sha256"]:
        raise ValueError("Heatmap identity or hash differs")
    heat=pd.read_parquet(heat_path)
    if not heat["Baseline start (s)"].eq(-15).all() or not heat["Baseline end (s)"].eq(0).all():
        raise ValueError("Inset requires the frozen [-15,0) s baseline")
    start,stop=-3.,5.
    bins=heat.loc[heat["Trial number"].eq(trial)&heat["Time bin center (s)"].between(start,stop)].sort_values("Time bin center (s)")
    centers=bins["Time bin center (s)"].to_numpy(float)
    np.testing.assert_array_equal(centers,np.arange(start+.25,stop,.5))
    values=bins["Signed log vigor"].to_numpy(float)
    raw=frames.loc[frames["Trial number"].eq(trial)&frames[TIME].between(start,stop)]
    cap=math.ceil(float(raw.Vigor.quantile(.995))*10)/10
    finite=values[np.isfinite(values)]
    lower=min(-.1,math.floor(float(finite.min())*10)/10-.1)
    upper=max(.1,math.ceil(float(finite.max())*10)/10+.1)
    theme=apply_theme()
    mpl.rcParams.update({"svg.fonttype":"none","path.simplify":False})
    fig=plt.figure(figsize=(8.15,2.9),layout="none")
    fig.text(.10,.96,f"E inset · {stage}, trial {trial} · −3 to +5 s",fontsize=12,weight="bold",va="top")
    fig.text(.10,.85,"Black: raw vigor    Orange: exact 0.5 s heatmap values",fontsize=9,color="#52606a")
    axis=fig.add_axes([.12,.31,.73,.44])
    bars=axis.twinx()
    bars.set_zorder(1);axis.set_zorder(2)
    axis.patch.set_visible(False);bars.patch.set_visible(False)
    for center,value in zip(centers,values):
        if np.isfinite(value):
            rectangle=Rectangle((center-.25,min(0,value)),.5,abs(value),
                                facecolor=mpl.colors.to_rgba(ORANGE,.14),edgecolor=ORANGE,lw=.8)
            rectangle.set_gid(f"heatmap-bin-{center:g}")
            bars.add_patch(rectangle)
    bars.axhline(0,color=ORANGE,lw=.5,alpha=.6)
    bars.set_ylim(lower,upper)
    bars.tick_params(axis="y",colors=ORANGE,labelsize=8,length=2)
    bars.set_ylabel("Signed log vigor",color=ORANGE,fontsize=9,labelpad=5)
    bars.spines["top"].set_visible(False);bars.spines["left"].set_visible(False)
    bars.spines["right"].set_color(ORANGE)
    line,=axis.plot(raw[TIME],raw.Vigor,color="#111111",lw=.4,rasterized=False)
    line.set_gid("raw-vigor-untransformed")
    axis.axvline(0,color=theme.cs_color,lw=1)
    axis.text(0,1.035,"CS onset",transform=axis.get_xaxis_transform(),fontsize=8,color=theme.cs_color,ha="center")
    axis.set(xlim=(start,stop),ylim=(0,cap),ylabel="Raw vigor (rad/ms)")
    axis.yaxis.label.set_size(9)
    axis.tick_params(labelbottom=False,labelsize=8,length=2)
    axis.spines["top"].set_visible(False);axis.spines["right"].set_visible(False)
    # An exact colour row makes the relationship between inset bars and F explicit.
    strip=fig.add_axes([.12,.205,.73,.052])
    cmap=heatmap_cmap(theme.single_fish_scaled_vigor_cmap,theme)
    norm=Normalize(-.25,.25,clip=True)
    for center,value in zip(centers,values):
        strip.add_patch(Rectangle((center-.25,0),.5,1,
                        facecolor=cmap(norm(value)) if np.isfinite(value) else "black",edgecolor="none"))
    strip.set(xlim=(start,stop),ylim=(0,1),yticks=[],xticks=np.arange(-3,6))
    strip.tick_params(labelsize=8,length=2)
    strip.set_xlabel("Time relative to CS onset (s)",fontsize=9,labelpad=3)
    strip.text(-.02,.5,"F row",ha="right",va="center",transform=strip.transAxes,fontsize=8)
    fig.text(.10,.025,f"Inset y limit: {cap:g} rad/ms; full peaks in overview. Empty bars / black cells: missing bin value.",fontsize=7.5,color="#52606a")
    detail=OUTPUT/"Fig1_PanelE_HeatmapBinInset_legacy-vigor_v3.svg"
    fig.savefig(detail,format="svg",metadata={"Title":"Figure 1E raw vigor and exact heatmap-bin inset"})
    plt.close(fig)
    overview=OUTPUT/"Fig1_PanelE_RawVigor_InsetWindow_legacy-vigor_v3.svg"
    draw(frames,events,panel="E",output=overview,inset_selection=(trial,start,stop))
    data=detail.with_suffix(".parquet");bins.to_parquet(data,index=False)
    stored=pd.read_parquet(data)
    np.testing.assert_array_equal(stored["Signed log vigor"].to_numpy(),values)
    sidecar={**meta,"version":3,"selection_status":"review", "baseline_s":[-15,0],
        "inset_trial":trial,"inset_window_s":[start,stop],"raw_vigor_transform":"none",
        "inset_raw_y_range":[0,cap],"bar_value_source":str(heat_path),"bar_value_source_sha256":digest(heat_path),
        "bar_semantics":"Exact stored signed log-vigor heatmap bins; no new transformation or normalization",
        "bar_edges":"0.5 s heatmap bin boundaries, not individual detected bout boundaries",
        "colour_strip_range":[-.25,.25],"colour_strip_palette":theme.single_fish_scaled_vigor_cmap,
        "panel_data":str(data),"panel_data_sha256":digest(data)}
    for path in (detail,overview):
        path.with_suffix(".svg.json").write_text(json.dumps({**sidecar,"svg":str(path),"svg_sha256":digest(path)},indent=2)+"\n",encoding="utf-8")
        print(path)


if __name__=="__main__":
    main()
