"""All five Figure 1E trials: raw vigor over exact, baseline-centred heatmap bins."""
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from build_figure1_trace_panels import OUTPUT, FISH, TRIALS, STAGES, digest
from classical_conditioning.figures.theme import apply_theme

TIME = "Time relative to CS onset (s)"
ORANGE = "#c85a17"


def main():
    meta = json.loads((OUTPUT / "Fig1_PanelE_RawVigor_legacy-vigor_v1.svg.json").read_text(encoding="utf-8"))
    for key in ("frames", "events"):
        if digest(Path(meta[key])) != meta[f"{key}_sha256"]:
            raise ValueError(f"Changed raw source: {key}")
    frames, events = pd.read_parquet(meta["frames"]), pd.read_parquet(meta["events"])
    from classical_conditioning.analysis.bout_vigor import VIGOR_SAMPLE_POLICY
    if "Vigor sample policy" not in frames or not frames["Vigor sample policy"].eq(VIGOR_SAMPLE_POLICY).all():
        raise ValueError("Superseded all-frame vigor cache: rebuild traces with the shared bout mask first")
    heat_path = OUTPUT.parent / "heatmaps/Fig1_PanelF_Delay_legacy-vigor_v2.parquet"
    heat_meta = json.loads(heat_path.with_suffix(".svg.json").read_text(encoding="utf-8"))
    if heat_meta["recording_id"] != FISH or digest(heat_path) != heat_meta["panel_data_sha256"]:
        raise ValueError("Heatmap source identity/hash mismatch")
    heat = pd.read_parquet(heat_path)
    if not heat["Baseline start (s)"].eq(-15).all() or not heat["Baseline end (s)"].eq(0).all():
        raise ValueError("Frozen baseline must be [-15,0) s")
    selected = heat.loc[heat["Trial number"].isin(TRIALS)].copy()
    finite = selected["Signed log vigor"].dropna()
    low = min(-.25, math.floor(float(finite.min())*4)/4)
    high = max(.25, math.ceil(float(finite.max())*4)/4)
    cap = math.ceil(float(frames.Vigor.quantile(.995))*10)/10
    theme = apply_theme()
    mpl.rcParams.update({"svg.fonttype":"none", "path.simplify":False})
    fig, axes = plt.subplots(5,1,figsize=(8.15,4.8),sharex=True,layout="none")
    fig.subplots_adjust(left=.21,right=.89,top=.86,bottom=.14,hspace=.13)
    fig.text(.09,.97,f"Raw vigor · Delay fish {FISH}",fontsize=13,weight="bold",va="top")
    fig.text(.09,.91,"Black: raw vigor (rad/ms, left)",fontsize=8.5,va="top")
    fig.text(.49,.91,"Orange: 0.5 s heatmap values (right)",fontsize=8.5,color=ORANGE,va="top")
    clipped = {}
    for axis,trial,stage in zip(axes,TRIALS,STAGES,strict=True):
        part = frames.loc[frames["Trial number"].eq(trial)]
        bins = selected.loc[selected["Trial number"].eq(trial)].sort_values("Time bin center (s)")
        centers = bins["Time bin center (s)"].to_numpy(float)
        np.testing.assert_array_equal(centers,np.arange(-19.75,20,.5))
        values = bins["Signed log vigor"].to_numpy(float)
        bars = axis.twinx()
        bars.set_zorder(1); axis.set_zorder(2)
        bars.patch.set_visible(False); axis.patch.set_visible(False)
        for center,value in zip(centers,values):
            if np.isfinite(value):
                rectangle = Rectangle((center-.25,min(0,value)),.5,abs(value),
                    facecolor=mpl.colors.to_rgba(ORANGE,.16),edgecolor=ORANGE,lw=.65)
                rectangle.set_gid(f"heatmap-trial-{trial}-bin-{center:g}")
                bars.add_patch(rectangle)
        bars.axhline(0,color=ORANGE,lw=.45,alpha=.55)
        bars.set_ylim(low,high); bars.set_yticks([low,0,high])
        bars.tick_params(axis="y",labelsize=7,colors=ORANGE,length=2)
        bars.spines["top"].set_visible(False);bars.spines["left"].set_visible(False)
        bars.spines["right"].set_color(ORANGE);bars.spines["right"].set_linewidth(.6)
        line, = axis.plot(part[TIME],part.Vigor,color="#111111",lw=.35,rasterized=False)
        line.set_gid(f"raw-vigor-trial-{trial}")
        for event in events.loc[events["Trial number"].eq(trial)].itertuples(index=False):
            axis.axvline(float(event[2]),color=theme.us_color if event.Event=="actual US onset" else theme.cs_color,
                         lw=.7,linestyle="--" if event.Event=="CS offset" else "-")
        axis.set(xlim=(-20,20),ylim=(0,cap),yticks=[0,cap])
        axis.tick_params(labelsize=7,length=2)
        axis.spines["top"].set_visible(False);axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(.6);axis.spines["bottom"].set_linewidth(.6)
        axis.text(-.19,.5,stage,transform=axis.transAxes,ha="right",va="center",fontsize=9,weight="bold")
        clipped[str(trial)] = int(part.Vigor.gt(cap).sum())
    axes[-1].set_xticks([-20,-10,0,10,20])
    axes[-1].set_xlabel(TIME,fontsize=10,weight="bold")
    fig.text(.09,.015,f"Y zoom only: raw peaks above {cap:g} rad/ms are clipped. Missing heatmap bins remain gaps.",fontsize=7.5,color="#52606a")
    svg = OUTPUT / "Fig1_PanelE_RawVigor_HeatmapOverlay_allTrials_v4.svg"
    fig.savefig(svg,format="svg",metadata={"Title":"Figure 1E raw vigor over exact heatmap bins, all five trials",
        "Description":"Full -20 to 20 s range. Raw trace untransformed. Bars use stored heatmap bins with [-15,0) s baseline."})
    plt.close(fig)
    data = svg.with_suffix(".parquet")
    selected.to_parquet(data,index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(data),selected.reset_index(drop=True))
    sidecar = {**meta,"version":4,"selection_status":"review","baseline_s":[-15,0],
        "raw_vigor_transform":"none","display_x_range_s":[-20,20],"raw_y_range":[0,cap],
        "raw_y_cap_rule":"Pooled five trials 99.5th percentile rounded up to 0.1 rad/ms",
        "clipped_raw_frames_per_trial":clipped,"bar_y_range":[low,high],
        "bar_semantics":"Exact stored 0.5 s signed log-vigor heatmap bins; gaps are missing values",
        "bar_source":str(heat_path),"bar_source_sha256":digest(heat_path),
        "bar_edges":"Half-second bin boundaries","bar_render_order":"behind raw black trace",
        "panel_data":str(data),"panel_data_sha256":digest(data),"svg":str(svg),"svg_sha256":digest(svg)}
    svg.with_suffix(".svg.json").write_text(json.dumps(sidecar,indent=2)+"\n",encoding="utf-8")
    print(svg)
    print(f"Raw axis: 0 to {cap:g}; heatmap axis: {low:g} to {high:g}; all five trials")


if __name__ == "__main__":
    main()
