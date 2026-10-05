"""Figure 1E: raw overview plus two vector y-only zooms with detected-bout bars."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classical_conditioning.figures.theme import apply_theme
from build_figure1_trace_panels import OUTPUT, PROJECT, FISH, TRIALS, STAGES, digest
from build_figure1_legacy_vigor_heatmaps import read_windows

ORANGE = "#c85a17"
TIME = "Time relative to CS onset (s)"


def bout_bars(part: pd.DataFrame) -> pd.DataFrame:
    """Same per-bout median as the heatmap kernel, on detected temporal support."""
    x = part[TIME].to_numpy(float)
    y = part["Vigor"].to_numpy(float)
    ids = part["bout_id"].to_numpy(int)
    support = part["valid"].to_numpy(bool) & part["moving"].to_numpy(bool) & (ids > 0)
    usable = support & np.isfinite(y) & (y > 0)
    baseline = np.log(y[usable & (x >= -15) & (x < 0)])
    if not len(baseline):
        raise ValueError("No usable pre-CS baseline for zoom")
    baseline_median = float(np.median(baseline))
    medians = {bid: float(np.median(np.log(y[usable & (ids == bid)])) - baseline_median)
               for bid in np.unique(ids[usable])}
    # Half-open sample intervals: the offset is the next observed frame time.
    change = np.r_[True, (ids[1:] != ids[:-1]) | (support[1:] != support[:-1]), True]
    bounds = np.flatnonzero(change)
    rows = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if support[start] and ids[start] in medians:
            rows.append({"start_s": float(x[start]),
                         "stop_s": float(x[stop]) if stop < len(x) else 20.,
                         "bout_id": int(ids[start]), "signed_log_vigor": medians[ids[start]],
                         "baseline_median_log_vigor": baseline_median})
    return pd.DataFrame(rows)


def style(axis, theme, events, trial, *, ylim, ticks):
    axis.set_xlim(-20, 20)
    axis.set_ylim(*ylim)
    axis.set_yticks(ticks)
    axis.set_xticks([-20, -10, 0, 10, 20])
    axis.tick_params(labelsize=8, length=2.5, width=.65)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(.65)
    for event in events.loc[events["Trial number"].eq(trial)].itertuples(index=False):
        axis.axvline(float(event[2]), color=theme.us_color if event.Event == "actual US onset"
                     else theme.cs_color, lw=.7,
                     linestyle="--" if event.Event == "CS offset" else "-", zorder=4)



def add_zoom(fig, rect, trial, stage, *, frames, bars, events, theme, cap, bout_cap):
    axis=fig.add_axes(rect)
    right=axis.twinx()
    right.set_zorder(1)
    axis.set_zorder(2)
    axis.patch.set_visible(False)
    right.patch.set_visible(False)
    for bar in bars[trial].itertuples(index=False):
        right.add_patch(Rectangle((bar.start_s,min(0,bar.signed_log_vigor)),
            bar.stop_s-bar.start_s,abs(bar.signed_log_vigor),
            facecolor=mpl.colors.to_rgba(ORANGE,.12),edgecolor=ORANGE,linewidth=.85))
    right.axhline(0,color=ORANGE,lw=.5,alpha=.35)
    right.set_ylim(-bout_cap,bout_cap)
    right.set_yticks([-bout_cap,0,bout_cap])
    right.tick_params(axis="y",labelsize=8,colors=ORANGE,length=2)
    right.spines["top"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.spines["right"].set_color(ORANGE)
    right.spines["right"].set_linewidth(.6)
    part=frames.loc[frames["Trial number"].eq(trial)]
    axis.plot(part[TIME],part.Vigor,color="#111111",lw=.5,rasterized=False,zorder=3)
    style(axis,theme,events,trial,ylim=(0,cap),ticks=[0,cap/2,cap])
    axis.set_title(stage,fontsize=10,weight="bold",loc="left",pad=6)
    axis.set_xlabel("Time from CS onset (s)",fontsize=8)

def main():
    frames_path = OUTPUT / "Fig1_PanelsD-E_Delay_legacy-vigor_frames_v1.parquet"
    events_path = OUTPUT / "Fig1_PanelsD-E_Delay_events_v1.parquet"
    old = json.loads((OUTPUT / "Fig1_PanelE_RawVigor_legacy-vigor_v1.svg.json").read_text())
    if digest(frames_path) != old["frames_sha256"] or digest(events_path) != old["events_sha256"]:
        raise ValueError("Cached trace data changed")
    frames, events = pd.read_parquet(frames_path), pd.read_parquet(events_path)
    protocol = pd.read_parquet(PROJECT / "Processed data" / FISH / "stimulus_events.parquet")
    cycles = protocol.loc[protocol.Type.eq("Cycle")].sort_values("Beg").reset_index(drop=True)
    movement_path = PROJECT / "Processed data" / FISH / "movement_state_candidates-corrected-v2.parquet"
    marker_path = PROJECT / "Metadata" / f"{FISH}_movement-candidate-corrected-v2_complete.json"
    marker = json.loads(marker_path.read_text())
    if marker["status"] != "complete" or digest(movement_path) != marker["movement_sha256"]:
        raise ValueError("Movement artifact differs from completion marker")
    intervals = [(int(cycles.iloc[t-1].Beg)-20000, int(cycles.iloc[t-1].Beg)+20000) for t in (17,63)]
    movement = read_windows(movement_path, ["FrameID", "AbsoluteTime", "valid", "moving", "bout_id"], intervals)
    joined = frames.loc[frames["Trial number"].isin([17,63]) & frames[TIME].lt(20)].merge(
        movement, on="FrameID", how="left", validate="one_to_one")
    if joined["valid"].isna().any():
        raise ValueError("Missing movement frames")
    bars = {t: bout_bars(joined.loc[joined["Trial number"].eq(t)]) for t in (17,63)}
    heatmap_path = OUTPUT.parent / "heatmaps" / "Fig1_PanelF_Delay_legacy-vigor_v2.parquet"
    heatmap_meta = json.loads(heatmap_path.with_suffix(".svg.json").read_text())
    if digest(heatmap_path) != heatmap_meta["panel_data_sha256"]:
        raise ValueError("Corrected heatmap data changed")
    heatmap = pd.read_parquet(heatmap_path)
    if not heatmap["Baseline start (s)"].eq(-15).all() or not heatmap["Baseline end (s)"].eq(0).all():
        raise ValueError("Heatmap uses a different baseline")
    for trial in (17, 63):
        usable = joined.loc[joined["Trial number"].eq(trial) & joined.valid & joined.moving
                            & joined.bout_id.gt(0) & np.isfinite(joined.Vigor) & joined.Vigor.gt(0)].copy()
        mapping = bars[trial].drop_duplicates("bout_id").set_index("bout_id").signed_log_vigor
        usable["bar_value"] = usable.bout_id.map(mapping)
        usable["bin"] = np.floor((usable[TIME]+20)/.5)*.5-19.75
        reconstructed = usable.groupby("bin").bar_value.mean().reindex(np.arange(-19.75,20,.5))
        expected = heatmap.loc[heatmap["Trial number"].eq(trial)].sort_values("Time bin center (s)")
        np.testing.assert_allclose(reconstructed.to_numpy(), expected["Signed log vigor"].to_numpy(),
                                   atol=1e-12, rtol=0, equal_nan=True)
    all_bars = pd.concat([part.assign(trial=t) for t,part in bars.items()], ignore_index=True)
    data_path = OUTPUT / "Fig1_PanelE_BoutBars_pre15_v2.parquet"
    all_bars.to_parquet(data_path, index=False)
    theme = apply_theme()
    mpl.rcParams.update({"svg.fonttype":"none", "path.simplify":False})
    fig = plt.figure(figsize=(8.15,7.0), layout="none")
    fig.text(.09,.974,"Frame vigor (rad/ms) · Delay fish 20221115_07", fontsize=13, weight="bold", va="top")
    fig.text(.09,.938,"Raw legacy distal angular speed", fontsize=9, color="#52606a", va="top")
    full_max = math.ceil(float(frames.Vigor.max())*2)/2
    for i,(trial,stage) in enumerate(zip(TRIALS,STAGES)):
        axis=fig.add_axes([.21,.823-i*.084,.765,.066])
        part=frames.loc[frames["Trial number"].eq(trial)]
        axis.plot(part[TIME],part.Vigor,color="#111111",lw=.48,rasterized=False,zorder=3)
        style(axis,theme,events,trial,ylim=(0,full_max),ticks=[0,full_max])
        axis.text(-.19,.5,stage,transform=axis.transAxes,ha="right",va="center",fontsize=9,weight="bold")
        axis.tick_params(labelbottom=i==4)
        if i==4:
            axis.set_xlabel(TIME,fontsize=9)
    # The two inset views use identical x ranges and y limits.
    cap=math.ceil(float(joined.Vigor.quantile(.995))*10)/10
    bout_cap=math.ceil(float(all_bars.signed_log_vigor.abs().max())*2)/2
    fig.text(.09,.365,"Enlarged y scale · Early and Late Train",fontsize=11,weight="bold")
    fig.text(.09,.337,f"Black: raw vigor (0–{cap:g} rad/ms); orange: bout log vigor Δ (right axis)",fontsize=8.5,color="#52606a")
    for trial,stage,left in ((17,"Early Train",.11),(63,"Late Train",.59)):
        add_zoom(fig,[left,.105,.32,.19],trial,stage,frames=frames,bars=bars,
                 events=events,theme=theme,cap=cap,bout_cap=bout_cap)
    fig.text(.09,.033,"Zoom only: peaks above the y limit are clipped; full amplitudes are shown above.",fontsize=8,color="#52606a")
    svg=OUTPUT / "Fig1_PanelE_RawVigor_withTrainZoom_legacy-vigor_v2.svg"
    fig.savefig(svg,format="svg",metadata={"Title":"Figure 1E raw vigor and training-trial y zooms",
        "Description":"Full 40 s windows. Bout bars on right axes are behind raw traces; baseline [-15,0) s."})
    plt.close(fig)
    sidecar={**old,"svg":str(svg),"svg_sha256":digest(svg),"version":2,
        "zoom_trials":[17,63],"zoom_window_s":[-20,20],"zoom_raw_y_range":[0,cap],
        "zoom_cap_rule":"ceil to 0.1 rad/ms of pooled selected trials' 99.5th percentile",
        "bout_bar_range":[-bout_cap,bout_cap],"baseline_s":[-15,0],
        "bars":"Detected-bout median natural-log vigor minus trial moving-frame median log vigor; before heatmap bin averaging",
        "bar_edges":"Half-open detected moving intervals, ending at the next observed frame time",
        "bout_data":str(data_path),"bout_data_sha256":digest(data_path),
        "movement":str(movement_path),"movement_sha256":marker["movement_sha256"]}
    sidecar.update({"heatmap_data":str(heatmap_path),"heatmap_data_sha256":digest(heatmap_path),
                    "heatmap_agreement":"Frame-weighted bar values reproduce both heatmap rows within 1e-12"})
    svg.with_suffix(".svg.json").write_text(json.dumps(sidecar,indent=2)+"\n",encoding="utf-8")
    # A separate detail strip lets the assembler place the two E insets below
    # the aligned D/E overview traces without shrinking any lettering.
    detail = plt.figure(figsize=(16.9,2.8),layout="none")
    detail.text(.035,.94,"E · enlarged-y details",fontsize=13,weight="bold",va="top")
    detail.text(.035,.84,f"Black: raw vigor (0–{cap:g} rad/ms) · Orange: bout log vigor Δ (right axis); baseline −15 to 0 s",
                fontsize=9,color="#52606a",va="top")
    for trial,stage,left in ((17,"Early Train",.07),(63,"Late Train",.57)):
        add_zoom(detail,[left,.24,.37,.48],trial,stage,frames=frames,bars=bars,
                 events=events,theme=theme,cap=cap,bout_cap=bout_cap)
    detail.text(.035,.025,"Only the y scale is enlarged. Clipped raw peaks remain visible in the overview; bars end at detected bout boundaries.",
                fontsize=8,color="#52606a")
    detail_svg=OUTPUT / "Fig1_PanelE_TrainZoomDetail_legacy-vigor_v2.svg"
    detail.savefig(detail_svg,format="svg")
    plt.close(detail)
    detail_svg.with_suffix(".svg.json").write_text(json.dumps({**sidecar,
        "svg":str(detail_svg),"svg_sha256":digest(detail_svg),"role":"E training-trial zoom inset strip"},indent=2)+"\n",encoding="utf-8")
    print(detail_svg)
    print(svg)


if __name__=="__main__":
    main()
