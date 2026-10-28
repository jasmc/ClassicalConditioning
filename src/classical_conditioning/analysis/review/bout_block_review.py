"""Compare selected D/E ratios with the legacy log-median outcome on corrected frames.

This review leaves the selected panels and all freeze records unchanged. It
implements the legacy outcome, not obsolete interpolation/smoothing/downsampling.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import gc
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu, wilcoxon, skew, binomtest
from statsmodels.stats.multitest import multipletests



SELECT = Path('J:/ClassicalConditioning Outputs/figures/figure-02/panels-d-e-f/versions/row2-block-ratio-review/analysis-selection.json')
COL = "legacy_distal_angular_speed_rad_per_ms"
VALUE = "Fish median response / baseline"
BLOCKS = [(10, 14), (65, 69), (90, 94)]
PAIRS = [(0, 1), (1, 2), (0, 2)]
LABELS = ["PT", "ET", "LT"]
COLORS = {"control": "#29abe2", "delay": "#e90e8b", "trace": "#f15b2d"}
NAMES = {"control": "Control", "delay": "Delay", "trace": "3 s Trace"}


@lru_cache(maxsize=None)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(str(path))}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stars(p):
    return "****" if p < .0001 else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def read_windows(path, columns, intervals):
    """Prune row groups and retain original floating-point measured timestamps."""
    parquet = pq.ParquetFile(path)
    index = parquet.schema.names.index("AbsoluteTime")
    pieces = []
    for i in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(i).column(index).statistics
        assert stats is not None and stats.min is not None
        overlap = [(a,b) for a,b in intervals if stats.max >= a and stats.min < b]
        if not overlap: continue
        d = parquet.read_row_group(i, columns=columns).to_pandas()
        times = d.AbsoluteTime.to_numpy(dtype=float)
        keep = np.zeros(len(d), bool)
        for a,b in overlap: keep |= (times >= a) & (times < b)
        pieces.append(d.loc[keep])
    assert pieces
    return pd.concat(pieces, ignore_index=True)


def calculate_trials(metrics, movement, cycles, trial_ids, end):
    """Legacy median(log positive moving vigor) difference, with measured-time windows."""
    assert np.array_equal(metrics[["FrameID", "AbsoluteTime"]].to_numpy(), movement[["FrameID", "AbsoluteTime"]].to_numpy())
    absolute = metrics.AbsoluteTime.to_numpy(dtype=float)
    assert np.all(np.diff(absolute) >= 0)
    raw = metrics[COL].to_numpy(dtype=float)
    eligible = movement.valid.to_numpy(bool) & movement.moving.to_numpy(bool) & np.isfinite(raw) & (raw > 0)
    logged = np.full(len(raw), np.nan)
    logged[eligible] = np.log(raw[eligible])
    rows = []
    for trial in trial_ids:
        onset = float(cycles.iloc[trial - 1].Beg)
        masks = [(absolute >= onset - 15000) & (absolute < onset), (absolute >= onset) & (absolute < onset + end * 1000)]
        samples = [logged[m][np.isfinite(logged[m])] for m in masks]
        medians = [float(np.median(x)) if len(x) else np.nan for x in samples]
        rows.append({"trial_number": trial, "baseline_log_median": medians[0], "response_log_median": medians[1],
                     "legacy_log_median_difference": medians[1] - medians[0],
                     "baseline_positive_moving_samples": len(samples[0]), "response_positive_moving_samples": len(samples[1]),
                     "baseline_measured_frames": int(masks[0].sum()), "response_measured_frames": int(masks[1].sum())})
    return pd.DataFrame(rows)


def block_statistics(data, version):
    rows, changes, diagnostics = [], [], []
    for panel, d in data.items():
        conditioned = "delay" if panel == "D" else "trace"
        w = d.loc[d.Eligible].pivot(index=["condition_id", "fish_id"], columns="Selected block order", values=VALUE).reindex(columns=[0, 1, 2])
        for block in range(3):
            t, c = w.xs(conditioned)[block].dropna(), w.xs("control")[block].dropna()
            r = mannwhitneyu(t, c, alternative="two-sided", method="asymptotic", use_continuity=True)
            rows.append(dict(version=version, panel=panel, family="between", condition=conditioned+" vs control", left=block, right=block,
                             test="Mann-Whitney U", n_control=len(c), n_conditioned=len(t), n_pairs=np.nan, statistic=r.statistic, p_raw=r.pvalue,
                             method="asymptotic; tie/continuity corrected", rank_biserial=2*r.statistic/(len(t)*len(c))-1))
        for lo, hi in PAIRS:
            deltas = {}
            for condition in [conditioned, "control"]:
                paired = w.xs(condition)[[lo, hi]].dropna()
                delta = (paired[hi] - paired[lo]).to_numpy()
                deltas[condition] = delta
                nonzero = delta[delta != 0]
                ties = len(np.unique(np.abs(nonzero))) != len(nonzero)
                method = "exact" if len(nonzero) == len(delta) and not ties else "asymptotic"
                r = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method=method) if len(nonzero) else None
                rows.append(dict(version=version, panel=panel, family="within", condition=condition, left=lo, right=hi,
                                 test="paired Wilcoxon signed-rank", n_control=np.nan, n_conditioned=np.nan, n_pairs=len(delta),
                                 statistic=float(r.statistic) if r else 0., p_raw=float(r.pvalue) if r else 1., method=method+"; zero_method=wilcox",
                                 median_change=float(np.median(delta)), zero_differences=int((delta == 0).sum()), absolute_difference_ties=ties))
                sign_p = binomtest(int((nonzero > 0).sum()), len(nonzero), .5).pvalue if len(nonzero) else 1.
                diagnostics.append(dict(version=version, panel=panel, condition=condition, left=lo, right=hi, n=len(delta),
                                        skewness=float(skew(delta, bias=False)) if len(delta)>2 and np.std(delta)>0 else 0.,
                                        median_change=float(np.median(delta)), minimum=float(delta.min()), maximum=float(delta.max()),
                                        sign_test_raw_p=sign_p, zero_differences=int((delta==0).sum()), absolute_difference_ties=ties))
                for (fish, values), change in zip(paired.iterrows(), delta):
                    changes.append(dict(version=version, panel=panel, condition=condition, left=lo, right=hi, fish_id=fish,
                                        left_value=float(values[lo]), right_value=float(values[hi]), change=float(change)))
            t, c = deltas[conditioned], deltas["control"]
            r = mannwhitneyu(t, c, alternative="two-sided", method="asymptotic", use_continuity=True)
            rows.append(dict(version=version, panel=panel, family="change", condition=conditioned+" vs control", left=lo, right=hi,
                             test="Mann-Whitney U on within-fish changes", n_control=len(c), n_conditioned=len(t), n_pairs=np.nan,
                             statistic=r.statistic, p_raw=r.pvalue, method="asymptotic; tie/continuity corrected",
                             median_change_conditioned=float(np.median(t)), median_change_control=float(np.median(c)),
                             descriptive_difference_of_median_changes=float(np.median(t)-np.median(c)), rank_biserial=2*r.statistic/(len(t)*len(c))-1))
    tests = pd.DataFrame(rows)
    assert len(tests) == 24 and not tests.duplicated(["panel", "family", "condition", "left", "right"]).any()
    assert np.isfinite(tests.p_raw).all()
    tests["p_holm24"] = multipletests(tests.p_raw, method="holm")[1]
    tests["stars"] = tests.p_holm24.map(stars)
    diag = pd.DataFrame(diagnostics)
    diag["sign_test_holm12_p"] = multipletests(diag.sign_test_raw_p, method="holm")[1]
    return tests, pd.DataFrame(changes), diag


def render_panel(panel, data, tests, version, out, all_tests=False):
    conditioned = "delay" if panel == "D" else "trace"
    ratio = version == "ratio"
    ref = 1. if ratio else 0.
    finite = data.loc[data.Eligible, VALUE].dropna().to_numpy()
    if ratio:
        limits = (min(.5, np.floor(finite.min()*10)/10-.05), max(1.6, np.ceil(finite.max()*10)/10+.05))
        ticks = MaxNLocator(nbins=5).tick_values(*limits)
        ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    else:
        radius = max(.2, np.ceil(np.max(np.abs(finite)) * 10) / 10 + .05)
        limits = (-radius, radius)
        ticks = MaxNLocator(nbins=5).tick_values(*limits)
        ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    assert finite.min() > limits[0] and finite.max() < limits[1]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 6.1), sharey=True)
    between_significant = bool((tests.panel.eq(panel) & tests.family.eq("between") & tests.p_holm24.lt(.05)).any())
    fig.subplots_adjust(left=.205 if ratio else .235, right=.975, bottom=.135,
                        top=.60 if all_tests or between_significant else .73, wspace=.15)
    texts, headings = [], []
    for ai, (ax, condition) in enumerate(zip(axes, [conditioned, "control"])):
        sub = data.loc[data.condition_id.eq(condition) & data.Eligible]
        wide = sub.pivot(index="fish_id", columns="Selected block order", values=VALUE).reindex(columns=[0, 1, 2])
        ax.axhline(ref, color="#000000", alpha=1, lw=.85, zorder=0).set_gid(f"fig2-{panel}-{version}-{condition}-reference")
        for fish, row in wide.iterrows():
            ax.plot(range(3), row, color=to_rgba(COLORS[condition], .37), lw=.9, zorder=1)
            ax.scatter(range(3), row, s=17, facecolors=to_rgba(COLORS[condition], .30), edgecolors=(0, 0, 0, .40), linewidths=.65, zorder=2)
        q = np.array([sub.loc[sub["Selected block order"].eq(i), VALUE].quantile([.25, .5, .75]).to_numpy() for i in range(3)])
        ax.errorbar(range(3), q[:, 1], yerr=[q[:, 1]-q[:, 0], q[:, 2]-q[:, 1]], fmt="o-", color="black", alpha=.72,
                    lw=2.1, elinewidth=1.45, capsize=3.8, capthick=1.45, ms=5.5, zorder=5)
        ax.set_xlim(-.15, 2.15); ax.set_ylim(*limits)
        ax.set_xticks(range(3), LABELS, fontsize=22, fontweight="bold")
        ax.set_yticks(ticks, [f"{x:g}" for x in ticks], fontsize=17)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        if ai == 0:
            ax.spines["left"].set_linewidth(1.4)
            ax.set_ylabel("Response / baseline" if ratio else "Median ln vigor\n(response − baseline)", fontsize=18 if ratio else 16, fontweight="bold", labelpad=9)
        else:
            ax.spines["left"].set_visible(False); ax.tick_params(axis="y", left=False, labelleft=False)
        headings.append(fig.text(ax.get_position().x0+ax.get_position().width/2, .95, NAMES[condition], ha="center", color=COLORS[condition], fontsize=23))
        counts = [int(wide[i].notna().sum()) for i in range(3)]
        fig.text(ax.get_position().x0+ax.get_position().width/2, .905, "n="+"/".join(map(str, counts))+" at PT/ET/LT", ha="center", fontsize=10)
        within = tests.loc[tests.panel.eq(panel) & tests.family.eq("within") & tests.condition.eq(condition)]
        for j, (lo, hi) in enumerate(PAIRS):
            r = within.loc[within.left.eq(lo) & within.right.eq(hi)].iloc[0]
            if not all_tests and r.p_holm24 >= .05: continue
            level = 1.025 + .09*j
            ax.plot([lo, hi], [level, level], transform=ax.get_xaxis_transform(), color="#252525", lw=1.25, clip_on=False)
            texts.append(ax.text((lo+hi)/2, level+.006, r.stars, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=15, clip_on=False))
    between = tests.loc[tests.panel.eq(panel) & tests.family.eq("between")]
    for j in range(3):
        r = between.loc[between.left.eq(j)].iloc[0]
        if not all_tests and r.p_holm24 >= .05: continue
        level = 1.315 + .09*j
        fig.add_artist(ConnectionPatch((j, level), (j, level), coordsA=axes[0].get_xaxis_transform(), coordsB=axes[1].get_xaxis_transform(), color="#252525", lw=1.25, clip_on=False))
        a = fig.transFigure.inverted().transform(axes[0].get_xaxis_transform().transform((j, level)))
        b = fig.transFigure.inverted().transform(axes[1].get_xaxis_transform().transform((j, level)))
        texts.append(fig.text((a[0]+b[0])/2, a[1]+.006, r.stars, ha="center", va="bottom", fontsize=15))
    fig.text(.205, .045, "Fish median and IQR · Holm24 · "+("all block tests" if all_tests else "significant block tests"), fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in texts+headings+[axes[0].yaxis.label]:
        b = text.get_window_extent(renderer)
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height, (panel, version, text.get_text(), tuple(b.extents), tuple(fig.bbox.extents))
    for text in texts:
        assert not any(text.get_window_extent(renderer).overlaps(h.get_window_extent(renderer)) for h in headings)
    stem = f"Fig2_{panel}_{version}" + ("_all_tests" if all_tests else "")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    write(out/f"{stem}.figure.json", {"panel":panel, "version":version, "status":"exploratory candidate; not frozen/selected",
          "baseline_s":[-15,0], "response_s":[0,9 if panel=="D" else 13], "blocks":BLOCKS, "data_limits":list(limits), "ticks":list(ticks),
          "reference":ref, "summary":"equal-fish median and fish IQR; not CI", "data_clipping":False,
          "data":artifact(out/f"{panel}_{version}_fish_blocks.parquet"), "statistics":artifact(out/f"{version}_tests.csv"),
          "code":artifact(__file__), "outputs":[artifact(out/f"{stem}.{ext}") for ext in ["png","svg","pdf"]]})


def render_changes(changes, tests, version, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for panel, ax in zip(["D", "E"], axes):
        conditioned = "delay" if panel=="D" else "trace"
        d = changes.loc[changes.panel.eq(panel)]
        for j, (lo, hi) in enumerate(PAIRS):
            for c, offset in [(conditioned,-.16),("control",.16)]:
                values = d.loc[d.condition.eq(c) & d.left.eq(lo) & d.right.eq(hi),"change"].to_numpy()
                ax.scatter(j+offset+np.linspace(-.05,.05,len(values)), values, color=COLORS[c], alpha=.4, s=16)
                q = np.quantile(values,[.25,.5,.75])
                ax.errorbar(j+offset,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt="o",color="black",alpha=.72,capsize=3)
            r=tests.loc[tests.panel.eq(panel)&tests.family.eq("change")&tests.left.eq(lo)&tests.right.eq(hi)].iloc[0]
            ax.text(j,1.02,r.stars+f"\np={r.p_holm24:.3g}",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=10)
        ax.axhline(0,color="black",alpha=1,lw=.85,zorder=0)
        ax.set_xticks(range(3),["ET − PT","LT − ET","LT − PT"])
        ax.set_ylabel("Within-fish change ("+("ratio" if version=="ratio" else "ln vigor difference")+")")
        ax.set_title(f"{panel}: {NAMES[conditioned]} / control",pad=53)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/f"{version}_direct_changes.png",dpi=170)
    fig.savefig(out/f"{version}_direct_changes.pdf")
    plt.close(fig)


def validate_calculation():
    times=np.array([-14000.,-1000.,0.,2000.,8999.,9000.])
    m=pd.DataFrame({"FrameID":range(6),"AbsoluteTime":times,COL:[1.,4.,2.,8.,0.,999.]})
    state=pd.DataFrame({"FrameID":range(6),"AbsoluteTime":times,"valid":[True]*6,"moving":[True]*6})
    r=calculate_trials(m,state,pd.DataFrame({"Beg":[0.]}),[1],9).iloc[0]
    assert np.isclose(r.legacy_log_median_difference,np.log(2.))
    assert r.response_positive_moving_samples==2
    scaled=m.copy(); scaled[COL]*=180/np.pi
    s=calculate_trials(scaled,state,pd.DataFrame({"Beg":[0.]}),[1],9).iloc[0]
    assert np.isclose(r.legacy_log_median_difference,s.legacy_log_median_difference)
    state.loc[2:4,"moving"]=False
    assert np.isnan(calculate_trials(m,state,pd.DataFrame({"Beg":[0.]}),[1],9).iloc[0].legacy_log_median_difference)
