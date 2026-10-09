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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
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

from populate_figure2_available import load_delay, load_trace, verified_heatmap_paths

SELECT = Path("J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row2-block-ratio-review/analysis-selection.json")
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
        limits, ticks = (.5, 1.6), [.5, 1., 1.5]
    else:
        radius = max(.2, np.ceil(np.max(np.abs(finite)) * 10) / 10 + .05)
        limits = (-radius, radius)
        ticks = MaxNLocator(nbins=5).tick_values(*limits)
        ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    assert finite.min() > limits[0] and finite.max() < limits[1]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 6.1), sharey=True)
    fig.subplots_adjust(left=.205, right=.975, bottom=.135, top=.60, wspace=.15)
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
            ax.set_ylabel("Response / baseline" if ratio else "Response − baseline\n(median ln vigor)", fontsize=18, fontweight="bold", labelpad=9)
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
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height, (panel, version, text.get_text())
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--render-only",action="store_true")
    args=parser.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    validate_calculation()
    current, logdata, inputs, records = {}, {}, [], {}
    if not args.render_only:
        selection=json.loads(SELECT.read_text()); inputs.append(artifact(SELECT))
        for panel in ["D","E"]:
            record_path=selection["current_panels"][panel]
            assert sha(record_path["path"])==record_path["sha256"]
            record=json.loads(Path(record_path["path"]).read_text()); records[panel]=record
            src=next(x for x in record["outputs"] if x["path"].endswith(".parquet"))
            assert sha(src["path"])==src["sha256"]
            current[panel]=pd.read_parquet(src["path"])
            shutil.copyfile(src["path"],out/f"{panel}_ratio_fish_blocks.parquet")
            inputs.extend([record_path,src])
            for ext in ["png","svg"]:
                src_export=next(x for x in record["outputs"] if x["path"].endswith("."+ext))
                assert sha(src_export["path"])==src_export["sha256"]
                shutil.copyfile(src_export["path"],out/f"Fig2_{panel}_selected_original.{ext}")
        for panel,exp,project,loader,end,min_trials in [
            ("D","allDelay",Path("J:/Digested Data/allDelay-full-v1"),load_delay,9,3),
            ("E","all3sTrace",Path("F:/Digested Data/all3sTrace-full-v1"),load_trace,13,1)]:
            outcomes,cohort,cohort_hash,source_inputs=loader(project)
            assert cohort_hash==records[panel]["analysis_identity"]["cohort_hash"]
            inputs.extend(source_inputs)
            eligible=outcomes.loc[outcomes.metric_id.eq("legacy_distal_angular_speed") & outcomes.alignment.eq("CS")].copy()
            eligible["current_trial_eligible"]=np.isfinite(eligible.response_total_activity)&np.isfinite(eligible.baseline_total_activity)&eligible.baseline_total_activity.gt(0)&eligible.response_total_activity.ge(0)
            trial_ids=[x for lo,hi in BLOCKS for x in range(lo,hi+1)]
            rows=[]
            for i,r in enumerate(cohort.itertuples(index=False),1):
                rid=str(r.recording_id); cache=out/f"cache_{panel}_{rid}.parquet"; metadata=cache.with_suffix(".json")
                paths,evidence=verified_heatmap_paths(project,rid,exp); inputs.extend(evidence)
                signature={"sources":[artifact(p) for p in paths],"code":artifact(__file__)}
                if cache.exists() and metadata.exists() and json.loads(metadata.read_text())==signature:
                    trials=pd.read_parquet(cache)
                else:
                    metrics_path,movement_path,protocol_path=paths
                    protocol=pd.read_parquet(protocol_path)
                    cycles=protocol.loc[protocol.Type.eq("Cycle")].sort_values("Beg").reset_index(drop=True)
                    assert len(cycles)>=94
                    intervals=[(float(cycles.iloc[t-1].Beg)-15000,float(cycles.iloc[t-1].Beg)+end*1000) for t in trial_ids]
                    metrics=read_windows(metrics_path,["FrameID","AbsoluteTime",COL],intervals)
                    movement=read_windows(movement_path,["FrameID","AbsoluteTime","valid","moving"],intervals)
                    trials=calculate_trials(metrics,movement,cycles,trial_ids,end)
                    trials.to_parquet(cache,index=False);write(metadata,signature)
                    del metrics,movement;gc.collect()
                trials["fish_id"]=str(r.fish_id);trials["recording_id"]=rid;trials["condition_id"]=str(r.condition_id)
                trials=trials.merge(eligible.loc[eligible.recording_id.eq(rid),["trial_number","current_trial_eligible"]],on="trial_number",validate="one_to_one")
                trials["log_trial_eligible"]=trials.current_trial_eligible&np.isfinite(trials.legacy_log_median_difference)
                rows.append(trials)
                print(f"{panel}: {i}/{len(cohort)} {rid}",flush=True)
            trials=pd.concat(rows,ignore_index=True);trials.to_parquet(out/f"{panel}_log_trials.parquet",index=False)
            blocks=[]
            for (fish,condition),d in trials.groupby(["fish_id","condition_id"]):
                for order,(lo,hi) in enumerate(BLOCKS):
                    sub=d.loc[d.trial_number.between(lo,hi)];valid=sub.loc[sub.log_trial_eligible]
                    ok=len(valid)>=min_trials
                    blocks.append({"fish_id":fish,"condition_id":condition,"Selected block order":order,"Selected block":LABELS[order],
                                   VALUE:float(valid.legacy_log_median_difference.median()) if ok else np.nan,"Eligible":ok,
                                   "eligible_log_trials":len(valid),"eligible_current_trials":int(sub.current_trial_eligible.sum()),"minimum_trials":min_trials})
            logdata[panel]=pd.DataFrame(blocks);logdata[panel].to_parquet(out/f"{panel}_log_fish_blocks.parquet",index=False)
        write(out/"inputs.json",inputs)
        write(out/"method.json",{"metric":"legacy_distal_angular_speed","variant":"legacy log-median outcome on current corrected measured-time frames",
              "formula":"median(log positive valid moving vigor in response) - median(log positive valid moving vigor in baseline); median eligible trials within fish/block",
              "baseline_s":[-15,0],"response_s":{"D":[0,9],"E":[0,13]},"intervals":"left-closed right-open","blocks":BLOCKS,
              "min_trials":{"D":3,"E":1},"eligibility":"current valid positive-baseline/nonnegative-response trial plus finite positive moving samples in both windows",
              "upstream_changes":"No additional historical rolling median/downsampling; no interpolation or historical detector restored; current corrected metric/detector used",
              "legacy_differences":"Historical code smoothed/downsampled, used inclusive window endpoints and optional >90% NaN filter; these are not imported into this current-panel outcome comparison",
              "statistics":"paired Wilcoxon; independent Mann-Whitney at blocks and on within-fish changes; Holm24 separately for each version",
              "scientific_status":"exploratory; unselected; no freeze","F":"placeholder; no authenticated inputs"})
    else:
        current={p:pd.read_parquet(out/f"{p}_ratio_fish_blocks.parquet") for p in ["D","E"]}
        logdata={p:pd.read_parquet(out/f"{p}_log_fish_blocks.parquet") for p in ["D","E"]}
    all_tests=[];summary=[];missing=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","axes.unicode_minus":False})
    for version,data in [("ratio",current),("log",logdata)]:
        tests,changes,diag=block_statistics(data,version)
        tests.to_csv(out/f"{version}_tests.csv",index=False);changes.to_csv(out/f"{version}_fish_changes.csv",index=False);diag.to_csv(out/f"{version}_paired_diagnostics.csv",index=False)
        if version=="ratio":
            prior=pd.read_csv(records["D"]["results"]["path"]) if records else pd.read_csv(out/"selected_statistics.csv")
            if records: shutil.copyfile(records["D"]["results"]["path"],out/"selected_statistics.csv")
            keys=["panel","family","condition","left","right"]
            joined=tests.merge(prior,on=keys,suffixes=("_new","_old"),validate="one_to_one")
            assert len(joined)==24
            np.testing.assert_allclose(joined.p_raw_new,joined.p_raw_old,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(joined.p_holm24,joined.p_holm_row24,rtol=1e-12,atol=1e-12)
        all_tests.append(tests)
        for panel,d in data.items():
            for (condition,block),group in d.loc[d.Eligible].groupby(["condition_id","Selected block order"]):
                q=group[VALUE].quantile([.25,.5,.75]).to_numpy()
                summary.append(dict(version=version,panel=panel,condition=condition,block=LABELS[block],n_fish=len(group),q25=q[0],median=q[1],q75=q[2]))
            render_panel(panel,d,tests,version,out)
            render_panel(panel,d,tests,version,out,True)
        render_changes(changes,tests,version,out)
    combined=pd.concat(all_tests,ignore_index=True);combined.to_csv(out/"all_statistics.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    for panel in ["D","E"]:
        keys=["fish_id","condition_id","Selected block order"]
        merged=current[panel][keys+[VALUE,"Eligible"]].merge(logdata[panel][keys+[VALUE,"Eligible","eligible_log_trials","eligible_current_trials"]],on=keys,suffixes=("_ratio","_log"),validate="one_to_one")
        merged["panel"]=panel;missing.append(merged)
    pd.concat(missing).to_csv(out/"paired_version_comparison.csv",index=False)
    # Pure log of the existing block ratio: control for the display transform alone.
    pure={p:d.assign(**{VALUE:np.log(d[VALUE])}) for p,d in current.items()}
    puretests,_,_=block_statistics(pure,"log_existing_ratio")
    puretests.to_csv(out/"sensitivity_log_existing_block_ratio_tests.csv",index=False)
    grid=[]
    for p in ["D","E"]:
        for v,label in [("ratio","A · Current mean-activity ratio"),("log","B · Legacy log-median outcome")]:
            grid.append(f'<article><h2>{p} · {label}</h2><img class="main" src="Fig2_{p}_{v}.png"><img class="all" src="Fig2_{p}_{v}_all_tests.png"><p><a href="Fig2_{p}_{v}.svg">SVG</a> · <a href="Fig2_{p}_{v}.pdf">PDF</a> · <a href="{v}_tests.csv">Statistics</a></p></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Figure 2 D/E · ratio versus legacy log median</title><style>
    *{box-sizing:border-box}body{max-width:1450px;margin:28px auto;padding:0 25px;font:16px/1.5 system-ui;background:#f3f5f7;color:#20242a}
    h1{font-size:28px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}article,section{background:white;border:1px solid #dfe3e8;padding:18px;border-radius:10px;margin:18px 0}h2{font-size:18px;margin:0 0 12px}img{width:100%;height:auto}a{color:#235fa4}.all{display:none}body.show-all .all{display:block}body.show-all .main{display:none}table{border-collapse:collapse;font-size:12px}th,td{padding:6px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
    <h1>Figure 2 D/E: choose the outcome</h1><p>PT 10–14 · ET 65–69 · LT 90–94. Same source cohorts, metric, windows and paired style. Each version has fresh fish-level tests and Holm correction across its 24 D/E comparisons. Black whiskers are fish IQR, not confidence intervals.</p>
    <p>A measures total activity, including valid stationary frames. B measures the typical positive moving-frame vigor relative to baseline using median natural logs. B therefore changes the biological outcome, not just the vertical scale. Current corrected preprocessing is retained; historical smoothing/downsampling is not added.</p>
    <label><input type="checkbox" onchange="document.body.classList.toggle('show-all',this.checked)"> Show all block comparisons, including ns</label><div class="grid">'''+"\n".join(grid)+'''</div><section><h2>Direct comparisons of changes</h2><p>These compare the change in conditioned fish with the change in controls; block stars alone do not establish differential change.</p><div class="grid"><div><h3>A · ratio changes</h3><img src="ratio_direct_changes.png"></div><div><h3>B · log-median changes</h3><img src="log_direct_changes.png"></div></div></section><section><h2>All 48 comparison results</h2><p>Holm24 is applied separately by version. These exploratory alternative outcomes are not independent confirmations or a license to choose by p-value. <a href="all_statistics.csv">Download CSV</a> · <a href="critique.md">Scientific critique and recommendation</a> · <a href="summary.csv">Medians and sample counts</a> · <a href="paired_version_comparison.csv">Missingness and matched values</a></p><div class="scroll">'''+combined[["version","panel","family","condition","left","right","n_control","n_conditioned","n_pairs","p_raw","p_holm24","stars"]].to_html(index=False,float_format=lambda x:f"{x:.5g}",na_rep="—")+'''</div></section><section><h2>F · 10 s Trace</h2><p>Placeholder/inconclusive: no authenticated cohort and processed inputs for this comparison.</p></section></html>'''
    (out/"comparison.html").write_text(page,encoding="utf-8")
    write(out/"verification.json",{"synthetic_estimator_checks":"passed: window boundaries, zero exclusion, missing response, unit invariance",
          "current_24_statistics_reproduced":True,"tests_per_version":24,"no_day_variable_used":True,"LMM_used":False,
          "data_clipping":False,"selected_sources_preserved":True,"freeze_invoked":False})
    print("OUTPUT",out,flush=True)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(combined.loc[combined.p_holm24.lt(.05),["version","panel","family","condition","left","right","p_raw","p_holm24"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
