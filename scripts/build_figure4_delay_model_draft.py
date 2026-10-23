"""Guarded exploratory Delay CR profile pilot; never selects a learner method.

Run extract, model, render, then publish after exact browser/visual verification.
Permanent data, SVG and HTML are stored only on the authenticated external SSD.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from classical_conditioning.analysis.figure4 import verify_expected_us
from classical_conditioning.analysis.metric_comparison import _verify_temporal_profiles
from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source
from classical_conditioning.analysis.temporal_profiles import _signed_bout_log_vigor
from classical_conditioning.analysis.review.figure4_profiles import (
    GROUPS, CONTRASTS, model_data, fit_bootstrap, differences, peaks, individual_predictions,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config import Alignment, get_experiment_spec
from classical_conditioning.figures.example_traces import METRIC_COLUMNS
from classical_conditioning.figures.pooled_log_heatmap import read_windows
from classical_conditioning.figure_workspace import Snapshot, publish_state, state_lock

TICKET = ROOT / "records/figure-reviews/figure-04/delay-model-draft-start-20261022.json"
STATE = ROOT / "docs/paper/figures/figure-04/delay-model-draft-state.json"
ATLAS = Path(r"J:\ClassicalConditioning Outputs\current\reviews\figure3-learner-exploration-g-style-20261015")
PROJECT = Path(r"J:\Digested Data\allDelay-full-v1")
OUT = Path(r"J:\ClassicalConditioning Outputs\current\reviews\figure4-delay-model-draft-20261022")
METRIC = "legacy_distal_angular_speed"
SVG = "{http://www.w3.org/2000/svg}"
COLORS = ("#00aeef", "#ec008c", "#E69F00")
PAIR_NAMES = ("Control − L", "Control − N", "N − L")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, value):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write("\n")


def ref(p):
    return {"path": str(Path(p).resolve()), "sha256": sha256_file(Path(p))}


def verify(r):
    if sha256_file(Path(r["path"])) != r["sha256"]:
        raise ValueError("Changed bound source: " + r["path"])


def guards():
    ticket = load(TICKET)
    for r in ticket["protected"]:
        p = Path(r["path"])
        actual = sha256_file(p) if p.is_file() else None
        if actual != r["sha256"]:
            raise ValueError("Changed starting authority: " + str(p))
    for r in ticket["inputs"]:
        verify(r)
    if not OUT.parent.is_dir() or not PROJECT.is_dir():
        raise FileNotFoundError("JOAQUIM and its authenticated Delay project are required")
    return ticket


def membership():
    inventory = pd.read_csv(ATLAS / "fish-inventory.csv")
    labels = pd.read_parquet(ATLAS / "classification-manifest.parquet")
    labels = labels.loc[labels.variant.eq("legacy-wip")]
    result = inventory.merge(labels[["fish_id", "condition_id", "decision", "eligibility_reason"]],
                             on=["fish_id", "condition_id"], validate="one_to_one")
    if len(result) != 57 or result.condition_id.value_counts().to_dict() != {"delay": 29, "control": 28}:
        raise ValueError("Delay source roster differs from the selected atlas")
    result["display_group"] = "Unclassified Delay"
    result.loc[result.condition_id.eq("control"), "display_group"] = GROUPS[0]
    result.loc[result.condition_id.eq("delay") & result.decision.eq("Learner"), "display_group"] = GROUPS[1]
    result.loc[result.condition_id.eq("delay") & result.decision.eq("Nonlearner"), "display_group"] = GROUPS[2]
    if result.display_group.value_counts().to_dict() != {GROUPS[0]: 28, GROUPS[1]: 16, GROUPS[2]: 10, "Unclassified Delay": 3}:
        raise ValueError("Saved legacy-WIP membership changed")
    return result


def extract():
    guards()
    OUT.mkdir(exist_ok=True)
    cache = OUT / "fish"
    cache.mkdir(exist_ok=True)
    members = membership()
    if not (OUT / "membership.csv").exists():
        members.to_csv(OUT / "membership.csv", index=False)
    else:
        old = pd.read_csv(OUT / "membership.csv")
        pd.testing.assert_frame_equal(old, members, check_dtype=False)
    lineage = load(ATLAS / "input-lineage.json")
    # Exact historical recipe pairing recorded in the selected atlas; never
    # rename sources or redirect the current unversioned production resolver.
    source = replace(resolve_candidate_metric_source(metric_recipe="tail-candidate-corrected"),
        metric_recipe="tail-candidate-corrected-v1",
        metrics_name="frame_activity_candidates-corrected-v1.parquet",
        movement_recipe="movement-candidate-corrected-v2",
        movement_artifact_name="movement_state_candidates-corrected-v2.parquet",
        temporal_recipe="candidate-temporal-outcomes-corrected-v3",
        temporal_artifact_name="candidate_temporal_outcomes-corrected-v3.parquet",
        temporal_summary_name="candidate-corrected-v3_temporal_outcomes_summary.json",
        temporal_marker_suffix="candidate-temporal-outcomes-corrected-v3_complete.json")
    catches = tuple(get_experiment_spec("allDelay").catch_trial_numbers(Alignment.CS))
    times = np.arange(-20, 20, .5) + .25
    for number, fish in enumerate(members.itertuples(index=False), 1):
        rid = fish.fish_id
        dest, report = cache / (rid + ".parquet"), cache / (rid + ".json")
        if report.exists():
            verify(load(report)["output"])
            print(f"Reusing authenticated catch bins {number}/57: {rid}", flush=True)
            continue
        directory = PROJECT / "Processed data" / rid
        mp = directory / source.metrics_name
        sp = directory / source.movement_artifact_name
        pp = directory / "stimulus_events.parquet"
        inputs = []
        for p in (mp, sp, pp):
            suffix = "/Processed data/" + rid + "/" + p.name
            selected = [r for r in lineage if r["path"].replace("\\", "/").endswith(suffix)]
            if not selected or len({r["sha256"] for r in selected}) != 1:
                raise ValueError("No unique atlas-bound source: " + str(p))
            record = ref(p)
            if record["sha256"] != selected[0]["sha256"]:
                raise ValueError("Processed source differs from the selected Delay atlas: " + str(p))
            inputs.append(record)
        temporal = _verify_temporal_profiles(PROJECT, rid, source)
        upstream = load(temporal.summary_path)["inputs"]
        for key, record in zip(("candidate_frames", "movement_state", "protocol"), inputs):
            if upstream[key]["sha256"] != record["sha256"]:
                raise ValueError("Temporal coverage uses different upstream data: " + rid)
        inputs += [ref(p) for p in (temporal.path, temporal.summary_path, temporal.marker_path)]
        protocol = pd.read_parquet(pp)
        cycles = protocol.loc[protocol.Type.eq("Cycle")].sort_values("Beg", kind="stable").reset_index(drop=True)
        timing = verify_expected_us(protocol, "allDelay") if fish.condition_id == "delay" else None
        if timing and abs(timing[0] - 9) > .1:
            raise ValueError("Delay expected-US latency differs from 9 s")
        windows = [(int(cycles.iloc[t - 1].Beg) - 20001, int(cycles.iloc[t - 1].Beg) + 20001) for t in catches]
        frames = read_windows(mp, ["FrameID", "AbsoluteTime", METRIC_COLUMNS[METRIC]], windows)
        moving = read_windows(sp, ["FrameID", "AbsoluteTime", "valid", "moving", "bout_id"], windows)
        np.testing.assert_array_equal(frames[["FrameID", "AbsoluteTime"]], moving[["FrameID", "AbsoluteTime"]])
        profiles = pd.read_parquet(temporal.path)
        profiles = profiles.loc[profiles["Trial type"].eq("CS") & profiles["Metric ID"].eq(METRIC)]
        rows = []
        for trial in catches:
            onset = int(cycles.iloc[trial - 1].Beg)
            left, right = np.searchsorted(frames.AbsoluteTime.to_numpy(), [onset - 20000, onset + 20000])
            fm, mv = frames.iloc[left:right], moving.iloc[left:right]
            seconds = (fm.AbsoluteTime.to_numpy() - onset) / 1000
            bin_indices = np.floor((seconds + 20) / .5).astype(int)
            signed = _signed_bout_log_vigor(fm[METRIC_COLUMNS[METRIC]].to_numpy(), seconds, bin_indices,
                mv.valid.to_numpy(bool), mv.moving.to_numpy(bool), mv.bout_id.to_numpy(int),
                bin_count=80, baseline_start_s=-20, baseline_end_s=0)
            stored = profiles.loc[profiles["Trial number"].eq(trial)].set_index("Time bin center (s)").reindex(times)
            if stored["Valid expected fraction"].isna().any():
                raise ValueError("Incomplete temporal grid")
            for i, time in enumerate(times):
                coverage = float(stored["Valid expected fraction"].iloc[i])
                rows.append({"fish_id": rid, "condition_id": fish.condition_id,
                    "display_group": fish.display_group, "trial_number": trial, "time_s": float(time),
                    "coverage": coverage, "signed": float(signed[i]) if coverage >= .9 else np.nan,
                    "movement": float(stored["Movement probability"].iloc[i]) if coverage >= .9 else np.nan})
        bins = pd.DataFrame(rows)
        if number == 1:
            from classical_conditioning.figures.signed_bout_heatmap import calculate_fish_heatmaps
            check = calculate_fish_heatmaps(pd.read_parquet(mp, columns=list(frames.columns)),
                pd.read_parquet(sp, columns=list(moving.columns)), cycles,
                recording_id=rid, metric_ids=(METRIC,))
            check = check.loc[check["Trial number"].isin(catches)].sort_values(["Trial number", "Time bin center (s)"])
            expected = check["Signed log vigor"].to_numpy().copy()
            expected[bins.coverage.to_numpy() < .9] = np.nan
            np.testing.assert_allclose(bins.signed, expected, atol=1e-12, equal_nan=True)
        bins.to_parquet(dest, index=False)
        save(report, {"output": ref(dest), "inputs": inputs, "expected_us": timing,
            "catches": catches, "equivalence_to_supported_reader": number == 1,
            "estimator": "Supported half-second signed bout-log-vigor; pre [-20,0); coverage >=0.9"})
        print(f"Extracted authenticated catch bins {number}/57: {rid}", flush=True)
        del frames, moving, profiles
        gc.collect()
    guards()
    parts = [pd.read_parquet(cache / (rid + ".parquet")) for rid in members.fish_id]
    trials = pd.concat(parts, ignore_index=True)
    if trials.duplicated(["fish_id", "trial_number", "time_s"]).any() or len(trials) != 57 * len(catches) * 80:
        raise ValueError("Catch grid identity/cardinality failure")
    fish = trials.groupby(["fish_id", "condition_id", "display_group", "time_s"], sort=False).agg(
        signed=("signed", "median"), signed_trials=("signed", "count"),
        movement=("movement", "median"), movement_trials=("movement", "count")).reset_index()
    trials.to_parquet(OUT / "trial-bins.parquet", index=False)
    fish.to_parquet(OUT / "fish-bins.parquet", index=False)
    save(OUT / "data-complete.json", {"artifacts": [ref(OUT / f) for f in ("membership.csv", "trial-bins.parquet", "fish-bins.parquet")],
        "fish_reports": [ref(cache / (rid + ".json")) for rid in members.fish_id],
        "roster": 57, "modeled_groups": {g: int((members.display_group == g).sum()) for g in GROUPS},
        "unclassified_delay": 3, "catches": catches, "source_atlas": ref(ATLAS / "artifact-manifest.json")})


def model(repetitions):
    guards()
    for r in load(OUT / "data-complete.json")["artifacts"]:
        verify(r)
    fish = pd.read_parquet(OUT / "fish-bins.parquet")
    data = model_data(fish)
    times = np.sort(data.time_s.unique())
    result, grid, curves, draws, diagnostics = fit_bootstrap(data, times, repetitions=repetitions,
        progress=lambda n, ok: print(f"Whole-fish LMM refits {n}/{repetitions}: {ok} accepted", flush=True))
    pre_us = (times >= 0) & (times < 9)
    contrast = differences(curves)
    cd = np.asarray([differences(d) for d in draws])
    sd = cd[:, :, pre_us].std(axis=0, ddof=1)
    standardized = np.abs(cd[:, :, pre_us] - contrast[None, :, pre_us]) / np.maximum(sd, 1e-10)
    critical = float(np.quantile(standardized.max(axis=(1, 2)), .95))
    amplitude, latency = peaks(contrast[:, pre_us], times[pre_us])
    ampd, latd = peaks(cd[:, :, pre_us], times[pre_us])
    predictions = individual_predictions(result, data, times, diagnostics["random"],
        next(x["residual_variance"] for x in reversed(diagnostics["full_fit_attempts"]) if x["diagnostic_status"] == "ok"))
    prediction_rows = [{"fish_id": rid, "time_s": float(t), "fitted": float(v)}
                       for rid, values in predictions.items() for t, v in zip(times, values)]
    pd.DataFrame(prediction_rows).to_parquet(OUT / "individual-predictions.parquet", index=False)
    np.savez_compressed(OUT / "model-arrays.npz", times=times, group_curves=curves, bootstrap_curves=draws,
        contrasts=contrast, bootstrap_contrasts=cd, contrast_sd=sd, contrast_critical=critical,
        peak_amplitude=amplitude, peak_latency=latency, bootstrap_peak_amplitude=ampd, bootstrap_peak_latency=latd)
    peak_rows = []
    for i, name in enumerate(PAIR_NAMES):
        aq, tq = np.quantile(ampd[:, i], [.025, .975]), np.quantile(latd[:, i], [.025, .975])
        peak_rows.append({"contrast": name, "maximum": float(amplitude[i]), "grid_peak_s": float(latency[i]),
            "amplitude_range": aq.tolist(), "timing_range_s": tq.tolist(),
            "bootstrap_fraction_nonpositive_maximum": float(np.mean(ampd[:, i] <= 0)),
            "bootstrap_fraction_boundary_maximum": float(np.mean((latd[:, i] == times[pre_us][0]) | (latd[:, i] == times[pre_us][-1]))),
            "status": "Descriptive fitted extremum; timing not validated or biologically localized"})
    summary = {"scope": "Delay-only exploratory Figure 4 model pilot", "metric": METRIC,
        "profile": "0.5 s mean of bout-median log-vigor values, centered by trial pre [-20,0) median; valid coverage >=0.9",
        "aggregation": "Median over five configured catches within fish; Gaussian spline LMM of fish medians",
        "membership": "Saved legacy-wip exploratory flags, fixed for this retrospective pilot; all controls unsplit",
        "counts": load(OUT / "data-complete.json")["modeled_groups"], "all_roster_fish": 57,
        "unclassified_delay_not_in_modeled_groups": 3, "cs_s": [0, 10], "expected_us_s": 9,
        "fit_window_s": [-5, 12], "peak_search_bin_centers_s": times[pre_us].tolist(),
        "model": "Group-specific degree-3 B-splines, df=6; Gaussian fish intercept/time slope if supported; ML",
        "diagnostics": diagnostics, "residual_excess_kurtosis": float(kurtosis(result.resid, fisher=True)),
        "nominal_simultaneous_family": "Three pairwise contrasts across 18 half-second pre-US bin centers",
        "nominal_simultaneous_critical": critical, "peaks": peak_rows,
        "movement_panel": "Observed equal-fish mean of within-fish catch medians; whole-fish percentile bootstrap, no Gaussian movement LMM",
        "limitations": diagnostics["limitations"] + ["Source groups use acquisition plus recovery, overlapping evaluation trials",
            "No p-values or validated CR peak/onset claim; all bands and extrema are exploratory",
            "Controls have unpaired US events; source cohort and protocol flags retained",
            "Missing conditional intensity is not zero activity; no-bout prediction is conditional, not reconstructed movement"],
        "implementation": [ref(ROOT / "src/classical_conditioning/analysis/review/figure4_profiles.py"),
                           ref(ROOT / "src/classical_conditioning/analysis/inference/panel_g_fast_lmm.py")],
        "artifacts": [ref(OUT / n) for n in ("model-arrays.npz", "individual-predictions.parquet")]}
    save(OUT / "analysis-summary.json", summary)
    guards()
    print(json.dumps({"accepted": diagnostics["bootstrap_accepted"], "peaks": peak_rows}), flush=True)


def render():
    # The existing shared review-page module has archived Figure 1 imports.
    # Resolve its recorded archive on this host; no Figure 1 build is invoked.
    os.environ.setdefault("CLASSICAL_CONDITIONING_ARCHIVE_ROOT",
        r"J:\ClassicalConditioning-Archive\repository-cleanup\2026-10-10")
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from classical_conditioning.figures.theme import apply_theme, style_axes
    from classical_conditioning.figures.presentation_profiles import load_presentation_profile
    from classical_conditioning.figures.layout_frames import embed_spec, load_policy
    from build_figure_legend_review import svg_editing_page
    guards()
    summary = load(OUT / "analysis-summary.json")
    for r in summary["implementation"] + summary["artifacts"]:
        verify(r)
    arrays = np.load(OUT / "model-arrays.npz")
    fish = pd.read_parquet(OUT / "fish-bins.parquet")
    individual = pd.read_parquet(OUT / "individual-predictions.parquet")
    times = arrays["times"]; curves = arrays["group_curves"]; draws = arrays["bootstrap_curves"]
    mask = (times >= 0) & (times < 9)
    profile = load_presentation_profile("nature-main")
    width, height, key_height = 183., 135., 9.
    cellw, cellh = width / 3, (height - key_height) / 2
    roles = profile.contract["styles"]
    registry = {"scientific_status": "Exploratory; no learner or timing validation", "sources":
        [ref(OUT / p) for p in ("analysis-summary.json", "model-arrays.npz", "fish-bins.parquet", "membership.csv")],
        "presentation_profile": profile.inputs, "elements": {}}
    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        mpl.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "figure4-delay-model-draft",
            "figure.constrained_layout.use": False, "savefig.bbox": None, "savefig.pad_inches": 0})
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        allocations = [{"id": "key", "kind": "shared", "box_mm": [0, 0, width, key_height], "group_ids": ["condition-key"]}]
        axes = []
        titles = ("Pooled catch response", "Control-relative suppression", "Fitted maximum", "Movement occurrence", "Illustrative fish fits", "Timing of fitted maximum")
        for i, title in enumerate(titles):
            row, col = divmod(i, 3); x, y = col * cellw, key_height + row * cellh
            ax = fig.add_axes([(x + 15) / width, 1 - (y + cellh - 12) / height, (cellw - 19) / width, (cellh - 24) / height])
            letter = chr(97 + i); ax.set_gid("panel-" + letter + "-axes")
            ax.set_title(title, pad=8, fontsize=theme.axes_titlesize)
            fig.text((x + 2) / width, 1 - (y + 5) / height, letter, fontsize=roles["panel_letter"]["font_size_pt"], fontweight="bold", gid="panel-" + letter + "-letter")
            allocations.append({"id": letter, "kind": "panel", "box_mm": [x, y, cellw, cellh],
                "group_ids": ["panel-" + letter + "-axes", "panel-" + letter + "-letter"]})
            axes.append(ax)
        def time_axis(ax, ylabel, pre_only=False):
            ax.axhline(0, color="#999999", lw=.35, zorder=0)
            ax.axvline(0, color=theme.cs_color, lw=theme.cs_linewidth, zorder=0)
            ax.axvline(9, color=theme.us_color, ls=":", lw=theme.us_linewidth, zorder=0)
            ax.set_xlim(0 if pre_only else -5, 9 if pre_only else 12)
            ax.set_xticks([0, 3, 6, 9] if pre_only else [-5, 0, 5, 10])
            style_axes(ax, theme=theme, xlabel="Time from CS (s)", ylabel=ylabel)
        lines = []
        lo, hi = np.quantile(draws, [.025, .975], axis=0)
        for i, group in enumerate(GROUPS):
            line, = axes[0].plot(times, curves[i], color=COLORS[i], label=group,
                lw=theme.lines_linewidth, ls="--" if i == 0 else "-", gid="response-" + str(i))
            axes[0].fill_between(times, lo[i], hi[i], color=COLORS[i], alpha=.12, linewidth=0, gid="response-band-" + str(i))
            observed = fish.loc[fish.display_group.eq(group)].groupby("time_s").signed.mean().reindex(times)
            axes[0].plot(times, observed, ".", color=COLORS[i], ms=1.8, alpha=.55, gid="observed-" + str(i))
            lines.append(line)
            registry["elements"]["response-" + str(i)] = {"role": "curve", "quantity": summary["profile"], "group": group, "estimator": summary["model"]}
        time_axis(axes[0], "Signed bout ln vigor")
        pair_colors = (COLORS[1], COLORS[2], "#555555")
        c = arrays["contrasts"][:, mask]; sd = arrays["contrast_sd"]; critical = float(arrays["contrast_critical"])
        for i in range(3):
            axes[1].plot(times[mask], c[i], color=pair_colors[i], lw=theme.lines_linewidth, ls=("-", "--", ":")[i], gid="contrast-" + str(i))
            axes[1].fill_between(times[mask], c[i] - critical * sd[i], c[i] + critical * sd[i], color=pair_colors[i], alpha=.10, linewidth=0)
            registry["elements"]["contrast-" + str(i)] = {"role": "condition_contrast", "groups": [GROUPS[k] for k in CONTRASTS[i]], "family": summary["nominal_simultaneous_family"]}
        time_axis(axes[1], "Difference (ln vigor)", pre_only=True)
        for j, (ax, field, range_key, xlabel) in enumerate(((axes[2], "maximum", "amplitude_range", "Maximum difference (ln vigor)"),
                                                        (axes[5], "grid_peak_s", "timing_range_s", "Time from CS (s)"))):
            for i, p in enumerate(summary["peaks"]):
                low, high = p[range_key]
                ax.plot([low, high], [2 - i] * 2, color=pair_colors[i], lw=theme.lines_linewidth)
                ax.plot(p[field], 2 - i, "o", ms=3, color=pair_colors[i])
            ax.set_yticks([2, 1, 0], PAIR_NAMES); ax.set_ylim(-.65, 2.65)
            if j:
                ax.set_xlim(0, 9); ax.set_xticks([0, 3, 6, 9])
            else:
                ax.axvline(0, color="#999999", lw=.35)
            style_axes(ax, theme=theme, xlabel=xlabel)
        rng = np.random.default_rng(20261022)
        for i, group in enumerate(GROUPS):
            matrix = fish.loc[fish.display_group.eq(group)].pivot(index="fish_id", columns="time_s", values="movement").reindex(columns=times).to_numpy()
            movement_draws = np.asarray([np.nanmean(matrix[rng.choice(len(matrix), len(matrix), replace=True)], axis=0) for _ in range(999)])
            mean = np.nanmean(matrix, axis=0); a, b = np.nanquantile(movement_draws, [.025, .975], axis=0)
            axes[3].plot(times, mean, color=COLORS[i], lw=theme.lines_linewidth, ls="--" if i == 0 else "-")
            axes[3].fill_between(times, a, b, color=COLORS[i], alpha=.12, linewidth=0)
        time_axis(axes[3], "Moving fraction")
        axes[3].set_ylim(bottom=0)
        for rid, color, marker in (("20240618_08", COLORS[1], "o"), ("20240612_08", COLORS[2], "s")):
            obs = fish.loc[fish.fish_id.eq(rid)].set_index("time_s").reindex(times)
            pred = individual.loc[individual.fish_id.eq(rid)].set_index("time_s").reindex(times)
            axes[4].plot(times, pred.fitted, color=color, lw=theme.lines_linewidth)
            axes[4].plot(times, obs.signed, marker, color=color, ms=2, markeredgewidth=.35, markerfacecolor="white", ls="none")
        time_axis(axes[4], "Signed bout ln vigor")
        key = fig.legend(lines, GROUPS, loc="upper right", bbox_to_anchor=(1 - 2 / width, 1 - 1 / height),
                         ncol=3, frameon=False, fontsize=theme.legend_fontsize, handlelength=2)
        key.set_gid("condition-key")
        stream = io.BytesIO(); fig.savefig(stream, format="svg", metadata={"Date": None}); plt.close(fig)
        root = ET.fromstring(stream.getvalue())
        root.set("width", f"{width:g}mm")
        root.set("height", f"{height:g}mm")
        clip_ids = [n.get("id") for n in root.iter(SVG + "clipPath")]
        spec = {"scope": "figure", "final_size_mm": [width, height], "inset_mm": load_policy()[0]["selected_inset_mm"],
            "allocations": allocations, "background_ids": ["patch_1"],
            "scientific_clips": [{"id": k, "evidence": "Explicit data-axis clipping rectangle of the named model-profile panel"} for k in clip_ids]}
        svg = embed_spec(ET.tostring(root), spec)
    legend = ("Figure 4 Delay | Exploratory CR profile model draft. "
        "Saved legacy-WIP groups: 16 learner-flagged (L), 10 nonlearner-flagged (N), all 28 controls; "
        "three unclassified Delay fish remain in the inventory. Groups are retrospective, not validated learner classes. "
        "Catches 25, 39, 53, 59 and 65 are pooled by median within fish. Conditional signed bout-ln-vigor uses "
        "0.5-s bins, each trial's [−20,0) s baseline and coverage ≥0.9; no-bout intensity stays unavailable. "
        "a, Group spline-LMM means (lines), observed across-fish means (dots), pointwise whole-fish bootstrap ranges (shading). "
        "b, Direct contrasts: Control−L (magenta), Control−N (orange), N−L (gray); nominal joint 95% bands cover three contrasts at 18 pre-US bin centers. "
        "c,f, Fitted-contrast maxima and times on 0.25–8.75 s; bars are bootstrap 2.5–97.5% ranges. "
        "These are unvalidated extrema summaries, not localized CR-peak confidence intervals. "
        "d, Observed mean moving fraction with pointwise fish-bootstrap ranges. "
        "e, Conditional fits and observed catch medians for illustrative fish 20240618_08 (circles) and "
        "20240612_08 (squares); fitted gaps do not reconstruct actual movement. "
        "Green marks CS onset; violet marks expected US at 9 s (omitted in Delay catches; CS ends at 10 s). "
        f"The fixed-basis Gaussian ML model uses {'fish intercepts' if summary['diagnostics']['random'] == '1' else 'fish intercepts and time slopes'}; {summary['diagnostics']['bootstrap_accepted']} "
        "whole-fish refits passed. Labels overlap evaluation data; no independent learner comparison or timing claim is made. "
        "Controls have unpaired reinforcers. LME/GAMM comparison, informative availability and calibration remain open.")
    registry["remaining_coverage"] = "Named scientific curves/contrasts; no full freeze-role inventory or production certification"
    files = {"figure4-delay.svg": svg, "legend.md": (legend + "\n").encode(),
        "figure4-delay.html": svg_editing_page(svg, scope_label="Figure 4 Delay exploratory model draft", legend=legend).encode(),
        "semantic-registry.json": (json.dumps(registry, indent=2) + "\n").encode()}
    with state_lock(OUT / "render"):
        if (OUT / "render-manifest.json").exists():
            prior = load(OUT / "render-manifest.json")
            for r in prior["artifacts"]:
                verify(r)
            history = OUT / "history" / prior["artifacts"][0]["sha256"][:12]
            history.mkdir(parents=True, exist_ok=True)
            for r in prior["artifacts"]:
                p = Path(r["path"]); target = history / p.name
                if not target.exists():
                    target.write_bytes(p.read_bytes())
        for name, content in files.items():
            (OUT / name).write_bytes(content)
        meta = {"artifacts": [ref(OUT / p) for p in files], "implementation": ref(Path(__file__)),
                "profile": profile.inputs, "legend_words": len(legend.split())}
        (OUT / "render-manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    guards()
    print(json.dumps({"svg": str(OUT / "figure4-delay.svg"), "legend_words": len(legend.split())}))


def publish(note):
    from classical_conditioning.figures.layout_frames import assess
    guards()
    rendered = load(OUT / "render-manifest.json")
    for r in rendered["artifacts"]:
        verify(r)
    svg = (OUT / "figure4-delay.svg").read_bytes()
    browser = load(OUT / "layout-browser.json")
    assessment = assess(svg, browser)
    if assessment["status"] != "passed":
        raise ValueError("Exact rendered layout unresolved: " + str(assessment["issues"]))
    # Reauthenticate all consumed original numerical sources before publication.
    bound = list(rendered["artifacts"])
    for p in (OUT / "fish").glob("*.json"):
        report = load(p)
        verify(report["output"])
        for r in report["inputs"]:
            verify(r)
        bound += [ref(p), report["output"], *report["inputs"]]
    bound += load(OUT / "data-complete.json")["artifacts"]
    bound += load(OUT / "analysis-summary.json")["implementation"]
    bound += load(OUT / "analysis-summary.json")["artifacts"]
    bound += [ref(OUT / n) for n in ("analysis-summary.json", "data-complete.json", "render-manifest.json", "layout-browser.json")]
    bound += [rendered["implementation"]]
    for r in bound:
        verify(r)
    evidence = OUT / "layout-evidence.json"
    save(evidence, assessment)
    review = OUT / "visual-review.json"
    save(review, {"artifact": ref(OUT / "figure4-delay.svg"), "note": note, "scope": "Delay only", "performed": True})
    ticket = load(TICKET)
    expected = ticket["protected"] + ticket["inputs"] + bound + [ref(evidence), ref(review)]
    snapshot = Snapshot([r["path"] for r in expected])
    for r in expected:
        snapshot.hashes[str(Path(r["path"]).resolve())] = r["sha256"]
    state = {"schema_version": 1, "figure_id": "fig4", "scope": "Delay exploratory model draft", "whole_figure_frozen": False,
        "default_display": "draft", "default_display_format": "html-svg-and-legend",
        "scientific_selection": {"status": "Unfrozen exploratory pilot; saved legacy groups, no method approval", "records": []},
        "draft": {"working_svg": ref(OUT / "figure4-delay.svg"), "review": ref(OUT / "figure4-delay.html"),
                  "legend": ref(OUT / "legend.md"), "semantic_registry": ref(OUT / "semantic-registry.json"),
                  "analysis": ref(OUT / "analysis-summary.json"), "layout_frames": ref(evidence), "visual_review": ref(review)},
        "start_ticket": ref(TICKET), "limitations": load(OUT / "analysis-summary.json")["limitations"]}
    publish_state(STATE, state, snapshot)
    print(json.dumps({"state": str(STATE), "review": state["draft"]["review"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "model", "render", "publish"))
    parser.add_argument("--bootstrap", type=int, default=399)
    parser.add_argument("--visual-review-note")
    args = parser.parse_args()
    if args.stage == "extract": extract()
    elif args.stage == "model": model(args.bootstrap)
    elif args.stage == "render": render()
    elif not args.visual_review_note: parser.error("Actual visual-review note required")
    else: publish(args.visual_review_note)
