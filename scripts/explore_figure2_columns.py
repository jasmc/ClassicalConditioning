"""Figure 2 presentation exploration from authenticated, completed results.

No acquisition processing, model fitting, resampling, or inferential calculation.
Build an independent draft and publish it through the Figure 2 state guard.
"""
from pathlib import Path
import argparse
import copy
import json
import re
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot, publish_state, sha, tag
from classical_conditioning.figures.theme import apply_theme, style_axes
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.export import FigureProvenance, FigureMode, export_matplotlib_figure
from classical_conditioning.figures.layout_frames import load_policy, embed_spec, assess
from main_paper_figures import render as render_page

BASE = Path("J:/ClassicalConditioning Outputs/current/reviews")
OUT = BASE / "figure-02-exploratory-range-20261020"
T3 = BASE / "figure-02-3strace/matched-delay-statistics"
T10 = BASE / "figure-02-10strace-20261020"
STATE = REPO / "docs/paper/figures/figure-02/current-state.json"
GSIDE = REPO / "records/frozen-analyses/figure-02/panel-g/2026-10-09-historical-log-median/sources/source-figure.json"
SIZE = (156., 138.)
CENTRES = (30., 76., 122.)
HEAT_W, HEAT_H, HEAT_PAIR = 12.451319, 37.907998, 28.985822
BLOCK_W, BLOCK_H, BLOCK_PAIR = 13.04813, 13.7820384, 28.44492
CURVE_W, CURVE_H = 35.5561477, 17.2275904
LIMITS = (-.1, .1)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def ref(p):
    return {"path": str(Path(p).resolve()), "sha256": sha256_file(Path(p))}


def encode(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write(p, data):
    with Path(p).open("xb") as f:
        f.write(data if isinstance(data, bytes) else encode(data))


def baseline():
    ticket = Snapshot([])
    ticket.hashes = load(OUT / "starting-ticket.json")
    ticket.check()
    return ticket


def counts(values):
    v = np.asarray(values, float)
    return {"finite": int(np.isfinite(v).sum()), "below_view": int((v < LIMITS[0]).sum()),
            "above_view": int((v > LIMITS[1]).sum()), "missing": int((~np.isfinite(v)).sum())}


def build():
    ticket = baseline()
    profile = load_presentation_profile("nature-main")
    policy, policy_hash = load_policy()
    source = load(OUT / "previous-selected/state.json")["draft"]
    # The saved selection is the source of the original G glyphs; the frozen
    # registry supplies each glyph's scientific mapping and exact p value.
    original = ET.parse(OUT / "previous-selected/figure2.svg").getroot()
    old_ids = {n.get("id"): n for n in original.iter() if n.get("id")}
    g_registry = load(GSIDE)["artist_registry"]
    panels = list("ABCDEFGHI")
    owned = {p: [] for p in panels + ["ABC-colorbar"]}
    maps, axis_ids, data_axes, numeric, displayed = {}, [], [], {}, {}
    inputs = [OUT / "source-data/G-bootstrap-summary.parquet", GSIDE,
              BASE / "figure-02-a-freeze-2026-10-14/pooled-bins.parquet",
              T3 / "B-pooled-bins.parquet", T10 / "C-pooled-bins.parquet",
              T10 / "DE-corrections/D-fish-blocks.parquet", T10 / "DE-corrections/E-fish-blocks.parquet",
              T10 / "DE-corrections/D-statistics-Holm18.csv", T10 / "DE-corrections/E-statistics-Holm18.csv",
              T10 / "F-fish-blocks.parquet", T10 / "F-statistics-Holm18.csv",
              T3 / "H-trajectory.parquet", T10 / "I-trajectory.parquet",
              *[folder / name for folder in [T3, T10 / "I-models"]
                for name in ["D-interaction-tests.csv", "MR-local-block-tests.csv", "phase-contrasts.csv"]]]
    build_ticket = Snapshot([*inputs, Path(__file__)])

    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        fig = plt.figure(figsize=(SIZE[0] / 25.4, SIZE[1] / 25.4), layout=None)
        fig.set_layout_engine(None)
        fig.patch.set_gid("explore-canvas")

        def identify(artist, key, panel, role, mapping=None):
            artist.set_gid(key)
            maps[key] = {"element_id": key, "panel_id": panel.lower(), "scientific_role": role,
                         "source_mapping": mapping or {"presentation_only": True}}
            return artist

        def ax_at(panel, name, box):
            x, y, w, h = box
            a = fig.add_axes([x / SIZE[0], 1 - (y + h) / SIZE[1], w / SIZE[0], h / SIZE[1]])
            axis_ids.append(name)
            owned[panel].append("axes__" + name.lower() + "__main")
            if not name.endswith("-statistics"):
                identify(a.patch, name + "-rectangle", panel, "background.data_axes", {"box_mm": box})
            return a

        def text_at(panel, key, x, y, text, *, size=None, ha="left", va="top", role="annotation.note", mapping=None, weight="normal", rotation=0):
            artist = fig.text(x / SIZE[0], 1 - y / SIZE[1], text, ha=ha, va=va,
                              fontsize=theme.tick_labelsize if size is None else size, color="black", weight=weight, rotation=rotation)
            identify(artist, key, panel, role, mapping)
            owned[panel].append(key)
            return artist

        for row, top in enumerate([1.2, 55.2, 91.2]):
            for col in range(3):
                p = panels[row * 3 + col]
                # Saved browser glyph bounds: Arial bold f extends 0.75 pt
                # left of its anchor. Compensate to align visible letter edges.
                letter_x = 8 + 46 * col + (.75 * 25.4 / 72 if p == "F" else 0)
                text_at(p, p + "-letter", letter_x, top, p.lower(),
                        size=profile.contract["styles"]["panel_letter"]["font_size_pt"],
                        weight="bold", role="panel.letter", mapping={"panel": p})

        cmap = mpl.colormaps["managua_r"].copy()
        cmap.set_bad("black")
        norm = Normalize(*LIMITS, clip=False)
        for col, (p, path, conditioned, name, us) in enumerate([
            ("A", inputs[2], "delay", "Delay", 9), ("B", inputs[3], "trace", "3sTrace", 13),
            ("C", inputs[4], "trace", "10sTrace", 20)]):
            table = pd.read_parquet(path)
            numeric[p] = {"source": ref(path), "conditions": {}}
            rects = []
            for i, condition in enumerate(["control", conditioned]):
                x = CENTRES[col] - HEAT_PAIR / 2 + i * (HEAT_PAIR - HEAT_W)
                ax = ax_at(p, p.lower() + str(i), [x, 6., HEAT_W, HEAT_H])
                key = p + "-heatmap-" + condition
                matrix = table.loc[table.condition_id.eq(condition)].pivot(
                    index="acquisition_trial", columns="bin_index", values="mean_delta_log_vigor"
                ).reindex(index=range(5, 95), columns=range(40)).to_numpy()
                im = ax.imshow(np.ma.masked_invalid(matrix), origin="upper", aspect="auto", interpolation="nearest",
                               extent=[-20, 20, 90.5, .5], cmap=cmap, norm=norm)
                identify(im, key, p, "heatmap.vigor", {"artifact": ref(path), "condition": condition,
                         "matrix_sha256": sha(np.asarray(matrix, dtype="<f8").tobytes()),
                         "field": "mean_delta_log_vigor", "acquisition_trials": [5, 94],
                         "display_trials": [1, 90], "time_edges_s": [-20, 20], "limits": list(LIMITS),
                         "colormap": "managua_r", "missing": "black", "out_of_range": "saturate"})
                ax.set(xticks=[-20, 0, 20], yticks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                       xlim=(-20, 20), ylim=(90.5, .5))
                style_axes(ax, theme=theme, xlabel="", ylabel="Trial" if p == "A" and i == 0 else "",
                           show_yticklabels=p == "A" and i == 0)
                for side in ["top", "right"]:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(theme.axes_linewidth)
                for t, label, ls in [(0, "CS-onset", "-"), (10, "CS-offset", "--")]:
                    identify(ax.axvline(t, color=theme.cs_color, lw=theme.cs_linewidth, ls=ls),
                             key + "-" + label, p, "stimulus.guide", {"time_s": t, "event": label})
                if i:
                    identify(ax.plot([us, us], [10.5, 60.5], color=theme.us_color, lw=theme.us_linewidth,
                                     ls=":", clip_on=False)[0], key + "-US", p, "stimulus.guide",
                             {"expected_training_US_onset_s": us, "display_trial_span": [10.5, 60.5]})
                for trial in [10.5, 60.5]:
                    identify(ax.axhline(trial, color="white", lw=theme.axes_linewidth), key + "-phase-" + str(trial),
                             p, "boundary.phase", {"display_trial": trial})
                text_at(p, p + "-heading-" + condition, x + HEAT_W / 2, 3.1,
                        "Control" if i == 0 else name, size=theme.axes_titlesize, ha="center",
                        role="title.condition", mapping={"condition": condition})
                numeric[p]["conditions"][condition] = {**counts(matrix), "matrix_sha256": maps[key]["source_mapping"]["matrix_sha256"]}
                rects.append(p.lower() + str(i) + "-rectangle")
                data_axes.append({"id": p + str(i), "element_id": rects[-1]})
            data_axes.append({"id": p + "-pair", "element_ids": rects})
            text_at(p, p + "-time-title", CENTRES[col], 50.5, "Time from CS onset (s)",
                    size=theme.axes_labelsize, ha="center", role="axis.label",
                    mapping={"quantity": "time from CS onset", "units": "s", "shared_axes": [p + "0", p + "1"]})
            if p == "A":
                for phase, start, end in [("Pre", 0, 10), ("Train", 10, 60), ("Test", 60, 90)]:
                    text_at(p, "A-phase-label-" + phase, 4.8, 6. + HEAT_H * (start + end) / 180,
                            phase, ha="center", va="center", rotation=90, role="title.phase",
                            mapping={"display_trial_span": [start + 1, end], "acquisition_trial_span": [start + 5, end + 4]})
        cbax = ax_at("ABC-colorbar", "shared-colourbar", [143.4, 12., 1.5, 28.])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax, ticks=[-.1, 0, .1])
        cb.solids.set_rasterized(True)
        style_axes(cbax, theme=theme, ylabel="Mean Δ ln vigor")
        cbax.yaxis.set_label_position("right")
        cbax.yaxis.tick_right()
        cb.outline.set_linewidth(theme.axes_linewidth)

        # Existing fish/block values, IQRs, and separate Holm18 result tables.
        for col, (p, conditioned, name, path, tests_path) in enumerate([
            ("D", "delay", "Delay", inputs[5], inputs[7]),
            ("E", "trace", "3sTrace", inputs[6], inputs[8]),
            ("F", "trace", "10sTrace", inputs[9], inputs[10])]):
            data, tests = pd.read_parquet(path), pd.read_csv(tests_path)
            assert len(tests) == 18 and "p_holm18" in tests
            numeric[p] = {"source": ref(path), "statistics": ref(tests_path),
                          "all_eligible": counts(data.loc[data.Eligible, "Fish median response / baseline"]),
                          "eligible_blocks": int(data.Eligible.sum()), "family_size": 18}
            displayed[p] = []
            axes, rects = [], []
            for i, c in enumerate(["control", conditioned]):
                x = CENTRES[col] - BLOCK_PAIR / 2 + i * (BLOCK_PAIR - BLOCK_W)
                ax = ax_at(p, p.lower() + "-" + c, [x, 71.5, BLOCK_W, BLOCK_H])
                axes.append(ax)
                rects.append(p.lower() + "-" + c + "-rectangle")
                sub = data.loc[data.Eligible & data.condition_id.eq(c)]
                color = "#00aeef" if c == "control" else "#ec008c" if c == "delay" else "#f15a29"
                wide = sub.pivot(index="fish_id", columns="Selected block order", values="Fish median response / baseline").reindex(columns=[0, 1, 2])
                for fish, row in wide.iterrows():
                    info = {"artifact": ref(path), "fish_id": fish, "condition": c,
                            "field": "Fish median response / baseline", "values": [float(v) if np.isfinite(v) else None for v in row]}
                    identify(ax.plot(range(3), row, color=color, lw=.5, alpha=.3, zorder=1)[0],
                             p + "-" + c + "-fish-" + fish, p, "trajectory.fish", info)
                    identify(ax.scatter(range(3), row, s=4, color=color, alpha=.3, linewidths=.25, zorder=2),
                             p + "-" + c + "-points-" + fish, p, "observation.fish", info)
                q = np.array([sub.loc[sub["Selected block order"].eq(i), "Fish median response / baseline"].quantile([.25, .5, .75]).to_numpy() for i in range(3)])
                artists = ax.errorbar(range(3), q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                                        fmt="o-", color="black", alpha=.72, lw=profile.contract["styles"]["median_iqr"]["linewidth_pt"],
                                        capsize=1.5, markersize=2, zorder=5).lines
                identify(artists[0], p + "-" + c + "-median", p, "summary.median", {"artifact": ref(path), "quantiles": q.tolist()})
                for j, a in enumerate([*artists[1], *artists[2]]):
                    identify(a, p + "-" + c + "-iqr-" + str(j), p, "uncertainty.iqr", {"artifact": ref(path), "quantiles": q.tolist()})
                identify(ax.axhline(0, color="black", lw=theme.axes_linewidth, zorder=0), p + "-" + c + "-zero", p, "reference.signal.zero", {"value": 0})
                ax.set(xlim=(-.15, 2.15), ylim=LIMITS, xticks=[0, 1, 2], xticklabels=["PT", "ET", "LT"], yticks=[-.1, 0, .1])
                style_axes(ax, theme=theme, xlabel="", ylabel="Δ median ln vigor" if p == "D" and i == 0 else "",
                           show_yticklabels=p == "D" and i == 0)
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", bottom=False, length=0)
                text_at(p, p + "-heading-" + c, x + BLOCK_W / 2, 57.7, "Control" if i == 0 else name,
                        ha="center", size=theme.axes_titlesize, role="title.condition", mapping={"condition": c})
                selected = tests.loc[tests.condition.eq(c) & tests.p_holm18.lt(.05)]
                for level, row in enumerate(selected.loc[selected.family.eq("within")].itertuples()):
                    frac = 1 + (1.6 + 2.4 * level) / BLOCK_H
                    key = p + "-" + c + "-within-" + str(row.left) + "-" + str(row.right)
                    info = {"artifact": ref(tests_path), "family": "within", "condition": c, "left": row.left,
                            "right": row.right, "p_holm18": row.p_holm18, "glyph": row.stars}
                    identify(ax.plot([row.left, row.right], [frac, frac], transform=ax.get_xaxis_transform(),
                                     color="black", lw=.5, clip_on=False)[0], key, p, "annotation.statistical_comparison", info)
                    identify(ax.text((row.left + row.right) / 2, frac + .025, row.stars, transform=ax.get_xaxis_transform(),
                                     ha="center", va="bottom", fontsize=5, clip_on=False), key + "-label", p, "annotation.statistical_label", info)
                    displayed[p].append(info)
                for row in selected.loc[selected.family.eq("zero")].itertuples():
                    info = {"artifact": ref(tests_path), "family": "zero", "condition": c, "left": row.left,
                            "p_holm18": row.p_holm18, "glyph": "†" + row.stars}
                    identify(ax.text(row.left, .83, "†" + row.stars, transform=ax.get_xaxis_transform(), ha="center", fontsize=5),
                             p + "-" + c + "-zero-label-" + str(row.left), p, "annotation.statistical_label", info)
                    displayed[p].append(info)
            for row in tests.loc[tests.family.eq("between") & tests.p_holm18.lt(.05)].itertuples():
                frac = 1 + 9.4 / BLOCK_H
                info = {"artifact": ref(tests_path), "family": "between", "left": row.left,
                        "right": row.right, "p_holm18": row.p_holm18, "glyph": row.stars}
                key = p + "-between-" + str(row.left)
                identify(fig.add_artist(ConnectionPatch((row.left, frac), (row.right, frac),
                         coordsA=axes[0].get_xaxis_transform(), coordsB=axes[1].get_xaxis_transform(),
                         color="black", lw=.5, clip_on=False)), key, p, "annotation.statistical_comparison", info)
                owned[p].append(key)
                a = fig.transFigure.inverted().transform(axes[0].get_xaxis_transform().transform((row.left, frac)))
                b = fig.transFigure.inverted().transform(axes[1].get_xaxis_transform().transform((row.right, frac)))
                text_at(p, key + "-label", (a[0] + b[0]) / 2 * SIZE[0], 71.5 - 9.4 - .35,
                        row.stars, size=5, ha="center", va="bottom", role="annotation.statistical_label", mapping=info)
                displayed[p].append(info)
            data_axes.append({"id": p + "-pair", "element_ids": rects})

        for col, (p, name, condition, path, folder) in enumerate([
            ("G", "Delay", "delay", inputs[0], None),
            ("H", "3sTrace", "trace", inputs[11], T3),
            ("I", "10sTrace", "trace", inputs[12], T10 / "I-models")]):
            data = pd.read_parquet(path)
            numeric[p] = {"source": ref(path), "median": counts(data["median"]),
                          "ci_lower": counts(data.ci_lower), "ci_upper": counts(data.ci_upper)}
            x = CENTRES[col] - CURVE_W / 2
            ax = ax_at(p, p.lower() + "-observed", [x, 112., CURVE_W, CURVE_H])
            stat = ax_at(p, p.lower() + "-statistics", [x, 96., CURVE_W, 12.])
            stat.patch.set_visible(False)
            stat.set(xlim=(4, 95), ylim=(-.5, 5.5), xticks=[], yticks=[])
            stat.spines[:].set_visible(False)
            for c in ["control", condition]:
                sub = data.loc[data.condition_id.eq(c)].sort_values("trial_number")
                color = "#00b5eb" if c == "control" else "#ec008c" if c == "delay" else "#f15a29"
                info = {"artifact": ref(path), "condition": c, "fields": ["trial_number", "median"]}
                identify(ax.plot(sub.trial_number, sub["median"], color=color,
                                lw=profile.contract["styles"]["summary_curve"]["linewidth_pt"], label="Control" if c == "control" else name)[0],
                         p + "-" + c + "-median", p, "summary.curve", info)
                identify(ax.fill_between(sub.trial_number, sub.ci_lower, sub.ci_upper, color=color, alpha=.2, lw=0),
                         p + "-" + c + "-CI", p, "uncertainty.confidence_interval",
                         {"artifact": ref(path), "condition": c, "fields": ["ci_lower", "ci_upper"], "resamples": 5000, "seed": 10})
            ax.set(xlim=(4, 95), ylim=LIMITS, xticks=[5, 25, 45, 65, 85, 94], yticks=[-.1, 0, .1])
            style_axes(ax, theme=theme, xlabel="Global CS trial", ylabel="Δ median ln vigor" if p == "G" else "",
                       show_yticklabels=p == "G")
            identify(ax.axhline(0, color="black", lw=theme.axes_linewidth), p + "-zero", p, "reference.signal.zero", {"value": 0})
            for boundary in [14.5, 64.5]:
                identify(ax.axvline(boundary, color="black", lw=theme.axes_linewidth, ls=":"),
                         p + "-phase-" + str(boundary), p, "boundary.phase", {"acquisition_trial": boundary})
            legend = ax.legend(loc="lower right", bbox_to_anchor=(1, 1.03), ncol=2, frameon=False,
                               fontsize=theme.legend_fontsize, handlelength=1, columnspacing=.6, handletextpad=.3, borderaxespad=0)
            identify(legend, "legend__" + p.lower() + "-observed__main", p, "legend.key", {"series": ["Control", name], "right_reference": p + "-observed"})
            data_axes.append({"id": p + "-observed", "element_id": p.lower() + "-observed-rectangle"})
            displayed[p] = []

            def mark(key, trial, y, glyph, info):
                identify(stat.text(trial, y, glyph, ha="center", va="center", fontsize=5), key, p, "annotation.statistical_label", info)
                displayed[p].append({"key": key, "trial": float(trial), "row": float(y), "glyph": glyph, "source": info})

            if p == "G":
                for key, item in g_registry.items():
                    if item.get("scientific_role") != "annotation.statistical_label" or "source_result" not in item:
                        continue
                    old_key = "edit-G-" + key
                    if old_key not in old_ids:
                        raise ValueError("Current G statistical glyph missing: " + old_key)
                    glyph = "".join(old_ids[old_key].itertext()).strip()
                    result = item["source_result"]
                    if "__D-" in key:
                        trial, y = 19.5 + 10 * result["row"], 4
                    elif "__trial-" in key:
                        trial, y = 5 + result["row"], 0
                    else:
                        trial = 9.5 + 10 * result["row"]
                        y = {"mean_p_fdr": 3, "slope_p_raw": 2, "slope_p_fdr": 1}[result["column"]]
                    mark(key, trial, y, glyph, {"source_svg": source["working_svg"], "source_element_id": old_key,
                                             "selected_registry": ref(GSIDE), "selected_result": result})
            else:
                def glyph(q):
                    return "*" * min(3, 4 if q < .0001 else 3 if q < .001 else 2 if q < .01 else 1 if q < .05 else 0)
                dt, local, trial = [pd.read_csv(folder / name) for name in
                                    ["D-interaction-tests.csv", "MR-local-block-tests.csv", "phase-contrasts.csv"]]
                assert (len(dt), len(local), len(trial)) == (8, 9, 90)
                for row in dt.itertuples():
                    if row.p_holm < .05:
                        mark(p + "-D-" + row.block_10_name.replace(" ", "-"), row.center, 4, glyph(row.p_holm),
                             {"artifact": ref(folder / "D-interaction-tests.csv"), "block": row.block_10_name, "p_holm": row.p_holm})
                for row in local.itertuples():
                    for field, y, label in [("mean_p_fdr", 3, "M"), ("slope_p_raw", 2, "R-raw"), ("slope_p_fdr", 1, "R-FDR")]:
                        q = getattr(row, field)
                        text = "n/a" if not np.isfinite(q) else glyph(q)
                        if text:
                            mark(p + "-" + label + "-" + row.block_10_name.replace(" ", "-"), row.center, y, text,
                                 {"artifact": ref(folder / "MR-local-block-tests.csv"), "field": field, "block": row.block_10_name})
                sig = trial.loc[trial.change_p_fdr.lt(.05)]
                identify(stat.scatter(sig.trial_number, np.zeros(len(sig)), marker="*", s=6, color="black", lw=0),
                         p + "-trial-stars", p, "annotation.statistical_label",
                         {"artifact": ref(folder / "phase-contrasts.csv"), "field": "change_p_fdr", "selected_trials": sig.trial_number.astype(int).tolist()})
                displayed[p].append({"key": p + "-trial-stars", "glyph": "scatter-star", "selected_trials": sig.trial_number.astype(int).tolist()})
            for i, label in enumerate(["Pre", "Tr1", "Tr2", "Tr3", "Tr4", "Tr5", "Te1", "Te2", "Te3"]):
                mark(p + "-phase-header-" + label, 9.5 + 10 * i, 5.25, label,
                     {"presentation_role": "title.phase", "acquisition_trial_span": [5 + 10 * i, 14 + 10 * i]})
            if p == "G":
                for i, label in enumerate(["D", "M", "R raw", "R BH", "LMM"]):
                    # Shared row identities; full definitions remain in the legend.
                    text_at(p, "G-stat-row-" + str(i), x - 1.0, 99. + 2 * i, label, size=5, ha="right", va="center",
                            role="annotation.statistical_row_label", mapping={"shared_panels": ["G", "H", "I"], "row_index": i})

        allocations = []
        for row, (y, height) in enumerate([(0., 54.), (54., 36.), (90., 48.)]):
            boxes = [(0., 53.), (53., 46.), (99., 43. if row == 0 else 57.)]
            for col, (x, width) in enumerate(boxes):
                p = panels[row * 3 + col]
                allocations.append({"id": p, "kind": "panel", "box_mm": [x, y, width, height], "group_ids": owned[p]})
        allocations.append({"id": "ABC-colorbar", "kind": "shared", "box_mm": [142., 0., 14., 54.], "group_ids": owned["ABC-colorbar"]})
        relationships = [{"id": "fig2-exploratory-column-" + name, "type": "layout", "panels": list(group),
                          "properties": ["equal measured data-body x centre", "panel-letter left alignment"],
                          "data_axes_ids": [group[0] + "-pair", group[1] + "-pair", group[2] + "-observed"],
                          "target_centre_mm": CENTRES[i], "tolerance_mm": .02}
                         for i, (name, group) in enumerate([("Delay", "ADG"), ("3sTrace", "BEH"), ("10sTrace", "CFI")])]
        frames = {"scope": "figure", "final_size_mm": list(SIZE), "inset_mm": policy["selected_inset_mm"],
                  "allocations": allocations, "background_ids": ["explore-canvas"], "data_axes": data_axes,
                  "alignments": [{"id": "row-" + group, "target": "data_axes", "property": prop,
                                   "ids": [p + ("-pair" if p < "G" else "-observed") for p in group]}
                                  for group in ["ABC", "DEF", "GHI"] for prop in ["top", "bottom", "width", "height"]],
                  "spacing_groups": [{"id": "equal-data-gaps-" + group, "target": "data_axes", "axis": "x",
                                      "ids": [p + ("-pair" if p < "G" else "-observed") for p in group], "minimum_gap_mm": 2.}
                                     for group in ["ABC", "DEF", "GHI"]],
                  "panel_relationships": relationships, "scientific_clips": []}
        provenance = FigureProvenance(figure_id="Figure 2 exploratory columns and display range", analysis_recipe="selected-results-presentation-only",
            source_file=str(Path(__file__)), source_symbol="build", source_hash=ref(Path(__file__))["sha256"],
            reproduction_snippet=".venv-trace/Scripts/python.exe scripts/explore_figure2_columns.py build",
            input_artifacts=tuple(ref(p) for p in inputs), artist_mappings=maps,
            analysis_identity={"scientific_changes": False, "statistics_recomputed": False, "exploratory_display_limits": list(LIMITS),
                               "source_svg": source["working_svg"], "profile_inputs": profile.inputs,
                               "columns": {"Delay": "ADG", "3sTrace": "BEH", "10sTrace": "CFI"},
                               "statistics_font_pt": 5., "row_pitch_mm": 2., "whole_figure_frozen": False})
        export_matplotlib_figure(fig, OUT / "figure2-render", provenance, mode=FigureMode.PUBLICATION,
                                 panel_ids=axis_ids, formats=("svg",), allow_dirty_publication=True, layout_frames=frames)
        plt.close(fig)

    raw = (OUT / "figure2-render.svg").read_bytes()
    root = ET.fromstring(raw)
    for node in root.iter(tag("clipPath")):
        frames["scientific_clips"].append({"id": node.get("id"), "evidence": {
            "purpose": "Author-requested exploratory display limits; full observations retained in immutable numerical sources",
            "definition_sha256": sha(ET.tostring(node)), "source_svg": ref(OUT / "figure2-render.svg"), "limits": list(LIMITS)}})
    svg = embed_spec(raw, frames)
    write(OUT / "figure2.svg", svg)
    legend = (OUT / "previous-selected/legend.md").read_text(encoding="utf-8")
    legend = legend.replace("Colours saturate at ±0.25", "Exploratory colours saturate at −0.1 and +0.1")
    legend = legend.replace("Display ±0.25; outside-view values remain in analysis.", "Exploratory display −0.1 to +0.1; out-of-range observations and connecting lines are clipped only in the artwork and remain in all numerical records and statistical calculations.")
    legend = legend.replace("G row labels apply to H/I;", "G row labels apply to H/I: D, block interaction; M, local mean; R raw/BH, local slope; LMM, trial contrast. All three trajectory axes display −0.1 to +0.1; curves and confidence bands outside this range are clipped only in the artwork, with full values retained in the numerical records. Statistical labels are 5 pt with 2 mm row spacing.")
    legend += "\n\nThis separate exploratory presentation aligns the data-body centres of A/D/G (Delay), B/E/H (3sTrace), and C/F/I (10sTrace). It preserves the selected numerical outcomes, statistics, cohorts, and learner classifications. Saturated heatmap cells include all finite values beyond the colour limits; black denotes missing values. The previous selected draft is preserved as a recoverable snapshot. No freeze is created.\n"
    write(OUT / "legend.md", legend.encode())
    page = render_page([{"id": "fig2", "svg_bytes": svg, "legend_text": legend, "svg": ref(OUT / "figure2.svg"),
                         "legend": ref(OUT / "legend.md"), "whole_figure_frozen": False}])[0]
    write(OUT / "figure2.html", page)
    write(OUT / "numerical-display-evidence.json", {"scientific_changes": False, "statistics_recomputed": False,
          "display_limits": list(LIMITS), "panels": numeric, "statistical_annotations": displayed,
          "bound_inputs": [ref(p) for p in inputs], "source_svg": source["working_svg"],
          "G_statistics": "Exact selected glyph strings and source_result mappings copied from authenticated current SVG/frozen registry; no new MR result inferred"})
    write(OUT / "build-ticket.json", build_ticket.hashes)
    write(OUT / "candidate.json", {"status": "built; saved-render measurements and visual review pending", "svg": ref(OUT / "figure2.svg"),
          "html": ref(OUT / "figure2.html"), "legend": ref(OUT / "legend.md"), "render_registry": ref(OUT / "figure2-render.figure.json"),
          "numerical_evidence": ref(OUT / "numerical-display-evidence.json"), "source": source,
          "profile_inputs": profile.inputs, "layout_policy_sha256": policy_hash, "generator": ref(Path(__file__))})
    ticket.check()
    build_ticket.check()
    print(OUT / "figure2.html")


def publish(note):
    ticket = baseline()
    b = Snapshot([])
    b.hashes = load(OUT / "build-ticket.json")
    b.check()
    ticket.hashes.update(b.hashes)
    candidate = load(OUT / "candidate.json")
    for key in ["svg", "html", "legend", "render_registry", "numerical_evidence", "generator"]:
        assert ref(candidate[key]["path"])["sha256"] == candidate[key]["sha256"], key
    svg = Path(candidate["svg"]["path"]).read_bytes()
    report = load(OUT / "browser.json")
    layout = assess(svg, report)
    assert layout["status"] == "passed", layout["issues"]
    alignment = load(OUT / "column-measurements.json")
    assert alignment["status"] == "passed" and alignment["artifact_sha256"] == sha(svg)
    registry = load(OUT / "previous-selected/semantic-registry.json")
    rendered = load(OUT / "figure2-render.figure.json")
    registry["inherited_previous_registry"] = {"source": candidate["source"]["semantic_registry"],
        "coverage": "Historical scientific context retained; previous SVG artist addresses and measurements are superseded for this separate presentation"}
    registry["element_registry"] = rendered["artist_registry"]
    registry["axis_inventory"] = {"source_svg_sha256": sha(svg), "source": candidate["render_registry"],
                                  "display_limits": list(LIMITS), "coverage": "Explicit new axes and source-bound data/statistical artists; no new scientific definitions"}
    frames = json.loads(ET.fromstring(svg).find(tag("metadata") + "[@id='layout-frames']").text)
    registry["presentation_layout"] = frames
    registry["panel_relationships"] = frames["panel_relationships"]
    evidence = {"status": "verified exploratory presentation; unfrozen", "candidate": candidate,
        "layout_frames": ref(OUT / "layout-evidence.json"), "column_measurements": ref(OUT / "column-measurements.json"),
        "numerical_display_evidence": candidate["numerical_evidence"], "visual_review": note,
        "supersedes_presentation": "Prior outer-content spacing is superseded only for this exploratory version by three aligned assay columns",
        "coverage": "Saved 600 dpi painted/unclipped frames, data-body centres, panel letters, numerical source identity and copied statistical annotations. Production PDF/font embedding and wider scientific/model audits remain separate."}
    registry["exploratory_columns_and_limits_20261020"] = evidence
    registry["whole_figure_frozen"] = False
    state = load(STATE)
    old_draft = copy.deepcopy(state["draft"])
    state.setdefault("preserved_drafts", []).append({"id": "figure2-before-exploratory-range-20261020",
        "state_snapshot": ref(OUT / "previous-selected/state.json"),
        "working_svg": ref(OUT / "previous-selected/figure2.svg"), "review": ref(OUT / "previous-selected/figure2.html"),
        "legend": ref(OUT / "previous-selected/legend.md"), "semantic_registry": ref(OUT / "previous-selected/semantic-registry.json"),
        "original_paths": {k: old_draft[k] for k in ["working_svg", "review", "legend", "semantic_registry"]}})
    draft = state["draft"]
    draft.update(working_svg=candidate["svg"], review=candidate["html"],
                 legend={**candidate["legend"], "panel_ids": list("ABCDEFGHI")},
                 semantic_registry={"path": str(OUT / "semantic-registry.json"), "sha256": sha(encode(registry))},
                 layout_frames=ref(OUT / "layout-evidence.json"),
                 assembly_scale={"final_width_mm": SIZE[0], "final_height_mm": SIZE[1], "plot_body_sizes_preserved": True},
                 status="Separate exploratory presentation: aligned assay columns, display −0.1 to +0.1; scientific selections unchanged; no freeze")
    draft["exploratory_columns_and_limits"] = evidence
    draft["verification"] = {"layout_frames": ref(OUT / "layout-evidence.json"),
        "columns": ref(OUT / "column-measurements.json"), "numerical_display": candidate["numerical_evidence"],
        "visual_review": {"status": "performed", "note": note}, "coverage": evidence["coverage"]}
    draft.setdefault("consistency_requests", []).append({"id": "fig2-exploratory-columns-limits-20261020",
        "scope": list("ABCDEFGHI"), "status": "verified", "properties": {"columns": ["ADG", "BEH", "CFI"],
        "display_limits": list(LIMITS), "statistics_font_pt": 5., "statistical_row_pitch_mm": 2.},
        "evidence": ref(OUT / "column-measurements.json"), "scientific_changes": False})
    draft["reproduction"]["last_scoped_edit"] = {"generator": ref(Path(__file__)),
        "command": ".venv-trace/Scripts/python.exe scripts/explore_figure2_columns.py build",
        "starting_ticket": ref(OUT / "starting-ticket.json"), "source_snapshot": ref(OUT / "previous-selected/state.json")}
    ticket.hashes.update(Snapshot([OUT / "candidate.json", OUT / "layout-evidence.json", OUT / "column-measurements.json",
                                  OUT / "browser.json", *[Path(candidate[k]["path"]) for k in ["svg", "html", "legend", "render_registry", "numerical_evidence"]]]).hashes)
    publish_state(STATE, state, ticket, artifacts=[(OUT / "semantic-registry.json", encode(registry)),
                                                   (OUT / "presentation-evidence.json", encode(evidence))])
    write(OUT / "publication.json", {"status": "published separate exploratory draft", "state": ref(STATE),
          "svg": candidate["svg"], "html": candidate["html"], "legend": candidate["legend"], "no_freeze": True,
          "previous_selected_preserved": state["preserved_drafts"][-1], "visual_review": note})
    print(OUT / "figure2.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["build", "publish"])
    p.add_argument("--visual-review-note")
    a = p.parse_args()
    if a.operation == "build":
        build()
    elif a.visual_review_note:
        publish(a.visual_review_note)
    else:
        p.error("Publication requires a note describing actual saved-render visual review")
