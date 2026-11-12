"""Render stimulus evidence only; no event selection or scientific recomputation."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

from classical_conditioning.figures.theme import apply_theme


@contextmanager
def timing_style(profile):
    """One caller-owned contract snapshot, isolated from other renderers."""
    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        yield {"CS": theme.cs_color, "US": theme.us_color,
               "training": theme.us_color, "maintenance": "#4477AA",
               "viability": "#CC6677", "unknown": "#777777"}


def _save(fig, path, title, footer=""):
    fig.suptitle(title)
    if footer:
        fig.text(.01, .005, footer, fontsize=7, va="bottom", wrap=True)
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _events(report, kind=None):
    return sorted([e for e in report["events"] if e["beg_ms"] is not None and (kind is None or e["type"] == kind)], key=lambda e: (e["beg_ms"], e["source_row"]))


def _boundaries(axis, events, *, x="event_number"):
    previous = None
    for event in events:
        if event["phase"] != previous:
            if event.get(x) is not None:
                axis.axvline(event[x], color=".7", lw=.5)
            previous = event["phase"]


def render_recording(report: dict, settings: dict, output: Path, *, profile) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    identity = f"{report['recording_id']} · {report['condition_id']}"
    with timing_style(profile) as colors:
        events = _events(report)
        # Whole recording: true interval widths, with visible endpoint markers.
        fig, axis = plt.subplots(figsize=(13, 4), layout="constrained")
        lanes = ["CS", *dict.fromkeys(e["role"] for e in events if e["type"] == "US"), "other"]
        lane_ids = {name: i for i, name in enumerate(lanes)}
        origin = report["origin_ms"]
        for event in events:
            lane = "CS" if event["type"] == "CS" else event["role"] if event["type"] == "US" else "other"
            lane = lane if lane in lane_ids else "unknown"
            y = lane_ids[lane]
            begin = (event["beg_ms"]-origin)/60000
            end = (event["end_ms"]-origin)/60000 if event["end_ms"] is not None else begin
            color = colors.get(lane, colors["US"] if lane != "other" else colors["unknown"])
            axis.plot([begin, end], [y, y], color=color, lw=3, solid_capstyle="butt")
            axis.plot(begin, y, "|", color=color, ms=9)
            axis.plot(end, y, "|" if end > begin else "x", color=color, ms=6)
            if event["is_catch"]:
                axis.plot(begin, y-.2, "v", color=color, ms=4)
        bounds = report["acquisition_bounds_ms"]
        if bounds:
            for t in bounds:
                axis.axvline((t-origin)/60000, color="black", ls="--", lw=.8)
        for begin, end in report["camera_gaps_ms"]:
            axis.axvspan((begin-origin)/60000, (end-origin)/60000, color=".6", alpha=.4)
        for phase in dict.fromkeys(e["phase"] for e in events):
            if bounds is not None and phase in report["metadata"].get("phase_bounds_s", {}):
                continue
            group = [e for e in events if e["phase"] == phase]
            if group:
                t = (min(e["beg_ms"] for e in group)-origin)/60000
                axis.axvline(t, color=".6", ls=":", lw=.6)
                axis.text(t, -.6, phase, rotation=30, fontsize=7, va="top")
        for phase, interval in report["metadata"].get("phase_bounds_s", {}).items():
            if bounds is None:
                continue
            provenance = report["metadata"].get("phase_bounds_provenance", {}).get(phase, {}).get("kind", "configured")
            for t in interval:
                axis.axvline(t/60, color=".4", ls="-." if provenance == "recorded" else ":", lw=.7)
            axis.text(interval[0]/60, -.6, f"{phase} ({provenance})", rotation=30, fontsize=7, va="top")
        axis.set(yticks=range(len(lanes)), yticklabels=lanes, ylim=(len(lanes)-.5, -1.2), xlabel=f"Minutes from {report['origin_provenance']}")
        count_text = "; ".join(f"{kind}: {v['observed']}/{v['expected'] if v['expected'] is not None else '?'} expected ({v['usable_onsets']} usable onsets)" for kind, v in report["counts"].items())
        paths.append(_save(fig, output/"timeline.png", identity+" · logged stimulus timeline", count_text+f"\nUnplottable source rows: {report['unplottable_source_rows']}. Phase guides may be configured ordinal assignments; triangles denote configured catch slots."))

        # Dual alignment; each event retains identity even when repeated in rows.
        anchors_by_kind = {kind: _events(report, kind) for kind in ("CS", "US")}
        height = max(4, max(map(len, anchors_by_kind.values()), default=0)*.12+2)
        fig, axes = plt.subplots(1, 2, figsize=(13, height), layout="constrained", squeeze=False)
        appearances = defaultdict(int)
        for axis, (kind, anchors) in zip(axes[0], anchors_by_kind.items()):
            previous_phase, previous_block = None, None
            for row, anchor in enumerate(anchors):
                if anchor["phase"] != previous_phase:
                    axis.axhline(row-.5, color=".4", lw=.6)
                    axis.text(settings["local_window_s"][0], row-.5, anchor["phase"], fontsize=6, va="bottom")
                elif anchor["block"] != previous_block:
                    axis.axhline(row-.5, color=".8", lw=.5, ls=":")
                previous_phase, previous_block = anchor["phase"], anchor["block"]
                for event in events:
                    begin = (event["beg_ms"]-anchor["beg_ms"])/1000
                    end = (event["end_ms"]-anchor["beg_ms"])/1000 if event["end_ms"] is not None else begin
                    lo, hi = settings["local_window_s"]
                    if max(begin, end) < lo or min(begin, end) > hi:
                        continue
                    appearances[event["source_row"]] += 1
                    y = row+(-.13 if event["type"] == "CS" else .13)
                    color = colors["CS"] if event["type"] == "CS" else colors.get(event["role"], colors["US"])
                    axis.plot([begin, end], [y, y], color=color, lw=1.8, solid_capstyle="butt")
                    axis.plot(begin, y, "|", color=color, ms=4)
                    axis.plot(end, y, "|" if end > begin else "x", color=color, ms=3)
                for pair in report["expected_pairs"]:
                    slot = pair["cs" if kind == "CS" else "us"]
                    if anchor["expected_slot"] == slot:
                        axis.plot(pair["latency_s"]*(1 if kind == "CS" else -1), row, "+", color=".2", ms=3)
                if anchor["is_catch"]:
                    axis.plot(0, row, "v", color="black", ms=3)
            step = max(1, len(anchors)//18)
            ticks = list(range(0, len(anchors), step))
            axis.set(xlim=settings["local_window_s"], ylim=(max(len(anchors), 1)-.5, -1.5),
                     yticks=ticks, yticklabels=[str(anchors[i]["event_number"]) for i in ticks],
                     xlabel=f"Seconds from {kind} onset", ylabel=f"Observed {kind} event number", title=f"{kind}-aligned · {len(anchors)} anchors")
            axis.axvline(0, color=".3", ls=":", lw=.6)
        repeated = sum(n > 1 for n in appearances.values())
        coverage = report["local_window_coverage"]
        paths.append(_save(fig, output/"trials.png", identity+" · individual stimulus windows",
            f"Actual onsets/offsets; + expected pairing reference; ▼ configured catch slot. {repeated} distinct events appear in multiple aligned windows.\nCS-window clipping: expected pair references {coverage['expected_pairs_outside_cs_window']}, observed pairs {coverage['observed_pairs_outside_cs_window']}. Window/acquisition coverage is recorded in evidence.json."))

        fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout="constrained")
        for column, kind in enumerate(("CS", "US")):
            group = anchors_by_kind[kind]
            for event in group:
                color = colors["CS"] if kind == "CS" else colors.get(event["role"], colors["US"])
                if event.get("duration_ms") is not None:
                    axes[0, column].plot(event["event_number"], event["duration_ms"], ".", color=color)
                if event["expected_duration_ms"] is not None:
                    axes[0, column].plot(event["event_number"], event["expected_duration_ms"], "_", color=".4", ms=5)
            axes[0, column].set(title=f"{kind} duration", ylabel="Milliseconds")
            for field, label in (("onset_interval_ms", "Onset → onset"), ("offset_gap_ms", "Offset → next onset")):
                valid = [e for e in group if e.get(field) is not None]
                axes[1, column].plot([e["event_number"] for e in valid], [e[field]/1000 for e in valid], ".-", lw=.5, ms=2, label=label)
            axes[1, column].set(title=f"Consecutive {kind} intervals", ylabel="Seconds")
            axes[1, column].legend(fontsize=7)
            for axis in axes[:2, column]:
                _boundaries(axis, group)
                axis.set_xlabel(f"Observed {kind} event number")
        pairs = [p for p in report["expected_pairs"] if p["latency_ms"] is not None]
        for axis, field, label in ((axes[2, 0], "latency_ms", "Expected-pair onset latency"), (axes[2, 1], "offset_gap_ms", "Signed CS-offset → US-onset gap")):
            good = [p for p in pairs if p[field] is not None]
            axis.plot([p["cs"] for p in good], [p[field]/1000 for p in good], "o", ms=3, label="Logged")
            if field == "latency_ms":
                axis.plot([p["cs"] for p in good], [p["latency_s"] for p in good], "_", color="black", label="Expected")
                for p in good:
                    axis.vlines(p["cs"], p["latency_s"]-settings["timing_tolerance_ms"]/1000, p["latency_s"]+settings["timing_tolerance_ms"]/1000, color=".6", lw=2, zorder=0)
            else:
                nominal = [p for p in good if p["expected_cs_duration_ms"] is not None]
                axis.plot([p["cs"] for p in nominal], [p["latency_s"]-p["expected_cs_duration_ms"]/1000 for p in nominal], "_", color="black", label="Expected")
            if not good:
                axis.text(.5, .5, "No configured, checkable pairs\nControls use all-pairs lag views", transform=axis.transAxes, ha="center")
            else:
                axis.legend()
            axis.set(title=label, xlabel="Expected CS slot", ylabel="Seconds")
        paths.append(_save(fig, output/"timing.png", identity+" · ordered timing measurements",
            "US role colors: training purple; maintenance blue; viability rose; unresolved grey. Nominal durations are references, not exclusion thresholds.\nPhase guides are configured unless independently annotated; unresolved correspondence is not silently repaired."))

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), layout="constrained")
        alignment = report["alignment"]
        for axis, kind in zip(axes, ("CS", "US")):
            for edge in ("beg", "end"):
                matched = [r for r in alignment["matches"] if r.get("type") == kind and r.get(f"{edge}_residual_ms") is not None]
                axis.plot([r.get("event_number", r["source_row"]) for r in matched], [r[f"{edge}_residual_ms"] for r in matched], ".", label=edge)
            axis.axhline(0, color=".5", lw=.5)
            axis.set(title=kind, xlabel="Processed event number / source row", ylabel="Processed − raw (ms)")
            axis.legend()
            if alignment["status"] != "available":
                axis.text(.5, .6, alignment["status"], ha="center", transform=axis.transAxes)
        unmatched = sum(r["status"] != "matched" for r in alignment["matches"])
        paths.append(_save(fig, output/"alignment.png", identity+" · processed timing comparison",
            f"Status: {alignment['status']}; unmatched entries: {unmatched}; phase-label mismatches: {alignment.get('phase_label_mismatch_count', '?')}; labels: {alignment.get('label_status', 'unavailable')}.\n{alignment.get('limitation', 'Missing processed timing is unavailable, never zero.')}"))
    return paths


def render_collection(groups: list[dict], comparisons: list[dict], output: Path, *, profile) -> list[Path]:
    """Render precomputed fish/schedule summaries; never choose aggregation here."""
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    with timing_style(profile):
        for index, group in enumerate(groups):
            experiment, condition, phase, role = (group[k] for k in ("experiment_id", "condition_id", "phase", "us_role"))
            population = group["recordings"]
            edges = np.array(group["edges_s"])
            curves = np.array(group["curves"], dtype=float)
            n = group["contributing_fish"]
            fig, axes = plt.subplots(4, 1, figsize=(12, max(10, len(population)*.15+7)), layout="constrained", gridspec_kw={"height_ratios": [2, 1, 1, .8]})
            cmap = mpl.colormaps["viridis"].copy()
            cmap.set_bad(".8")
            maximum = float(np.nanmax(curves)) if np.isfinite(curves).any() else 1
            mesh = axes[0].imshow(np.ma.masked_invalid(curves), aspect="auto", origin="upper", extent=(edges[0], edges[-1], len(population)-.5, -.5), cmap=cmap, interpolation="nearest", vmin=0, vmax=maximum if maximum > 0 else 1)
            axes[0].set_yticks(range(len(population)), [f"{r['recording_id']} | {r['day']} | batch {r['batch_id']} | CS {r['anchor_count']}" for r in population], fontsize=6)
            fig.colorbar(mesh, ax=axes[0], label="US counts / CS anchor (uncorrected coverage)")
            centers = (edges[:-1]+edges[1:])/2
            axes[1].stairs(group["equal_fish_counts_per_anchor"], edges, label=f"Equal fish · n={n}")
            if group["equal_intended_schedule_counts_per_anchor"] is not None:
                axes[1].stairs(group["equal_intended_schedule_counts_per_anchor"], edges, ls="--", label=f"Equal intended schedules · {group['known_intended_schedule_count']} (known subset)")
            axes[1].stairs(group["shifted_equal_fish_counts_per_anchor"], group["shifted_edges_s"], color=".5", ls=":", label="Half-bin shift · equal fish")
            for stratum in group["strata"]:
                axes[2].plot(centers, stratum["counts_per_anchor"], lw=.7, label=f"{stratum['day']} / {stratum['batch_id']} · n={stratum['fish_count']}")
            axes[1].legend(fontsize=7)
            axes[2].legend(fontsize=6, ncol=2)
            axes[2].set_title("Acquisition-day / verified-batch strata (unknown retained)")
            for axis in axes:
                axis.set(xlabel="US onset − CS onset (s)", xlim=(edges[0], edges[-1]))
                axis.axvline(0, color=".5", ls=":", lw=.5)
            for axis in axes[1:3]:
                axis.set_ylabel("US counts / CS anchor")
            known_exposure = group["coverage_fish_count"]
            if group["observable_fraction_equal_fish"] is not None:
                axes[3].stairs(group["observable_fraction_equal_fish"], edges, label=f"Equal-fish mean · n={known_exposure}")
                axes[3].legend(fontsize=7)
            else:
                axes[3].text(.5, .5, "Observable-time coverage unavailable", ha="center", transform=axes[3].transAxes)
            axes[3].set(ylabel="Observable fraction", ylim=(0, 1.05), title="Camera coverage; unverified phase boundaries are not imposed")
            name = f"lags-{index:03d}.png"
            exact = "; ".join(f"{r['recording_id']}: {r['target_coincidences']}" for r in population if r["target_coincidences"] is not None)
            if len(exact) > 180:
                exact = "Per-fish exact target counts in evidence.json (independent of bins)"
            paths.append(_save(fig, output/name, f"{experiment} · {condition} · {phase} · US {role}",
                f"Contributing fish {n}/{len(population)}; acquisition exposure available {known_exposure}/{n}. Grey rows unavailable.\n{group['lag_scope']}. No rate correction or randomness verdict. Exact target counts: {exact}"))
        # One comparison page per phase, with separate CS/US matrices and residuals.
        by_phase = defaultdict(dict)
        for comparison in comparisons:
            by_phase[(comparison["experiment_id"], comparison["phase"])][comparison["type"]] = comparison
        for index, ((experiment, phase), kinds) in enumerate(sorted(by_phase.items())):
            fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
            for column, kind in enumerate(("CS", "US")):
                item = kinds[kind]
                matrix = np.array([[np.nan if n is None else n for n in row] for row in item["maximum_residual_ms"]])
                maximum = max(1, float(np.nanmax(matrix))) if np.isfinite(matrix).any() else 1
                cmap = mpl.colormaps["magma"].copy()
                cmap.set_bad(".8")
                mesh = axes[0, column].imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=SymLogNorm(linthresh=1, vmin=0, vmax=maximum), interpolation="nearest")
                ids = item["recording_labels"]
                axes[0, column].set(xticks=range(len(ids)), xticklabels=ids, yticks=range(len(ids)), yticklabels=ids, title=f"{kind} schedule similarity")
                axes[0, column].tick_params(axis="x", rotation=90, labelsize=5)
                axes[0, column].tick_params(axis="y", labelsize=5)
                fig.colorbar(mesh, ax=axes[0, column], label="Maximum residual after common offset (ms)")
                for pair in item["closest_pairs"]:
                    axes[1, column].plot(range(1, len(pair["residuals_ms"])+1), pair["residuals_ms"], ".-", ms=2, lw=.5, label=f"{pair['first']} / {pair['second']}")
                if item["closest_pairs"]:
                    axes[1, column].legend(fontsize=6)
                axes[1, column].set(xlabel=f"Observed {kind} position within phase", ylabel="Residual (ms)", title="Up to three most similar recording pairs")
            paths.append(_save(fig, output/f"schedules-{index:03d}.png", f"{experiment} · {phase} · schedule comparisons",
                "Rows/columns ordered by condition, acquisition day, verified batch and recording ID (full identities in JSON).\nGrey: unequal counts or fewer than two events. Translation only; drift is retained. Logged similarity does not establish intended schedule reuse."))
    return paths
