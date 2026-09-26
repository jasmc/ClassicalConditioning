"""Build matching, fully vector Figure 1 single-fish legacy-vigor heatmaps.

The signal is signed bout-log vigor from the historical distal angular speed
metric, centered on each trial's -20 to 0 s moving-bout median. The three
example fish share one palette and fixed -0.25 to +0.25 display scale.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.colors import Normalize, to_hex

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classical_conditioning.analysis.figure4 import verify_expected_us  # noqa: E402
from classical_conditioning.figures.example_traces import METRIC_COLUMNS  # noqa: E402
from classical_conditioning.figures.signed_bout_heatmap import calculate_fish_heatmaps  # noqa: E402
from classical_conditioning.figures.theme import apply_theme, heatmap_cmap  # noqa: E402

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
METRIC = "legacy_distal_angular_speed"
ROOT = Path("J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly")
PHASES = (("Pre-Train", 5, 14), ("Train", 15, 64), ("Test", 65, 94))
FISH = (
    ("F", "Delay", "20221115_07", "J:/Digested Data/allDelay-full-v1",
     "frame_activity_candidates-corrected-v1.parquet",
     "movement_state_candidates-corrected-v2.parquet", "delay", 9),
    ("G", "3 s Trace", "20230307_12", "F:/Digested Data/all3sTrace-full-v1",
     "frame_activity_candidates-corrected.parquet",
     "movement_state_candidates-corrected.parquet", "trace", 13),
    ("H", "Control", "20221115_09", "J:/Digested Data/allDelay-full-v1",
     "frame_activity_candidates-corrected-v1.parquet",
     "movement_state_candidates-corrected-v2.parquet", "control", None),
)


def q(name: str) -> str:
    return f"{{{SVG}}}{name}"


def add(parent: ET.Element, name: str, **attributes: object) -> ET.Element:
    return ET.SubElement(parent, q(name), {
        key.replace("_", "-"): str(value) for key, value in attributes.items()
    })


def text(parent: ET.Element, value: str, x: float, y: float, *, size: int = 14,
         weight: str = "normal", fill: str = "#252d34", anchor: str = "start") -> None:
    add(parent, "text", x=x, y=y, fill=fill, font_size=size,
        font_weight=weight, text_anchor=anchor).text = value


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hashed.update(block)
    return hashed.hexdigest()


def read_windows(path: Path, columns: list[str], intervals: list[tuple[int, int]]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    time_index = parquet.schema.names.index("AbsoluteTime")
    parts = []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(time_index).statistics
        if stats is None or stats.min is None or stats.max is None:
            raise ValueError(f"No AbsoluteTime row-group statistics: {path}")
        overlapping = [(start, stop) for start, stop in intervals
                       if stats.max >= start and stats.min < stop]
        if not overlapping:
            continue
        part = parquet.read_row_group(row_group, columns=columns).to_pandas()
        times = part["AbsoluteTime"].to_numpy(dtype=np.int64)
        keep = np.zeros(len(part), dtype=bool)
        for start, stop in overlapping:
            keep |= (times >= start) & (times < stop)
        if keep.any():
            parts.append(part.loc[keep])
    if not parts:
        raise ValueError(f"No recorded frames in requested CS windows: {path}")
    return pd.concat(parts, ignore_index=True)


def load_fish(spec: tuple) -> tuple[pd.DataFrame, list[dict], float | None]:
    panel, name, recording, project_name, metric_name, movement_name, role, us_s = spec
    project = Path(project_name)
    processed = project / "Processed data" / recording
    paths = {
        "metrics": processed / metric_name,
        "movement": processed / movement_name,
        "protocol": processed / "stimulus_events.parquet",
    }
    manifest_path = project / "Metadata" / f"{recording}_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["recording_id"] != recording or manifest["condition_id"] != role:
        raise ValueError(f"{panel}: source manifest disagrees with example fish identity")
    for kind in ("metrics", "movement"):
        marker_name = (
            f"{recording}_candidate-corrected-v1_complete.json" if kind == "metrics" and panel != "G" else
            f"{recording}_movement-candidate-corrected-v2_complete.json" if kind == "movement" and panel != "G" else
            f"{recording}_candidate-corrected_complete.json" if kind == "metrics" else
            f"{recording}_movement-candidate-corrected_complete.json"
        )
        marker = json.loads((project / "Metadata" / marker_name).read_text(encoding="utf-8"))
        expected = marker["metrics_sha256" if kind == "metrics" else "movement_sha256"]
        if marker["status"] != "complete" or digest(paths[kind]) != expected:
            raise ValueError(f"{panel}: {kind} artifact differs from its completion marker")
    protocol_sha = digest(paths["protocol"])
    if protocol_sha != manifest["artifacts"]["protocol"]["sha256"]:
        raise ValueError(f"{panel}: stimulus events differ from the source manifest")
    protocol = pq.read_table(paths["protocol"]).to_pandas()
    verified_us_s = None
    if panel in {"F", "G"}:
        verified_us_s, paired_trials = verify_expected_us(
            protocol, "allDelay" if panel == "F" else "all3sTrace")
        if paired_trials != 46 or abs(verified_us_s - us_s) > 0.1:
            raise ValueError(f"{panel}: measured paired US onset differs from Figure 1B")
    cycles = protocol.loc[protocol["Type"].eq("Cycle")].sort_values("Beg").reset_index(drop=True)
    if len(cycles) != 94:
        raise ValueError(f"{panel}: expected 94 measured CS cycles, found {len(cycles)}")
    intervals = [(int(cycles.iloc[trial - 1]["Beg"] - 20_000),
                  int(cycles.iloc[trial - 1]["Beg"] + 20_000))
                 for trial in range(5, 95)]
    metrics = read_windows(paths["metrics"],
                           ["FrameID", "AbsoluteTime", METRIC_COLUMNS[METRIC]], intervals)
    movement = read_windows(paths["movement"],
                            ["FrameID", "AbsoluteTime", "valid", "moving", "bout_id"],
                            intervals)
    bins = calculate_fish_heatmaps(metrics, movement, cycles,
                                   recording_id=recording, metric_ids=(METRIC,))
    del metrics, movement
    gc.collect()
    inputs = [{"path": str(path), "sha256": digest(path)}
              for path in (paths["protocol"], manifest_path)]
    for kind in ("metrics", "movement"):
        inputs.append({"path": str(paths[kind]), "sha256":
                       json.loads((project / "Metadata" / (
                           f"{recording}_candidate-corrected-v1_complete.json" if kind == "metrics" and panel != "G" else
                           f"{recording}_movement-candidate-corrected-v2_complete.json" if kind == "movement" and panel != "G" else
                           f"{recording}_candidate-corrected_complete.json" if kind == "metrics" else
                           f"{recording}_movement-candidate-corrected_complete.json"
                       )).read_text(encoding="utf-8"))[
                           "metrics_sha256" if kind == "metrics" else "movement_sha256"]})
    return bins, inputs, verified_us_s


def draw(spec: tuple, bins: pd.DataFrame) -> ET.Element:
    panel, name, recording, _, _, _, _, us_s = spec
    theme = apply_theme()
    cmap = heatmap_cmap(theme.single_fish_scaled_vigor_cmap, theme)
    scale = Normalize(theme.single_fish_scaled_vigor_vmin,
                      theme.single_fish_scaled_vigor_vmax, clip=True)
    root = ET.Element(q("svg"), {
        "width": "530", "height": "395", "viewBox": "0 0 530 395",
        "font-family": "DejaVu Sans", "role": "img",
        "aria-label": f"Figure 1{panel} {name} single-fish legacy vigor heatmap",
        "data-metric-id": METRIC, "data-recording-id": recording,
        "data-signal": "signed_bout_log_vigor_pre20_median",
    })
    add(root, "title").text = f"{name} fish · {recording}"
    add(root, "desc").text = (
        "Measured single-fish legacy distal angular speed. Moving-bout log vigor "
        "is centered on each trial's -20 to 0 second pre-CS median, then "
        "averaged in 0.5 second bins. Missing bins are black. All three fish "
        "share managua_r at -0.25 to +0.25. The green guides mark 0 and 10 s."
    )
    text(root, f"{name} fish", 62, 28, size=19, weight="bold")
    text(root, recording, 506, 28, size=15, anchor="end", fill="#52606a")
    text(root, "Legacy distal angular speed", 62, 49, size=13, fill="#52606a")
    x0, x1 = 84.0, 420.0
    bin_width = (x1 - x0) / 80
    row_height = 2.6
    phase_top = (66.0, 99.0, 236.0)
    expected_times = np.arange(-20.0, 20.0, 0.5) + 0.25
    for (phase, first, last), top in zip(PHASES, phase_top, strict=True):
        section = add(root, "g", id=f"phase-{phase.lower().replace('-', '-')}",
                      data_trial_first=first, data_trial_last=last)
        values = bins.pivot(index="Trial number", columns="Time bin center (s)",
                            values="Signed log vigor").reindex(
                                index=range(first, last + 1), columns=expected_times)
        if values.isna().all().all():
            raise ValueError(f"{panel}: {phase} has no vigor values")
        matrix = values.to_numpy(dtype=float)
        for row in range(matrix.shape[0]):
            y = top + row * row_height
            for col, value in enumerate(matrix[row]):
                fill = "#000000" if not np.isfinite(value) else to_hex(cmap(scale(value)), keep_alpha=False)
                add(section, "rect", x=f"{x0 + col * bin_width:.3f}",
                    y=f"{y:.3f}", width=f"{bin_width + .02:.3f}",
                    height=f"{row_height + .02:.3f}", fill=fill)
        bottom = top + (last - first + 1) * row_height
        for seconds in (0, 10):
            x = x0 + (seconds + 20) / 0.5 * bin_width
            add(section, "line", x1=x, x2=x, y1=top, y2=bottom,
                stroke="#0d7f3c", stroke_width=2.0)
        if phase == "Train" and us_s is not None:
            x = x0 + (us_s + 20) / 0.5 * bin_width
            add(section, "line", x1=x, x2=x, y1=top, y2=bottom,
                stroke="#78358c", stroke_width=1.9, stroke_dasharray="3 2")
        text(root, phase, 75, top + (bottom - top) / 2 + 5,
             size=13, weight="bold", anchor="end")
        add(root, "rect", x=x0, y=top, width=x1-x0, height=bottom-top,
            fill="none", stroke="#8e979d", stroke_width=.75)
    for seconds in (-20, -10, 0, 10, 20):
        x = x0 + (seconds + 20) / 0.5 * bin_width
        add(root, "line", x1=x, x2=x, y1=321, y2=326,
            stroke="#252d34", stroke_width=1)
        text(root, str(seconds), x, 343, size=12, anchor="middle")
    text(root, "Time relative to CS onset (s)", (x0+x1)/2, 371,
         size=14, weight="bold", anchor="middle")
    text(root, "Signed log vigor", 506, 51, size=12, anchor="end")
    bar_x, bar_y, bar_w, bar_h = 455, 68, 10, 246
    for index in range(120):
        fraction = index / 119
        value = theme.single_fish_scaled_vigor_vmax - fraction * (
            theme.single_fish_scaled_vigor_vmax - theme.single_fish_scaled_vigor_vmin)
        add(root, "rect", x=bar_x, y=bar_y + index * bar_h / 120,
            width=bar_w, height=bar_h/120 + .1,
            fill=to_hex(cmap(scale(value)), keep_alpha=False))
    add(root, "rect", x=bar_x, y=bar_y, width=bar_w, height=bar_h,
        fill="none", stroke="#8e979d", stroke_width=.75)
    for value, label in ((.25, "+0.25"), (0, "0"), (-.25, "−0.25")):
        y = bar_y + (.25 - value) / .5 * bar_h
        add(root, "line", x1=bar_x+bar_w, x2=bar_x+bar_w+5,
            y1=y, y2=y, stroke="#252d34", stroke_width=1)
        text(root, label, bar_x+bar_w+8, y+4, size=11)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("F", "G", "H"), action="append",
                        help="Build one or more selected fish; default all three")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "heatmaps")
    args = parser.parse_args()
    selected = set(args.panel or ("F", "G", "H"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in FISH:
        panel, name, recording, *_ = spec
        if panel not in selected:
            continue
        print(f"Calculating Figure 1{panel}: {name} {recording}", flush=True)
        bins, inputs, verified_us_s = load_fish(spec)
        if len(bins) != 90 * 80 or set(bins["Metric ID"]) != {METRIC}:
            raise ValueError(f"{panel}: wrong signed bin count or metric")
        svg = args.output_dir / f"Fig1_Panel{panel}_{name.replace(' ', '')}_legacy-vigor_v1.svg"
        data = svg.with_suffix(".parquet")
        pq.write_table(pa.Table.from_pandas(bins, preserve_index=False), data)
        ET.ElementTree(draw(spec, bins)).write(svg, encoding="utf-8", xml_declaration=True)
        sidecar = {
            "panel": panel, "recording_id": recording, "condition": name,
            "metric_id": METRIC, "signal": "signed_bout_log_vigor_pre20_median",
            "baseline_s": [-20, 0], "bin_width_s": 0.5,
            "palette": "managua_r", "display_range": [-0.25, 0.25],
            "missing_color": "black", "input_artifacts": inputs,
            "verified_paired_us_onset_s": verified_us_s,
            "panel_data": str(data), "panel_data_sha256": digest(data),
            "svg": str(svg), "svg_sha256": digest(svg),
        }
        svg.with_suffix(".svg.json").write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        print(svg, flush=True)
        del bins
        gc.collect()


if __name__ == "__main__":
    main()
