"""Draw an exploratory, fully vector Figure 1B from active experiment timing.

The panel compares CS–US contingency within a training trial. It deliberately
does not encode session phases, trial counts, or US pulse duration.
"""

from __future__ import annotations

import argparse
import sys
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The top-level package imports analysis libraries that this vector-only
# generator does not need. Load its declarative config modules directly.
package_root = REPO_ROOT / "src" / "classical_conditioning"
package = types.ModuleType("classical_conditioning")
package.__path__ = [str(package_root)]
sys.modules.setdefault("classical_conditioning", package)
config_package = types.ModuleType("classical_conditioning.config")
config_package.__path__ = [str(package_root / "config")]
sys.modules.setdefault("classical_conditioning.config", config_package)

from classical_conditioning.config.domain import ConditionRole  # noqa: E402
from classical_conditioning.config.experiments import get_experiment_spec  # noqa: E402

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)

OUTPUT_DIRECTORY = Path(
    "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/"
    "schemes"
)
CS_COLOR = "#0d7f3c"
US_COLOR = "#78358c"
TEXT = "#252d34"
MUTED = "#52606a"
RULE = "#dce2e5"


def tag(name: str) -> str:
    return f"{{{SVG}}}{name}"


def add(parent: ET.Element, name: str, **attrs: object) -> ET.Element:
    return ET.SubElement(parent, tag(name), {key.replace("_", "-"): str(value)
                                              for key, value in attrs.items()})


def label(parent: ET.Element, value: str, x: float, y: float, *, size: int = 18,
          color: str = TEXT, weight: str = "normal", anchor: str = "start") -> ET.Element:
    element = add(parent, "text", x=x, y=y, fill=color, font_size=size,
                  font_weight=weight, text_anchor=anchor)
    element.text = value
    return element


@dataclass(frozen=True)
class Row:
    key: str
    title: str
    accent: str
    cs_duration_s: float
    us_onset_s: float | None
    annotation: str
    detail: str


def protocol_rows() -> list[Row]:
    result: list[Row] = []
    specifications = [
        ("allDelay", "Delay", "delay"),
        ("all3sTrace", "Trace (3 s)", "trace-3s"),
        ("all10sTrace", "Trace (10 s)", "trace-10s"),
    ]
    controls = []
    paired = []
    for experiment_id, title, key in specifications:
        spec = get_experiment_spec(experiment_id)
        control = next(c for c in spec.conditions if c.role is ConditionRole.CONTROL)
        condition = next(c for c in spec.conditions if c.role is ConditionRole.CONDITIONED)
        if control.us_latency_s:
            raise ValueError(f"{experiment_id}: control acquired a fixed US latency")
        if len(set(condition.us_latency_s)) != 1:
            raise ValueError(f"{experiment_id}: cannot show one paired US onset")
        controls.append((spec, control))
        paired.append((spec, condition, title, key))
    durations = {spec.cs_duration_s for spec, _ in controls}
    if len(durations) != 1:
        raise ValueError("Assays no longer share one CS duration")
    cs_duration = durations.pop()
    control_color = controls[0][1].color_rgb_255
    if any(control.color_rgb_255 != control_color for _, control in controls):
        raise ValueError("Control palettes differ; one shared control row is ambiguous")
    result.append(Row("control", "Control", rgb(control_color), cs_duration, None,
                      "US unpaired", "No fixed CS-relative onset"))
    for spec, condition, title, key in paired:
        onset = condition.us_latency_s[0]
        if onset < cs_duration:
            detail = "During CS"
        elif onset == cs_duration:
            detail = "At CS offset"
        else:
            detail = f"{number(onset - cs_duration)} s after CS offset"
        result.append(Row(key, title, rgb(condition.color_rgb_255),
                          spec.cs_duration_s, onset,
                          f"US at {number(onset)} s", detail))
    return result


def rgb(channels: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{value:02x}" for value in channels)


def number(value: float) -> str:
    return f"{value:g}"


def draw_v1(rows: list[Row]) -> ET.Element:
    width, height = 1110, 355
    x_zero, pixels_per_second = 260.0, 26.5
    x_end = x_zero + 22 * pixels_per_second
    row_centers = (80, 149, 218, 287)
    root = ET.Element(tag("svg"), {
        "width": str(width), "height": str(height),
        "viewBox": f"0 0 {width} {height}",
        "font-family": "DejaVu Sans", "role": "img",
        "aria-label": "Figure 1B exploratory within-trial CS and US timing",
    })
    add(root, "title").text = "Figure 1B: CS–US timing by condition"
    add(root, "desc").text = (
        "The four rows share a 0–22 second scale from CS onset. All CS bars last "
        "10 seconds. In Delay, 3-second trace, and 10-second trace, US onset is "
        "at 9, 13, and 20 seconds. Control US timing is unpaired with the CS "
        "and has no fixed onset on this scale. Purple marks show onset, not duration."
    )
    heading = add(root, "g", id="heading")
    label(heading, "Time relative to CS onset (s)", x_zero, 27,
          size=17, color=MUTED, weight="bold")
    add(heading, "rect", x=632, y=15, width=23, height=10, fill=CS_COLOR)
    label(heading, "CS", 664, 26, size=17)
    add(heading, "line", x1=746, y1=11, x2=746, y2=25,
        stroke=US_COLOR, stroke_width=2.6)
    add(heading, "polygon", points="741,24 751,24 746,30", fill=US_COLOR)
    label(heading, "US onset", 763, 26, size=17)
    label(heading, "Relationship", 875, 27, size=17,
          color=MUTED, weight="bold")
    add(root, "line", x1=20, y1=42, x2=1090, y2=42,
        stroke=RULE, stroke_width=1.2)
    add(root, "line", x1=858, y1=42, x2=858, y2=310,
        stroke=RULE, stroke_width=1.2)

    guides = add(root, "g", id="time-guides")
    for value in (0, 10, 20):
        x = x_zero + value * pixels_per_second
        add(guides, "line", x1=x, y1=52, x2=x, y2=310,
            stroke="#e8edef", stroke_width=1.1, stroke_dasharray="3 5")

    for index, (row, cy) in enumerate(zip(rows, row_centers, strict=True)):
        group = add(root, "g", id=f"row-{row.key}",
                    data_cs_duration_s=number(row.cs_duration_s))
        if index:
            add(group, "line", x1=20, y1=cy - 34, x2=1090, y2=cy - 34,
                stroke=RULE, stroke_width=1)
        add(group, "rect", x=24, y=cy - 13, width=6, height=27,
            rx=2, fill=row.accent)
        label(group, row.title, 43, cy + 7, size=22, weight="bold")
        add(group, "line", x1=x_zero, y1=cy + 11, x2=x_end, y2=cy + 11,
            stroke="#738089", stroke_width=1.2)
        add(group, "rect", id=f"cs-{row.key}", x=x_zero, y=cy - 6,
            width=row.cs_duration_s * pixels_per_second, height=16,
            fill=CS_COLOR, data_start_s="0", data_end_s=number(row.cs_duration_s))
        if row.us_onset_s is not None:
            x_us = x_zero + row.us_onset_s * pixels_per_second
            add(group, "line", id=f"us-onset-{row.key}",
                x1=x_us, y1=cy - 18, x2=x_us, y2=cy + 16,
                stroke=US_COLOR, stroke_width=2.7,
                data_time_s=number(row.us_onset_s))
            add(group, "polygon",
                points=f"{x_us-5},{cy+15} {x_us+5},{cy+15} {x_us},{cy+22}",
                fill=US_COLOR)
        label(group, row.annotation, 875, cy - 1, size=18, weight="bold")
        label(group, row.detail, 875, cy + 20, size=15, color=MUTED)

    axis = add(root, "g", id="shared-time-axis")
    add(axis, "line", x1=x_zero, y1=324, x2=x_end, y2=324,
        stroke=TEXT, stroke_width=1.5)
    add(axis, "polygon", points=f"{x_end},320 {x_end+7},324 {x_end},328", fill=TEXT)
    for value in (0, 10, 20):
        x = x_zero + value * pixels_per_second
        add(axis, "line", x1=x, y1=320, x2=x, y2=330,
            stroke=TEXT, stroke_width=1.5)
        label(axis, str(value), x, 349, size=18, anchor="middle")
    return root


def draw_v2(rows: list[Row], *, full_control_bar: bool = False,
            separate_control_trials: bool = False) -> ET.Element:
    """Use C-compatible US dots and illustrative control-trial alternatives."""
    width, height = 1110, 420 if separate_control_trials else 355
    x_zero, pixels_per_second = 340.0, 22.5
    x_min = x_zero - 5 * pixels_per_second
    x_end = x_zero + 22 * pixels_per_second
    row_centers = (104, 211, 277, 343) if separate_control_trials else (80, 149, 218, 287)
    root = ET.Element(tag("svg"), {
        "width": str(width), "height": str(height),
        "viewBox": f"0 0 {width} {height}",
        "font-family": "DejaVu Sans", "role": "img",
        "aria-label": "Figure 1B exploratory within-trial CS and US timing, "
                      + ("version 4" if separate_control_trials else
                         "version 3" if full_control_bar else "version 2"),
    })
    add(root, "title").text = "Figure 1B: trial timing and variable control US onset"
    add(root, "desc").text = (
        "Each row shares a minus-five to twenty-two second scale aligned to CS "
        "onset. The CS lasts ten seconds. Filled purple US-onset dots occur at "
        "nine, thirteen, and twenty seconds in the paired conditions. In the "
        + ("control row, three separate example trials each have a full-sized "
           "CS bar and one US-onset dot; question-mark ticks signify that "
           "control US latency varies from trial to trial. " if separate_control_trials else
           "control row, one full-sized CS bar is shown with three alternative "
           "filled US dots from separate illustrative trials: before, during, "
           "or after the CS. " if full_control_bar else
           "control row, three small CS-aligned trial examples each contain one "
           "filled US dot: before, during, or after the CS. ") + "These positions are "
        "illustrative, not an empirical distribution or one fixed control schedule."
    )
    heading = add(root, "g", id="heading")
    label(heading, "Training trial", 24, 27, size=17, weight="bold")
    label(heading, "Time relative to CS onset (s)", x_min, 27,
          size=17, color=MUTED, weight="bold")
    add(heading, "rect", x=629, y=15, width=23, height=10, fill=CS_COLOR)
    label(heading, "CS", 661, 26, size=17)
    add(heading, "circle", cx=746, cy=20, r=6.4, fill=US_COLOR)
    label(heading, "US onset", 763, 26, size=17)
    label(heading, "Relationship", 875, 27, size=17,
          color=MUTED, weight="bold")
    add(root, "line", x1=20, y1=42, x2=1090, y2=42,
        stroke=RULE, stroke_width=1.2)
    add(root, "line", x1=858, y1=42, x2=858,
        y2=366 if separate_control_trials else 310,
        stroke=RULE, stroke_width=1.2)
    guides = add(root, "g", id="time-guides")
    for value in (0, 10, 20):
        x = x_zero + value * pixels_per_second
        add(guides, "line", x1=x, y1=52, x2=x,
            y2=366 if separate_control_trials else 310,
            stroke="#e8edef", stroke_width=1.1, stroke_dasharray="3 5")

    for index, (row, cy) in enumerate(zip(rows, row_centers, strict=True)):
        group = add(root, "g", id=f"row-{row.key}",
                    data_cs_duration_s=number(row.cs_duration_s))
        if index:
            add(group, "line", x1=20, y1=cy - 34, x2=1090, y2=cy - 34,
                stroke=RULE, stroke_width=1)
        add(group, "rect", x=24, y=cy - 13, width=6, height=27,
            rx=2, fill=row.accent)
        label(group, row.title, 43, cy + 7, size=22, weight="bold")
        if row.us_onset_s is not None:
            add(group, "line", x1=x_min, y1=cy + 11, x2=x_end, y2=cy + 11,
                stroke="#738089", stroke_width=1.2)
            add(group, "rect", id=f"cs-{row.key}", x=x_zero, y=cy - 6,
                width=row.cs_duration_s * pixels_per_second, height=16,
                fill=CS_COLOR, data_start_s="0", data_end_s=number(row.cs_duration_s))
            x_us = x_zero + row.us_onset_s * pixels_per_second
            add(group, "circle", id=f"us-onset-{row.key}",
                cx=x_us, cy=cy + 2, r=7.2, fill=US_COLOR,
                stroke="white", stroke_width=1.3,
                data_time_s=number(row.us_onset_s))
            label(group, row.annotation, 875, cy - 1, size=18, weight="bold")
            label(group, row.detail, 875, cy + 20, size=15, color=MUTED)
        else:
            if separate_control_trials:
                for example, (time_s, relation, example_y) in enumerate(
                    ((-3, "before", 64), (5, "during", 104), (16, "after", 144)), start=1
                ):
                    trial_group = add(group, "g", id=f"control-trial-{example}",
                                      data_example_trial=example,
                                      data_relation=relation)
                    label(trial_group, f"Trial {example}", 213, example_y + 6,
                          size=15, color=MUTED, anchor="end")
                    add(trial_group, "line", x1=x_min, y1=example_y + 11,
                        x2=x_end, y2=example_y + 11,
                        stroke="#738089", stroke_width=1.1)
                    add(trial_group, "rect", id=f"cs-control-trial-{example}",
                        x=x_zero, y=example_y - 6,
                        width=row.cs_duration_s * pixels_per_second, height=16,
                        fill=CS_COLOR, data_start_s="0",
                        data_end_s=number(row.cs_duration_s))
                    x_us = x_zero + time_s * pixels_per_second
                    add(trial_group, "circle", id=f"us-control-trial-{example}",
                        cx=x_us, cy=example_y + 2, r=7.2, fill=US_COLOR,
                        stroke="white", stroke_width=1.3,
                        data_example_time_s=time_s)
                    add(trial_group, "line", x1=x_us, x2=x_us,
                        y1=example_y + 12, y2=example_y + 18,
                        stroke=US_COLOR, stroke_width=1.2)
                    label(trial_group, "?", x_us, example_y + 30,
                          size=16, color=US_COLOR, weight="bold", anchor="middle")
                label(group, "Unpaired Control", 875, cy - 8,
                      size=18, weight="bold")
                label(group, "One US in each example trial", 875, cy + 14,
                      size=15, color=MUTED)
                label(group, "Before / during / after CS", 875, cy + 35,
                      size=14, color=MUTED)
            elif full_control_bar:
                add(group, "line", x1=x_min, y1=cy + 11,
                    x2=x_end, y2=cy + 11,
                    stroke="#738089", stroke_width=1.2)
                add(group, "rect", id="cs-control", x=x_zero, y=cy - 6,
                    width=row.cs_duration_s * pixels_per_second, height=16,
                    fill=CS_COLOR, data_start_s="0",
                    data_end_s=number(row.cs_duration_s))
            examples = (() if separate_control_trials else
                        ((-3, "before", -14), (5, "during", 0), (16, "after", 14)))
            for index, (time_s, relation, offset) in enumerate(examples, start=1):
                row_y = cy + (0 if full_control_bar else offset)
                if not full_control_bar:
                    add(group, "line", x1=x_min, y1=row_y + 3,
                        x2=x_end, y2=row_y + 3,
                        stroke="#9ba6ac", stroke_width=0.9)
                    add(group, "rect", id=f"cs-control-example-{index}",
                        x=x_zero, y=row_y - 2.5,
                        width=row.cs_duration_s * pixels_per_second, height=5,
                        fill=CS_COLOR, data_start_s="0",
                        data_end_s=number(row.cs_duration_s))
                x_us = x_zero + time_s * pixels_per_second
                add(group, "circle", id=f"control-example-{relation}",
                    cx=x_us, cy=row_y + (2 if full_control_bar else 0),
                    r=7.2 if full_control_bar else 5.2, fill=US_COLOR,
                    stroke="white", stroke_width=1,
                    data_example_time_s=time_s,
                    data_example_trial=index,
                    aria_label=f"Illustrative control trial {index}: US onset {relation} the CS")
            if not separate_control_trials:
                label(group, "Alternative trials" if full_control_bar else "Three example trials", 875, cy - 1,
                      size=18, weight="bold")
                label(group, "One US: before / during / after CS" if full_control_bar
                      else "US before / during / after CS", 875, cy + 20,
                      size=14, color=MUTED)

    axis = add(root, "g", id="shared-time-axis")
    axis_y = 385 if separate_control_trials else 324
    add(axis, "line", x1=x_min, y1=axis_y, x2=x_end, y2=axis_y,
        stroke=TEXT, stroke_width=1.5)
    add(axis, "polygon", points=f"{x_min},{axis_y-4} {x_min-7},{axis_y} {x_min},{axis_y+4}", fill=TEXT)
    add(axis, "polygon", points=f"{x_end},{axis_y-4} {x_end+7},{axis_y} {x_end},{axis_y+4}", fill=TEXT)
    for value in (-5, 0, 10, 20):
        x = x_zero + value * pixels_per_second
        add(axis, "line", x1=x, y1=axis_y-4, x2=x, y2=axis_y+6,
            stroke=TEXT, stroke_width=1.5)
        label(axis, str(value), x, axis_y + 25, size=18, anchor="middle")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v1", "v2", "v3", "v4"), default="v4")
    parser.add_argument("--output", type=Path,
                        help="SVG destination, defaulting to the figure SSD folder")
    args = parser.parse_args()
    output = args.output or OUTPUT_DIRECTORY / (
        "Fig1_PanelB_ConditionTiming_exploratory.svg" if args.variant == "v1"
        else f"Fig1_PanelB_ConditionTiming_exploratory_{args.variant}.svg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    root = (draw_v1(protocol_rows()) if args.variant == "v1" else
            draw_v2(protocol_rows(), full_control_bar=args.variant == "v3",
                    separate_control_trials=args.variant == "v4"))
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
