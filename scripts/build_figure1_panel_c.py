"""Build an editable SVG schematic of the Figure 1 session protocol.

Counts come from the explicit figure specification and are checked against the
active experiment definitions. The phase widths are schematic, not minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "configs" / "paper-figures" / "figure1-session-protocol.json"
OUTPUT_DIRECTORY = Path(
    "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/"
    "schemes"
)

# Import only declarative config; the package root loads optional analysis
# libraries that this vector generator does not need.
package_root = REPO_ROOT / "src" / "classical_conditioning"
package = types.ModuleType("classical_conditioning")
package.__path__ = [str(package_root)]
sys.modules.setdefault("classical_conditioning", package)
config_package = types.ModuleType("classical_conditioning.config")
config_package.__path__ = [str(package_root / "config")]
sys.modules.setdefault("classical_conditioning.config", config_package)

from classical_conditioning.config.domain import Alignment, ConditionRole, Phase  # noqa: E402
from classical_conditioning.config.experiments import get_experiment_spec  # noqa: E402

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
CS_GREEN = "#0d7f3c"
US_PURPLE = "#78358c"
CONTROL_BLUE = "#00aeef"
INK = "#252d34"
SECONDARY = "#52606a"
BORDER = "#adb5bb"
LIGHT_BORDER = "#dce2e5"


def tag(name: str) -> str:
    return f"{{{SVG}}}{name}"


def add(parent: ET.Element, name: str, **attrs: object) -> ET.Element:
    return ET.SubElement(parent, tag(name), {
        key.replace("_", "-"): str(value) for key, value in attrs.items()
    })


def text(parent: ET.Element, value: str, x: float, y: float, *, size: int = 20,
         weight: str = "normal", color: str = INK, anchor: str = "start") -> ET.Element:
    element = add(parent, "text", x=x, y=y, fill=color, font_size=size,
                  font_weight=weight, text_anchor=anchor)
    element.text = value
    return element


def validate(specification: dict) -> dict[str, dict]:
    phases = specification["phases"]
    if [phase["id"] for phase in phases] != ["priming", "pretraining", "training", "testing"]:
        raise ValueError("Figure 1C requires four phases in chronological order")
    by_id = {phase["id"]: phase for phase in phases}
    train = by_id["training"]
    if train["paired_long_us_count"] + train["catch_cs_count"] != train["cs_count"]:
        raise ValueError("Training CS count must equal paired plus catch trials")
    if sum(phase["cs_count"] for phase in phases) != 94:
        raise ValueError("The current protocol requires 94 CS/CS-like presentations")
    if (by_id["priming"]["short_us_count"]
            + by_id["pretraining"]["short_us_count"]
            + train["paired_long_us_count"]
            + by_id["testing"]["short_us_count"] != 78):
        raise ValueError("The current protocol requires 78 regular US events")
    if [specification[key] for key in (
        "short_us_duration_ms", "long_us_duration_ms", "final_check_us_duration_ms"
    )] != [50, 100, 500]:
        raise ValueError("US pulse durations differ from the reviewed paper methods")
    for experiment_id in ("allDelay", "all3sTrace", "all10sTrace"):
        experiment = get_experiment_spec(experiment_id)
        if experiment.cs_duration_s != specification["cs_duration_s"]:
            raise ValueError(f"{experiment_id}: CS duration does not match Figure 1C")
        cs_counts = {
            phase: sum(trial.alignment is Alignment.CS and trial.phase is phase
                       for trial in experiment.analysis_trials)
            for phase in (Phase.PRE, Phase.TRAIN, Phase.TEST)
        }
        if cs_counts != {
            Phase.PRE: by_id["pretraining"]["cs_count"],
            Phase.TRAIN: train["cs_count"],
            Phase.TEST: by_id["testing"]["cs_count"],
        }:
            raise ValueError(f"{experiment_id}: phase CS counts do not match Figure 1C")
        conditioned = next(c for c in experiment.conditions
                           if c.role is ConditionRole.CONDITIONED)
        if len(conditioned.us_latency_s) != train["paired_long_us_count"]:
            raise ValueError(f"{experiment_id}: paired training US count differs")
        catch_count = sum(trial.alignment is Alignment.CS and trial.phase is Phase.TRAIN
                          and trial.catch for trial in experiment.analysis_trials)
        if catch_count != train["catch_cs_count"]:
            raise ValueError(f"{experiment_id}: training catch count differs")
    return by_id


def cs_mark(parent: ET.Element, x: float, y: float) -> None:
    add(parent, "rect", x=x, y=y - 10, width=19, height=10, fill=CS_GREEN)


def short_us_mark(parent: ET.Element, x: float, y: float) -> None:
    add(parent, "circle", cx=x + 9, cy=y - 5, r=6.5, fill="white",
        stroke=US_PURPLE, stroke_width=2.3)


def long_us_mark(parent: ET.Element, x: float, y: float) -> None:
    add(parent, "circle", cx=x + 9, cy=y - 5, r=6.8, fill=US_PURPLE)


def check_mark(parent: ET.Element, cx: float, cy: float) -> None:
    add(parent, "polygon", points=f"{cx},{cy-10} {cx+10},{cy} {cx},{cy+10} {cx-10},{cy}",
        fill=US_PURPLE)


def arrow(parent: ET.Element, x1: float, x2: float, y: float) -> None:
    add(parent, "line", x1=x1, y1=y, x2=x2 - 4, y2=y,
        stroke=BORDER, stroke_width=1.6)
    add(parent, "polygon", points=f"{x2-7},{y-5} {x2},{y} {x2-7},{y+5}", fill=BORDER)


def card(root: ET.Element, phase: dict, x: int, width: int) -> ET.Element:
    group = add(root, "g", id=f"phase-{phase['id']}",
                data_cs_count=phase["cs_count"])
    for key in ("short_us_count", "paired_long_us_count", "catch_cs_count"):
        if key in phase:
            group.set(key.replace("_", "-"), str(phase[key]))
    add(group, "rect", x=x, y=101, width=width, height=209, rx=9,
        fill="white", stroke=LIGHT_BORDER, stroke_width=1.5)
    if phase["id"] == "priming":
        text(group, "Priming /", x + 18, 135, size=23, weight="bold")
        text(group, "habituation", x + 18, 160, size=23, weight="bold")
    else:
        text(group, phase["name"], x + 18, 144, size=23, weight="bold")
    cs_mark(group, x + 20, 199)
    cs_word = "CS-like" if phase["id"] == "priming" else "CS presentations"
    text(group, f"{phase['cs_count']} {cs_word}", x + 49, 199, size=19)
    if phase["id"] == "priming":
        text(group, "includes left-side light", x + 49, 221,
             size=15, color=SECONDARY)
    if phase["id"] in {"priming", "pretraining", "testing"}:
        short_us_mark(group, x + 20, 265)
        text(group, f"{phase['short_us_count']} separate short US",
             x + 49, 266, size=18)
    else:
        long_us_mark(group, x + 20, 242)
        text(group, f"{phase['paired_long_us_count']} paired long US",
             x + 49, 243, size=19)
        add(group, "rect", x=x + 22, y=271, width=14, height=14,
            fill="white", stroke=CS_GREEN, stroke_width=2)
        text(group, f"{phase['catch_cs_count']} CS-only catch trials",
             x + 49, 284, size=17)
        text(group, "Trial timing in B", x + width - 18, 143,
             size=16, color=US_PURPLE, weight="bold", anchor="end")
    return group


def draw(specification: dict, *, show_viability_check: bool = False) -> ET.Element:
    phases = validate(specification)
    root = ET.Element(tag("svg"), {
        "width": "1690", "height": "427", "viewBox": "0 0 1690 427",
        "font-family": "DejaVu Sans", "role": "img",
        "aria-label": "Figure 1C exploratory conditioning-session sequence",
    })
    add(root, "title").text = "Conditioning-session sequence"
    add(root, "desc").text = (
        "Priming has four CS-like presentations and fifteen short US pulses; "
        "pre-training has ten CS and three separate short US pulses; training "
        "has fifty CS, including forty-six paired long US trials and four "
        "CS-only catch trials; testing has thirty CS and fourteen separate "
        "short US pulses. "
        + ("A final 500 ms US checks viability. " if show_viability_check else "")
        + "Control fish "
        "receive the same long-US times but randomized training CS onsets. "
        "Phase widths are schematic, not a minute scale."
    )
    text(root, "Session sequence", 95, 41, size=26, weight="bold")
    text(root, "Schematic · not to scale", 1655, 41,
         size=17, color=SECONDARY, anchor="end")
    legend = add(root, "g", id="stimulus-key")
    cs_mark(legend, 97, 75)
    text(legend, "CS (10 s)", 125, 76, size=17)
    short_us_mark(legend, 295, 80)
    text(legend, "short US (50 ms)", 323, 76, size=17)
    long_us_mark(legend, 550, 80)
    text(legend, "long US (100 ms)", 578, 76, size=17)
    if show_viability_check:
        check_mark(legend, 850, 72)
        text(legend, "final check (500 ms)", 870, 76, size=17)

    positions = ({
        "priming": (95, 260),
        "pretraining": (385, 260),
        "training": (675, 455),
        "testing": (1160, 330),
    } if show_viability_check else {
        "priming": (95, 300),
        "pretraining": (425, 300),
        "training": (755, 500),
        "testing": (1285, 300),
    })
    for phase_id in ("priming", "pretraining", "training", "testing"):
        x, width = positions[phase_id]
        card(root, phases[phase_id], x, width)
    arrows = add(root, "g", id="phase-arrows")
    phase_order = ("priming", "pretraining", "training", "testing")
    for earlier, later in zip(phase_order, phase_order[1:]):
        start_x, earlier_width = positions[earlier]
        next_x, _ = positions[later]
        arrow(arrows, start_x + earlier_width + 3, next_x - 4, 205)
    if show_viability_check:
        arrow(arrows, 1493, 1516, 205)
        final = add(root, "g", id="final-check", data_us_duration_ms=500)
        add(final, "rect", x=1520, y=101, width=145, height=209, rx=9,
            fill="white", stroke=LIGHT_BORDER, stroke_width=1.5)
        text(final, "Viability", 1592, 145, size=20, weight="bold", anchor="middle")
        text(final, "check", 1592, 169, size=20, weight="bold", anchor="middle")
        check_mark(final, 1592, 215)
        text(final, "500 ms US", 1592, 255, size=17, anchor="middle")
        text(final, "after Test", 1592, 279, size=15,
             color=SECONDARY, anchor="middle")

    notes = add(root, "g", id="protocol-notes")
    short_us_mark(notes, 95, 354)
    text(notes, "Short US pulses are interspersed; they are not conditioning pairings.",
         123, 351, size=17, color=SECONDARY)
    add(notes, "rect", x=840, y=332, width=5, height=26,
        rx=2, fill=CONTROL_BLUE)
    text(notes, "Control:", 857, 351, size=18, weight="bold")
    text(notes, specification["control_training_rule"], 947, 351,
         size=17, color=SECONDARY)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--variant", choices=("v1", "v2"), default="v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    output = args.output or OUTPUT_DIRECTORY / f"Fig1_PanelC_SessionProtocol_exploratory_{args.variant}.svg"
    root = draw(specification, show_viability_check=args.variant == "v1")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
