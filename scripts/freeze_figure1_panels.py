"""Snapshot the user-approved Figure 1 A-D selection and pre-CS baseline."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from build_figure1_trace_panels import draw, digest, OUTPUT

REPO = Path(__file__).resolve().parents[1]


def main():
    layout_path = REPO / "configs/paper-figures/figure1-assembly.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    root = Path(layout["storage_root"])
    target = root / "frozen/2026-10-05-ABCD"
    manifest_path = target / "freeze.json"
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in frozen["panels"]:
            if digest(root / item["source"]) != item["sha256"]:
                raise ValueError("Frozen artwork changed")
        print("Existing freeze verified:", manifest_path)
        return
    target.mkdir(parents=True, exist_ok=True)
    old_meta = json.loads((OUTPUT / "Fig1_PanelD_TailAngle_legacy-vigor_v1.svg.json").read_text(encoding="utf-8"))
    frames_path, events_path = Path(old_meta["frames"]), Path(old_meta["events"])
    assert digest(frames_path) == old_meta["frames_sha256"]
    assert digest(events_path) == old_meta["events_sha256"]
    frames, events = pd.read_parquet(frames_path), pd.read_parquet(events_path)
    offsets = {}
    for trial, part in frames.groupby("Trial number"):
        before = part.loc[part["Time relative to CS onset (s)"].ge(-15)
                          & part["Time relative to CS onset (s)"].lt(0), "Tail angle (rad)"]
        median = float(before.median())
        frames.loc[part.index, "Tail angle (rad)"] -= median
        offsets[int(trial)] = median * 180 / np.pi
    frozen_data = target / "Fig1_PanelD_TailAngle_pre15_frames.parquet"
    frames.to_parquet(frozen_data, index=False)
    frozen_events = target / "Fig1_PanelD_events.parquet"
    shutil.copy2(events_path, frozen_events)
    records = []
    for panel in layout["panels"]:
        if panel["id"] not in "ABCD" or len(panel["id"]) != 1:
            continue
        original = root / panel["source"]
        destination = target / (f"Fig1_Panel{panel['id']}_frozen.svg")
        if panel["id"] == "D":
            draw(frames, events, panel="D", output=destination, baseline_s=(-15,0))
            metadata = {**old_meta,"baseline_s":[-15,0],"baseline_interval":"[-15,0)",
                        "angle_reference_shift_deg":offsets,
                        "frames":str(frozen_data),"frames_sha256":digest(frozen_data),
                        "events":str(frozen_events),"events_sha256":digest(frozen_events),
                        "svg":str(destination),"svg_sha256":digest(destination)}
            destination.with_suffix(".svg.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
        else:
            shutil.copy2(original,destination)
        record={"id":panel["id"],"role":panel["role"],"source":destination.relative_to(root).as_posix(),
                "sha256":digest(destination),"previous_source":str(original),"previous_sha256":digest(original)}
        records.append(record)
        panel.update(source=record["source"],frozen_sha256=record["sha256"],selection_status="frozen")
    freeze={"figure":"Figure 1","approved_panels":["A","B","C","D"],
            "baseline_s":[-15,0],"baseline_interval":"[-15,0) relative to CS onset",
            "font":"DejaVu Sans","panels":records,
            "pending_panels":{"E":"raw vigor and exact heatmap-bin inset design","F-H":"single-fish heatmap review deferred"},
            "lettering":"A setup; B contingencies; C session; D tail angle; E raw vigor; F/G/H heatmaps",
            "prior_thread":"01a0d967-3f9c-7bc3-b97b-0451f556de3d uses historical lettering and separate scaled-log alternatives"}
    manifest_path.write_text(json.dumps(freeze,indent=2)+"\n",encoding="utf-8")
    layout.update(baseline_s=[-15,0],frozen_baseline_s=[-15,0],freeze_manifest=str(manifest_path))
    layout_path.write_text(json.dumps(layout,indent=2)+"\n",encoding="utf-8")
    (REPO / "configs/paper-figures/figure1-freeze.json").write_text(json.dumps(freeze,indent=2)+"\n",encoding="utf-8")
    print(manifest_path)
    print("D angle reference shifts in degrees:",offsets)


if __name__ == "__main__":
    main()
