"""Prepare an all-fish fixed-trace pipeline config from the raw folder.

This command reads raw files and writes only to the repository's config folder.
Run it again immediately before starting the pipelines if raw files have changed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


SUFFIXES = {
    "camera": "_cam.txt",
    "tracking": "_mp tail tracking.txt",
    "protocol": "_stim control.txt",
}
RECORDING_ID = re.compile(r"^\d{8}_\d{2}$")


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.iterdir()
        if path.is_file()
    }


def training_latencies(path: Path) -> list[float | None]:
    cycles: list[int] = []
    reinforcers: list[int] = []
    with path.open(encoding="utf-8") as source:
        next(source)
        for line in source:
            fields = line.split()
            if len(fields) < 3:
                continue
            if fields[0] == "Cycle":
                cycles.append(int(Decimal(fields[1])))
            elif fields[0] == "Reinforcer":
                reinforcers.append(int(Decimal(fields[1])))
    if len(cycles) != 94 or len(reinforcers) != 79:
        raise ValueError(f"Unexpected protocol event counts: {path.name}")
    latencies: list[float | None] = []
    for index in range(14, 64):
        next_reinforcer = next((time for time in reinforcers if time >= cycles[index]), None)
        if next_reinforcer is None:
            raise ValueError(f"No subsequent reinforcer: {path.name}")
        latency = (next_reinforcer - cycles[index]) / 1000
        latencies.append(round(latency, 3) if latency <= 20 else None)
    if sum(value is not None for value in latencies) != 46:
        raise ValueError(f"Expected 46 reinforced training trials: {path.name}")
    return latencies


def schedule(latencies: list[float | None]) -> str:
    reinforced = [value for value in latencies if value is not None]
    if all(12.8 <= value <= 13.2 for value in reinforced):
        return "fixed"
    first_ten = reinforced[:10]
    middle = reinforced[10:-10]
    last_ten = reinforced[-10:]
    if (
        all(10.3 <= value <= 10.7 for value in first_ten)
        and all(12.8 <= value <= 13.2 for value in last_ten)
        and min(middle) > max(first_ten)
        and max(middle) < min(last_ten) + 0.1
        and max(left - right for left, right in zip(reinforced, reinforced[1:])) < 0.1
    ):
        return "increasing"
    return "unknown"


def pipeline_config(raw: Path, output: Path, experiment: str, ids: list[str]) -> dict:
    return {
        "raw_dir": str(raw),
        "save_dir": str(output),
        "experiment": experiment,
        "analysis_id": f"{experiment}-full-v1",
        "routes": ["candidate"],
        "recording_ids": sorted(ids),
        "candidate_runner_recipe": "candidate-corrected-runner-v1",
        "run_inventory": True,
        "run_intake": True,
        "run_figures": True,
        "figure_outcomes": [
            "total-activity", "movement-probability", "fraction-time-moving",
            "conditional-intensity", "bout-rate",
        ],
        "overwrite": False,
        "continue_on_error": False,
    }


def prepare(raw: Path, digested: Path, config_dir: Path) -> dict:
    raw = raw.resolve()
    digested = digested.resolve()
    if not raw.is_dir() or not digested.is_dir():
        raise ValueError("Raw and digested directories must already exist")
    before = snapshot(raw)
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    ignored = []
    for name in sorted(before):
        kind = next((kind for kind, suffix in SUFFIXES.items() if name.endswith(suffix)), None)
        if kind is None:
            ignored.append(name)
            continue
        stem = name[: -len(SUFFIXES[kind])]
        grouped[stem][kind] = raw / name

    selected: dict[str, list[str]] = {"fixed": [], "increasing": [], "control": []}
    incomplete = []
    classified = []
    seen_ids: set[str] = set()
    for stem, parts in sorted(grouped.items()):
        fields = stem.split("_")
        recording_id = "_".join(fields[:2])
        if not RECORDING_ID.fullmatch(recording_id) or recording_id in seen_ids:
            raise ValueError(f"Invalid or colliding recording ID: {stem}")
        seen_ids.add(recording_id)
        missing = sorted(set(SUFFIXES) - set(parts))
        if missing:
            incomplete.append({"recording_id": recording_id, "condition": fields[2], "missing": missing})
            continue
        condition = fields[2].lower()
        if condition == "control":
            selected["control"].append(recording_id)
            continue
        if condition != "trace":
            raise ValueError(f"Unexpected complete condition {condition!r}: {recording_id}")
        latencies = training_latencies(parts["protocol"])
        kind = schedule(latencies)
        if kind == "unknown":
            raise ValueError(f"Unrecognized trace schedule for {recording_id}")
        selected[kind].append(recording_id)
        reinforced = [value for value in latencies if value is not None]
        classified.append({
            "recording_id": recording_id,
            "schedule": kind,
            "reinforced_training_trials": len(reinforced),
            "catch_training_trials": latencies.count(None),
            "latency_range_s": [min(reinforced), max(reinforced)],
        })

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "all3sTrace-full-windows.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw),
        "raw_file_count": len(before),
        "fixed_trace_count": len(selected["fixed"]),
        "matched_control_count": len(selected["control"]),
        "incomplete": incomplete,
        "ignored_files": ignored,
        "classified_trace": classified,
        "config_path": str(config_path),
    }
    changed = snapshot(raw) != before
    blockers = []
    if changed:
        blockers.append("Raw folder changed during preparation")
    if selected["increasing"]:
        blockers.append(f"Increasing trace fish remain in fixed-trace folder: {selected['increasing']}")
    if incomplete:
        blockers.append(f"Incomplete recordings: {[item['recording_id'] for item in incomplete]}")
    if not selected["fixed"]:
        blockers.append("No complete fixed-trace recordings found")
    report["status"] = "blocked" if blockers else "ready"
    report["blockers"] = blockers
    (config_dir / "trace-preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if blockers:
        # This config is generated by this script. Remove a stale selection so
        # it cannot be launched directly while raw recordings are still arriving.
        config_path.unlink(missing_ok=True)
        raise RuntimeError("Trace run is not ready: " + "; ".join(blockers))
    config = pipeline_config(raw, digested / "all3sTrace-full-v1", "all3sTrace", selected["fixed"] + selected["control"])
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"J:\Raw Data\all3sTtrace"))
    parser.add_argument("--digested-dir", type=Path, default=Path(r"J:\Digested Data"))
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    args = parser.parse_args()
    result = prepare(args.raw_dir, args.digested_dir, args.config_dir)
    print(json.dumps({key: result[key] for key in ("raw_file_count", "fixed_trace_count", "matched_control_count", "incomplete", "config_path")}, indent=2))
