"""Classify every incTrace protocol using only Cycle and Reinforcer event times."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from statistics import median


def protocol_evidence(path: Path) -> dict:
    cycles: list[int] = []
    reinforcers: list[int] = []
    with path.open(encoding="utf-8") as stream:
        header = next(stream).split()
        if header != ["Type", "Beg", "End"]:
            raise ValueError(f"Unexpected protocol header: {path.name}: {header}")
        for line in stream:
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Malformed protocol event in {path.name}: {line!r}")
            if fields[0] == "Cycle":
                cycles.append(int(Decimal(fields[1])))
            elif fields[0] == "Reinforcer":
                reinforcers.append(int(Decimal(fields[1])))
            else:
                raise ValueError(f"Unknown protocol event in {path.name}: {fields[0]}")

    if len(cycles) != 94 or len(reinforcers) != 79:
        raise ValueError(f"Unexpected event counts in {path.name}: {len(cycles)} cycles, {len(reinforcers)} reinforcers")
    training_latencies: list[float | None] = []
    for index in range(14, 64):
        next_reinforcer = next((time for time in reinforcers if time >= cycles[index]), None)
        latency = (next_reinforcer - cycles[index]) / 1000 if next_reinforcer is not None else None
        training_latencies.append(round(latency, 3) if latency is not None and latency <= 20 else None)
    reinforced = [value for value in training_latencies if value is not None]
    if len(reinforced) != 46:
        raise ValueError(f"Expected 46 reinforced training cycles in {path.name}; found {len(reinforced)}")
    first_ten = reinforced[:10]
    middle = reinforced[10:-10]
    last_ten = reinforced[-10:]
    drops = [left - right for left, right in zip(reinforced, reinforced[1:])]
    increasing = (
        all(10.3 <= value <= 10.7 for value in first_ten)
        and all(12.8 <= value <= 13.2 for value in last_ten)
        and min(middle) > max(first_ten)
        and max(middle) < min(last_ten) + 0.1
        and max(drops) < 0.1
        and len({round(value, 1) for value in reinforced}) >= 20
    )
    return {
        "protocol_file": path.name,
        "stim_control_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cycle_count": len(cycles),
        "reinforcer_count": len(reinforcers),
        "training_reinforced_count": len(reinforced),
        "training_catch_count": training_latencies.count(None),
        "first_ten_median_s": median(first_ten),
        "middle_min_s": min(middle),
        "middle_max_s": max(middle),
        "last_ten_median_s": median(last_ten),
        "largest_downward_step_s": max(drops),
        "classification": "increasing" if increasing else "not_increasing",
        "training_latency_s": training_latencies,
    }


def audit(root: Path) -> dict:
    protocols = sorted(root.glob("*_stim control.txt"))
    if not protocols:
        raise ValueError(f"No stim control files found in {root}")
    records = [protocol_evidence(path) for path in protocols]
    return {
        "source_dir": str(root.resolve()),
        "basis": "Stim control Cycle and Reinforcer timestamps only; 50 training cycles at indices 14-63.",
        "protocol_count": len(records),
        "all_increasing": all(record["classification"] == "increasing" for record in records),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"J:\Raw Data\incTrace"))
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "incTrace-stim-audit.json")
    arguments = parser.parse_args()
    result = audit(arguments.raw_dir)
    arguments.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for record in result["records"]:
        print(f"{record['protocol_file'][:11]}: {record['classification']} "
              f"{record['first_ten_median_s']:.3f} -> "
              f"{record['middle_min_s']:.3f}..{record['middle_max_s']:.3f} -> "
              f"{record['last_ten_median_s']:.3f} s")
    print(f"All {result['protocol_count']} stim control files increasing: {result['all_increasing']}")
    if not result["all_increasing"]:
        raise SystemExit(1)
