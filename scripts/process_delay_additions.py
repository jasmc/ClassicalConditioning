"""Process an audited list of missing Delay fish without replacing existing files.

Uses the supported individual processing stages so shared inventory, intake
ledgers, and earlier cohort outputs in an existing project remain intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from classical_conditioning.intake import intake_recording
from classical_conditioning.preprocessing.corrected_frame_preprocessing import build_corrected_preprocessing
from classical_conditioning.preprocessing.candidate_metrics_from_corrected_frames import build_candidate_activity_metrics_from_corrected
from classical_conditioning.analysis.movement_state import build_candidate_movement_state
from classical_conditioning.analysis.temporal_profiles import build_candidate_temporal_profiles
from classical_conditioning.analysis.trial_outcomes import build_candidate_trial_outcomes


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.discovery.read_text(encoding="utf-8"))
    project = Path(report["current_project"]).resolve()
    if project != Path(r"F:\Digested Data\allDelay-full-v1").resolve():
        raise ValueError("This run is restricted to the author's existing Delay project")
    prefix = project / "Metadata" / args.run_id
    rows = report["fish"]
    current = json.loads((project / "Metadata/recording_inventory.json").read_text())
    old_records = {r["recording_id"]: r for r in current["records"]}
    targets = []
    table = []
    for row in rows:
        rid = row["recording_id"]
        paths = [Path(p) for p in row["raw_files"]]
        protocols = row["protocols"]
        issues = []
        notes = []
        if len(paths) != 3 or len(protocols) != 1:
            issues.append("raw triplet is missing or ambiguous")
        if len(protocols) == 1:
            protocol = protocols[0]
            if protocol.get("counts") != {"Cycle": 94, "Reinforcer": 79}:
                issues.append("unexpected protocol event counts")
            durations = protocol.get("cs_duration_s_range", [])
            if len(durations) != 2 or not (9.8 <= durations[0] <= durations[1] <= 10.2):
                issues.append("CS duration is not approximately 10 seconds")
            if (row["condition"] == "delay" and protocol.get("near9") != 46) or (row["condition"] == "control" and protocol.get("near9", 0) > 5):
                issues.append("condition and observed 9-second pairing count disagree")
            if row["condition"] == "control" and 0 < protocol.get("near9", 0) <= 5:
                notes.append("isolated near-9s coincidence in otherwise unpaired control: " + str(protocol["near9"]))
            if row["condition"] == "delay" and protocol.get("within20") != 46:
                issues.append("unexpected additional training reinforcers within 20 seconds")
        for path in paths:
            if path.drive.lower() != "d:":
                issues.append("raw input is not on D:")
            raw_condition = path.name.split("_")[2].lower()
            normalized_condition = {"delaynomk801": "delay", "controlnomk801": "control"}.get(raw_condition, raw_condition)
            if normalized_condition != row["condition"]:
                issues.append("raw filename condition differs from pickle")
            if raw_condition != normalized_condition:
                notes.append("raw NoMK801 condition suffix absent from historical pickle; protocol hash matches current inventory")
            suffix = next((s for s in ("_cam.txt", "_mp tail tracking.txt", "_stim control.txt") if path.name.endswith(s)), None)
            normalized_name = path.name[:-len(suffix)].lower().replace("_delaynomk801_", "_delay_").replace("_controlnomk801_", "_control_") if suffix else ""
            if suffix is None or normalized_name != Path(row["pkl_name"]).stem.lower():
                issues.append("raw recording name differs from pickle")
        protocol_hash = digest(Path(protocols[0]["path"])) if len(protocols) == 1 else ""
        if rid in old_records:
            old_hashes = {p["sha256"] for p in old_records[rid]["components"]["protocol"]}
            if protocol_hash not in old_hashes:
                issues.append("D: protocol differs from current analysis source")
        is_new = not row["current_trial_marker"]
        if is_new:
            # Reject partial existing processing rather than replacing any artifact.
            existing = [project / "Processed data" / rid, project / "Quality checks" / rid]
            existing += list((project / "Metadata").glob(f"{rid}_*"))
            if any(p.exists() for p in existing):
                issues.append("partial processing already exists; preserve it for reconciliation")
            if not issues:
                targets.append(row)
        table.append({
            "recording_id": rid, "condition": row["condition"],
            "processing": "missing" if is_new else "existing",
            "protocol_check": "PASS" if not issues else "REVIEW",
            "issues": "; ".join(issues), "pkl_path": row["pkl_path"],
            "notes": "; ".join(sorted(set(notes))),
            "protocol_path": protocols[0]["path"] if len(protocols) == 1 else "",
            "protocol_sha256": protocol_hash,
            "training_pairings_at_9s": protocols[0].get("near9") if len(protocols) == 1 else "",
            "raw_files": " | ".join(map(str, paths)),
        })
    with prefix.with_suffix(".csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with prefix.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("# Delay fish inventory and protocol audit\n\n")
        stream.write(f"All {len(rows)} pickle-identified fish, including all subfolders. "
                     f"Existing processing: {sum(r['processing'] == 'existing' for r in table)}; "
                     f"new fish ready: {len(targets)}.\n\n")
        stream.write("Protocol checks compare full raw/pickle identity, condition, 94 CS and 79 US events, "
                     "approximately 10 s CS, and the 50 training-cycle next-US latencies. "
                     "Delay requires 46 pairings within 9 ± 0.2 s; controls are checked for systematic pairing; "
                     "isolated coincidences (up to five) are retained as notes. Raw NoMK801 suffixes map explicitly "
                     "to the historical pickle's control/delay label, with all remaining name fields unchanged. "
                     "Current fish also require identical protocol SHA-256 to the existing inventory. "
                     "This audit does not decide scientific exclusion.\n\n")
        stream.write("| Fish | Condition | Processing | Protocol check | Issues | Notes |\n|---|---|---|---|---|---|\n")
        for row in table:
            stream.write("| " + " | ".join(str(row[k]) for k in ("recording_id", "condition", "processing", "protocol_check", "issues", "notes")) + " |\n")
    print(json.dumps({"fish": len(rows), "conditions": dict(Counter(r["condition"] for r in rows)),
                      "new": len(targets), "new_conditions": dict(Counter(r["condition"] for r in targets)),
                      "review": [r for r in table if r["issues"]]}, indent=2), flush=True)
    if args.audit_only:
        return
    if any(r["issues"] and r["processing"] == "missing" for r in table):
        raise ValueError("A new fish has unresolved issues; no processing launched")

    baseline = {p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in project.rglob("*") if p.is_file()}
    protected_root = Path(report["pkl_root"]).resolve()

    def protect(event: str, values: tuple) -> None:
        candidates = []
        if event == "open" and isinstance(values[0], (str, bytes, os.PathLike)):
            flags = values[2] if len(values) > 2 else 0
            mode = values[1] or ""
            if any(c in mode for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                candidates = [values[0]]
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime"}:
            candidates = [values[0]]
        elif event == "os.rename":
            candidates = [values[0], values[1]]
        for value in candidates:
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value)).resolve()
                if path.drive.lower() == "d:" or path == protected_root or protected_root in path.parents or path in baseline:
                    raise PermissionError(f"Preservation guard blocked {event}: {path}")

    sys.addaudithook(protect)
    results = []
    started = time.time()
    for i, row in enumerate(targets, 1):
        rid = row["recording_id"]
        result = {"recording_id": rid, "condition": row["condition"], "stages": []}
        print(f"START {rid} [{i}/{len(targets)}]", flush=True)
        try:
            intake_recording(Path(row["protocols"][0]["path"]).parent, project,
                             recording_id=rid, overwrite=False)
            result["stages"].append("intake")
            for name, function, kwargs in (
                ("corrected-preprocess", build_corrected_preprocessing, {}),
                ("candidate-metrics", build_candidate_activity_metrics_from_corrected, {}),
                ("movement-state", build_candidate_movement_state, {"metric_recipe": "tail-candidate-corrected"}),
                ("temporal-profiles", build_candidate_temporal_profiles, {"metric_recipe": "tail-candidate-corrected", "experiment_name": "allDelay"}),
                ("trial-outcomes", build_candidate_trial_outcomes, {"metric_recipe": "tail-candidate-corrected", "experiment_name": "allDelay"}),
            ):
                print(f"STAGE {rid} {name}", flush=True)
                function(project, rid, overwrite=False, **kwargs)
                result["stages"].append(name)
            result["status"] = "completed"
        except Exception as error:
            import traceback
            traceback.print_exc()
            result.update(status="failed", error=str(error))
        results.append(result)
        with Path(str(prefix) + "-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        print(f"DONE {rid} {result['status']} elapsed={time.time()-started:.1f}s", flush=True)
    changed = [str(p) for p, stat in baseline.items()
               if not p.exists() or (p.stat().st_size, p.stat().st_mtime_ns) != stat]
    summary = {"results": results, "existing_files_changed": changed, "elapsed_seconds": time.time()-started}
    with Path(str(prefix) + "-completion.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if changed or any(r["status"] != "completed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
